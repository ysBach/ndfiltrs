"""Compare exact filter methods with prior calls and external libraries.

Run only with a release extension and no concurrent CPU-heavy work.
Full calls include validation and allocation. Astro-SCRAPPY copies borders;
its exact-median comparison is restricted to matching float32 interiors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import astroscrappy
import ndfiltrs as nf
import numpy as np
import scipy
from astroscrappy.utils import median_utils
from ndfiltrs import _api, _core, _validation
from numpy.typing import NDArray
from scipy import ndimage

from benchmarks._common import load_baseline, make_input

FloatArray = NDArray[np.float32] | NDArray[np.float64]


def prior_candidate(
    package: ModuleType, array: FloatArray, size: int, kind: str
) -> FloatArray:
    """Time the old native candidates with their original argument validation."""
    arguments = package._validation.validate_filter_arguments(
        array, size, "reflect", 0.0, min_count=1
    )
    return package._api._window_filter(kind, arguments)


def measure(
    runners: dict[str, Callable[[], FloatArray]],
    array: FloatArray,
    operation: str,
    size: int,
    repeats: int,
    preflight_only: bool,
) -> dict[str, Any]:
    """Check matching results and ownership, then rotate full-call timing order.

    Notes
    -----
    Median parity is bitwise over complete outputs, except Astro-SCRAPPY's
    copied borders. Boxcar tolerance follows the existing SciPy benchmark's
    64-epsilon input-scale allowance; it is a finite-input smoke check, not
    proof of exact rounding. Fraction-oracle tests establish rolling accuracy.
    """
    original = array.copy()
    bits = np.uint32 if array.dtype == np.dtype(np.float32) else np.uint64
    reference = runners["current_direct"]()
    np.testing.assert_array_equal(array.view(bits), original.view(bits))
    tolerance = (
        0.0
        if operation == "median"
        else float(64 * np.finfo(array.dtype).eps * np.max(np.abs(array)))
    )
    validation = {}
    for name, runner in runners.items():
        actual = runner()
        if actual.shape != array.shape or actual.dtype != array.dtype:
            raise AssertionError(f"{name}: output shape or dtype mismatch")
        if np.shares_memory(actual, array):
            raise AssertionError(f"{name}: output aliases input")
        compared, expected = actual, reference
        if name == "astroscrappy_exact_interior_only":
            radius = size // 2
            interior = (slice(radius, -radius),) * array.ndim
            compared, expected = actual[interior], reference[interior]
            border = np.ones(array.shape, dtype=bool)
            border[interior] = False
            np.testing.assert_array_equal(
                actual[border].view(bits), array[border].view(bits)
            )
        if operation == "median":
            np.testing.assert_array_equal(
                compared.view(bits), expected.view(bits), err_msg=name
            )
        else:
            np.testing.assert_allclose(
                compared,
                expected,
                rtol=0,
                atol=tolerance,
                equal_nan=False,
                err_msg=name,
            )
        validation[name] = {
            "parity": "bitwise" if operation == "median" else "within_tolerance",
            "comparison_region": "valid_interior"
            if name == "astroscrappy_exact_interior_only"
            else "complete_image",
            "compared_pixels": int(expected.size),
            "max_absolute_error": float(
                np.max(
                    np.abs(compared.astype(np.float64) - expected.astype(np.float64))
                )
            ),
        }
        np.testing.assert_array_equal(array.view(bits), original.view(bits))
    samples: dict[str, list[float]] = {name: [] for name in runners}
    orders = []
    if not preflight_only:
        names = list(runners)
        for repeat in range(repeats):
            shift = repeat % len(names)
            order = names[shift:] + names[:shift]
            orders.append(order)
            for name in order:
                start = time.perf_counter()
                result = runners[name]()
                samples[name].append(time.perf_counter() - start)
                del result
            np.testing.assert_array_equal(array.view(bits), original.view(bits))
    return {
        "absolute_tolerance": tolerance,
        "relative_tolerance": 0.0,
        "timing_order": orders,
        "runners": {
            name: {
                **validation[name],
                "samples_seconds": values,
                "median_seconds": statistics.median(values) if values else None,
            }
            for name, values in samples.items()
        },
    }


def benchmark(
    *,
    quick: bool,
    repeats: int,
    preflight_only: bool,
    baseline: ModuleType | None,
) -> dict[str, Any]:
    """Compare fixed small/large windows across both dtypes and three ranks."""
    boxcars = (
        ((128,) if quick else (16384,), (31, 127)),
        ((32, 32) if quick else (256, 256), (15, 31, 63)),
        ((4, 5, 6) if quick else (16, 20, 24), (7,)),
    )
    medians = (
        ((64,) if quick else (1024,), (15, 31)),
        ((32, 32) if quick else (256, 256), (3, 5, 7, 15, 31)),
        ((4, 5, 6) if quick else (16, 20, 24), (3, 7)),
    )
    records = []
    for operation, cases in (("median", medians), ("boxcar", boxcars)):
        for shape, sizes in cases:
            for dtype in (np.float32, np.float64):
                array = make_input(shape, dtype)
                for size in sizes:
                    public_name = (
                        "median_filter" if operation == "median" else "mean_filter"
                    )
                    kind = "median_network" if operation == "median" else "mean_rolling"
                    method = "sortnet" if operation == "median" else "boxcar"
                    runners = {
                        "current_direct": partial(
                            getattr(nf, public_name), array, size, mode="reflect"
                        ),
                        f"current_{method}": partial(
                            getattr(nf, public_name), array, size, method=method
                        ),
                        "scipy": partial(
                            ndimage.median_filter
                            if operation == "median"
                            else ndimage.uniform_filter,
                            array,
                            size=size,
                            mode="reflect",
                        ),
                    }
                    if operation == "boxcar":
                        runners["current_separable"] = partial(
                            nf.mean_filter,
                            array,
                            size,
                            mode="reflect",
                            method="separable",
                        )
                    if baseline is not None:
                        runners[f"prior_{method}"] = partial(
                            prior_candidate, baseline, array, size, kind
                        )
                        runners["prior_direct"] = partial(
                            getattr(baseline, public_name), array, size, mode="reflect"
                        )
                        if operation == "boxcar":
                            runners["prior_separable"] = partial(
                                baseline.mean_filter,
                                array,
                                size,
                                mode="reflect",
                                method="separable",
                            )
                    if (
                        operation == "median"
                        and array.ndim == 2
                        and dtype is np.float32
                        and size in (3, 5, 7)
                    ):
                        runners["astroscrappy_exact_interior_only"] = partial(
                            getattr(median_utils, f"medfilt{size}"), array
                        )
                    records.append(
                        {
                            "operation": operation,
                            "shape": list(shape),
                            "size": size,
                            "dtype": np.dtype(dtype).name,
                            "mode": "reflect",
                            "origin": 0,
                            **measure(
                                runners, array, operation, size, repeats, preflight_only
                            ),
                        }
                    )
    root = Path(__file__).resolve().parents[1]
    paths = {
        "benchmark": Path(__file__),
        "common": root / "benchmarks/_common.py",
        "current_extension": Path(_core.__file__),
        "current_api": Path(_api.__file__),
        "current_validation": Path(_validation.__file__),
        "astroscrappy_extension": Path(median_utils.__file__),
        "scipy_filters": Path(ndimage._filters.__file__),
        "scipy_extension": Path(ndimage._nd_image.__file__),
    }
    for name in (
        "median_network",
        "window",
        "filter_nd",
        "boxcar",
        "exact_sum",
        "mean",
        "separable",
    ):
        paths[f"checkout_{name}"] = root / f"src/{name}.rs"
    if baseline is not None:
        paths["baseline_extension"] = Path(baseline._core.__file__)
        paths["baseline_api"] = Path(baseline.__file__).with_name("_api.py")
    return {
        "scope": __doc__,
        "quick": quick,
        "preflight_only": preflight_only,
        "repeats": 0 if preflight_only else repeats,
        "warmup_calls_per_runner": 1,
        "extra_reference_calls_per_case": 1,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "versions": {
            "ndfiltrs": nf.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "astroscrappy": astroscrappy.__version__,
            "baseline": baseline.__version__ if baseline else None,
        },
        "rayon_threads": nf.get_num_threads(),
        "baseline_rayon_threads": baseline.get_num_threads() if baseline else None,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "RAYON_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OMP_DYNAMIC",
                "OMP_THREAD_LIMIT",
                "OMP_PROC_BIND",
                "OMP_PLACES",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "actual_openmp_team_size": "unknown; environment records requested policy",
        "source_note": "checkout hashes do not establish loaded-extension provenance",
        "sources": {
            name: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
        "records": records,
    }


def main() -> None:
    """Write a report; preflight-only mode collects no timing samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, help="prior installed ndfiltrs package directory"
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    report = benchmark(
        quick=args.quick,
        repeats=args.repeats,
        preflight_only=args.preflight_only,
        baseline=load_baseline(args.baseline) if args.baseline is not None else None,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logging.basicConfig(level=logging.INFO)
    logging.info("Saved candidate benchmark to %s", args.output)


if __name__ == "__main__":
    main()
