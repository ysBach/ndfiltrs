"""Compare exact median interiors; report separable approximations separately.

Only finite C-contiguous 2-D float32 inputs have matching native contracts.
Timings cover full calls: Astro-SCRAPPY copies its borders, while ndfiltrs
filters them. No full-image Astro-SCRAPPY equivalence or float64 speed claim
is implied. An optional prior ndfiltrs installation must match current
ndfiltrs bitwise over the complete image, including reflected borders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import astroscrappy
import ndfiltrs as nf
import numpy as np
from astroscrappy.utils import median_utils
from ndfiltrs import _core
from numpy.typing import NDArray

from benchmarks._common import load_baseline, make_input

FloatArray = NDArray[np.float32]
SOURCE_ROOT = "https://github.com/astropy/astroscrappy/blob/v1.3.0/"


def benchmark(
    *,
    quick: bool,
    repeats: int,
    preflight_only: bool,
    baseline: ModuleType | None = None,
) -> dict[str, Any]:
    """Check exact interiors and approximate errors, then optionally time calls.

    Parameters
    ----------
    quick : bool
        Use one 32 by 32 image instead of 128 and 256 square images.
    repeats : int
        Timed calls per implementation, after one parity/warmup call.
    preflight_only : bool
        Check outputs without reading a timer or collecting timing samples.
    baseline : types.ModuleType or None, optional
        Prior installed ndfiltrs package loaded with ``load_baseline()``.
        Compare complete outputs bitwise before timing and recheck after timing.

    Returns
    -------
    dict
        Separate exact and approximate records, with environment and identities.
    """
    exact_records = []
    approximate_records = []
    for side in (32,) if quick else (128, 256):
        array = make_input((side, side), np.float32)
        original = array.copy()
        for size in (3, 5, 7):
            radius = size // 2
            interior = (slice(radius, -radius),) * 2
            windows = np.lib.stride_tricks.sliding_window_view(array, (size, size))
            reference = np.median(windows, axis=(-2, -1))
            # Independent row-then-column definition of the approximation.
            row_windows = np.lib.stride_tricks.sliding_window_view(array, size, axis=1)
            row_medians = np.median(row_windows, axis=-1)
            column_windows = np.lib.stride_tricks.sliding_window_view(
                row_medians, size, axis=0
            )
            separable_reference = np.median(column_windows, axis=-1)
            runners: dict[str, Callable[[], FloatArray]] = {
                "ndfiltrs_exact": partial(
                    nf.median_filter, array, size, mode="reflect"
                ),
                "astroscrappy_exact": partial(
                    getattr(median_utils, f"medfilt{size}"), array
                ),
                "astroscrappy_separable_approximate": partial(
                    getattr(median_utils, f"sepmedfilt{size}"), array
                ),
            }
            if baseline is not None:
                runners["ndfiltrs_previous_exact"] = partial(
                    baseline.median_filter, array, size, mode="reflect"
                )
            full_results = {}
            errors = {}
            border = np.ones(array.shape, dtype=bool)
            border[interior] = False
            for name, runner in runners.items():
                result = runner()
                if result.shape != array.shape or result.dtype != array.dtype:
                    raise AssertionError(f"{name}: shape or dtype mismatch")
                if np.shares_memory(result, array):
                    raise AssertionError(f"{name}: output aliases input")
                expected = separable_reference if "approximate" in name else reference
                # Bit parity is meaningful here: odd finite windows select a sample.
                np.testing.assert_array_equal(
                    result[interior].view(np.uint32),
                    expected.view(np.uint32),
                    err_msg=f"{name}, size={size}, side={side}",
                )
                if name.startswith("astroscrappy"):
                    np.testing.assert_array_equal(result[border], array[border])
                elif baseline is not None:
                    full_results[name] = result
                difference = result[interior].astype(np.float64) - reference
                errors[name] = {
                    "different_pixels": int(np.count_nonzero(difference)),
                    "max_absolute_error": float(np.max(np.abs(difference))),
                    "rmse": float(np.sqrt(np.mean(difference**2))),
                }
                np.testing.assert_array_equal(array, original)
            if baseline is not None:
                np.testing.assert_array_equal(
                    full_results["ndfiltrs_exact"].view(np.uint32),
                    full_results["ndfiltrs_previous_exact"].view(np.uint32),
                    err_msg=f"Full-image baseline mismatch, size={size}, side={side}",
                )
            samples: dict[str, list[float]] = {name: [] for name in runners}
            if not preflight_only:
                # Rotate positions so the approximate runner is not always last.
                names = list(runners)
                for repeat in range(repeats):
                    shift = repeat % len(names)
                    for name in names[shift:] + names[:shift]:
                        start = time.perf_counter()
                        result = runners[name]()
                        samples[name].append(time.perf_counter() - start)
                        del result
                np.testing.assert_array_equal(array, original)
                # These untimed calls must still match the preflight snapshots.
                for name, expected_full in full_results.items():
                    result = runners[name]()
                    if result.shape != array.shape or result.dtype != array.dtype:
                        raise AssertionError(
                            f"{name}: postflight shape or dtype mismatch"
                        )
                    np.testing.assert_array_equal(
                        result.view(np.uint32),
                        expected_full.view(np.uint32),
                        err_msg=f"{name}: full-image postflight mismatch",
                    )
                    np.testing.assert_array_equal(array, original)
            for name in runners:
                approximate = "approximate" in name
                record = {
                    "implementation": name,
                    "shape": list(array.shape),
                    "dtype": "float32",
                    "size": size,
                    "excluded_border_width": radius,
                    "interior_pixels": int(reference.size),
                    "equivalent_to_true_median_on_interior": not approximate,
                    "parity_preflight": "passed",
                    "full_image_baseline_bitwise_parity": "passed"
                    if name in full_results
                    else "not checked",
                    "full_image_postflight": "passed"
                    if name in full_results and not preflight_only
                    else "not checked",
                    "parity_reference": "row_then_column"
                    if approximate
                    else "true_median",
                    "error_vs_true_median": errors[name],
                    "samples_seconds": samples[name],
                    "median_seconds": statistics.median(samples[name])
                    if samples[name]
                    else None,
                }
                (approximate_records if approximate else exact_records).append(record)
    paths = {
        "benchmark": Path(__file__),
        "common": Path(__file__).with_name("_common.py"),
        "ndfiltrs_extension": Path(_core.__file__),
        "ndfiltrs_api": Path(nf.__file__).with_name("_api.py"),
        "astroscrappy_extension": Path(median_utils.__file__),
    }
    if baseline is not None:
        paths["baseline_extension"] = Path(baseline._core.__file__)
        paths["baseline_api"] = Path(baseline.__file__).with_name("_api.py")
    return {
        "scope": __doc__,
        "preflight_only": preflight_only,
        "quick": quick,
        "repeats": 0 if preflight_only else repeats,
        "warmup_calls_per_runner": 1,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "ndfiltrs": nf.__version__,
        "baseline_version": baseline.__version__ if baseline is not None else None,
        "baseline_rayon_threads": baseline.get_num_threads()
        if baseline is not None
        else None,
        "astroscrappy": astroscrappy.__version__,
        "rayon_threads": nf.get_num_threads(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "RAYON_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OMP_DYNAMIC",
                "OMP_THREAD_LIMIT",
                "OMP_PROC_BIND",
                "OMP_PLACES",
            )
        },
        "astroscrappy_actual_openmp_threads": "unknown; requested policy only",
        "sources": {
            name: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
        "reviewed_official_sources": [
            SOURCE_ROOT + "astroscrappy/utils/median_utils.pyx",
            SOURCE_ROOT + "astroscrappy/utils/medutils.c",
        ],
        "exact_records": exact_records,
        "approximate_records_not_equivalent": approximate_records,
    }


def main() -> None:
    """Write a JSON report after parity checks and optional timing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="prior installed ndfiltrs package directory; full-image bitwise parity",
    )
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
    logging.info("Saved report to %s", args.output)


if __name__ == "__main__":
    main()
