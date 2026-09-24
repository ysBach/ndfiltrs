"""Compare equivalent finite-input filters with SciPy and NumPy.

Timings include validation, copies, allocation, and boundary padding. NumPy
sliding windows are a baseline, not a claim about the fastest NumPy algorithm.
Run with fixed RAYON_NUM_THREADS and no other CPU-heavy jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import ndfiltrs as nf
import numpy as np
import scipy
from ndfiltrs import _core
from numpy.typing import NDArray
from scipy import ndimage

from benchmarks._common import load_baseline, make_input

FloatArray = NDArray[np.float32] | NDArray[np.float64]


def numpy_window(array: FloatArray, size: int, statistic: str) -> FloatArray:
    """Reduce reflected sliding windows, including padding in the measured call."""
    padded = np.pad(array, size // 2, mode="symmetric")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size,) * array.ndim)
    axes = tuple(range(-array.ndim, 0))
    if statistic == "mean":
        return np.mean(windows, axis=axes, dtype=np.float64).astype(array.dtype)
    return getattr(np, statistic)(windows, axis=axes)


def measure(
    runners: dict[str, Callable[[], FloatArray]],
    reference: FloatArray,
    *,
    tolerance: float,
    repeats: int,
) -> dict[str, Any]:
    """Check numerical parity, warm each call, then alternate timing order."""
    errors = {}
    for name, runner in runners.items():
        actual = runner()
        if actual.shape != reference.shape or actual.dtype != reference.dtype:
            raise AssertionError(f"{name}: shape or dtype differs from reference")
        np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=tolerance)
        errors[name] = float(np.max(np.abs(actual - reference)))
    samples: dict[str, list[float]] = {name: [] for name in runners}
    for repeat in range(repeats):
        order = list(runners)
        if repeat % 2:
            order.reverse()
        for name in order:
            start = time.perf_counter()
            result = runners[name]()
            samples[name].append(time.perf_counter() - start)
            del result
    return {
        name: {
            "median_seconds": statistics.median(values),
            "samples_seconds": values,
            "max_absolute_error": errors[name],
        }
        for name, values in samples.items()
    }


def benchmark(repeats: int, baseline: ModuleType | None = None) -> dict[str, Any]:
    """Measure a fixed 1-D/2-D/3-D matrix with both floating dtypes."""
    records = []
    for shape, sizes in (
        ((4096,), (3, 15)),
        ((256, 256), (3, 15)),
        ((16, 20, 24), (3, 7)),
    ):
        for dtype in (np.float32, np.float64):
            array = make_input(shape, dtype)
            original = array.copy()
            tolerance = float(64 * np.finfo(dtype).eps * np.max(np.abs(array)))
            for size in sizes:
                for statistic, nf_name, scipy_name in (
                    ("median", "median_filter", "median_filter"),
                    ("min", "min_filter", "minimum_filter"),
                    ("max", "max_filter", "maximum_filter"),
                    ("mean", "mean_filter", "uniform_filter"),
                    ("gaussian", "gaussian_filter", "gaussian_filter"),
                    ("convolve", "convolve", "convolve"),
                ):
                    kwargs: dict[str, Any] = {"size": size, "mode": "reflect"}
                    if statistic == "gaussian":
                        kwargs = {"sigma": 1.0, "radius": size // 2, "mode": "reflect"}
                    elif statistic == "convolve":
                        weights = make_input((size,) * array.ndim, np.float64)
                        weights /= np.sum(np.abs(weights))
                        kwargs = {"weights": weights, "mode": "reflect"}
                    runners = {
                        "ndfiltrs": partial(getattr(nf, nf_name), array, **kwargs),
                        "scipy": partial(getattr(ndimage, scipy_name), array, **kwargs),
                    }
                    if statistic in ("mean", "gaussian"):
                        runners["ndfiltrs_separable"] = partial(
                            getattr(nf, nf_name), array, method="separable", **kwargs
                        )
                    if statistic in ("median", "min", "max", "mean"):
                        runners["numpy_sliding_window"] = partial(
                            numpy_window, array, size, statistic
                        )
                    if baseline is not None:
                        runners["previous"] = partial(
                            getattr(baseline, nf_name), array, **kwargs
                        )
                        if statistic in ("mean", "gaussian"):
                            runners["previous_separable"] = partial(
                                getattr(baseline, nf_name),
                                array,
                                method="separable",
                                **kwargs,
                            )
                        for suffix in ("", "_separable"):
                            if "previous" + suffix in runners:
                                np.testing.assert_array_equal(
                                    runners["ndfiltrs" + suffix]().view(
                                        f"u{array.dtype.itemsize}"
                                    ),
                                    runners["previous" + suffix]().view(
                                        f"u{array.dtype.itemsize}"
                                    ),
                                    err_msg=f"changed output bits: {statistic}{suffix}",
                                )
                    timings = measure(
                        runners,
                        runners["scipy"](),
                        repeats=repeats,
                        tolerance=0.0
                        if statistic in ("median", "min", "max")
                        else tolerance,
                    )
                    np.testing.assert_array_equal(array, original)
                    records.append(
                        {
                            "operation": statistic,
                            "shape": shape,
                            "size": size,
                            "dtype": np.dtype(dtype).name,
                            "mode": "reflect",
                            "timings": timings,
                        }
                    )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "ndfiltrs": nf.__version__,
        "rayon_threads": nf.get_num_threads(),
        "repeats": repeats,
        "warmup_calls": 1,
        "baseline_bitwise_parity": "passed" if baseline is not None else "not checked",
        "baseline_extension_sha256": hashlib.sha256(
            Path(baseline._core.__file__).read_bytes()
        ).hexdigest()
        if baseline is not None
        else None,
        "scope": "finite contiguous arrays, odd windows, reflect boundary; no IRAF",
        "sources": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in (("extension", _core.__file__), ("benchmark", __file__))
        },
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--baseline", type=Path, help="prior installed package directory"
    )
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    arguments.output.write_text(
        json.dumps(
            benchmark(
                arguments.repeats,
                load_baseline(arguments.baseline) if arguments.baseline else None,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logging.basicConfig(level=logging.INFO)
    logging.info("Saved matched benchmark to %s", arguments.output)
