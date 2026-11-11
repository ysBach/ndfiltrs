"""Time variance-propagating convolution against Astropy and a previous build.

Astropy is not a project dependency. Run, for example::

    uv run --frozen --extra dev --with astropy python \
        -m benchmarks.benchmark_variance --output result.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import ndfiltrs as nf
import numpy as np

from benchmarks._common import load_baseline, make_input

# Gaussian with FWHM 3 px (sigma = 3 / 2.3548), support 4 sigma.
# Moffat with FWHM 4 px and beta 2.5: alpha = 4 / (2 sqrt(2**(1/2.5) - 1)).
KERNELS = {
    "gaussian_fwhm3_13x13": nf.kernels.gaussian(3 / 2.3548, ndim=2, radius=6),
    "moffat_fwhm4_25x25": nf.kernels.moffat(
        4 / (2 * np.sqrt(2 ** (1 / 2.5) - 1)), 2.5, ndim=2, radius=12
    ),
}


def _timings(function: Callable[[], object], repeats: int) -> list[float]:
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        elapsed.append(time.perf_counter() - start)
    return elapsed


def _summary(elapsed: list[float]) -> dict:
    return {
        "median": float(np.median(elapsed)),
        "min": float(min(elapsed)),
        "max": float(max(elapsed)),
    }


def _check(actual: np.ndarray, expected: np.ndarray) -> float:
    """Return the largest difference relative to the array scale; fail above 1e-12."""
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.isfinite(actual), finite)
    scale = np.abs(expected[finite]).max()
    error = float(np.abs(actual[finite] - expected[finite]).max() / scale)
    assert error < 1e-12, error
    return error


def _raw_ours(data: np.ndarray, variance: np.ndarray, kernel: np.ndarray) -> tuple:
    return nf.convolve_variance(data, variance, kernel, mode="constant", cval=0.0)


def _raw_astropy(
    convolve: Callable, data: np.ndarray, variance: np.ndarray, kernel: np.ndarray
) -> tuple:
    options = {"boundary": "fill", "fill_value": 0.0, "normalize_kernel": False}
    return convolve(data, kernel, **options), convolve(variance, kernel**2, **options)


def _mean_ours(missing: np.ndarray, variance: np.ndarray, kernel: np.ndarray) -> tuple:
    return nf.nanconvolve_variance(
        missing, variance, kernel, mode="constant", cval=np.nan
    )


def _mean_astropy(
    convolve: Callable,
    missing: np.ndarray,
    masked_variance: np.ndarray,
    valid: np.ndarray,
    kernel: np.ndarray,
) -> tuple:
    """Weighted mean, then sum(w**2 v) and sum(w) over valid pixels."""
    value = convolve(
        missing, kernel, boundary="fill", fill_value=np.nan, nan_treatment="interpolate"
    )
    options = {"boundary": "fill", "fill_value": 0.0, "normalize_kernel": False}
    numerator = convolve(masked_variance, kernel**2, **options)
    weight = convolve(valid, kernel, **options)
    return value, numerator / weight**2


def astropy_comparison(sizes: list[int], repeats: int) -> list[dict]:
    """Compare one ndfiltrs pass with repeated Astropy passes after validation.

    Parameters
    ----------
    sizes : list of int
        Square image side lengths.
    repeats : int
        Positive number of timed calls per method.

    Returns
    -------
    list of dict
        One record per size, kernel, and missing-data case.
    """
    from astropy.convolution import convolve

    rng = np.random.default_rng(20261112)
    results = []
    for size in sizes:
        data = make_input((size, size), np.float64)
        variance = rng.uniform(0.5, 2.0, size=data.shape)
        missing = data.copy()
        missing[rng.random(data.shape) < 0.1] = np.nan
        valid = np.isfinite(missing).astype(np.float64)
        masked_variance = np.where(valid > 0, variance, 0.0)
        for name, kernel in KERNELS.items():
            raw_ours = partial(_raw_ours, data, variance, kernel)
            raw_astropy = partial(_raw_astropy, convolve, data, variance, kernel)
            mean_ours = partial(_mean_ours, missing, variance, kernel)
            mean_astropy = partial(
                _mean_astropy, convolve, missing, masked_variance, valid, kernel
            )
            for case, ours, theirs, calls in (
                ("finite_sum", raw_ours, raw_astropy, 2),
                ("nan_mean", mean_ours, mean_astropy, 3),
            ):
                expected = theirs()
                actual = ours()
                errors = [_check(a, e) for a, e in zip(actual, expected, strict=True)]
                ours_time = _timings(ours, repeats)
                theirs_time = _timings(theirs, repeats)
                results.append(
                    {
                        "size": size,
                        "kernel": name,
                        "kernel_shape": kernel.shape,
                        "case": case,
                        "astropy_calls": calls,
                        "max_scaled_difference": errors,
                        "ndfiltrs_seconds": _summary(ours_time),
                        "astropy_seconds": _summary(theirs_time),
                        "speedup": float(np.median(theirs_time) / np.median(ours_time)),
                    }
                )
    return results


def baseline_comparison(baseline: Path, size: int, repeats: int) -> list[dict]:
    """Time existing convolution paths in this build and a previous build.

    Calls alternate between builds so drifting machine load affects both.
    """
    previous = load_baseline(baseline)
    data = make_input((size, size), np.float64)
    missing = data.copy()
    missing[np.random.default_rng(3).random(data.shape) < 0.1] = np.nan
    results = []
    for name, kernel in KERNELS.items():
        for function, array in (("convolve", data), ("nanconvolve", missing)):
            current_call = getattr(nf, function)
            previous_call = getattr(previous, function)
            np.testing.assert_array_equal(
                current_call(array, kernel), previous_call(array, kernel)
            )
            current, prior = [], []
            for repeat in range(repeats):
                pair = [(current, current_call), (prior, previous_call)]
                for elapsed, call in pair if repeat % 2 == 0 else pair[::-1]:
                    elapsed.extend(_timings(partial(call, array, kernel), 1))
            results.append(
                {
                    "function": function,
                    "kernel": name,
                    "size": size,
                    "current_seconds": _summary(current),
                    "previous_seconds": _summary(prior),
                    "ratio_current_to_previous": float(
                        np.median(current) / np.median(prior)
                    ),
                }
            )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument(
        "--baseline", type=Path, help="installed previous ndfiltrs package directory"
    )
    parser.add_argument("--baseline-size", type=int, default=2048)
    parser.add_argument("--skip-astropy", action="store_true")
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    report = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rayon_threads": nf.get_num_threads(),
        "repeats": arguments.repeats,
    }
    if arguments.baseline is not None:
        report["baseline"] = baseline_comparison(
            arguments.baseline, arguments.baseline_size, arguments.repeats
        )
    if not arguments.skip_astropy:
        import astropy

        report["astropy"] = astropy.__version__
        report["astropy_threads"] = 1
        report["results"] = astropy_comparison(arguments.sizes, arguments.repeats)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
