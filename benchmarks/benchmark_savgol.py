"""Compare Savitzky-Golay filters with SciPy and a NumPy per-window reference."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path

import ndfiltrs as nf
import numpy as np
import scipy
from scipy.signal import savgol_filter

from tests._savgol_reference import savgol_reference


def _median_seconds(function: Callable[[], object], repeats: int) -> float:
    """Return the median wall time of `repeats` calls."""
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        elapsed.append(time.perf_counter() - start)
    return float(np.median(elapsed))


def _check(actual: np.ndarray, expected: np.ndarray, rtol: float) -> float:
    """Assert agreement relative to the largest magnitude; return the error."""
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = ~np.isnan(expected)
    scale = max(1.0, float(np.max(np.abs(expected[finite]))))
    error = float(np.max(np.abs(actual[finite] - expected[finite]))) / scale
    if error > rtol:
        raise AssertionError(f"relative error {error:.3g} exceeds {rtol:.3g}")
    return error


def benchmark(repeats: int) -> dict:
    """Measure median call times after checking agreement.

    Parameters
    ----------
    repeats : int
        Positive number of timed calls for each fast method.

    Returns
    -------
    dict
        Environment, workloads, agreement errors, and timings in seconds.
    """
    rng = np.random.default_rng(20261112)
    results = []

    line = rng.normal(size=1_000_000)
    for window, mode in ((11, "interp"), (11, "mirror"), (51, "interp")):
        ours = nf.savgol_filter(line, window, 3, mode=mode)
        error = _check(ours, savgol_filter(line, window, 3, mode=mode), 1e-11)
        results.append(
            {
                "case": f"1-D complete, n=1e6, window={window}, p=3, mode={mode}",
                "relative_error": error,
                "ndfiltrs": _median_seconds(
                    lambda w=window, m=mode: nf.savgol_filter(line, w, 3, mode=m),
                    repeats,
                ),
                "scipy": _median_seconds(
                    lambda w=window, m=mode: savgol_filter(line, w, 3, mode=m), repeats
                ),
            }
        )

    image = rng.normal(size=(1024, 1024))
    for window in (5, 11):

        def per_axis(w: int = window) -> np.ndarray:
            return savgol_filter(savgol_filter(image, w, 2, axis=0), w, 2, axis=1)

        def ours(w: int = window) -> np.ndarray:
            return nf.savgol_filter(image, w, 2, basis="per_axis")

        error = _check(ours(), per_axis(), 1e-10)
        results.append(
            {
                "case": f"2-D complete 1024^2, window={window}, per-axis p=2, interp",
                "relative_error": error,
                "ndfiltrs": _median_seconds(ours, repeats),
                "scipy": _median_seconds(per_axis, repeats),
                "note": "SciPy applies two separable 1-D passes; ndfiltrs evaluates "
                "the full 2-D window directly",
            }
        )
        masked = image.copy()
        masked[rng.random(masked.shape) < 0.1] = np.nan
        results.append(
            {
                "case": f"2-D 1024^2 with 10% NaN, window={window}, total p=2, interp",
                "ndfiltrs": _median_seconds(
                    lambda w=window, m=masked: nf.nansavgol_filter(m, w, 2), repeats
                ),
                "complete_data_ndfiltrs": _median_seconds(
                    lambda w=window: nf.savgol_filter(image, w, 2), repeats
                ),
            }
        )

    small = rng.normal(size=(64, 64))
    small[rng.random(small.shape) < 0.1] = np.nan
    for window in (5, 7):
        modes = ("interp", "interp")
        reference_start = time.perf_counter()
        reference = savgol_reference(
            small, (window, window), "total", 2, (0, 0), (1.0, 1.0), modes, min_count=6
        )
        reference_seconds = time.perf_counter() - reference_start
        error = _check(nf.nansavgol_filter(small, window, 2), reference, 1e-9)
        results.append(
            {
                "case": f"2-D 64^2 with 10% NaN, window={window}, total p=2, interp",
                "relative_error": error,
                "ndfiltrs": _median_seconds(
                    lambda w=window: nf.nansavgol_filter(small, w, 2), repeats
                ),
                "numpy_per_window": reference_seconds,
                "note": "NumPy reference: Python loop, SVD rank test, lstsq per window",
            }
        )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "rayon_threads": nf.get_num_threads(),
        "repeats": repeats,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    arguments.output.write_text(
        json.dumps(benchmark(arguments.repeats), indent=2) + "\n"
    )
