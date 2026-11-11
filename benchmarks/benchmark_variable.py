"""Compare per-pixel-window means with a vectorized NumPy summed-area table."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import ndfiltrs as nf
import numpy as np
from numpy.typing import NDArray

# SciPy-style boundary names mapped to the equivalent ``np.pad`` modes.
_PAD_MODES = {
    "reflect": "symmetric",
    "mirror": "reflect",
    "nearest": "edge",
    "wrap": "wrap",
    "constant": "constant",
}


def _corner_sum(
    table: NDArray[np.floating], radius: NDArray[np.integer], pad: int
) -> NDArray[np.float64]:
    """Combine the ``2**ndim`` window corners of a padded summed-area table."""
    centers = np.indices(radius.shape, sparse=True)
    total = np.zeros(radius.shape)
    for corner in np.ndindex((2,) * radius.ndim):
        index = tuple(
            center + pad + (radius + 1 if upper else -radius)
            for center, upper in zip(centers, corner, strict=True)
        )
        sign = -1.0 if (radius.ndim - sum(corner)) % 2 else 1.0
        total += sign * table[index]
    return total


def _table(padded: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return a float64 summed-area table with a leading zero on every axis."""
    table = np.zeros(tuple(length + 1 for length in padded.shape))
    table[(slice(1, None),) * padded.ndim] = padded
    for axis in range(padded.ndim):
        np.cumsum(table, axis=axis, out=table)
    return table


def numpy_variable_mean(
    data: NDArray[np.floating],
    radius: NDArray[np.integer],
    mode: str,
    *,
    omit_nan: bool = False,
) -> NDArray[np.float64]:
    """Average data over centered per-pixel windows with NumPy.

    Parameters
    ----------
    data : numpy.ndarray
        Input of any rank without infinities. NaN is allowed only with
        `omit_nan`.
    radius : numpy.ndarray
        Nonnegative integer half-width per pixel, with the shape of `data`.
    mode : str
        SciPy boundary mode. `constant` pads with zero.
    omit_nan : bool, optional
        Skip NaN samples with a second table of valid-sample counts.

    Returns
    -------
    numpy.ndarray
        Float64 means. The tables are plain float64 cumulative sums over the
        input padded by the largest radius.
    """
    pad = int(radius.max())
    padded = np.pad(data.astype(np.float64), pad, mode=_PAD_MODES[mode])
    if not omit_nan:
        total = _corner_sum(_table(padded), radius, pad)
        return total / (2 * radius + 1).astype(np.float64) ** data.ndim
    valid = ~np.isnan(padded)
    total = _corner_sum(_table(np.where(valid, padded, 0.0)), radius, pad)
    count = _corner_sum(_table(valid.astype(np.float64)), radius, pad)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _radial_radius(shape: tuple[int, ...], maximum: int) -> NDArray[np.int64]:
    """Return ``floor(1 + maximum * rho / rho_max)``, a comet-like radius map.

    The linear growth with distance `rho` from the array center is an
    arbitrary benchmark choice.
    """
    grids = np.indices(shape, sparse=True)
    squares = (
        (grid - (length - 1) / 2) ** 2
        for grid, length in zip(grids, shape, strict=True)
    )
    rho = np.sqrt(sum(squares))
    return np.floor(1 + maximum * rho / rho.max()).astype(np.int64)


def _median_time(call, repeats: int) -> float:
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        elapsed.append(time.perf_counter() - start)
    return float(np.median(elapsed))


def benchmark(repeats: int) -> dict:
    """Measure median call times after checking ndfiltrs against NumPy.

    Parameters
    ----------
    repeats : int
        Positive number of timed calls per implementation.

    Returns
    -------
    dict
        Environment, workload parameters, and timings in seconds.
    """
    rng = np.random.default_rng(20261112)
    results = []
    # The 1% NaN fraction is an arbitrary choice that forces the NaN tables.
    cases = (
        ((2048, 2048), np.float32, "radial", 32, 0.0),
        ((2048, 2048), np.float64, "radial", 32, 0.0),
        ((2048, 2048), np.float64, "radial", 32, 0.01),
        ((2048, 2048), np.float64, "random", 15, 0.0),
        ((128, 128, 128), np.float64, "radial", 8, 0.0),
    )
    for shape, dtype, kind, maximum, missing in cases:
        data = (1000.0 + rng.normal(size=shape)).astype(dtype)
        data[rng.random(shape) < missing] = np.nan
        if kind == "radial":
            radius = _radial_radius(shape, maximum)
        else:
            radius = rng.integers(0, maximum + 1, size=shape)
        omit = missing > 0

        def native(data=data, radius=radius, omit=omit):
            function = nf.nanvariable_mean_filter if omit else nf.variable_mean_filter
            return function(data, radius=radius)

        def reference(data=data, radius=radius, omit=omit):
            return numpy_variable_mean(data, radius, "reflect", omit_nan=omit)

        tolerance = 1e-5 if dtype is np.float32 else 1e-9
        np.testing.assert_allclose(native(), reference(), rtol=tolerance)
        timings = {
            "ndfiltrs": _median_time(native, repeats),
            "numpy": _median_time(reference, repeats),
        }
        results.append(
            {
                "shape": shape,
                "dtype": np.dtype(dtype).name,
                "radius_map": kind,
                "max_radius": int(radius.max()),
                "nan_fraction": missing,
                "seconds": timings,
            }
        )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
        "ndfiltrs_threads": nf.get_num_threads(),
        "repeats": repeats,
        "results": results,
    }


def main() -> None:
    """Run the benchmark and print or save a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    report = json.dumps(benchmark(arguments.repeats), indent=2)
    if arguments.output is None:
        print(report)
    else:
        arguments.output.write_text(report + "\n")


if __name__ == "__main__":
    main()
