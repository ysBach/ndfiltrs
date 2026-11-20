"""Direct NumPy statement of the ``mesh_background`` pipeline.

Each stage follows the definitions in ``docs/background.md`` with plain loops
and full sorts, so it is slow but easy to audit. Keyword names and defaults are
those of ``ndfiltrs.mesh_background``.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

# Phi^-1(0.75): the MAD of a standard normal variable.
NORMAL_UPPER_QUARTILE = 0.6744897501960817

DEFAULTS: dict[str, Any] = {
    "mask": None,
    "filter_size": 3,
    "estimator": "sextractor",
    "sigma_lower": 3.0,
    "sigma_upper": 3.0,
    "maxiters": 10,
    "center": "median",
    "spread": "std",
    "statistic": "mode",
    "mode_estimator": "sex",
    "pearson": None,
    "rms": "std",
    "min_valid_fraction": 0.5,
    "valid_count": "finite",
    "valid_base": "mesh",
    "fill": "nearest",
    "idw_neighbors": 10,
    "idw_power": 1.0,
    "filter_edge": "shrink",
    "filter_threshold": None,
    "threshold_kind": "deviation",
    "interpolation": "spline",
}


def _median(values: NDArray[np.float64]) -> float:
    return float(np.median(values)) if values.size else math.nan


def _std(values: NDArray[np.float64]) -> float:
    return float(np.std(values)) if values.size else math.nan


def _mode(mean: float, median: float, std: float, estimator: str, pearson: float):
    pearson_mode = median + (pearson - 1.0) * (median - mean)
    if estimator == "mmm":
        return pearson_mode
    if estimator == "iraf":
        return mean if mean < median else pearson_mode
    if std == 0.0:
        return mean
    return median if abs(median - mean) / std >= 0.3 else pearson_mode


def _clip(values, center, spread, lower, upper, maxiters):
    """Astropy-style iterative clipping: inclusive bounds, stop when stable."""
    iteration = 0
    while values.size and (maxiters is None or iteration < maxiters):
        middle = _median(values) if center == "median" else float(np.mean(values))
        if spread == "std":
            width = _std(values)
        else:
            width = _median(np.abs(values - _median(values))) / NORMAL_UPPER_QUARTILE
        kept = values[
            (values >= middle - lower * width) & (values <= middle + upper * width)
        ]
        iteration += 1
        if kept.size == values.size:
            break
        values = kept
    return values


def _sextractor_stats(values, pearson):
    """Exact-sample form of SExtractor's mesh estimate (see the guide)."""
    mean, std = float(np.mean(values)), _std(values)
    first = values[(values >= mean - 2.0 * std) & (values <= mean + 2.0 * std)]
    mean, std = float(np.mean(first)), _std(first)
    span = values[(values >= mean - 5.0 * std) & (values <= mean + 5.0 * std)]
    kept = _clip(span, "median", "std", 3.0, 3.0, 100)
    mean, median, std = float(np.mean(kept)), _median(kept), _std(kept)
    return _mode(mean, median, std, "sex", pearson), std, kept.size


def _mesh_stats(values, options):
    if options["estimator"] == "sextractor":
        pearson = 2.5 if options["pearson"] is None else options["pearson"]
        return _sextractor_stats(values, pearson)
    kept = _clip(
        values,
        options["center"],
        options["spread"],
        options["sigma_lower"],
        options["sigma_upper"],
        options["maxiters"],
    )
    if kept.size == 0:
        return math.nan, math.nan, 0
    mean, median = float(np.mean(kept)), _median(kept)
    statistic = options["statistic"]
    if statistic == "mean":
        location = mean
    elif statistic == "median":
        location = median
    else:
        estimator = options["mode_estimator"]
        pearson = options["pearson"]
        if pearson is None:
            pearson = 2.5 if estimator == "sex" else 3.0
        location = _mode(mean, median, _std(kept), estimator, pearson)
    if options["rms"] == "std":
        rms = _std(kept)
    else:
        rms = _median(np.abs(kept - median)) / NORMAL_UPPER_QUARTILE
    return location, rms, kept.size


def _fill(back, rms, valid, options):
    grid = np.argwhere(np.ones(back.shape, dtype=bool))
    good = np.flatnonzero(valid.ravel())
    out_back, out_rms = back.ravel().copy(), rms.ravel().copy()
    if good.size == 0:
        return np.full(back.shape, np.nan), np.full(back.shape, np.nan)
    for flat in np.flatnonzero(~valid.ravel()):
        d2 = ((grid[good] - grid[flat]) ** 2).sum(axis=1).astype(np.float64)
        if options["fill"] == "nearest":
            nearest = good[d2 == d2.min()]
            out_back[flat] = back.ravel()[nearest].mean()
            out_rms[flat] = rms.ravel()[nearest].mean()
        else:
            # Every mesh tied with the last of the k nearest is included.
            last = np.sort(d2)[min(options["idw_neighbors"], d2.size) - 1]
            order = np.flatnonzero(d2 <= last)
            weights = 1.0 / np.sqrt(d2[order]) ** options["idw_power"]
            chosen = good[order]
            out_back[flat] = (weights * back.ravel()[chosen]).sum() / weights.sum()
            out_rms[flat] = (weights * rms.ravel()[chosen]).sum() / weights.sum()
    return out_back.reshape(back.shape), out_rms.reshape(back.shape)


def _filter(back, rms, size, options):
    if all(length == 1 for length in size):
        return back, rms
    out_back, out_rms = back.copy(), rms.copy()
    threshold = options["filter_threshold"]
    for index in itertools.product(*(range(n) for n in back.shape)):
        slices = []
        for k, n, length in zip(index, back.shape, size, strict=True):
            half = length // 2
            if options["filter_edge"] == "shrink":
                half = min(half, k, n - 1 - k)
                slices.append(slice(k - half, k + half + 1))
            else:
                slices.append(slice(max(0, k - half), min(n, k + half + 1)))
        median_back = _median(back[tuple(slices)].ravel())
        if threshold is None:
            replace = True
        elif options["threshold_kind"] == "deviation":
            replace = abs(median_back - back[index]) >= threshold
        else:
            replace = back[index] > threshold
        if replace:
            out_back[index] = median_back
            out_rms[index] = _median(rms[tuple(slices)].ravel())
    return out_back, out_rms


def interpolate(grid, shape, box_size):
    """Evaluate SciPy's natural cubic spline at pixel centers along each axis."""
    result = np.asarray(grid, dtype=np.float64)
    if not np.isfinite(result).any():
        return np.full(shape, np.nan)
    for axis, (length, box) in enumerate(zip(shape, box_size, strict=True)):
        nodes = result.shape[axis]
        if nodes == 1:
            result = np.repeat(result, length, axis=axis)
        else:
            centers = (np.arange(nodes) + 0.5) * box - 0.5
            result = CubicSpline(centers, result, axis=axis, bc_type="natural")(
                np.arange(length)
            )
    return result


def mesh_background(data, box_size, **overrides) -> dict[str, Any]:
    """Run every stage and return full maps plus mesh-level grids."""
    options = {**DEFAULTS, **overrides}
    data = np.asarray(data)
    ndim = data.ndim
    box = (box_size,) * ndim if np.ndim(box_size) == 0 else tuple(box_size)
    size = options["filter_size"]
    size = (size,) * ndim if np.ndim(size) == 0 else tuple(size)
    valid_data = np.isfinite(data)
    if options["mask"] is not None:
        valid_data &= ~np.asarray(options["mask"], dtype=bool)
    mesh_shape = tuple(-(-n // b) for n, b in zip(data.shape, box, strict=True))
    back = np.full(mesh_shape, np.nan)
    rms = np.full(mesh_shape, np.nan)
    finite_count = np.zeros(mesh_shape, dtype=np.int64)
    retained = np.zeros(mesh_shape, dtype=np.int64)
    valid = np.zeros(mesh_shape, dtype=bool)
    for index in itertools.product(*(range(n) for n in mesh_shape)):
        region = tuple(
            slice(k * b, (k + 1) * b) for k, b in zip(index, box, strict=True)
        )
        values = np.sort(data[region][valid_data[region]].astype(np.float64))
        finite_count[index] = values.size
        if values.size == 0:
            continue
        location, spread, kept = _mesh_stats(values, options)
        retained[index] = kept
        count = values.size if options["valid_count"] == "finite" else kept
        base = data[region].size if options["valid_base"] == "mesh" else math.prod(box)
        good = count > 0 and count >= options["min_valid_fraction"] * base
        if good and math.isfinite(location) and math.isfinite(spread):
            back[index], rms[index], valid[index] = location, spread, True
    filled_back, filled_rms = _fill(back, rms, valid, options)
    final_back, final_rms = _filter(filled_back, filled_rms, size, options)
    if options["interpolation"] != "spline":
        raise ValueError("The NumPy reference supports the default spline only.")
    return {
        "background": interpolate(final_back, data.shape, box),
        "rms": interpolate(final_rms, data.shape, box),
        "mesh_background": final_back,
        "mesh_rms": final_rms,
        "mesh_valid": valid,
        "mesh_finite_count": finite_count,
        "mesh_retained_count": retained,
        "global_background": _median(final_back.ravel()),
        "global_rms": _median(final_rms.ravel()),
    }
