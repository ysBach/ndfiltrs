"""Independent literal N-D reference implementation for tests."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

MODES = ("reflect", "constant", "nearest", "mirror", "wrap")


def _map_index(index: int, length: int, mode: str) -> int | None:
    if 0 <= index < length:
        return index
    if mode == "constant":
        return None
    if mode == "nearest":
        return min(max(index, 0), length - 1)
    if mode == "wrap":
        return index % length
    if length == 1:
        return 0
    if mode == "reflect":
        period = 2 * length
        reduced = index % period
        return reduced if reduced < length else period - 1 - reduced
    if mode == "mirror":
        period = 2 * (length - 1)
        reduced = index % period
        return reduced if reduced < length else period - reduced
    raise ValueError(f"unsupported mode: {mode}")


def _safe_midpoint(lower: float, upper: float) -> float:
    if math.copysign(1.0, lower) == math.copysign(1.0, upper) and lower == upper:
        return lower
    if lower == 0.0 and upper == 0.0:
        return 0.0
    if math.isinf(lower) or math.isinf(upper):
        if math.isinf(lower) and math.isinf(upper):
            return math.nan
        return lower if math.isinf(lower) else upper
    if math.copysign(1.0, lower) == math.copysign(1.0, upper):
        return lower + (upper - lower) / 2.0
    return (lower + upper) / 2.0


def _median(values: list[float]) -> float:
    if not values:
        return math.nan
    values.sort(
        key=lambda value: (
            np.float64(value).view(np.uint64)
            ^ (
                np.uint64(0xFFFFFFFFFFFFFFFF)
                if np.signbit(value)
                else np.uint64(0x8000000000000000)
            )
        )
    )
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return _safe_midpoint(values[middle - 1], values[middle])


def median_filter_reference(
    array: NDArray[np.float32] | NDArray[np.float64],
    size: Sequence[int],
    mode: Sequence[str],
    *,
    cval: float,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: int,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Evaluate the specified median filter with explicit coordinate loops."""
    output = np.empty_like(array)
    radii = tuple(value // 2 for value in size)
    for output_index in np.ndindex(array.shape):
        values: list[float] = []
        propagate_nan = False
        for footprint_index in np.ndindex(tuple(size)):
            mapped: list[int] = []
            use_constant = False
            for axis, footprint_coordinate in enumerate(footprint_index):
                coordinate = output_index[axis] + footprint_coordinate - radii[axis]
                mapped_coordinate = _map_index(
                    coordinate, array.shape[axis], mode[axis]
                )
                if mapped_coordinate is None:
                    use_constant = True
                    break
                mapped.append(mapped_coordinate)
            value = float(cval if use_constant else array[tuple(mapped)])
            if math.isnan(value):
                if omit_nan:
                    continue
                propagate_nan = True
                break
            if ignore_inf and math.isinf(value):
                continue
            values.append(value)
        result = (
            math.nan if propagate_nan or len(values) < min_count else _median(values)
        )
        output[output_index] = result
    return output
