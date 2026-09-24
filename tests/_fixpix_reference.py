"""Independent reference implementation for :func:`ndfiltrs.fixpix`."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import find_objects, label


def _axis_connectivity(ndim: int, axis: int) -> NDArray[np.int8]:
    """Return connectivity that joins only neighboring positions on `axis`."""
    structure = np.zeros((3,) * ndim, dtype=np.int8)
    center = [1] * ndim
    structure[tuple(center)] = 1
    for offset in (-1, 1):
        neighbor = center.copy()
        neighbor[axis] += offset
        structure[tuple(neighbor)] = 1
    return structure


def _axis_runs(
    mask: NDArray[np.bool_],
) -> tuple[list[NDArray[np.int32]], list[list[tuple[slice, ...] | None]]]:
    """Label original mask runs independently along every array axis."""
    labels_by_axis: list[NDArray[np.int32]] = []
    objects_by_axis: list[list[tuple[slice, ...] | None]] = []
    for axis in range(mask.ndim):
        labels, _ = label(mask, structure=_axis_connectivity(mask.ndim, axis))
        labels_by_axis.append(labels)
        objects_by_axis.append(find_objects(labels))
    return labels_by_axis, objects_by_axis


def _interpolate_span(
    output: NDArray[np.float32] | NDArray[np.float64],
    position: tuple[int, ...],
    axis: int,
    start: int,
    stop: int,
    *,
    use_scipy: bool = True,
) -> None:
    """Apply one legacy span interpolation to `output` in place."""
    initial = start - 1
    final = stop + 1
    delta = final - initial
    axis_length = output.shape[axis]

    if initial < 0 and final >= axis_length:
        return

    at_edge = initial < 0 or final >= axis_length
    if initial < 0:
        initial = final
    elif final >= axis_length:
        final = initial

    initial_position = list(position)
    final_position = list(position)
    initial_position[axis] = initial
    final_position[axis] = final
    initial_value = np.float64(output.item(tuple(initial_position)))
    final_value = np.float64(output.item(tuple(final_position)))

    grid = np.arange(start, stop + 1, dtype=np.float64)
    with np.errstate(all="ignore"):
        difference = final_value - initial_value
        comparable = (
            not at_edge
            and np.isfinite(initial_value)
            and np.isfinite(final_value)
            and np.isfinite(difference)
        )
        if comparable and use_scipy:
            interpolator = interp1d(
                np.asarray([initial, final], dtype=np.float64),
                np.asarray([initial_value, final_value], dtype=np.float64),
                kind="linear",
                assume_sorted=True,
            )
            values = interpolator(grid)
        else:
            values = (
                difference / np.float64(delta) * np.arange(1, delta, dtype=np.float64)
                + initial_value
            )

    span = [slice(coordinate, coordinate + 1) for coordinate in position]
    span[axis] = slice(start, stop + 1)
    output[tuple(span)].flat = values


def fixpix_reference(
    array: NDArray[np.float32] | NDArray[np.float64],
    mask: NDArray[np.bool_] | NDArray[np.number] | None = None,
    *,
    priority: Sequence[int] | None = None,
    use_scipy: bool = True,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Evaluate the requested fixpix contract with SciPy labeling primitives.

    Parameters
    ----------
    array : numpy.ndarray
        The input array. The reference assumes the public API has already
        validated its floating dtype and rank.
    mask : numpy.ndarray or None, optional
        Bad-pixel locations. Numeric values are cast to boolean.
    priority : sequence of int or None, optional
        Axis order from highest to lowest priority. `None` uses reversed axis
        order.
    use_scipy : bool, optional
        Use SciPy interpolation for finite interior anchors by default.
        `False` uses the literal legacy arithmetic order with the same
        independent SciPy run labeling; SciPy can differ by an ULP because
        its interpolation implementation may order floating operations
        differently.

    Returns
    -------
    numpy.ndarray
        A C-contiguous copy after replaying every original bad coordinate in
        C order.
    """
    output = np.array(array, copy=True, order="C", subok=False)
    if mask is None:
        return output

    original_mask = np.asarray(mask).astype(bool, copy=False)
    if original_mask.size == 0 or not np.any(original_mask):
        return output

    axis_priority = (
        tuple(range(output.ndim - 1, -1, -1))
        if priority is None
        else tuple(int(axis) for axis in priority)
    )
    labels_by_axis, objects_by_axis = _axis_runs(original_mask)
    bad_positions = np.flatnonzero(original_mask.ravel(order="C"))
    coordinates = np.unravel_index(bad_positions, output.shape, order="C")

    for position_values in zip(*coordinates, strict=True):
        position = tuple(int(value) for value in position_values)
        selected_axis = axis_priority[0]
        selected_start = 0
        selected_stop = output.shape[selected_axis] - 1
        selected_length = np.iinfo(np.intp).max
        for axis in axis_priority:
            component = int(labels_by_axis[axis][position])
            span = objects_by_axis[axis][component - 1]
            if span is None:
                raise AssertionError("a labeled bad pixel has no bounding slice")
            axis_slice = span[axis]
            start = axis_slice.start
            stop = axis_slice.stop - 1
            length = stop - start + 1
            if length < selected_length:
                selected_axis = axis
                selected_start = start
                selected_stop = stop
                selected_length = length

        _interpolate_span(
            output,
            position,
            selected_axis,
            selected_start,
            selected_stop,
            use_scipy=use_scipy,
        )

    return output
