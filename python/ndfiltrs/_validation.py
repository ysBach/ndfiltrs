"""Validation and normalization for the public filtering API."""

from __future__ import annotations

import numbers
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_MODES = ("reflect", "constant", "nearest", "mirror", "wrap")


def _integer_index(value: object, message: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(message)
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(message) from None


@dataclass(frozen=True)
class FilterArguments:
    """Normalized arguments passed to the native extension."""

    array: NDArray[np.float32] | NDArray[np.float64]
    size: tuple[int, ...]
    mode: tuple[str, ...]
    cval: float
    min_count: int
    ddof: int = 0
    footprint: list[bool] | None = None
    origin: tuple[int, ...] = ()
    count: int = 1


def _normalize_size(size: object, ndim: int) -> tuple[int, ...]:
    try:
        scalar = _integer_index(
            size, "size must be an integer or a sequence of integers"
        )
    except TypeError:
        if isinstance(size, (str, bytes)) or not isinstance(size, Sequence):
            raise TypeError(
                "size must be an integer or a sequence of integers"
            ) from None
        values = tuple(size)
        if len(values) != ndim:
            raise ValueError(
                "size sequence must have one entry per array dimension"
            ) from None
        values = tuple(
            _integer_index(value, "size entries must be integers, not booleans")
            for value in values
        )
    else:
        values = (scalar,) * ndim
    if any(value <= 0 for value in values):
        raise ValueError("size entries must be positive integers")
    return values


def _normalize_window(
    size: object, footprint: object, origin: object, ndim: int
) -> tuple[tuple[int, ...], list[bool] | None, tuple[int, ...], int]:
    """Return window shape, flat footprint, origins, and selected-element count.

    `size` and `footprint` are mutually exclusive; with neither, the window is
    3 along every axis. Nonzero footprint entries are selected, as in SciPy.
    """
    if footprint is None:
        shape = _normalize_size(3 if size is None else size, ndim)
        flat = None
        count = 1
        for length in shape:
            count *= length
    else:
        if size is not None:
            raise ValueError("pass either size or footprint, not both")
        if np.ma.isMaskedArray(footprint):
            raise TypeError("footprint must not be a masked array")
        entries = np.asarray(footprint)
        if entries.dtype.kind not in "biuf":
            raise TypeError("footprint must have a boolean or real numeric dtype")
        if entries.dtype.kind == "f" and not np.isfinite(entries).all():
            raise ValueError("footprint entries must be finite")
        selected = entries != 0
        if selected.ndim != ndim:
            raise ValueError("footprint must have the same rank as the array")
        if selected.size == 0:
            raise ValueError("footprint axes must be nonempty")
        count = int(np.count_nonzero(selected))
        if count == 0:
            raise ValueError("footprint must select at least one element")
        shape = tuple(int(length) for length in selected.shape)
        flat = selected.ravel().tolist()
    if np.ndim(origin) == 0:
        origins = (_integer_index(origin, "origin must be an integer or a sequence"),)
        origins *= ndim
    else:
        if isinstance(origin, np.ndarray):
            origin = origin.tolist()
        origins = tuple(
            _integer_index(value, "origin entries must be integers, not booleans")
            for value in origin
        )
        if len(origins) != ndim:
            raise ValueError("origin sequence must have one entry per array dimension")
    for length, shift in zip(shape, origins, strict=True):
        if not -(length // 2) <= shift <= (length - 1) // 2:
            raise ValueError(
                "origin must satisfy -(size // 2) <= origin <= (size - 1) // 2"
            )
    return shape, flat, origins, count


def _normalize_mode(mode: object, ndim: int) -> tuple[str, ...]:
    if isinstance(mode, str):
        values = (mode,) * ndim
    else:
        if not isinstance(mode, Sequence):
            raise TypeError("mode must be a string or a sequence of strings")
        values = tuple(mode)
        if len(values) != ndim:
            raise ValueError("mode sequence must have one entry per array dimension")
        if any(not isinstance(value, str) for value in values):
            raise TypeError("mode entries must be strings")
    if any(value not in _MODES for value in values):
        raise ValueError(f"mode must use only {_MODES}")
    return values


def validate_array(array: object) -> NDArray[np.float32] | NDArray[np.float64]:
    """Validate array rank, floating dtype, byte order, and alignment."""
    if np.ma.isMaskedArray(array):
        raise TypeError("masked arrays are not supported")
    value = np.asarray(array)
    if value.ndim == 0:
        raise ValueError("array must have at least one dimension")
    if value.dtype.kind != "f" or value.dtype.itemsize not in (4, 8):
        raise TypeError("array dtype must be float32 or float64")
    if not value.dtype.isnative:
        raise TypeError("array dtype must be native-endian")
    if not value.flags.aligned:
        raise ValueError("array must be aligned")
    return value


def validate_filter_arguments(
    array: object,
    size: object,
    mode: object,
    cval: object,
    *,
    min_count: object,
    ddof: object = 0,
    footprint: object = None,
    origin: object = 0,
    normalize_array: bool = True,
) -> FilterArguments:
    """Validate public arguments and return native-extension inputs."""
    value = validate_array(array)

    normalized_size, flat_footprint, origins, count = _normalize_window(
        size, footprint, origin, value.ndim
    )
    normalized_mode = _normalize_mode(mode, value.ndim)

    if isinstance(cval, (bool, np.bool_)) or not isinstance(cval, numbers.Real):
        raise TypeError("cval must be a non-boolean real scalar")
    normalized_cval = float(value.dtype.type(cval))

    normalized_min_count = _integer_index(
        min_count, "min_count must be an integer, not a boolean"
    )
    volume = 1
    native_max = np.iinfo(np.uintp).max
    for length in normalized_size:
        volume *= length
        if volume > native_max:
            raise OverflowError("window volume exceeds the native index range")
    if normalized_min_count < 1 or normalized_min_count > count:
        raise ValueError(
            "min_count must be between 1 and the number of window elements"
        )

    normalized_ddof = _integer_index(ddof, "ddof must be an integer, not a boolean")
    if normalized_ddof < 0:
        raise ValueError("ddof must be nonnegative")
    if normalized_ddof > native_max:
        raise OverflowError("ddof exceeds the native index range")

    if normalize_array and (
        not value.flags.c_contiguous or type(value) is not np.ndarray
    ):
        value = np.require(value, dtype=value.dtype, requirements=("C", "A", "O", "W"))

    return FilterArguments(
        array=value,
        size=normalized_size,
        mode=normalized_mode,
        cval=normalized_cval,
        min_count=normalized_min_count,
        ddof=normalized_ddof,
        footprint=flat_footprint,
        origin=origins,
        count=count,
    )
