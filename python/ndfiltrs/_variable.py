"""Mean filters whose window size changes from pixel to pixel."""

from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._api import BoundaryMode
from ._validation import _integer_index, _normalize_mode, validate_array

_SIZE_MAX = np.iinfo(np.uint64).max


def _window_sizes(
    size: object, radius: object, shape: tuple[int, ...]
) -> tuple[NDArray[np.uint64], bool]:
    """Return a C-contiguous ``uint64`` size map and whether it is per axis.

    Exactly one of `size` and `radius` must be given. A radius ``r`` becomes
    the window size ``2 * r + 1``.
    """
    if (size is None) == (radius is None):
        raise ValueError("pass exactly one of size or radius")
    name, value = ("size", size) if radius is None else ("radius", radius)
    if np.ma.isMaskedArray(value):
        raise TypeError(f"{name} must not be a masked array")
    entries = np.asarray(value)
    if entries.dtype.kind not in "iu":
        raise TypeError(f"{name} must have an integer dtype")
    if entries.shape == shape:
        per_axis = False
    elif entries.shape == (len(shape), *shape):
        per_axis = True
    else:
        raise ValueError(f"{name} must have the array's shape or (ndim, *array.shape)")
    if entries.size and entries.min() < (1 if radius is None else 0):
        raise ValueError(
            "size entries must be positive"
            if radius is None
            else "radius entries must be nonnegative"
        )
    if radius is None:
        return np.ascontiguousarray(entries, dtype=np.uint64), per_axis
    if entries.size and int(entries.max()) > (_SIZE_MAX - 1) // 2:
        raise OverflowError("radius exceeds the native index range")
    sizes = np.ascontiguousarray(entries, dtype=np.uint64)
    return sizes * np.uint64(2) + np.uint64(1), per_axis


def _variable_mean(
    array: ArrayLike,
    size: object,
    radius: object,
    mode: object,
    cval: object,
    *,
    ignore_inf: bool | None,
    min_count: object,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Validate arguments and run the native mean; `ignore_inf=None` propagates."""
    value = validate_array(array)
    sizes, per_axis = _window_sizes(size, radius, value.shape)
    modes = _normalize_mode(mode, value.ndim)
    if isinstance(cval, (bool, np.bool_)) or not isinstance(cval, numbers.Real):
        raise TypeError("cval must be a non-boolean real scalar")
    count = _integer_index(min_count, "min_count must be an integer, not a boolean")
    if count < 1:
        raise ValueError("min_count must be at least 1")
    if count > _SIZE_MAX:
        raise OverflowError("min_count exceeds the native index range")
    if not value.flags.c_contiguous or type(value) is not np.ndarray:
        value = np.require(value, dtype=value.dtype, requirements=("C", "A", "O", "W"))
    return _core.variable_mean(
        value,
        sizes,
        per_axis,
        modes,
        float(value.dtype.type(cval)),
        ignore_inf is not None,
        bool(ignore_inf),
        count,
    )


def variable_mean_filter(
    array: ArrayLike,
    size: ArrayLike | None = None,
    *,
    radius: ArrayLike | None = None,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply a mean filter with a separate rectangular window at each pixel.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher.
    size : array-like of int, optional
        Window size per pixel. Shape ``array.shape`` uses one size on every
        axis. Shape ``(array.ndim, *array.shape)`` gives one size per axis:
        ``size[a]`` is the map for axis ``a``. Entries must be positive.
        Pixel ``i`` averages input indices ``i - s // 2`` through
        ``i - s // 2 + s - 1`` on each axis, as `mean_filter` places a
        window of size ``s`` with ``origin=0``. Pass `size` or `radius`, not
        both.
    radius : array-like of int, optional
        Window half-width per pixel, with the same shapes as `size`. A radius
        ``r`` gives the centered window size ``2 * r + 1``. Entries must be
        nonnegative.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. A scalar broadcasts to every axis. Default is
        `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries. Default is `0.0`.

    Returns
    -------
    numpy.ndarray
        New C-contiguous array with the input shape and dtype. A window with
        NaN gives NaN. Infinities follow `mean_filter`.

    Notes
    -----
    Size maps must have an integer dtype. Round or clip floating-point maps
    before the call. Windows can exceed the axis length. Boundary repetitions
    and `cval` count as separate samples.

    The filter sums in float64 with summed-area tables (integral images), so
    the cost per pixel does not depend on the window size. The tables are
    built with one compensated prefix pass per axis, rounded to float64. A
    window sum subtracts table entries, so its absolute error scales with the
    prefix sums rather than the window sum. Derived first-order bound for a
    window inside the array: ``2**-53 * 2**ndim * ndim * sum(abs(array))``
    per window sum. Boundary windows combine more table entries. Use
    ``mean_filter(method='boxcar')`` for exact fixed-size windows. A finite
    input whose absolute sum overflows float64 raises `ValueError`.

    One huge finite value, such as a ``1e30`` sentinel, enters most table
    entries. Its rounding error then appears in windows far from it. Set such
    values to NaN and use `nanvariable_mean_filter`.

    The tables use 8 bytes per element of a grid with shape
    ``array.shape + 1``. Each nonfinite category present in the input (NaN,
    ``+inf``, ``-inf``) adds one more such table.

    Examples
    --------
    >>> import numpy as np
    >>> import ndfiltrs as nf
    >>> row = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> nf.variable_mean_filter(row, radius=np.array([0, 1, 1, 2, 0]), mode="nearest")
    array([1. , 2. , 3. , 3.8, 5. ])
    """
    return _variable_mean(array, size, radius, mode, cval, ignore_inf=None, min_count=1)


def nanvariable_mean_filter(
    array: ArrayLike,
    size: ArrayLike | None = None,
    *,
    radius: ArrayLike | None = None,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply a per-pixel-window mean filter while omitting NaN values.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher.
    size : array-like of int, optional
        Window size per pixel, as in `variable_mean_filter`.
    radius : array-like of int, optional
        Window half-width per pixel, as in `variable_mean_filter`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. Default is `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries. It follows the same omission
        policy as input values. Default is `0.0`.
    ignore_inf : bool, optional
        Omit positive and negative infinity when `True`. Default is `False`.
    min_count : int, optional
        Minimum retained samples for a result. It must be at least `1`. A
        window with fewer elements than `min_count` gives NaN. Default is `1`.

    Returns
    -------
    numpy.ndarray
        New C-contiguous array with the input shape and dtype. Positions below
        `min_count` contain NaN.

    Notes
    -----
    Accuracy, memory, and window placement follow `variable_mean_filter`.
    NaNs are omitted without warnings for empty windows. Retained infinities
    of one sign give that infinity. Opposite signs give NaN.
    """
    if not isinstance(ignore_inf, (bool, np.bool_)):
        raise TypeError("ignore_inf must be a boolean")
    return _variable_mean(
        array,
        size,
        radius,
        mode,
        cval,
        ignore_inf=bool(ignore_inf),
        min_count=min_count,
    )
