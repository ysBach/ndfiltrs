"""Public Python filtering API."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._validation import FilterArguments, _integer_index, validate_filter_arguments

BoundaryMode = Literal["reflect", "constant", "nearest", "mirror", "wrap"]


def _window_filter(
    kind: str,
    arguments: FilterArguments,
    *,
    ignore_inf: bool | None = None,
    parameter: int = 0,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Run one native window reduction; `ignore_inf=None` propagates NaN."""
    return _core.window_filter(
        arguments.array,
        kind,
        arguments.size,
        arguments.footprint,
        arguments.origin,
        arguments.mode,
        arguments.cval,
        ignore_inf is not None,
        bool(ignore_inf),
        arguments.min_count,
        parameter,
    )


def _separable(method: object) -> bool:
    """Return whether `method` selects the separable mean."""
    if not isinstance(method, str) or method not in ("direct", "separable"):
        raise ValueError("method must be 'direct' or 'separable'")
    return method == "separable"


def _validate_method(method: object, allowed: tuple[str, ...]) -> None:
    """Reject non-string or unsupported algorithm choices before native dispatch."""
    if not isinstance(method, str) or method not in allowed:
        raise ValueError(f"method must be one of {allowed}")


def _separable_mean(
    arguments: FilterArguments, *, ignore_inf: bool | None
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Average rectangular windows one axis at a time with equal weights."""
    if (
        arguments.footprint is not None
        or any(arguments.origin)
        or any(length % 2 == 0 for length in arguments.size)
    ):
        raise ValueError("method='separable' needs an odd size, no footprint, origin 0")
    return _core.separable_smooth(
        arguments.array,
        tuple(range(arguments.array.ndim)),
        [[1.0] * length for length in arguments.size],
        arguments.mode,
        arguments.cval,
        ignore_inf is not None,
        bool(ignore_inf),
        arguments.min_count,
    )


def median_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    method: Literal["direct", "sortnet"] = "direct",
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply an N-dimensional median filter.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher. The function rejects unaligned arrays.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis. A tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank. The function uses positions where
        `footprint` is nonzero. Use it for
        disks, rings, or other non-rectangular windows, for example
        ``kernels.tophat(2, ndim=2)``. Pass `size` or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. `0` centers odd windows.
        Positive values move the window toward lower indices. Default is `0`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. A scalar broadcasts to every axis. Default is
        `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries. It is cast to the input dtype.
        Default is `0.0`.
    method : {'direct', 'sortnet'}, optional
        `'direct'` (default) selects each window's median independently.
        `'sortnet'` uses an exact comparator network with the same results,
        including even windows, footprints, and shifted origins.

    Returns
    -------
    numpy.ndarray
        New owned, writable, aligned, C-contiguous array with the input shape
        and dtype.

    Notes
    -----
    NaNs propagate. Infinities participate in the median. An even number of
    samples gives the midpoint of the two middle values. SciPy's
    `median_filter` returns the upper one instead.

    `'sortnet'` preserves the direct method's numerical semantics. It uses a
    comparator network and buffers samples across interior row runs, which can
    need more memory than `'direct'`. Neither method is fastest for every window.
    """
    arguments = validate_filter_arguments(
        array, size, mode, cval, footprint=footprint, origin=origin, min_count=1
    )
    _validate_method(method, ("direct", "sortnet"))
    return _window_filter(
        "median_network" if method == "sortnet" else "median", arguments
    )


def nanmedian_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ignore_inf: bool = False,
    min_count: int = 1,
    method: Literal["direct", "sortnet"] = "direct",
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply an N-dimensional median filter while omitting NaN values.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher. The function rejects unaligned arrays.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis. A tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank. The function uses positions where
        `footprint` is nonzero. Use it for
        disks, rings, or other non-rectangular windows, for example
        ``kernels.tophat(2, ndim=2)``. Pass `size` or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. `0` centers odd windows.
        Positive values move the window toward lower indices. Default is `0`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. A scalar broadcasts to every axis. Default is
        `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries. It follows the same omission
        policy as input values. Default is `0.0`.
    ignore_inf : bool, optional
        Omit positive and negative infinity when `True`. Default is `False`.
    min_count : int, optional
        Minimum retained samples required for a result. Default is `1`.
    method : {'direct', 'sortnet'}, optional
        `'direct'` (default) selects each window's median independently.
        `'sortnet'` uses an exact comparator network with the same omission
        rules and results, including footprints and shifted origins.

    Returns
    -------
    numpy.ndarray
        New owned, writable, aligned, C-contiguous array with the input shape
        and dtype. Positions below `min_count` contain canonical NaN.

    Notes
    -----
    This function always omits NaNs. It is not a drop-in SciPy replacement.
    SciPy's median filter does not expose these omission semantics.
    Both methods preserve the same retained-count, midpoint, and signed-zero
    rules. `'sortnet'` uses more scratch memory for interior row runs and is
    not universally faster than `'direct'`.
    """
    if not isinstance(ignore_inf, (bool, np.bool_)):
        raise TypeError("ignore_inf must be a boolean")
    arguments = validate_filter_arguments(
        array, size, mode, cval, footprint=footprint, origin=origin, min_count=min_count
    )
    _validate_method(method, ("direct", "sortnet"))
    return _window_filter(
        "median_network" if method == "sortnet" else "median",
        arguments,
        ignore_inf=bool(ignore_inf),
    )


def mean_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    method: Literal["direct", "separable", "boxcar"] = "direct",
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply an N-dimensional mean filter.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher. The function rejects unaligned arrays.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis. A tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank. The function uses positions where
        `footprint` is nonzero. Use it for
        disks, rings, or other non-rectangular windows, for example
        ``kernels.tophat(2, ndim=2)``. Pass `size` or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. `0` centers odd windows.
        Positive values move the window toward lower indices. Default is `0`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. A scalar broadcasts to every axis. Default is
        `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries. It is cast to the input dtype.
        Default is `0.0`.

    method : {'direct', 'separable', 'boxcar'}, optional
        `'direct'` (default) averages each window in one compensated pass.
        `'separable'` averages one axis at a time, costing about the sum of the
        window lengths per pixel instead of their product. It needs an odd
        `size`, no `footprint`, and `origin=0`, and can differ from `'direct'` in
        the last bits.
        `'boxcar'` uses exact rolling sums and rounds the finite mean once to
        the input dtype. It accepts even `size` and shifted `origin`, but no
        `footprint`. Results can differ from `'direct'` in the last bits.

    Returns
    -------
    numpy.ndarray
        New owned, writable, aligned, C-contiguous array with the input shape
        and dtype.

    Notes
    -----
    NaNs propagate only to windows containing them. Infinities of one sign
    produce that infinity. Opposite signs produce canonical NaN. With
    `method='direct'`, finite values use compensated `float64` accumulation
    and overflow rescaling, with one final cast to the input dtype.
    Subnormal contributions can underflow in
    overflow rescaling. Exact rounding is not guaranteed. Zero sums are `+0.0`.

    With `size` this is a boxcar (uniform) filter. `'direct'` evaluates each
    window independently in C order. `'boxcar'` keeps two full-image exact
    accumulator arrays plus support counts. Its memory depends on the input's
    exponent range and window support. Neither method is universally faster.
    Results for each method are independent of thread count.
    """
    arguments = validate_filter_arguments(
        array, size, mode, cval, footprint=footprint, origin=origin, min_count=1
    )
    _validate_method(method, ("direct", "separable", "boxcar"))
    if method == "separable":
        return _separable_mean(arguments, ignore_inf=None)
    if method == "boxcar" and arguments.footprint is not None:
        raise ValueError("method='boxcar' requires size, not footprint")
    return _window_filter("mean_rolling" if method == "boxcar" else "mean", arguments)


def nanmean_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ignore_inf: bool = False,
    min_count: int = 1,
    method: Literal["direct", "separable", "boxcar"] = "direct",
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply an N-dimensional mean filter while omitting NaN values.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher. The function rejects unaligned arrays.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis. A tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank. The function uses positions where
        `footprint` is nonzero. Use it for
        disks, rings, or other non-rectangular windows, for example
        ``kernels.tophat(2, ndim=2)``. Pass `size` or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. `0` centers odd windows.
        Positive values move the window toward lower indices. Default is `0`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. A scalar broadcasts to every axis. Default is
        `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries. It follows the same omission
        policy as input values. Default is `0.0`.
    ignore_inf : bool, optional
        Omit positive and negative infinity when `True`. Default is `False`.
    min_count : int, optional
        Minimum retained samples required for a result. Default is `1`.

    method : {'direct', 'separable', 'boxcar'}, optional
        `'direct'` (default) averages each window in one compensated pass.
        `'separable'` averages one axis at a time, costing about the sum of the
        window lengths per pixel instead of their product. It needs an odd
        `size`, no `footprint`, and `origin=0`, and can differ from `'direct'` in
        the last bits.
        `'boxcar'` uses exact rolling sums and rounds the retained finite mean
        once to the input dtype. It accepts even `size` and shifted `origin`,
        but no `footprint`. Results can differ from `'direct'` in the last bits.

    Returns
    -------
    numpy.ndarray
        New owned, writable, aligned, C-contiguous array with the input shape
        and dtype. Positions below `min_count` contain canonical NaN.

    Notes
    -----
    NaNs are always omitted, without warnings for empty neighborhoods. Retained
    infinities of one sign produce that infinity. Opposite signs produce
    canonical NaN. Boundary repetitions and `cval` count as separate samples.
    Accumulation, rounding, memory, and computational cost follow `mean_filter`.
    The filter retains finite hot or dead pixels. NaN omission does not identify
    bad pixels.
    """
    if not isinstance(ignore_inf, (bool, np.bool_)):
        raise TypeError("ignore_inf must be a boolean")
    arguments = validate_filter_arguments(
        array, size, mode, cval, footprint=footprint, origin=origin, min_count=min_count
    )
    _validate_method(method, ("direct", "separable", "boxcar"))
    if method == "separable":
        return _separable_mean(arguments, ignore_inf=bool(ignore_inf))
    if method == "boxcar" and arguments.footprint is not None:
        raise ValueError("method='boxcar' requires size, not footprint")
    return _window_filter(
        "mean_rolling" if method == "boxcar" else "mean",
        arguments,
        ignore_inf=bool(ignore_inf),
    )


def min_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply an N-dimensional minimum filter.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher. The function rejects unaligned arrays.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis. A tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank. The function uses positions where
        `footprint` is nonzero. Use it for
        disks, rings, or other non-rectangular windows, for example
        ``kernels.tophat(2, ndim=2)``. Pass `size` or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. `0` centers odd windows.
        Positive values move the window toward lower indices. Default is `0`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. A scalar broadcasts to every axis. Default is
        `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries. It is cast to the input dtype.
        Default is `0.0`.

    Returns
    -------
    numpy.ndarray
        New owned, writable, aligned, C-contiguous array with the input shape
        and dtype.

    Notes
    -----
    NaNs propagate as canonical NaN only to windows containing them. Infinities
    participate. Extrema use total ordering, with `-0.0` below `+0.0`, so a
    minimum over both signed zeros is `-0.0`.
    """
    arguments = validate_filter_arguments(
        array, size, mode, cval, footprint=footprint, origin=origin, min_count=1
    )
    return _window_filter("min", arguments)


def max_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply an N-dimensional maximum filter.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher. The function rejects unaligned arrays.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis. A tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank. The function uses positions where
        `footprint` is nonzero. Use it for
        disks, rings, or other non-rectangular windows, for example
        ``kernels.tophat(2, ndim=2)``. Pass `size` or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. `0` centers odd windows.
        Positive values move the window toward lower indices. Default is `0`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. A scalar broadcasts to every axis. Default is
        `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries. It is cast to the input dtype.
        Default is `0.0`.

    Returns
    -------
    numpy.ndarray
        New owned, writable, aligned, C-contiguous array with the input shape
        and dtype.

    Notes
    -----
    NaNs propagate as canonical NaN only to windows containing them. Infinities
    participate. Extrema use total ordering, with `-0.0` below `+0.0`, so a
    maximum over both signed zeros is `+0.0`.
    """
    arguments = validate_filter_arguments(
        array, size, mode, cval, footprint=footprint, origin=origin, min_count=1
    )
    return _window_filter("max", arguments)


def nanmin_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply an N-dimensional minimum filter while omitting NaN values.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher. The function rejects unaligned arrays.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis. A tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank. The function uses positions where
        `footprint` is nonzero. Use it for
        disks, rings, or other non-rectangular windows, for example
        ``kernels.tophat(2, ndim=2)``. Pass `size` or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. `0` centers odd windows.
        Positive values move the window toward lower indices. Default is `0`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. A scalar broadcasts to every axis. Default is
        `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries, cast to the input dtype.
        It follows the same omission policy as input values. Default is `0.0`.
    ignore_inf : bool, optional
        Omit positive and negative infinity when `True`. Default is `False`.
    min_count : int, optional
        Minimum retained samples required for a result, between `1` and the
        number of window elements. Default is `1`.

    Returns
    -------
    numpy.ndarray
        New owned, writable, aligned, C-contiguous array with the input shape
        and dtype. Positions below `min_count` contain canonical NaN.

    Notes
    -----
    NaNs are always omitted. Retained infinities participate. Extrema use total
    ordering, with `-0.0` below `+0.0`, so a minimum over both signed zeros is
    `-0.0`. Boundary repetitions and retained `cval` values count as separate
    samples toward `min_count`.
    """
    if not isinstance(ignore_inf, (bool, np.bool_)):
        raise TypeError("ignore_inf must be a boolean")
    arguments = validate_filter_arguments(
        array, size, mode, cval, footprint=footprint, origin=origin, min_count=min_count
    )
    return _window_filter("min", arguments, ignore_inf=bool(ignore_inf))


def nanmax_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply an N-dimensional maximum filter while omitting NaN values.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher. The function rejects unaligned arrays.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis. A tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank. The function uses positions where
        `footprint` is nonzero. Use it for
        disks, rings, or other non-rectangular windows, for example
        ``kernels.tophat(2, ndim=2)``. Pass `size` or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. `0` centers odd windows.
        Positive values move the window toward lower indices. Default is `0`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. A scalar broadcasts to every axis. Default is
        `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries, cast to the input dtype.
        It follows the same omission policy as input values. Default is `0.0`.
    ignore_inf : bool, optional
        Omit positive and negative infinity when `True`. Default is `False`.
    min_count : int, optional
        Minimum retained samples required for a result, between `1` and the
        number of window elements. Default is `1`.

    Returns
    -------
    numpy.ndarray
        New owned, writable, aligned, C-contiguous array with the input shape
        and dtype. Positions below `min_count` contain canonical NaN.

    Notes
    -----
    NaNs are always omitted. Retained infinities participate. Extrema use total
    ordering, with `-0.0` below `+0.0`, so a maximum over both signed zeros is
    `+0.0`. Boundary repetitions and retained `cval` values count as separate
    samples toward `min_count`.
    """
    if not isinstance(ignore_inf, (bool, np.bool_)):
        raise TypeError("ignore_inf must be a boolean")
    arguments = validate_filter_arguments(
        array, size, mode, cval, footprint=footprint, origin=origin, min_count=min_count
    )
    return _window_filter("max", arguments, ignore_inf=bool(ignore_inf))


def var_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ddof: int = 0,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply an N-dimensional variance filter.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher. The function rejects unaligned arrays.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis. A tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank. The function uses positions where
        `footprint` is nonzero. Use it for
        disks, rings, or other non-rectangular windows, for example
        ``kernels.tophat(2, ndim=2)``. Pass `size` or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. `0` centers odd windows.
        Positive values move the window toward lower indices. Default is `0`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. A scalar broadcasts to every axis. Default is
        `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries. It is cast to the input dtype.
        Default is `0.0`.
    ddof : int, optional
        Nonnegative degrees of freedom subtracted from the sample count in
        the variance divisor. Accepts the integer-index protocol, excluding
        booleans, and must fit the native unsigned index type. Default is `0`.

    Returns
    -------
    numpy.ndarray
        New owned, writable, aligned, C-contiguous array with the input shape
        and dtype. Windows containing NaN or infinity, or with sample count
        less than or equal to `ddof`, contain canonical NaN.

    Raises
    ------
    TypeError
        If `ddof` is a boolean or does not support the integer-index protocol.
    ValueError
        If `ddof` is negative.
    OverflowError
        If `ddof` exceeds the native unsigned index range.

    Notes
    -----
    Finite values use stable `float64` accumulation with one final cast to the
    input dtype. Boundary repetitions and `cval` count as separate samples.
    NaNs and infinities affect only windows containing them. The result measures
    local scatter, including spatial structure. It is not a measurement-noise
    estimate. A variance outside the output dtype's range can overflow.
    """
    arguments = validate_filter_arguments(
        array,
        size,
        mode,
        cval,
        footprint=footprint,
        origin=origin,
        min_count=1,
        ddof=ddof,
    )
    return _window_filter("var", arguments, parameter=arguments.ddof)


def std_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ddof: int = 0,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply an N-dimensional standard deviation filter.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher. The function rejects unaligned arrays.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis. A tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank. The function uses positions where
        `footprint` is nonzero. Use it for
        disks, rings, or other non-rectangular windows, for example
        ``kernels.tophat(2, ndim=2)``. Pass `size` or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. `0` centers odd windows.
        Positive values move the window toward lower indices. Default is `0`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. A scalar broadcasts to every axis. Default is
        `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries. It is cast to the input dtype.
        Default is `0.0`.
    ddof : int, optional
        Nonnegative degrees of freedom subtracted from the sample count in
        the variance divisor before taking its square root. Accepts the
        integer-index protocol, excluding booleans, and must fit the native
        unsigned index type. Default is `0`.

    Returns
    -------
    numpy.ndarray
        New owned, writable, aligned, C-contiguous array with the input shape
        and dtype. Windows containing NaN or infinity, or with sample count
        less than or equal to `ddof`, contain canonical NaN.

    Raises
    ------
    TypeError
        If `ddof` is a boolean or does not support the integer-index protocol.
    ValueError
        If `ddof` is negative.
    OverflowError
        If `ddof` exceeds the native unsigned index range.

    Notes
    -----
    Finite values use stable `float64` accumulation with one final cast to the
    input dtype. Boundary repetitions and `cval` count as separate samples.
    NaNs and infinities affect only windows containing them. The result measures
    local scatter, including spatial structure. It is not a measurement-noise
    estimate. A standard deviation outside the output dtype's range can overflow.
    """
    arguments = validate_filter_arguments(
        array,
        size,
        mode,
        cval,
        footprint=footprint,
        origin=origin,
        min_count=1,
        ddof=ddof,
    )
    return _window_filter("std", arguments, parameter=arguments.ddof)


def nanvar_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ddof: int = 0,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply an N-dimensional variance filter while omitting NaN values.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher. The function rejects unaligned arrays.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis. A tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank. The function uses positions where
        `footprint` is nonzero. Use it for
        disks, rings, or other non-rectangular windows, for example
        ``kernels.tophat(2, ndim=2)``. Pass `size` or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. `0` centers odd windows.
        Positive values move the window toward lower indices. Default is `0`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. A scalar broadcasts to every axis. Default is
        `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries, cast to the input dtype.
        It follows the same omission policy as input values. Default is `0.0`.
    ddof : int, optional
        Nonnegative degrees of freedom subtracted from the retained sample
        count in the variance divisor. Accepts the integer-index protocol,
        excluding booleans, and must fit the native unsigned index type.
        Default is `0`.
    ignore_inf : bool, optional
        Omit positive and negative infinity when `True`. Default is `False`.
    min_count : int, optional
        Minimum retained samples required for a result, between `1` and the
        number of window elements. Default is `1`.

    Returns
    -------
    numpy.ndarray
        New owned, writable, aligned, C-contiguous array with the input shape
        and dtype. Windows retaining infinity, fewer than `min_count` samples,
        or a sample count less than or equal to `ddof`, contain canonical NaN.

    Raises
    ------
    TypeError
        If `ddof` is a boolean or does not support the integer-index protocol.
    ValueError
        If `ddof` is negative.
    OverflowError
        If `ddof` exceeds the native unsigned index range.

    Notes
    -----
    NaNs are always omitted. Finite values use stable `float64` accumulation
    with one final cast to the input dtype. Boundary repetitions and retained
    `cval` values count as separate samples. The result measures local scatter,
    including spatial structure. It is not a measurement-noise estimate.
    A variance outside the output dtype's range can overflow.
    """
    if not isinstance(ignore_inf, (bool, np.bool_)):
        raise TypeError("ignore_inf must be a boolean")
    arguments = validate_filter_arguments(
        array,
        size,
        mode,
        cval,
        footprint=footprint,
        origin=origin,
        min_count=min_count,
        ddof=ddof,
    )
    return _window_filter(
        "var", arguments, ignore_inf=bool(ignore_inf), parameter=arguments.ddof
    )


def nanstd_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ddof: int = 0,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply an N-dimensional standard deviation filter while omitting NaNs.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher. The function rejects unaligned arrays.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis. A tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank. The function uses positions where
        `footprint` is nonzero. Use it for
        disks, rings, or other non-rectangular windows, for example
        ``kernels.tophat(2, ndim=2)``. Pass `size` or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. `0` centers odd windows.
        Positive values move the window toward lower indices. Default is `0`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. A scalar broadcasts to every axis. Default is
        `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries, cast to the input dtype.
        It follows the same omission policy as input values. Default is `0.0`.
    ddof : int, optional
        Nonnegative degrees of freedom subtracted from the retained sample
        count in the variance divisor before taking its square root. Accepts
        the integer-index protocol, excluding booleans, and must fit the native
        unsigned index type. Default is `0`.
    ignore_inf : bool, optional
        Omit positive and negative infinity when `True`. Default is `False`.
    min_count : int, optional
        Minimum retained samples required for a result, between `1` and the
        number of window elements. Default is `1`.

    Returns
    -------
    numpy.ndarray
        New owned, writable, aligned, C-contiguous array with the input shape
        and dtype. Windows retaining infinity, fewer than `min_count` samples,
        or a sample count less than or equal to `ddof`, contain canonical NaN.

    Raises
    ------
    TypeError
        If `ddof` is a boolean or does not support the integer-index protocol.
    ValueError
        If `ddof` is negative.
    OverflowError
        If `ddof` exceeds the native unsigned index range.

    Notes
    -----
    NaNs are always omitted. Finite values use stable `float64` accumulation
    with one final cast to the input dtype. Boundary repetitions and retained
    `cval` values count as separate samples. The result measures local scatter,
    including spatial structure. It is not a measurement-noise estimate.
    A standard deviation outside the output dtype's range can overflow.
    """
    if not isinstance(ignore_inf, (bool, np.bool_)):
        raise TypeError("ignore_inf must be a boolean")
    arguments = validate_filter_arguments(
        array,
        size,
        mode,
        cval,
        footprint=footprint,
        origin=origin,
        min_count=min_count,
        ddof=ddof,
    )
    return _window_filter(
        "std", arguments, ignore_inf=bool(ignore_inf), parameter=arguments.ddof
    )


def rank_filter(
    array: ArrayLike,
    rank: int,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Return the value of a given rank in each sorted window.

    Parameters
    ----------
    array : array-like
        Dense, aligned array with native-endian `float32` or `float64` dtype
        and rank one or higher. The function rejects unaligned arrays.
    rank : int
        Zero-based position in ascending order. `0` is the minimum. Negative
        values count from the maximum (`-1`), as in SciPy. It must satisfy
        ``-n <= rank < n``, where `n` is the number of window elements.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis. A tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank. The function uses positions where
        `footprint` is nonzero. Pass `size` or
        `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in `median_filter`. Default is `0`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. Default is `'reflect'`.
    cval : float, optional
        Value used beyond `constant` boundaries, cast to the input dtype.
        Default is `0.0`.

    Returns
    -------
    numpy.ndarray
        New owned, C-contiguous array with the input shape and dtype.

    Notes
    -----
    NaNs propagate. Ordering is total: `-0.0` sorts before `+0.0`, matching
    `min_filter` and `max_filter`. There is no NaN-omitting rank filter: after
    omission the sample count varies, so a fixed rank has no stable meaning.
    `nanmin_filter`, `nanmax_filter`, `nanmedian_filter`, and
    `nanpercentile_filter` cover the rank-like statistics.
    """
    position = _integer_index(rank, "rank must be an integer, not a boolean")
    arguments = validate_filter_arguments(
        array, size, mode, cval, footprint=footprint, origin=origin, min_count=1
    )
    if not -arguments.count <= position < arguments.count:
        raise ValueError("rank must satisfy -n <= rank < n for n window elements")
    return _window_filter("rank", arguments, parameter=position % arguments.count)


def get_num_threads() -> int:
    """Return the number of threads in Rayon's global pool.

    Returns
    -------
    int
        Current global Rayon thread count.

    Notes
    -----
    Set `RAYON_NUM_THREADS` before the first Rayon-using call in the process.
    """
    return _core.get_num_threads()
