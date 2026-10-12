"""Grey-scale morphology with flat structuring elements.

Erosion is `min_filter`. The functions here use it and `max_filter`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._api import BoundaryMode, max_filter, min_filter
from ._validation import _normalize_window, validate_array

FloatArray = NDArray[np.float32] | NDArray[np.float64]


def grey_dilation(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> FloatArray:
    """Dilate: take the maximum over the reflected structuring element.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Flat structuring element with the array's rank. Nonzero entries select
        samples. Pass `size` or `footprint`, not both.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. Default is `'reflect'`.
    cval : float, optional
        Value beyond `constant` boundaries. Default is `0.0`.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. Default is `0`.

    Returns
    -------
    numpy.ndarray
        New array with the input shape and dtype.

    Notes
    -----
    Erosion needs no separate function: it is `min_filter`. This function
    matches `scipy.ndimage.grey_dilation` for flat structuring elements.
    It reflects the footprint and negates the origin, then subtracts one
    on even axes.
    NaNs propagate.
    """
    value = validate_array(array)
    shape, _, origins, _ = _normalize_window(size, footprint, origin, value.ndim)
    if footprint is not None:
        footprint = np.asarray(footprint)[(slice(None, None, -1),) * value.ndim]
    reflected = tuple(
        -shift - (1 - length % 2) for length, shift in zip(shape, origins, strict=True)
    )
    return max_filter(
        value, size, footprint=footprint, mode=mode, cval=cval, origin=reflected
    )


def grey_opening(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> FloatArray:
    """Open: erode, then dilate. Removes bright features smaller than the window.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Flat structuring element with the array's rank. Nonzero entries select
        samples. Pass `size` or `footprint`, not both.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. Default is `'reflect'`.
    cval : float, optional
        Value beyond `constant` boundaries. Default is `0.0`.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. Default is `0`.

    Returns
    -------
    numpy.ndarray
        New array with the input shape and dtype.

    Notes
    -----
    NaNs propagate. The boundary mode also applies to the intermediate erosion.
    """
    return grey_dilation(
        min_filter(
            array, size, footprint=footprint, mode=mode, cval=cval, origin=origin
        ),
        size,
        footprint=footprint,
        mode=mode,
        cval=cval,
        origin=origin,
    )


def grey_closing(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> FloatArray:
    """Close: dilate, then erode. Removes dark features smaller than the window.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Flat structuring element with the array's rank. Nonzero entries select
        samples. Pass `size` or `footprint`, not both.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. Default is `'reflect'`.
    cval : float, optional
        Value beyond `constant` boundaries. Default is `0.0`.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. Default is `0`.

    Returns
    -------
    numpy.ndarray
        New array with the input shape and dtype.

    Notes
    -----
    NaNs propagate. The boundary mode also applies to the intermediate dilation.
    """
    return min_filter(
        grey_dilation(
            array, size, footprint=footprint, mode=mode, cval=cval, origin=origin
        ),
        size,
        footprint=footprint,
        mode=mode,
        cval=cval,
        origin=origin,
    )


def white_tophat(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> FloatArray:
    """Return the input minus its grey opening: bright features smaller than the window.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Flat structuring element with the array's rank. Nonzero entries select
        samples. Pass `size` or `footprint`, not both.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. Default is `'reflect'`.
    cval : float, optional
        Value beyond `constant` boundaries. Default is `0.0`.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. Default is `0`.

    Returns
    -------
    numpy.ndarray
        New array with the input shape and dtype.

    Notes
    -----
    Useful for removing a smooth background under compact bright sources.
    Opposite infinities give NaN. This function performs morphology.
    For the disk-shaped smoothing kernel Astropy calls a top-hat, use `kernels.tophat`.
    """
    value = validate_array(array)
    return value - grey_opening(
        value, size, footprint=footprint, mode=mode, cval=cval, origin=origin
    )


def black_tophat(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> FloatArray:
    """Return the grey closing minus the input: dark features smaller than the window.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Flat structuring element with the array's rank. Nonzero entries select
        samples. Pass `size` or `footprint`, not both.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. Default is `'reflect'`.
    cval : float, optional
        Value beyond `constant` boundaries. Default is `0.0`.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. Default is `0`.

    Returns
    -------
    numpy.ndarray
        New array with the input shape and dtype.

    Notes
    -----
    Opposite infinities give NaN.
    """
    value = validate_array(array)
    return (
        grey_closing(
            value, size, footprint=footprint, mode=mode, cval=cval, origin=origin
        )
        - value
    )


def morphological_gradient(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> FloatArray:
    """Return dilation minus erosion: the local max-min range.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Flat structuring element with the array's rank. Nonzero entries select
        samples. Pass `size` or `footprint`, not both.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. Default is `'reflect'`.
    cval : float, optional
        Value beyond `constant` boundaries. Default is `0.0`.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. Default is `0`.

    Returns
    -------
    numpy.ndarray
        New array with the input shape and dtype.

    Notes
    -----
    Opposite infinities give NaN.
    """
    value = validate_array(array)
    return grey_dilation(
        value, size, footprint=footprint, mode=mode, cval=cval, origin=origin
    ) - min_filter(
        value, size, footprint=footprint, mode=mode, cval=cval, origin=origin
    )


def morphological_laplace(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> FloatArray:
    """Return dilation plus erosion minus twice the input.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Flat structuring element with the array's rank. Nonzero entries select
        samples. Pass `size` or `footprint`, not both.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'} or tuple, optional
        Boundary mode per axis. Default is `'reflect'`.
    cval : float, optional
        Value beyond `constant` boundaries. Default is `0.0`.
    origin : int or tuple of int, optional
        Window shift per axis, as in SciPy. Default is `0`.

    Returns
    -------
    numpy.ndarray
        New array with the input shape and dtype.

    Notes
    -----
    Arithmetic uses the input dtype, as in SciPy. Infinities can give NaN.
    """
    value = validate_array(array)
    return (
        grey_dilation(
            value, size, footprint=footprint, mode=mode, cval=cval, origin=origin
        )
        + min_filter(
            value, size, footprint=footprint, mode=mode, cval=cval, origin=origin
        )
        - 2 * value
    )
