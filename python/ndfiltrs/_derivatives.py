"""Discrete N-dimensional derivative filters matching SciPy conventions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._api import BoundaryMode
from ._linear import correlate
from ._validation import _integer_index, validate_array

FloatArray = NDArray[np.float32] | NDArray[np.float64]

_DIFF_1D = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
_SOBEL_SMOOTH_1D = np.array([1.0, 2.0, 1.0], dtype=np.float64)
_PREWITT_SMOOTH_1D = np.array([1.0, 1.0, 1.0], dtype=np.float64)
_LAPLACE_DIFF2_1D = np.array([1.0, -2.0, 1.0], dtype=np.float64)


def _normalize_axis(axis: object, ndim: int) -> int:
    """Validate and wrap an array axis index."""
    index = _integer_index(axis, "axis must be an integer, not a boolean")
    if not -ndim <= index < ndim:
        raise ValueError(f"axis {index} is out of bounds for array of dimension {ndim}")
    return index % ndim


def _build_derivative_kernel(
    ndim: int, axis: int, smooth_1d: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Build an N-D separable derivative kernel via outer products."""
    kernel = np.array(1.0, dtype=np.float64)
    for dim in range(ndim):
        factor = _DIFF_1D if dim == axis else smooth_1d
        kernel = np.multiply.outer(kernel, factor)
    return np.ascontiguousarray(kernel, dtype=np.float64)


def _build_laplace_kernel(ndim: int) -> NDArray[np.float64]:
    """Build an N-D Laplace correlation kernel as sum of 1-D second differences."""
    shape = (3,) * ndim
    kernel = np.zeros(shape, dtype=np.float64)
    for dim in range(ndim):
        idx = [1] * ndim
        idx[dim] = slice(None)
        kernel[tuple(idx)] += _LAPLACE_DIFF2_1D
    return np.ascontiguousarray(kernel, dtype=np.float64)


def sobel(
    array: ArrayLike,
    axis: int = -1,
    *,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
) -> FloatArray:
    """Calculate an N-dimensional Sobel filter along a specified axis.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input of rank one or higher.
    axis : int, optional
        The axis of `array` along which to calculate the gradient. Default is `-1`.
    mode : str or tuple of str, optional
        Boundary mode per array axis: `'reflect'`, `'constant'`, `'nearest'`,
        `'mirror'`, or `'wrap'`. Default is `'reflect'`.
    cval : float, optional
        Constant boundary value, cast to the input dtype. Default is `0.0`.

    Returns
    -------
    numpy.ndarray
        Filtered array of the same shape and floating dtype as `array`.

    Notes
    -----
    The filter applies an N-dimensional correlation kernel formed by an outer
    product. Along `axis`, the factor is the central difference ``[-1, 0, 1]``.
    Every other axis uses the smoothing filter ``[1, 2, 1]``.
    For 1-D inputs, this reduces to ``[-1, 0, 1]``.

    The sign convention yields positive values where sample values increase with
    array index (gradient pointing in the positive coordinate direction).

    Sample NaNs follow standard correlation rules. Zero kernel coefficients
    omit their corresponding sample. Nonzero coefficients propagate NaNs.

    There are no NaN-aware variants. Use `nanconvolve` or `nancorrelate` with
    explicit kernels if normalized missing-data handling is required.
    """
    valid_array = validate_array(array)
    norm_axis = _normalize_axis(axis, valid_array.ndim)
    kernel = _build_derivative_kernel(valid_array.ndim, norm_axis, _SOBEL_SMOOTH_1D)
    return correlate(valid_array, kernel, mode=mode, cval=cval)


def prewitt(
    array: ArrayLike,
    axis: int = -1,
    *,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
) -> FloatArray:
    """Calculate an N-dimensional Prewitt filter along a specified axis.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input of rank one or higher.
    axis : int, optional
        The axis of `array` along which to calculate the gradient. Default is `-1`.
    mode : str or tuple of str, optional
        Boundary mode per array axis: `'reflect'`, `'constant'`, `'nearest'`,
        `'mirror'`, or `'wrap'`. Default is `'reflect'`.
    cval : float, optional
        Constant boundary value, cast to the input dtype. Default is `0.0`.

    Returns
    -------
    numpy.ndarray
        Filtered array of the same shape and floating dtype as `array`.

    Notes
    -----
    The filter applies an N-dimensional correlation kernel formed by an outer
    product. Along `axis`, the factor is the central difference ``[-1, 0, 1]``.
    Every other axis uses the smoothing filter ``[1, 1, 1]``.
    For 1-D inputs, this reduces to ``[-1, 0, 1]``.

    The sign convention yields positive values where sample values increase with
    array index (gradient pointing in the positive coordinate direction).

    Sample NaNs follow standard correlation rules. Zero kernel coefficients
    omit their corresponding sample. Nonzero coefficients propagate NaNs.

    There are no NaN-aware variants. Use `nanconvolve` or `nancorrelate` with
    explicit kernels if normalized missing-data handling is required.
    """
    valid_array = validate_array(array)
    norm_axis = _normalize_axis(axis, valid_array.ndim)
    kernel = _build_derivative_kernel(valid_array.ndim, norm_axis, _PREWITT_SMOOTH_1D)
    return correlate(valid_array, kernel, mode=mode, cval=cval)


def laplace(
    array: ArrayLike,
    *,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
) -> FloatArray:
    """Calculate an N-dimensional Laplace filter using second differences.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input of rank one or higher.
    mode : str or tuple of str, optional
        Boundary mode per array axis: `'reflect'`, `'constant'`, `'nearest'`,
        `'mirror'`, or `'wrap'`. Default is `'reflect'`.
    cval : float, optional
        Constant boundary value, cast to the input dtype. Default is `0.0`.

    Returns
    -------
    numpy.ndarray
        Filtered array of the same shape and floating dtype as `array`.

    Notes
    -----
    The filter applies an N-dimensional correlation kernel. It forms this kernel
    by summing the discrete second-derivative operator ``[1, -2, 1]`` along each
    axis. For 1-D inputs, this reduces to ``[1, -2, 1]``.

    The sign convention matches the standard positive discrete second difference
    ``f[i-1] - 2*f[i] + f[i+1]`` along each axis.

    Sample NaNs follow standard correlation rules. Zero kernel coefficients
    omit their corresponding sample. Nonzero coefficients propagate NaNs.

    There are no NaN-aware variants. Use `nanconvolve` or `nancorrelate` with
    explicit kernels if normalized missing-data handling is required.
    """
    valid_array = validate_array(array)
    kernel = _build_laplace_kernel(valid_array.ndim)
    return correlate(valid_array, kernel, mode=mode, cval=cval)
