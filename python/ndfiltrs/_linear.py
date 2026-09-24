"""Weighted convolution and normalized missing-data smoothing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._api import BoundaryMode
from ._validation import FilterArguments, _integer_index, validate_filter_arguments

FloatArray = NDArray[np.float32] | NDArray[np.float64]


@dataclass(frozen=True)
class KernelArguments:
    """Validated geometry and coefficients for one native convolution."""

    filtering: FilterArguments
    weights: NDArray[np.float64]
    origin: tuple[int, ...]
    min_count: int


def _kernel_arguments(
    array: ArrayLike,
    weights: ArrayLike,
    mode: BoundaryMode | tuple[BoundaryMode, ...],
    cval: float,
    origin: int | tuple[int, ...],
    *,
    normalized: bool = False,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> KernelArguments:
    """Validate inputs before allocating native snapshots or returning empties."""
    filtering = validate_filter_arguments(array, 1, mode, cval, min_count=1)
    if np.ma.isMaskedArray(weights):
        raise TypeError("masked kernel weights are not supported")
    kernel = np.asarray(weights)
    if kernel.dtype.kind not in "iuf":
        raise TypeError("weights must contain non-boolean real numbers")
    if kernel.ndim != filtering.array.ndim:
        raise ValueError("weights must have the same rank as array")
    if any(length == 0 for length in kernel.shape):
        raise ValueError("weights must have nonempty dimensions")
    with np.errstate(over="ignore", invalid="ignore"):
        kernel = np.require(kernel, dtype=np.float64, requirements=("C", "A"))
    if not np.isfinite(kernel).all():
        raise ValueError("weights must be finite and representable in float64")

    if isinstance(origin, Sequence) and not isinstance(origin, (str, bytes)):
        origins = tuple(origin)
        if len(origins) != kernel.ndim:
            raise ValueError("origin must have one entry per array dimension")
    else:
        origins = (origin,) * kernel.ndim
    origins = tuple(
        _integer_index(value, "origin entries must be integers, not booleans")
        for value in origins
    )
    if any(
        not 0 <= length // 2 + offset < length
        for length, offset in zip(kernel.shape, origins, strict=True)
    ):
        raise ValueError("origin must place the anchor inside the kernel")

    if not isinstance(ignore_inf, (bool, np.bool_)):
        raise TypeError("ignore_inf must be a boolean")
    count = _integer_index(min_count, "min_count must be an integer, not a boolean")
    if normalized:
        if np.any(kernel < 0) or not np.any(kernel > 0):
            raise ValueError(
                "normalized weights must be nonnegative with positive support"
            )
        if not 1 <= count <= np.count_nonzero(kernel):
            raise ValueError(
                "min_count must be between 1 and the positive kernel support"
            )
    return KernelArguments(filtering, kernel, origins, count)


def convolve(
    array: ArrayLike,
    weights: ArrayLike,
    *,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> FloatArray:
    """Convolve an array with a finite, real N-D kernel.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input, rank one or higher.
    weights : array-like
        Finite real coefficients with the input rank and nonempty dimensions.
        Odd, even, signed, and zero-sum kernels are supported. No normalization
        is applied. Use singleton dimensions to leave axes unfiltered.
    mode : str or tuple of str, optional
        Boundary mode per input axis: `'reflect'`, `'constant'`, `'nearest'`,
        `'mirror'`, or `'wrap'`. Default is `'reflect'`.
    cval : float, optional
        Constant boundary value, cast to the input dtype. Default is `0.0`.
    origin : int or tuple of int, optional
        Offset from each kernel's center index. The anchor is
        ``weights.shape[axis] // 2 + origin[axis]`` and must lie in the kernel.

    Returns
    -------
    numpy.ndarray
        New owned, writable, C-contiguous array with the input shape and dtype.

    Notes
    -----
    Each output sums ``weights[j] * array[i + anchor - j]`` after boundary
    mapping. Zero coefficients skip their samples, including NaN and infinity.
    Other NaNs propagate locally; opposite weighted infinities produce NaN.
    An all-zero kernel returns positive zero. Accumulation uses compensated
    `float64` arithmetic with overflow recovery and one final dtype cast.
    Exact rounding is not guaranteed; fallback scaling can lose tiny terms.
    """
    arguments = _kernel_arguments(array, weights, mode, cval, origin)
    filtering = arguments.filtering
    return _core.convolve(
        filtering.array,
        arguments.weights,
        filtering.mode,
        filtering.cval,
        arguments.origin,
        False,
        False,
        False,
        1,
    )


def correlate(
    array: ArrayLike,
    weights: ArrayLike,
    *,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> FloatArray:
    """Correlate an array with a finite, real N-D kernel.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    weights : array-like
        Finite real coefficients with the input rank. Signed, zero-sum, odd,
        and even kernels are supported without normalization.
    mode : str or tuple of str, optional
        Boundary mode per input axis. Default is `'reflect'`; supported modes
        are the same as `convolve`.
    cval : float, optional
        Constant boundary value, cast to the input dtype. Default is `0.0`.
    origin : int or tuple of int, optional
        Kernel anchor offset as in `convolve`. Default is `0`.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype.

    Notes
    -----
    Each output sums ``weights[j] * array[i + j - anchor]``. Numerical,
    zero-coefficient, and invalid-value rules follow `convolve`.
    """
    arguments = _kernel_arguments(array, weights, mode, cval, origin)
    filtering = arguments.filtering
    return _core.convolve(
        filtering.array,
        arguments.weights,
        filtering.mode,
        filtering.cval,
        arguments.origin,
        True,
        False,
        False,
        1,
    )


def nanconvolve(
    array: ArrayLike,
    weights: ArrayLike,
    *,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> FloatArray:
    """Smooth with a nonnegative kernel, omitting NaNs and normalizing weights.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    weights : array-like
        Finite nonnegative coefficients with the input rank and at least one
        positive entry. Signed derivative kernels are rejected.
    mode : str or tuple of str, optional
        Boundary mode per input axis, as in `convolve`. Default is `'reflect'`.
    cval : float, optional
        Constant boundary value, cast to the input dtype and subject to the
        same omission rules as data. Default is `0.0`.
    origin : int or tuple of int, optional
        Kernel anchor offset as in `convolve`. Default is `0`.
    ignore_inf : bool, optional
        Also omit infinities when `True`. Default is `False`.
    min_count : int, optional
        Required retained positive-weight samples. Boundary repetitions count
        separately. Must lie between `1` and the positive kernel support.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype. Insufficient support
        returns canonical NaN, without an empty-window warning.

    Notes
    -----
    Divide each retained weighted sum by its retained weight sum. For example,
    weights ``[1, 1, 1]`` produce a mean, even on entirely finite input.
    Retained infinities of one sign produce that infinity; opposing signs
    produce NaN. Zero-weight positions never participate in values or counts.
    """
    arguments = _kernel_arguments(
        array,
        weights,
        mode,
        cval,
        origin,
        normalized=True,
        ignore_inf=ignore_inf,
        min_count=min_count,
    )
    filtering = arguments.filtering
    return _core.convolve(
        filtering.array,
        arguments.weights,
        filtering.mode,
        filtering.cval,
        arguments.origin,
        False,
        True,
        bool(ignore_inf),
        arguments.min_count,
    )


def nancorrelate(
    array: ArrayLike,
    weights: ArrayLike,
    *,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> FloatArray:
    """Apply normalized nonnegative correlation while omitting NaNs.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    weights : array-like
        Finite nonnegative coefficients with the input rank and positive support.
    mode : str or tuple of str, optional
        Boundary mode per input axis, as in `convolve`. Default is `'reflect'`.
    cval : float, optional
        Constant boundary sample, cast to the input dtype. Default is `0.0`.
    origin : int or tuple of int, optional
        Kernel anchor offset as in `convolve`. Default is `0`.
    ignore_inf : bool, optional
        Omit infinities as well as NaNs when `True`. Default is `False`.
    min_count : int, optional
        Required retained positive-weight samples. Default is `1`.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype; insufficient support
        produces canonical NaN.

    Notes
    -----
    Uses the orientation of `correlate` and the normalization, omission, and
    support-count rules of `nanconvolve`. Signed kernels are rejected.
    """
    arguments = _kernel_arguments(
        array,
        weights,
        mode,
        cval,
        origin,
        normalized=True,
        ignore_inf=ignore_inf,
        min_count=min_count,
    )
    filtering = arguments.filtering
    return _core.convolve(
        filtering.array,
        arguments.weights,
        filtering.mode,
        filtering.cval,
        arguments.origin,
        True,
        True,
        bool(ignore_inf),
        arguments.min_count,
    )
