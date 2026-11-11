"""Convolution with variance propagation and matched filtering."""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from . import _core
from ._api import BoundaryMode
from ._linear import FloatArray, KernelArguments, _kernel_arguments

NoiseType = Literal["variance", "inverse_variance"]
_NOISE_TYPES = ("variance", "inverse_variance")
_WEIGHTINGS = ("inverse_variance", "kernel")


class MatchedFilterResult(NamedTuple):
    """Maps returned by `matched_filter`.

    Attributes
    ----------
    snr : numpy.ndarray
        Signal-to-noise ratio of the filtered sum.
    amplitude : numpy.ndarray
        Estimated template amplitude at each position.
    variance : numpy.ndarray
        Variance of `amplitude`.
    """

    snr: FloatArray
    amplitude: FloatArray
    variance: FloatArray


def _noise_array(noise: ArrayLike, arguments: KernelArguments) -> FloatArray:
    """Validate `noise` and cast it to the data dtype and layout."""
    if np.ma.isMaskedArray(noise):
        raise TypeError("masked noise arrays are not supported")
    value = np.asarray(noise)
    if value.dtype.kind not in "iuf":
        raise TypeError("noise must contain non-boolean real numbers")
    data = arguments.filtering.array
    if value.shape != data.shape:
        raise ValueError("noise must have the same shape as array")
    with np.errstate(over="ignore", invalid="ignore"):
        return np.require(value, dtype=data.dtype, requirements=("C", "A"))


def _noise_type(noise_type: object) -> bool:
    """Return `True` for inverse-variance noise."""
    if noise_type not in _NOISE_TYPES:
        raise ValueError(f"noise_type must be one of {_NOISE_TYPES}")
    return noise_type == "inverse_variance"


def convolve_variance(
    array: ArrayLike,
    noise: ArrayLike,
    weights: ArrayLike,
    *,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
    correlate: bool = False,
    noise_type: NoiseType = "variance",
) -> tuple[FloatArray, FloatArray]:
    """Convolve an array and propagate its per-pixel variance.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` data, rank one or higher.
    noise : array-like
        Per-pixel variance, or inverse variance when `noise_type` is
        ``"inverse_variance"``, with the shape of `array`. Cast to the dtype
        of `array`.
    weights : array-like
        Finite real coefficients with the input rank, as in `convolve`.
    mode : str or tuple of str, optional
        Boundary mode per input axis, as in `convolve`. Default is
        ``"reflect"``.
    cval : float, optional
        Constant boundary value. Constant-boundary samples have zero variance.
        Default is ``0.0``.
    origin : int or tuple of int, optional
        Kernel anchor offset as in `convolve`. Default is ``0``.
    correlate : bool, optional
        Use the orientation of `correlate` instead of `convolve`. Default is
        `False`.
    noise_type : {"variance", "inverse_variance"}, optional
        Meaning of `noise`. Default is ``"variance"``.

    Returns
    -------
    output : numpy.ndarray
        Weighted sum, equal to `convolve` (or `correlate`) bitwise.
    variance : numpy.ndarray
        Variance of `output`, with the shape and dtype of `array`.

    Notes
    -----
    For independent noise with variance ``v[p]`` at pixel ``p``, the output
    ``y = sum_j w[j] x[j]`` has variance ``sum_p c[p]**2 v[p]``. Here
    ``c[p]`` adds the weights of all taps that read pixel ``p``. Boundary
    modes other than ``"constant"`` can read one pixel through several taps.

    NaN data propagate to both outputs. Noise that is NaN or negative gives a
    NaN variance and leaves `output` unchanged. Infinite variance (inverse
    variance ``0``) on a nonzero tap gives an infinite variance. Infinite data
    give an infinite `output` with a finite variance.
    """
    arguments = _kernel_arguments(array, weights, mode, cval, origin)
    filtering = arguments.filtering
    value, variance = _core.convolve_noise(
        filtering.array,
        _noise_array(noise, arguments),
        arguments.weights,
        filtering.mode,
        filtering.cval,
        arguments.origin,
        bool(correlate),
        "sum",
        False,
        _noise_type(noise_type),
        False,
        False,
        1,
        False,
    )
    return value, variance


def nanconvolve_variance(
    array: ArrayLike,
    noise: ArrayLike,
    weights: ArrayLike,
    *,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
    correlate: bool = False,
    ignore_inf: bool = False,
    min_count: int = 1,
    noise_type: NoiseType = "variance",
) -> tuple[FloatArray, FloatArray]:
    """Smooth with a nonnegative kernel, omit NaNs, and propagate variance.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` data.
    noise : array-like
        Per-pixel variance, or inverse variance when `noise_type` is
        ``"inverse_variance"``, with the shape of `array`. Cast to the dtype
        of `array`.
    weights : array-like
        Finite nonnegative coefficients with the input rank and at least one
        positive entry, as in `nanconvolve`.
    mode : str or tuple of str, optional
        Boundary mode per input axis. Default is ``"reflect"``.
    cval : float, optional
        Constant boundary value. Constant-boundary samples have zero variance.
        Use ``cval=np.nan`` with ``mode="constant"`` to omit them. Default is
        ``0.0``.
    origin : int or tuple of int, optional
        Kernel anchor offset as in `convolve`. Default is ``0``.
    correlate : bool, optional
        Use the orientation of `nancorrelate`. Default is `False`.
    ignore_inf : bool, optional
        Also omit samples with infinite data. Default is `False`.
    min_count : int, optional
        Required retained positive-weight samples, between ``1`` and the
        positive kernel support. Default is ``1``.
    noise_type : {"variance", "inverse_variance"}, optional
        Meaning of `noise`. Default is ``"variance"``.

    Returns
    -------
    output : numpy.ndarray
        Weighted mean of the retained samples.
    variance : numpy.ndarray
        Variance of `output`. NaN wherever `output` is NaN.

    Notes
    -----
    Over the retained samples, ``y = sum_j w[j] x[j] / W`` with
    ``W = sum_j w[j]``, and ``var(y) = sum_p c[p]**2 v[p] / W**2``, where
    ``c[p]`` adds the weights of all taps that read pixel ``p``.

    A sample is retained when its data are not NaN (and not infinite with
    `ignore_inf`) and its variance is finite and nonnegative. Thus NaN or
    negative noise, and infinite variance (inverse variance ``0``), omit the
    sample from both outputs. Where the noise omits nothing, `output` equals
    `nanconvolve` (or `nancorrelate`) bitwise.
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
    value, variance = _core.convolve_noise(
        filtering.array,
        _noise_array(noise, arguments),
        arguments.weights,
        filtering.mode,
        filtering.cval,
        arguments.origin,
        bool(correlate),
        "mean",
        False,
        _noise_type(noise_type),
        True,
        bool(ignore_inf),
        arguments.min_count,
        False,
    )
    return value, variance


def matched_filter(
    array: ArrayLike,
    noise: ArrayLike,
    template: ArrayLike,
    *,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
    weighting: Literal["inverse_variance", "kernel"] = "inverse_variance",
    ignore_inf: bool = False,
    min_count: int = 1,
    noise_type: NoiseType = "variance",
) -> MatchedFilterResult:
    """Fit a template amplitude at every position and map its significance.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` data, for example a
        background-subtracted image.
    noise : array-like
        Per-pixel variance, or inverse variance when `noise_type` is
        ``"inverse_variance"``, with the shape of `array`. Cast to the dtype
        of `array`.
    template : array-like
        Finite real template, such as a PSF image, with the input rank and at
        least one nonzero entry. It is correlated, so it keeps its
        orientation.
    mode : str or tuple of str, optional
        Boundary mode per input axis. Default is ``"reflect"``.
    cval : float, optional
        Constant boundary value. Constant-boundary samples have zero variance.
        Default is ``0.0``.
    origin : int or tuple of int, optional
        Template anchor offset as in `correlate`. Default is ``0``.
    weighting : {"inverse_variance", "kernel"}, optional
        Coefficients ``template / variance`` (default) or ``template``.
    ignore_inf : bool, optional
        Also omit samples with infinite data. Default is `False`.
    min_count : int, optional
        Required retained nonzero-template samples, between ``1`` and the
        number of nonzero template entries. Default is ``1``.
    noise_type : {"variance", "inverse_variance"}, optional
        Meaning of `noise`. Default is ``"variance"``.

    Returns
    -------
    MatchedFilterResult
        Named tuple ``(snr, amplitude, variance)`` of arrays with the shape
        and dtype of `array`.

    Notes
    -----
    With coefficients ``a[j]`` equal to ``w[j] / v[j]`` (inverse-variance
    weighting) or ``w[j]`` (kernel weighting), over retained samples:

    - ``N = sum_j a[j] x[j]``;
    - ``amplitude = N / S`` with ``S = sum_j w[j] a[j]``;
    - ``variance = Var(N) / S**2`` with ``Var(N) = sum_p c[p]**2 v[p]``,
      where ``c[p]`` adds the coefficients of all taps that read pixel ``p``;
    - ``snr = N / sqrt(Var(N))``.

    If the data equal ``A * template`` plus independent zero-mean noise,
    `amplitude` is an unbiased estimate of ``A``. With inverse-variance
    weighting, it is the minimum-variance linear unbiased estimate, and
    `variance` equals ``1 / S`` away from boundary repeats. With uniform
    variance, both weightings give the same `snr`.

    NaN data, NaN or negative noise, and infinite variance (inverse variance
    ``0``) omit the sample. Inverse-variance weighting also omits samples with
    zero variance, whose weight would be infinite; this includes
    constant-boundary samples.
    """
    arguments = _kernel_arguments(
        array, template, mode, cval, origin, ignore_inf=ignore_inf, min_count=min_count
    )
    support = int(np.count_nonzero(arguments.weights))
    if support == 0:
        raise ValueError("template must have a nonzero entry")
    if not 1 <= arguments.min_count <= support:
        raise ValueError("min_count must be between 1 and the nonzero template support")
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"weighting must be one of {_WEIGHTINGS}")
    filtering = arguments.filtering
    amplitude, variance, snr = _core.convolve_noise(
        filtering.array,
        _noise_array(noise, arguments),
        arguments.weights,
        filtering.mode,
        filtering.cval,
        arguments.origin,
        True,
        "amplitude",
        weighting == "inverse_variance",
        _noise_type(noise_type),
        True,
        bool(ignore_inf),
        arguments.min_count,
        True,
    )
    return MatchedFilterResult(snr, amplitude, variance)
