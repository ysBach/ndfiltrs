"""Sampled Gaussian kernels, derivatives, and normalized smoothing."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._api import BoundaryMode
from ._linear import FloatArray, convolve, nanconvolve
from ._validation import (
    FilterArguments,
    _integer_index,
    _normalize_mode,
    validate_array,
    validate_filter_arguments,
)


@dataclass(frozen=True)
class GaussianArguments:
    """Selected-axis Gaussian parameters and validated input geometry."""

    filtering: FilterArguments
    axes: tuple[int, ...]
    sigma: tuple[float, ...]
    order: tuple[int, ...]
    radius: tuple[int, ...]


def _axis_values(value: object, count: int, name: str) -> tuple[object, ...]:
    """Broadcast a scalar or validate exactly one entry per selected axis."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(value)
        if len(result) != count:
            raise ValueError(f"{name} must have one entry per selected axis")
        return result
    return (value,) * count


def _nonnegative_real(value: object, name: str) -> float:
    """Validate a finite, nonnegative real without coercing booleans or text."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must contain non-boolean real numbers")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _gaussian_arguments(
    array: ArrayLike,
    sigma: float | tuple[float, ...],
    order: int | tuple[int, ...],
    mode: BoundaryMode | tuple[BoundaryMode, ...],
    cval: float,
    truncate: float,
    radius: int | tuple[int, ...] | None,
    axes: tuple[int, ...] | None,
    *,
    dense: bool = True,
) -> GaussianArguments:
    """Validate selected axes and finite support before creating any kernels."""
    value = validate_array(array)
    if axes is None:
        selected = tuple(range(value.ndim))
    else:
        if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)):
            raise TypeError("axes must be a sequence of integers")
        selected = tuple(
            _integer_index(axis, "axes entries must be integers, not booleans")
            for axis in axes
        )
        if any(not -value.ndim <= axis < value.ndim for axis in selected):
            raise ValueError("axes entries must be in the input dimension range")
        selected = tuple(axis % value.ndim for axis in selected)
        if len(set(selected)) != len(selected):
            raise ValueError("axes must not contain duplicates")
    if not selected:
        raise ValueError("axes must select at least one dimension")

    count = len(selected)
    sigmas = tuple(
        _nonnegative_real(entry, "sigma")
        for entry in _axis_values(sigma, count, "sigma")
    )
    orders = tuple(
        _integer_index(entry, "order entries must be integers, not booleans")
        for entry in _axis_values(order, count, "order")
    )
    if any(entry < 0 for entry in orders):
        raise ValueError("order must be nonnegative")
    if any(s == 0 and n != 0 for s, n in zip(sigmas, orders, strict=True)):
        raise ValueError("Gaussian derivatives require strictly positive sigma")

    truncation = _nonnegative_real(truncate, "truncate")
    if radius is None:
        supports = []
        for width in sigmas:
            extent = truncation * width
            if not math.isfinite(extent):
                raise OverflowError("Gaussian radius exceeds the native index range")
            supports.append(math.floor(extent + 0.5))
        radii = tuple(supports)
    else:
        radii = tuple(
            _integer_index(entry, "radius entries must be integers, not booleans")
            for entry in _axis_values(radius, count, "radius")
        )
    if any(entry < 0 for entry in radii):
        raise ValueError("radius must be nonnegative")
    volume = 1
    max_elements = np.iinfo(np.intp).max // np.dtype(np.float64).itemsize
    for entry in radii:
        volume *= 2 * entry + 1
        if 2 * entry + 1 > max_elements or (dense and volume > max_elements):
            raise OverflowError("Gaussian kernel exceeds the native allocation range")

    modes = ["reflect"] * value.ndim
    for axis, boundary in zip(selected, _normalize_mode(mode, count), strict=True):
        modes[axis] = boundary
    filtering = validate_filter_arguments(value, 1, tuple(modes), cval, min_count=1)
    return GaussianArguments(filtering, selected, sigmas, orders, radii)


def _hermite_factors(
    locations: NDArray[np.float64], order: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return signs and log magnitudes of He_n using binary-scaled recurrence."""
    previous = np.ones_like(locations)
    current = locations.copy()
    _, exponent = np.frexp(np.maximum(np.abs(previous), np.abs(current)))
    previous = np.ldexp(previous, -exponent)
    current = np.ldexp(current, -exponent)
    exponent = exponent.astype(np.int64)
    # Both successive polynomials share a binary exponent. Scale each recurrence
    # before multiplying, preserving representable final derivatives even when
    # the unscaled Hermite polynomial exceeds the floating-point range.
    for degree in range(2, order + 1):
        _, factor = np.frexp(np.maximum(np.abs(locations), degree - 1))
        following = (
            np.ldexp(locations, -factor) * current
            - np.ldexp(float(degree - 1), -factor) * previous
        )
        previous = np.ldexp(current, -factor)
        _, adjustment = np.frexp(np.maximum(np.abs(previous), np.abs(following)))
        previous = np.ldexp(previous, -adjustment)
        current = np.ldexp(following, -adjustment)
        exponent += factor.astype(np.int64) + adjustment
    return np.sign(current), np.log(np.abs(current)) + exponent * math.log(2.0)


def _gaussian_factors(
    sigma: float, order: int, radius: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Keep coefficient signs and log magnitudes until the full product exists."""
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    if sigma == 0:
        signs = np.zeros_like(offsets)
        logarithms = np.full_like(offsets, -np.inf)
        signs[radius], logarithms[radius] = 1.0, 0.0
        return signs, logarithms
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        locations = offsets / sigma
        logarithms = -0.5 * locations * locations
        logarithms -= math.log(math.fsum(np.exp(logarithms)))
        signs = np.ones_like(offsets)
        if order:
            # A finite log Gaussian can combine with a large derivative scale
            # even when exp(log Gaussian) alone would already have underflowed.
            active = np.isfinite(logarithms)
            hermite_sign, hermite_log = _hermite_factors(locations[active], order)
            signs[active] = hermite_sign * (-1.0 if order % 2 else 1.0)
            logarithms[active] += hermite_log - order * math.log(sigma)
            signs[~active] = 0.0
    return signs, logarithms


def _product_kernel(
    arguments: GaussianArguments, orders: tuple[int, ...]
) -> NDArray[np.float64]:
    """Combine all coefficient scales before rounding the dense product kernel."""
    signs = np.ones((1,) * arguments.filtering.array.ndim, dtype=np.float64)
    logarithms = np.zeros_like(signs)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for axis, width, degree, radius in zip(
            arguments.axes, arguments.sigma, orders, arguments.radius, strict=True
        ):
            axis_signs, axis_logs = _gaussian_factors(width, degree, radius)
            shape = [1] * signs.ndim
            shape[axis] = axis_signs.size
            signs = signs * axis_signs.reshape(shape)
            logarithms = logarithms + axis_logs.reshape(shape)
        logarithms[signs == 0] = -np.inf
        kernel = signs * np.exp(logarithms)
    if not np.isfinite(kernel).all():
        raise ValueError(
            "Gaussian product coefficients are not representable in float64"
        )
    return kernel


def _separable_smoothing(
    arguments: GaussianArguments,
    *,
    omit_nan: bool,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> FloatArray:
    """Pass axis weights and omission policy to native smoothing with retained mass."""
    if any(arguments.order):
        raise ValueError(
            "method='separable' supports smoothing only; "
            "use method='direct' for derivatives"
        )
    if not isinstance(ignore_inf, (bool, np.bool_)):
        raise TypeError("ignore_inf must be a boolean")
    count = _integer_index(min_count, "min_count must be an integer, not a boolean")
    if count < 1:
        raise ValueError("min_count must be positive")
    kernels = []
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for width, radius in zip(arguments.sigma, arguments.radius, strict=True):
            signs, logs = _gaussian_factors(width, 0, radius)
            kernels.append((signs * np.exp(logs)).tolist())
    filtering = arguments.filtering
    return _core.separable_smooth(
        filtering.array,
        arguments.axes,
        kernels,
        filtering.mode,
        filtering.cval,
        omit_nan,
        bool(ignore_inf),
        count,
    )


def _validate_method(method: str) -> None:
    """Reject unknown Gaussian execution methods before validating geometry."""
    if not isinstance(method, str):
        raise TypeError("method must be 'direct' or 'separable'")
    if method not in ("direct", "separable"):
        raise ValueError("method must be 'direct' or 'separable'")


def gaussian_filter(
    array: ArrayLike,
    sigma: float | tuple[float, ...],
    *,
    order: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
    radius: int | tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = None,
    method: Literal["direct", "separable"] = "direct",
) -> FloatArray:
    """Smooth an array or calculate a sampled Gaussian derivative.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    sigma : float or tuple of float
        Finite nonnegative Gaussian width in pixels per selected axis.
    order : int or tuple of int, optional
        Nonnegative derivative order per selected axis. Default is `0`
        (smoothing). A scalar broadcasts. ``order=(1, 0)`` differentiates only
        the first selected axis. Positive orders require positive `sigma`.
    mode : str or tuple of str, optional
        Boundary modes per selected axis, as in `convolve`.
    cval : float, optional
        Constant boundary value, cast to the input dtype. Default is `0.0`.
    truncate : float, optional
        Nonnegative support in units of `sigma`. Used only when `radius` is
        absent. Radius is ``floor(truncate * sigma + 0.5)``. Default is `4.0`.
    radius : int or tuple of int, optional
        Exact nonnegative radius per selected axis, overriding `truncate`.
        Kernel lengths are ``2 * radius + 1``.
    axes : tuple of int or None, optional
        Unique axes to filter. Negative axes are valid. Default selects
        every axis. Tuple parameters follow the order of `axes`.

    method : {'direct', 'separable'}, optional
        Default `'direct'` uses a dense N-D kernel. `'separable'` smooths one
        axis at a time, carrying retained weights and counts across passes.
        It supports order zero and requires normal-range positive product
        weights. Use `'direct'` for derivatives or underflowing support.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype.

    Notes
    -----
    Order-zero coefficients are normalized samples of a Gaussian. Derivatives
    use its analytic Hermite-polynomial derivatives, without forcing zero sum.
    Smoothing renormalizes the computed weights while still propagating NaNs,
    preserving finite constant neighborhoods despite coefficient rounding.
    `sigma=0` is an identity kernel for order zero. Zero coefficients do not
    propagate NaNs or infinities. Other nonfinite rules follow `convolve`.

    Direct evaluation costs work proportional to kernel volume. Separable
    smoothing costs work proportional to the sum of axis lengths. Rounding
    can differ because the separable method groups accumulation by axis.
    Both accumulate in `float64` and cast once to the input dtype.
    Neither method uses an FFT backend.
    """
    _validate_method(method)
    arguments = _gaussian_arguments(
        array,
        sigma,
        order,
        mode,
        cval,
        truncate,
        radius,
        axes,
        dense=method == "direct",
    )
    if method == "separable":
        return _separable_smoothing(arguments, omit_nan=False)
    filtering = arguments.filtering
    return _core.convolve(
        filtering.array,
        _product_kernel(arguments, arguments.order),
        filtering.mode,
        filtering.cval,
        (0,) * filtering.array.ndim,
        False,
        False,
        False,
        1,
        not any(arguments.order),
    )


def nangaussian_filter(
    array: ArrayLike,
    sigma: float | tuple[float, ...],
    *,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ignore_inf: bool = False,
    min_count: int = 1,
    truncate: float = 4.0,
    radius: int | tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = None,
    method: Literal["direct", "separable"] = "direct",
) -> FloatArray:
    """Smooth with a Gaussian while omitting NaNs and renormalizing support.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    sigma : float or tuple of float
        Finite nonnegative Gaussian width in pixels per selected axis.
    mode : str or tuple of str, optional
        Boundary modes per selected axis. Default is `'reflect'`.
    cval : float, optional
        Constant boundary sample, subject to omission. Default is `0.0`.
    ignore_inf : bool, optional
        Omit infinities as well as NaNs when `True`. Default is `False`.
    min_count : int, optional
        Required retained positive-weight samples. Default is `1`.
    truncate : float, optional
        Support in units of `sigma`, as in `gaussian_filter`. Default is `4.0`.
    radius : int or tuple of int, optional
        Exact per-selected-axis radius, overriding `truncate`.
    axes : tuple of int or None, optional
        Unique selected axes. Default selects every axis.

    method : {'direct', 'separable'}, optional
        Default `'direct'` uses a dense N-D kernel. `'separable'` smooths one
        axis at a time, carrying retained weights and counts across passes.
        It supports order zero and requires normal-range positive product
        weights. Use `'direct'` for derivatives or underflowing support.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype. Insufficient support
        produces canonical NaN without warnings.

    Notes
    -----
    Normalize once over the full N-D neighborhood's retained weights, using
    `nanconvolve` semantics. Derivatives are intentionally absent from this
    normalized smoothing API. Underflowed zero coefficients do not count as
    support. Separable smoothing carries unnormalized retained mass between
    passes. It does not average already-normalized rows equally. Support
    counts remain integers. Geometry and method limits follow `gaussian_filter`.
    """
    _validate_method(method)
    arguments = _gaussian_arguments(
        array, sigma, 0, mode, cval, truncate, radius, axes, dense=method == "direct"
    )
    if method == "separable":
        return _separable_smoothing(
            arguments, omit_nan=True, ignore_inf=ignore_inf, min_count=min_count
        )
    filtering = arguments.filtering
    return nanconvolve(
        filtering.array,
        _product_kernel(arguments, arguments.order),
        mode=filtering.mode,
        cval=filtering.cval,
        ignore_inf=ignore_inf,
        min_count=min_count,
    )


def gaussian_laplace(
    array: ArrayLike,
    sigma: float | tuple[float, ...],
    *,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
    radius: int | tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = None,
) -> FloatArray:
    """Apply the sum of selected-axis Gaussian second derivatives.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    sigma : float or tuple of float
        Finite strictly positive Gaussian width per selected axis.
    mode : str or tuple of str, optional
        Boundary modes per selected axis. Default is `'reflect'`.
    cval : float, optional
        Constant boundary value. Default is `0.0`.
    truncate : float, optional
        Support in units of `sigma`, as in `gaussian_filter`. Default is `4.0`.
    radius : int or tuple of int, optional
        Exact per-selected-axis radius, overriding `truncate`.
    axes : tuple of int or None, optional
        Unique axes contributing second derivatives to the sum. Default selects all.

    Returns
    -------
    numpy.ndarray
        New owned Laplacian map with the input shape and dtype.

    Notes
    -----
    Sum derivative product kernels before one convolution. A coefficient that
    cancels to zero therefore excludes its sample, even when the sample is
    nonfinite. Boundary and numerical rules follow `convolve`.
    """
    arguments = _gaussian_arguments(array, sigma, 2, mode, cval, truncate, radius, axes)
    kernel = None
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for position in range(len(arguments.axes)):
            orders = tuple(
                2 if index == position else 0 for index in range(len(arguments.axes))
            )
            component = _product_kernel(arguments, orders)
            if kernel is None:
                kernel = component
            else:
                kernel += component
    if kernel is None or not np.isfinite(kernel).all():
        raise ValueError(
            "Gaussian Laplace coefficients are not representable in float64"
        )
    filtering = arguments.filtering
    return convolve(filtering.array, kernel, mode=filtering.mode, cval=filtering.cval)


def gaussian_gradient_magnitude(
    array: ArrayLike,
    sigma: float | tuple[float, ...],
    *,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
    radius: int | tuple[int, ...] | None = None,
    axes: tuple[int, ...] | None = None,
) -> FloatArray:
    """Calculate the Euclidean magnitude of Gaussian first derivatives.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    sigma : float or tuple of float
        Finite strictly positive Gaussian width per selected axis.
    mode : str or tuple of str, optional
        Boundary modes per selected axis. Default is `'reflect'`.
    cval : float, optional
        Constant boundary value, cast to the input dtype. Default is `0.0`.
    truncate : float, optional
        Support in units of `sigma`, as in `gaussian_filter`. Default is `4.0`.
    radius : int or tuple of int, optional
        Exact per-selected-axis radius, overriding `truncate`.
    axes : tuple of int or None, optional
        Unique axes contributing to the norm. Default selects all axes.

    Returns
    -------
    numpy.ndarray
        New owned nonnegative map with the input shape and dtype.

    Notes
    -----
    Compute components in `float64` and combine with scaled hypotenuse
    arithmetic, avoiding overflow from squaring finite large derivatives.
    Cast once at the end. Any NaN component makes the magnitude canonical NaN.
    """
    arguments = _gaussian_arguments(array, sigma, 1, mode, cval, truncate, radius, axes)
    filtering = arguments.filtering
    # One owned snapshot keeps every derivative on the same original data.
    value = np.array(filtering.array, dtype=np.float64, order="C", copy=True)
    magnitude = np.zeros(value.shape, dtype=np.float64)
    invalid = np.zeros(value.shape, dtype=bool)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for position in range(len(arguments.axes)):
            orders = tuple(
                1 if index == position else 0 for index in range(len(arguments.axes))
            )
            component = convolve(
                value,
                _product_kernel(arguments, orders),
                mode=filtering.mode,
                cval=filtering.cval,
            )
            invalid |= np.isnan(component)
            np.hypot(magnitude, component, out=magnitude)
        magnitude[invalid] = np.nan
        return magnitude.astype(filtering.array.dtype)
