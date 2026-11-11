"""Savitzky-Golay filters: local least-squares polynomial fits."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from . import _core
from ._gaussian import _axis_values
from ._linear import FloatArray
from ._validation import _MODES, _integer_index, validate_array

SavgolMode = Literal["interp", "reflect", "constant", "nearest", "mirror", "wrap"]
_SAVGOL_MODES = ("interp", *_MODES)


def _selected_axes(axes: object, ndim: int) -> tuple[int, ...]:
    """Return unique nonnegative axes; `None` selects every axis."""
    if axes is None:
        return tuple(range(ndim))
    if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)):
        raise TypeError("axes must be a sequence of integers")
    selected = tuple(
        _integer_index(axis, "axes entries must be integers, not booleans")
        for axis in axes
    )
    if any(not -ndim <= axis < ndim for axis in selected):
        raise ValueError("axes entries must be in the input dimension range")
    selected = tuple(axis % ndim for axis in selected)
    if len(set(selected)) != len(selected):
        raise ValueError("axes must not contain duplicates")
    if not selected:
        raise ValueError("axes must select at least one dimension")
    return selected


def _nonnegative_integers(value: object, count: int, name: str) -> tuple[int, ...]:
    """Broadcast and validate nonnegative integers per selected axis."""
    result = tuple(
        _integer_index(entry, f"{name} entries must be integers, not booleans")
        for entry in _axis_values(value, count, name)
    )
    if any(entry < 0 for entry in result):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _savgol_arguments(
    array: ArrayLike,
    window_length: int | tuple[int, ...],
    polyorder: int | tuple[int, ...],
    deriv: int | tuple[int, ...],
    delta: float | tuple[float, ...],
    axes: tuple[int, ...] | None,
    basis: str,
    weights: ArrayLike | None,
    mode: SavgolMode | tuple[SavgolMode, ...],
    cval: float,
) -> tuple[FloatArray, dict[str, object], int, int]:
    """Validate public arguments and expand them to one entry per array axis.

    Returns the input array, native keyword arguments, the number of
    polynomial terms, and the number of positive weights.
    """
    value = validate_array(array)
    ndim = value.ndim
    selected = _selected_axes(axes, ndim)
    count = len(selected)
    windows = _nonnegative_integers(window_length, count, "window_length")
    if any(length % 2 == 0 for length in windows):
        raise ValueError("window_length entries must be odd positive integers")
    if not isinstance(basis, str):
        raise TypeError("basis must be 'total' or 'per_axis'")
    if basis == "total":
        degree = _integer_index(polyorder, "polyorder must be an integer for 'total'")
        if degree < 0:
            raise ValueError("polyorder must be nonnegative")
        filtered = sum(length > 1 for length in windows)
        terms = math.comb(degree + filtered, filtered)
        degrees: tuple[int, ...] = (degree,)
    elif basis == "per_axis":
        selected_degrees = _nonnegative_integers(polyorder, count, "polyorder")
        terms = math.prod(entry + 1 for entry in selected_degrees)
        full_degrees = [0] * ndim
        for axis, entry in zip(selected, selected_degrees, strict=True):
            full_degrees[axis] = entry
        degrees = tuple(full_degrees)
    else:
        raise ValueError("basis must be 'total' or 'per_axis'")
    orders = _nonnegative_integers(deriv, count, "deriv")
    steps = []
    for entry in _axis_values(delta, count, "delta"):
        if isinstance(entry, (bool, np.bool_)) or not isinstance(entry, numbers.Real):
            raise TypeError("delta must contain non-boolean real numbers")
        step = float(entry)
        if not math.isfinite(step) or step <= 0:
            raise ValueError("delta must be finite and positive")
        steps.append(step)
    if isinstance(mode, str):
        modes = (mode,) * count
    else:
        if not isinstance(mode, Sequence):
            raise TypeError("mode must be a string or a sequence of strings")
        modes = tuple(mode)
        if len(modes) != count:
            raise ValueError("mode must have one entry per selected axis")
        if any(not isinstance(entry, str) for entry in modes):
            raise TypeError("mode entries must be strings")
    if any(entry not in _SAVGOL_MODES for entry in modes):
        raise ValueError(f"mode must use only {_SAVGOL_MODES}")
    if isinstance(cval, (bool, np.bool_)) or not isinstance(cval, numbers.Real):
        raise TypeError("cval must be a non-boolean real scalar")

    full_windows = [1] * ndim
    full_orders = [0] * ndim
    full_steps = [1.0] * ndim
    full_modes = ["nearest"] * ndim
    for position, axis in enumerate(selected):
        full_windows[axis] = windows[position]
        full_orders[axis] = orders[position]
        full_steps[axis] = steps[position]
        full_modes[axis] = modes[position]

    flat_weights = None
    positive = math.prod(windows)
    if weights is not None:
        if np.ma.isMaskedArray(weights):
            raise TypeError("masked weights are not supported")
        kernel = np.asarray(weights)
        if kernel.dtype.kind not in "iuf":
            raise TypeError("weights must contain non-boolean real numbers")
        if kernel.shape != windows:
            raise ValueError("weights shape must equal window_length per selected axis")
        with np.errstate(over="ignore", invalid="ignore"):
            kernel = kernel.astype(np.float64)
        if not np.isfinite(kernel).all() or np.any(kernel < 0):
            raise ValueError("weights must be finite and nonnegative")
        # Reorder selected axes into array order, then insert unit axes.
        kernel = np.transpose(kernel, np.argsort(selected)).reshape(full_windows)
        flat_weights = np.ascontiguousarray(kernel).ravel().tolist()
        positive = int(np.count_nonzero(kernel))

    if not value.flags.c_contiguous or type(value) is not np.ndarray:
        value = np.require(value, dtype=value.dtype, requirements=("C", "A", "O", "W"))
    native = {
        "window": full_windows,
        "basis": basis,
        "degrees": list(degrees),
        "derivative": full_orders,
        "delta": full_steps,
        "weights": flat_weights,
        "mode": full_modes,
        "cval": float(value.dtype.type(cval)),
    }
    return value, native, terms, positive


def savgol_filter(
    array: ArrayLike,
    window_length: int | tuple[int, ...],
    polyorder: int | tuple[int, ...],
    deriv: int | tuple[int, ...] = 0,
    delta: float | tuple[float, ...] = 1.0,
    *,
    axes: tuple[int, ...] | None = None,
    basis: Literal["total", "per_axis"] = "total",
    weights: ArrayLike | None = None,
    mode: SavgolMode | tuple[SavgolMode, ...] = "interp",
    cval: float = 0.0,
) -> FloatArray:
    """Fit a local polynomial in each window; NaNs propagate.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input of rank one or
        higher.
    window_length : int or tuple of int
        Odd positive window length per selected axis. A scalar broadcasts.
        A length of one leaves that axis unfiltered.
    polyorder : int or tuple of int
        Polynomial degree. With ``basis='total'``, one integer: the largest
        total degree. With ``basis='per_axis'``, the largest degree per
        selected axis (a scalar broadcasts). Each degree must be less than
        the window length on its axis.
    deriv : int or tuple of int, optional
        Derivative order per selected axis. A scalar broadcasts, so
        ``deriv=1`` on two axes is the mixed derivative. Default is `0`
        (the fitted value). The derivative monomial must be in the basis.
    delta : float or tuple of float, optional
        Finite positive sample spacing per selected axis. A derivative of
        order ``k`` is divided by ``delta**k``. Default is `1.0`.
    axes : tuple of int or None, optional
        Unique axes to filter. Default selects every axis. Tuple parameters
        and `weights` axes follow the order of `axes`.
    basis : {'total', 'per_axis'}, optional
        Polynomial space. `'total'` (default) uses monomials whose exponents
        sum to at most `polyorder`. `'per_axis'` limits each exponent
        separately.
    weights : array-like or None, optional
        Finite nonnegative least-squares weights with shape `window_length`
        over the selected axes. A zero weight removes that window position.
        Default is uniform weights.
    mode : str or tuple of str, optional
        Edge treatment per selected axis. `'interp'` (default) keeps the
        window inside the array and evaluates the fit at the output
        position; the window length must not exceed the axis length. The
        modes `'reflect'`, `'constant'`, `'nearest'`, `'mirror'`, and
        `'wrap'` pad the array and fit the centered window.
    cval : float, optional
        Value outside the array for `'constant'`. Default is `0.0`.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype.

    See Also
    --------
    nansavgol_filter : Fit only the non-NaN samples of each window.

    Notes
    -----
    On complete data the filter is a fixed linear filter. In one dimension
    it equals `scipy.signal.savgol_filter` with the same arguments. SciPy
    filters along ``axis=-1`` only, while the default here filters every
    axis. SciPy also rejects `'reflect'` and accepts even windows.

    Any NaN sample in a window with positive weight gives NaN. An infinite
    sample gives an infinity with the sign of sample times coefficient.
    Infinities of both signs, or an infinite sample with a zero coefficient,
    give NaN.
    """
    value, native, _, _ = _savgol_arguments(
        array, window_length, polyorder, deriv, delta, axes, basis, weights, mode, cval
    )
    return _core.savgol_filter(
        value, **native, omit_nan=False, ignore_inf=False, min_count=1
    )


def nansavgol_filter(
    array: ArrayLike,
    window_length: int | tuple[int, ...],
    polyorder: int | tuple[int, ...],
    deriv: int | tuple[int, ...] = 0,
    delta: float | tuple[float, ...] = 1.0,
    *,
    axes: tuple[int, ...] | None = None,
    basis: Literal["total", "per_axis"] = "total",
    weights: ArrayLike | None = None,
    mode: SavgolMode | tuple[SavgolMode, ...] = "interp",
    cval: float = 0.0,
    mask: ArrayLike | None = None,
    ignore_inf: bool = False,
    min_count: int | None = None,
) -> FloatArray:
    """Fit a local polynomial to the retained samples of each window.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input of rank one or
        higher.
    window_length, polyorder, deriv, delta, axes, basis, weights, mode, cval
        As in `savgol_filter`. With ``mode='constant'``, ``cval=np.nan``
        leaves exterior positions out of the fit.
    mask : array-like or None, optional
        Boolean array with the input shape. `True` marks samples to leave
        out, in addition to NaNs. Default is no mask.
    ignore_inf : bool, optional
        Also leave out infinite samples. Default is `False`.
    min_count : int or None, optional
        Smallest number of retained positive-weight samples for a finite
        result. Default `None` uses the number of polynomial terms, the
        fewest samples that can determine the fit.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype.

    See Also
    --------
    savgol_filter : The same fit with NaN propagation.

    Notes
    -----
    A complete window uses the fixed coefficients of `savgol_filter`. A
    window with left-out samples solves its own least-squares problem in
    `float64` with only the retained samples. Its result is the value or
    derivative of that polynomial, not a renormalized weighted sum.

    A window gives NaN when fewer than `min_count` samples remain, or when
    the retained samples do not determine the polynomial. The polynomial
    is undetermined when the singular values of the weighted design matrix
    satisfy ``s_min <= s_max * max(n, r) * eps`` for ``n`` retained samples
    and ``r`` terms, with offsets scaled to ``[-1, 1]`` per axis. This is
    the default tolerance of `numpy.linalg.matrix_rank`.
    """
    value, native, terms, positive = _savgol_arguments(
        array, window_length, polyorder, deriv, delta, axes, basis, weights, mode, cval
    )
    if not isinstance(ignore_inf, (bool, np.bool_)):
        raise TypeError("ignore_inf must be a boolean")
    if min_count is None:
        # More terms than weights is rejected natively with its own message.
        count = min(terms, positive)
    else:
        count = _integer_index(min_count, "min_count must be an integer, not a boolean")
    if not 1 <= count <= positive:
        raise ValueError(
            "min_count must be between 1 and the number of positive window weights"
        )
    if mask is not None:
        if np.ma.isMaskedArray(mask):
            raise TypeError("mask must not be a masked array")
        flags = np.asarray(mask)
        if flags.dtype != np.bool_:
            raise TypeError("mask must have a boolean dtype")
        if flags.shape != value.shape:
            raise ValueError("mask must have the input shape")
        value = np.where(flags, value.dtype.type(np.nan), value)
    return _core.savgol_filter(
        value, **native, omit_nan=True, ignore_inf=bool(ignore_inf), min_count=count
    )
