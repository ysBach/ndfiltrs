"""Local percentiles, median absolute deviations, and sigma-clipped statistics."""

from __future__ import annotations

import math
import numbers
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._api import BoundaryMode
from ._validation import FilterArguments, _integer_index, validate_filter_arguments

FloatArray = NDArray[np.float32] | NDArray[np.float64]
ModeEstimator = Literal["sex", "mmm", "iraf"]
_MODE_ESTIMATORS = ("sex", "mmm", "iraf")


def _finite_real(value: object, name: str) -> float:
    """Reject booleans, non-real values, and nonfinite parameters."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a non-boolean real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def percentile_filter(
    array: ArrayLike,
    percentile: float,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
) -> FloatArray:
    """Calculate a linearly interpolated percentile in each neighborhood.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    percentile : float
        Percentile from `0` through `100`, inclusive. `50` gives the median.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis; a tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank; nonzero entries are used. Pass `size`
        or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in `median_filter`. Default is `0`.
    mode : str or tuple of str, optional
        Boundary mode per input axis, as in `median_filter`.
        Default is `'reflect'`.
    cval : float, optional
        Sample beyond constant boundaries, cast to the input dtype.
        Default is `0.0`.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype.

    Notes
    -----
    Sort each window and interpolate at rank ``(n - 1) * percentile / 100``.
    Endpoints select the minimum or maximum. NaNs propagate locally.
    Infinities participate; interpolation between opposite infinities gives
    NaN. Finite interpolation uses `float64` and casts once at the end.
    """
    quantile = _finite_real(percentile, "percentile")
    if not 0 <= quantile <= 100:
        raise ValueError("percentile must lie between 0 and 100")
    arguments = validate_filter_arguments(
        array, size, mode, cval, footprint=footprint, origin=origin, min_count=1
    )
    return _core.robust_filter(
        arguments.array,
        arguments.size,
        arguments.footprint,
        arguments.origin,
        arguments.mode,
        arguments.cval,
        kind="percentile",
        parameter=quantile,
        omit_nan=False,
        ignore_inf=False,
        min_count=1,
        sigma_lower=3.0,
        sigma_upper=3.0,
        maxiters=5,
        center="median",
        spread="std",
    )


def nanpercentile_filter(
    array: ArrayLike,
    percentile: float,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> FloatArray:
    """Calculate local percentiles after omitting NaNs.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    percentile : float
        Percentile from `0` through `100`, inclusive, with linear interpolation.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis; a tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank; nonzero entries are used. Pass `size`
        or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in `median_filter`. Default is `0`.
    mode : str or tuple of str, optional
        Boundary mode per input axis. Default is `'reflect'`.
    cval : float, optional
        Constant boundary sample, subject to omission. Default is `0.0`.
    ignore_inf : bool, optional
        Also omit infinities when `True`. Default is `False`.
    min_count : int, optional
        Required retained samples per window. Default is `1`.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype. Insufficient support
        produces canonical NaN without an empty-window warning.

    Notes
    -----
    Interpolation uses the retained population, which can have even length.
    Boundary repetitions count separately. Other interpolation and infinity
    rules follow `percentile_filter`.
    """
    quantile = _finite_real(percentile, "percentile")
    if not 0 <= quantile <= 100:
        raise ValueError("percentile must lie between 0 and 100")
    if not isinstance(ignore_inf, (bool, np.bool_)):
        raise TypeError("ignore_inf must be a boolean")
    arguments = validate_filter_arguments(
        array, size, mode, cval, footprint=footprint, origin=origin, min_count=min_count
    )
    return _core.robust_filter(
        arguments.array,
        arguments.size,
        arguments.footprint,
        arguments.origin,
        arguments.mode,
        arguments.cval,
        kind="percentile",
        parameter=quantile,
        omit_nan=True,
        ignore_inf=bool(ignore_inf),
        min_count=arguments.min_count,
        sigma_lower=3.0,
        sigma_upper=3.0,
        maxiters=5,
        center="median",
        spread="std",
    )


def mad_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    scale: float = 1.0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
) -> FloatArray:
    """Calculate the median absolute deviation in each neighborhood.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis; a tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank; nonzero entries are used. Pass `size`
        or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in `median_filter`. Default is `0`.
    scale : float, optional
        Finite nonnegative multiplier applied to the MAD. Default is `1.0`,
        giving the unscaled deviation in the same units as `array`.
    mode : str or tuple of str, optional
        Boundary mode per input axis. Default is `'reflect'`.
    cval : float, optional
        Constant boundary sample, cast to the input dtype. Default is `0.0`.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype.

    Notes
    -----
    The statistic is ``scale * median(abs(x - median(x)))`` in each window.
    NaNs propagate locally. Infinities participate; undefined deviations
    such as ``inf - inf`` produce NaN. Both medians use the average of the
    two central values when their population has even length.
    """
    multiplier = _finite_real(scale, "scale")
    if multiplier < 0:
        raise ValueError("scale must be nonnegative")
    arguments = validate_filter_arguments(
        array, size, mode, cval, footprint=footprint, origin=origin, min_count=1
    )
    return _core.robust_filter(
        arguments.array,
        arguments.size,
        arguments.footprint,
        arguments.origin,
        arguments.mode,
        arguments.cval,
        kind="mad",
        parameter=multiplier,
        omit_nan=False,
        ignore_inf=False,
        min_count=1,
        sigma_lower=3.0,
        sigma_upper=3.0,
        maxiters=5,
        center="median",
        spread="std",
    )


def nanmad_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    scale: float = 1.0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> FloatArray:
    """Calculate local median absolute deviations after omitting NaNs.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis; a tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank; nonzero entries are used. Pass `size`
        or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in `median_filter`. Default is `0`.
    scale : float, optional
        Finite nonnegative multiplier applied to the MAD. Default is `1.0`.
    mode : str or tuple of str, optional
        Boundary mode per input axis. Default is `'reflect'`.
    cval : float, optional
        Constant boundary sample, subject to omission. Default is `0.0`.
    ignore_inf : bool, optional
        Also omit infinities when `True`. Default is `False`.
    min_count : int, optional
        Required retained samples before calculating the MAD. Default is `1`.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype. Insufficient support
        produces canonical NaN without an empty-window warning.

    Notes
    -----
    Both medians use the same retained samples. NaNs created by an undefined
    deviation are not omitted again. Arithmetic and scaling follow `mad_filter`.
    Boundary repetitions count separately toward `min_count`.
    """
    multiplier = _finite_real(scale, "scale")
    if multiplier < 0:
        raise ValueError("scale must be nonnegative")
    if not isinstance(ignore_inf, (bool, np.bool_)):
        raise TypeError("ignore_inf must be a boolean")
    arguments = validate_filter_arguments(
        array, size, mode, cval, footprint=footprint, origin=origin, min_count=min_count
    )
    return _core.robust_filter(
        arguments.array,
        arguments.size,
        arguments.footprint,
        arguments.origin,
        arguments.mode,
        arguments.cval,
        kind="mad",
        parameter=multiplier,
        omit_nan=True,
        ignore_inf=bool(ignore_inf),
        min_count=arguments.min_count,
        sigma_lower=3.0,
        sigma_upper=3.0,
        maxiters=5,
        center="median",
        spread="std",
    )


def sigma_clipped_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    sigma: float = 3.0,
    sigma_lower: float | None = None,
    sigma_upper: float | None = None,
    maxiters: int | None = 5,
    center: Literal["mean", "median"] = "median",
    spread: Literal["std", "mad"] = "std",
    statistic: Literal["mean", "median", "mode"] = "mean",
    mode_estimator: Literal["sex", "mmm", "iraf"] = "sex",
    pearson: float | None = None,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    min_count: int = 1,
) -> FloatArray:
    """Reject local outliers iteratively, then summarize the retained samples.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input. NaNs and infinities
        are always omitted before clipping.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis; a tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank; nonzero entries are used. Pass `size`
        or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in `median_filter`. Default is `0`.
    sigma : float, optional
        Nonnegative finite clipping multiplier for either side without an
        explicit override. Default is `3.0`.
    sigma_lower, sigma_upper : float or None, optional
        Nonnegative finite lower and upper multipliers. `None` uses `sigma`.
        A zero multiplier places that bound at the center.
    maxiters : int or None, optional
        Maximum rejection passes. Default is `5`. `0` skips clipping and
        returns a statistic over the finite samples. `None` continues until
        no more samples are rejected.
    center : {'mean', 'median'}, optional
        Center recalculated from retained values before each rejection pass.
        Default is `'median'`.
    spread : {'std', 'mad'}, optional
        Population standard deviation or Gaussian-consistent MAD used for
        clipping. Default is `'std'`. MAD is divided by the standard normal
        distribution's 75th percentile, approximately `0.67449`.
    statistic : {'mean', 'median', 'mode'}, optional
        Statistic returned after clipping. Default is `'mean'`. `'mode'` with
        `mode_estimator='sex'` and `maxiters=None` gives a SExtractor-like local
        background: clip at median ± 3 std until stable, then take the mode.
    mode_estimator : {'sex', 'mmm', 'iraf'}, optional
        Mode formula used when `statistic='mode'`; see `mode_filter`.
        Default is `'sex'`.
    pearson : float or None, optional
        Pearson factor in ``mode = pearson * median - (pearson - 1) * mean``, at
        least `1` (SExtractor's `BACK_PEARSON`). `None` uses `2.5` for `'sex'` and `3`
        for `'mmm'` and `'iraf'`.
    mode : str or tuple of str, optional
        Boundary mode per input axis. Default is `'reflect'`.
    cval : float, optional
        Constant boundary sample, omitted if nonfinite. Default is `0.0`.
    min_count : int, optional
        Required retained samples after clipping. Default is `1`.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype. Insufficient support
        produces canonical NaN without an empty-window warning.

    Notes
    -----
    Retain values from ``center - sigma_lower * spread`` through
    ``center + sigma_upper * spread``, including both endpoints. Rejected
    values never re-enter a window. Stop when no more values are rejected or
    `maxiters` passes finish. With zero spread, only values equal to the
    center survive. Each boundary repetition counts as a separate sample.
    """
    threshold = _finite_real(sigma, "sigma")
    lower = (
        threshold if sigma_lower is None else _finite_real(sigma_lower, "sigma_lower")
    )
    upper = (
        threshold if sigma_upper is None else _finite_real(sigma_upper, "sigma_upper")
    )
    if threshold < 0 or lower < 0 or upper < 0:
        raise ValueError("sigma and its lower and upper overrides must be nonnegative")
    iterations = None
    if maxiters is not None:
        iterations = _integer_index(
            maxiters, "maxiters must be an integer, not a boolean"
        )
        if iterations < 0:
            raise ValueError("maxiters must be nonnegative")
        if iterations > np.iinfo(np.uintp).max:
            raise OverflowError("maxiters exceeds the native index range")
    for name, value, choices in (
        ("center", center, ("mean", "median")),
        ("spread", spread, ("std", "mad")),
        ("statistic", statistic, ("mean", "median", "mode")),
    ):
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"{name} must be one of {choices}")
    factor = _pearson(mode_estimator, pearson)
    arguments = validate_filter_arguments(
        array, size, mode, cval, footprint=footprint, origin=origin, min_count=min_count
    )
    return _core.robust_filter(
        arguments.array,
        arguments.size,
        arguments.footprint,
        arguments.origin,
        arguments.mode,
        arguments.cval,
        kind=f"sigma_{statistic}",
        parameter=0.0,
        omit_nan=True,
        ignore_inf=True,
        min_count=arguments.min_count,
        sigma_lower=lower,
        sigma_upper=upper,
        maxiters=iterations,
        center=center,
        spread=spread,
        estimator=mode_estimator,
        pearson=factor,
    )


def _positive_finite(value: object, name: str) -> float:
    """Return a finite positive real parameter."""
    number = _finite_real(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _pearson(estimator: object, pearson: object) -> float:
    """Validate the mode rule and return its Pearson factor (default per rule)."""
    if not isinstance(estimator, str) or estimator not in _MODE_ESTIMATORS:
        raise ValueError(f"mode_estimator must be one of {_MODE_ESTIMATORS}")
    if pearson is None:
        return 2.5 if estimator == "sex" else 3.0
    factor = _finite_real(pearson, "pearson")
    if factor < 1:
        raise ValueError("pearson must be at least 1")
    return factor


def _robust_estimate(
    arguments: FilterArguments,
    kind: str,
    *,
    omit_nan: bool,
    ignore_inf: object = False,
    parameter: float,
    estimator: str = "sex",
    modify_sample_size: bool = False,
    pearson: float = 2.5,
) -> FloatArray:
    """Run a native location estimator, propagating or omitting NaN."""
    if not isinstance(ignore_inf, (bool, np.bool_)):
        raise TypeError("ignore_inf must be a boolean")
    return _core.robust_filter(
        arguments.array,
        arguments.size,
        arguments.footprint,
        arguments.origin,
        arguments.mode,
        arguments.cval,
        kind=kind,
        parameter=parameter,
        omit_nan=omit_nan,
        ignore_inf=bool(ignore_inf),
        min_count=arguments.min_count,
        sigma_lower=3.0,
        sigma_upper=3.0,
        maxiters=None,
        center="median",
        spread="std",
        estimator=estimator,
        modify_sample_size=modify_sample_size,
        pearson=pearson,
    )


def mode_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode_estimator: ModeEstimator = "sex",
    pearson: float | None = None,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
) -> FloatArray:
    """Estimate the local mode from window mean, median, and spread.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis; a tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank; nonzero entries are used. Pass `size`
        or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in `median_filter`. Default is `0`.
    mode_estimator : {'sex', 'mmm', 'iraf'}, optional
        Mode formula. Default is `'sex'`.

        * `'sex'` (SExtractor): the Pearson mode (default ``2.5 * median -
          1.5 * mean``), or the median when ``abs(mean - median) / std >= 0.3``.
        * `'mmm'` (DAOPHOT MMM formula): the Pearson mode, default
          ``3 * median - 2 * mean``.
        * `'iraf'`: the mean when it is below the median, otherwise `'mmm'`.
    pearson : float or None, optional
        Pearson factor in ``mode = pearson * median - (pearson - 1) * mean``, at
        least `1` (SExtractor's `BACK_PEARSON`). `None` uses `2.5` for `'sex'` and `3`
        for `'mmm'` and `'iraf'`.
    mode : str or tuple of str, optional
        Boundary mode per input axis. Default is `'reflect'`.
    cval : float, optional
        Constant boundary sample, cast to the input dtype. Default is `0.0`.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype. Windows containing NaN
        or infinity contain NaN.

    Notes
    -----
    For `'sex'`, `std` is the population standard deviation (`ddof=0`), and a
    zero `std` returns the mean, as in SExtractor's `backguess`
    (`src/back.c`, default `BACK_PEARSON = 2.5`). `'mmm'` and `'iraf'` match
    astroimred's `sky_fit`. No samples are clipped; for a SExtractor-like
    background use `sigma_clipped_filter(..., maxiters=None, statistic="mode")`.
    """
    factor = _pearson(mode_estimator, pearson)
    arguments = validate_filter_arguments(
        array,
        size,
        mode,
        cval,
        footprint=footprint,
        origin=origin,
        min_count=1,
    )
    return _robust_estimate(
        arguments,
        "mode",
        omit_nan=False,
        parameter=0.0,
        estimator=mode_estimator,
        pearson=factor,
    )


def biweight_location_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    c: float = 6.0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
) -> FloatArray:
    """Calculate Tukey's biweight location in each window.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis; a tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank; nonzero entries are used. Pass `size`
        or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in `median_filter`. Default is `0`.
    c : float, optional
        Tuning constant in units of the MAD. Default is `6.0`, as in Astropy.
    mode : str or tuple of str, optional
        Boundary mode per input axis. Default is `'reflect'`.
    cval : float, optional
        Constant boundary sample, cast to the input dtype. Default is `0.0`.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype. Windows containing NaN
        or infinity contain NaN.

    Notes
    -----
    With `M` the window median and ``u = (x - M) / (c * MAD)``, the result is
    ``M + sum((x - M) * (1 - u**2)**2) / sum((1 - u**2)**2)`` over samples with
    ``|u| < 1``. A zero MAD returns `M`. This matches
    ``astropy.stats.biweight_location(M=None)`` up to rounding.
    """
    arguments = validate_filter_arguments(
        array,
        size,
        mode,
        cval,
        footprint=footprint,
        origin=origin,
        min_count=1,
    )
    return _robust_estimate(
        arguments,
        "biweight_location",
        omit_nan=False,
        parameter=_positive_finite(c, "c"),
    )


def biweight_scale_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    c: float = 9.0,
    modify_sample_size: bool = False,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
) -> FloatArray:
    """Calculate the local biweight scale (a robust spread).

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis; a tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank; nonzero entries are used. Pass `size`
        or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in `median_filter`. Default is `0`.
    c : float, optional
        Tuning constant in units of the MAD. Default is `9.0`, as in Astropy.
    modify_sample_size : bool, optional
        Use only samples with nonzero weight as the sample size, as in
        Astropy. Default is `False`.
    mode : str or tuple of str, optional
        Boundary mode per input axis. Default is `'reflect'`.
    cval : float, optional
        Constant boundary sample, cast to the input dtype. Default is `0.0`.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype. Windows containing NaN
        or infinity contain NaN.

    Notes
    -----
    With `M` the window median, ``d = x - M``, ``u = d / (c * MAD)``, and
    ``w = 1 - u**2``, the result is
    ``sqrt(n * sum(d**2 * w**4)) / abs(sum(w * (1 - 5 * u**2)))`` over samples
    with ``|u| < 1``. A zero MAD returns `0`. This matches
    ``astropy.stats.biweight_scale(M=None)`` up to rounding.
    """
    if not isinstance(modify_sample_size, (bool, np.bool_)):
        raise TypeError("modify_sample_size must be a boolean")
    arguments = validate_filter_arguments(
        array,
        size,
        mode,
        cval,
        footprint=footprint,
        origin=origin,
        min_count=1,
    )
    return _robust_estimate(
        arguments,
        "biweight_scale",
        omit_nan=False,
        parameter=_positive_finite(c, "c"),
        modify_sample_size=bool(modify_sample_size),
    )


def nanmode_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    mode_estimator: ModeEstimator = "sex",
    pearson: float | None = None,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> FloatArray:
    """Estimate the local mode from window mean, median, and spread, omitting NaNs.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis; a tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank; nonzero entries are used. Pass `size`
        or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in `median_filter`. Default is `0`.
    mode_estimator : {'sex', 'mmm', 'iraf'}, optional
        Mode formula. Default is `'sex'`.

        * `'sex'` (SExtractor): the Pearson mode (default ``2.5 * median -
          1.5 * mean``), or the median when ``abs(mean - median) / std >= 0.3``.
        * `'mmm'` (DAOPHOT MMM formula): the Pearson mode, default
          ``3 * median - 2 * mean``.
        * `'iraf'`: the mean when it is below the median, otherwise `'mmm'`.
    pearson : float or None, optional
        Pearson factor in ``mode = pearson * median - (pearson - 1) * mean``, at
        least `1` (SExtractor's `BACK_PEARSON`). `None` uses `2.5` for `'sex'` and `3`
        for `'mmm'` and `'iraf'`.
    mode : str or tuple of str, optional
        Boundary mode per input axis. Default is `'reflect'`.
    cval : float, optional
        Constant boundary sample, cast to the input dtype and subject to
        omission. Default is `0.0`.
    ignore_inf : bool, optional
        Also omit infinities when `True`. Default is `False`.
    min_count : int, optional
        Required retained samples per window. Default is `1`.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype. Windows with fewer than
        `min_count` retained samples, or retaining an infinity, contain NaN.

    Notes
    -----
    For `'sex'`, `std` is the population standard deviation (`ddof=0`), and a
    zero `std` returns the mean, as in SExtractor's `backguess`
    (`src/back.c`, default `BACK_PEARSON = 2.5`). `'mmm'` and `'iraf'` match
    astroimred's `sky_fit`. No samples are clipped; for a SExtractor-like
    background use `sigma_clipped_filter(..., maxiters=None, statistic="mode")`.
    """
    factor = _pearson(mode_estimator, pearson)
    arguments = validate_filter_arguments(
        array,
        size,
        mode,
        cval,
        footprint=footprint,
        origin=origin,
        min_count=min_count,
    )
    return _robust_estimate(
        arguments,
        "mode",
        omit_nan=True,
        ignore_inf=ignore_inf,
        parameter=0.0,
        estimator=mode_estimator,
        pearson=factor,
    )


def nanbiweight_location_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    c: float = 6.0,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> FloatArray:
    """Calculate Tukey's biweight location in each window, omitting NaNs.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis; a tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank; nonzero entries are used. Pass `size`
        or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in `median_filter`. Default is `0`.
    c : float, optional
        Tuning constant in units of the MAD. Default is `6.0`, as in Astropy.
    mode : str or tuple of str, optional
        Boundary mode per input axis. Default is `'reflect'`.
    cval : float, optional
        Constant boundary sample, cast to the input dtype and subject to
        omission. Default is `0.0`.
    ignore_inf : bool, optional
        Also omit infinities when `True`. Default is `False`.
    min_count : int, optional
        Required retained samples per window. Default is `1`.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype. Windows with fewer than
        `min_count` retained samples, or retaining an infinity, contain NaN.

    Notes
    -----
    With `M` the window median and ``u = (x - M) / (c * MAD)``, the result is
    ``M + sum((x - M) * (1 - u**2)**2) / sum((1 - u**2)**2)`` over samples with
    ``|u| < 1``. A zero MAD returns `M`. This matches
    ``astropy.stats.biweight_location(M=None)`` up to rounding.
    """
    arguments = validate_filter_arguments(
        array,
        size,
        mode,
        cval,
        footprint=footprint,
        origin=origin,
        min_count=min_count,
    )
    return _robust_estimate(
        arguments,
        "biweight_location",
        omit_nan=True,
        ignore_inf=ignore_inf,
        parameter=_positive_finite(c, "c"),
    )


def nanbiweight_scale_filter(
    array: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    *,
    footprint: ArrayLike | None = None,
    origin: int | tuple[int, ...] = 0,
    c: float = 9.0,
    modify_sample_size: bool = False,
    mode: BoundaryMode | tuple[BoundaryMode, ...] = "reflect",
    cval: float = 0.0,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> FloatArray:
    """Calculate the local biweight scale (a robust spread), omitting NaNs.

    Parameters
    ----------
    array : array-like
        Aligned, native-endian `float32` or `float64` input.
    size : int or tuple of int, optional
        Rectangular window size. A scalar broadcasts to every axis; a tuple
        has one entry per axis. Default is `3` when `footprint` is `None`.
    footprint : array-like, optional
        Window with the array's rank; nonzero entries are used. Pass `size`
        or `footprint`, not both.
    origin : int or tuple of int, optional
        Window shift per axis, as in `median_filter`. Default is `0`.
    c : float, optional
        Tuning constant in units of the MAD. Default is `9.0`, as in Astropy.
    modify_sample_size : bool, optional
        Use only samples with nonzero weight as the sample size, as in
        Astropy. Default is `False`.
    mode : str or tuple of str, optional
        Boundary mode per input axis. Default is `'reflect'`.
    cval : float, optional
        Constant boundary sample, cast to the input dtype and subject to
        omission. Default is `0.0`.
    ignore_inf : bool, optional
        Also omit infinities when `True`. Default is `False`.
    min_count : int, optional
        Required retained samples per window. Default is `1`.

    Returns
    -------
    numpy.ndarray
        New owned array with the input shape and dtype. Windows with fewer than
        `min_count` retained samples, or retaining an infinity, contain NaN.

    Notes
    -----
    With `M` the window median, ``d = x - M``, ``u = d / (c * MAD)``, and
    ``w = 1 - u**2``, the result is
    ``sqrt(n * sum(d**2 * w**4)) / abs(sum(w * (1 - 5 * u**2)))`` over samples
    with ``|u| < 1``. A zero MAD returns `0`. This matches
    ``astropy.stats.biweight_scale(M=None)`` up to rounding.
    """
    if not isinstance(modify_sample_size, (bool, np.bool_)):
        raise TypeError("modify_sample_size must be a boolean")
    arguments = validate_filter_arguments(
        array,
        size,
        mode,
        cval,
        footprint=footprint,
        origin=origin,
        min_count=min_count,
    )
    return _robust_estimate(
        arguments,
        "biweight_scale",
        omit_nan=True,
        ignore_inf=ignore_inf,
        parameter=_positive_finite(c, "c"),
        modify_sample_size=bool(modify_sample_size),
    )
