"""Exact per-pixel reference for convolution with noise propagation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from tests._reference import _map_index


def _ratio(numerator: Fraction, denominator: Fraction) -> float:
    if denominator == 0:
        return math.copysign(math.inf, numerator) if numerator else math.nan
    return float(numerator / denominator)


def noise_reference(
    array: np.ndarray,
    noise: np.ndarray,
    weights: np.ndarray,
    mode: Sequence[str],
    *,
    cval: float = 0.0,
    correlate: bool = False,
    estimator: str = "sum",
    inverse_weighting: bool = False,
    inverse_noise: bool = False,
    omit_nan: bool = False,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate estimate, variance, and signal-to-noise ratio pixel by pixel.

    Parameters
    ----------
    array, noise : numpy.ndarray
        Small same-shape data and variance (or inverse variance) arrays.
    weights : numpy.ndarray
        Same-rank real kernel with zero origin.
    mode : sequence of str
        Boundary mapping for each axis.
    cval : float, optional
        Exterior data value. Exterior samples have zero variance.
    correlate : bool, optional
        Use correlation coordinates when `True`.
    estimator : {"sum", "mean", "amplitude"}, optional
        Denominator ``1``, ``sum w``, or ``sum w a``.
    inverse_weighting : bool, optional
        Use coefficients ``w / v`` instead of ``w``.
    inverse_noise : bool, optional
        `noise` holds inverse variances.
    omit_nan, ignore_inf : bool, optional
        Omission rules. Without `omit_nan`, invalid samples propagate.
    min_count : int, optional
        Required retained samples when omitting.

    Returns
    -------
    tuple of numpy.ndarray
        `float64` estimate, variance, and signal-to-noise ratio. Products
        and sums are exact rationals, rounded once.
    """
    shape = array.shape
    outputs = [np.empty(shape) for _ in range(3)]
    centers = tuple(length // 2 for length in weights.shape)
    constant = float(array.dtype.type(cval))
    for index in np.ndindex(shape):
        result = _evaluate(
            array,
            noise,
            weights,
            mode,
            index,
            centers,
            constant,
            correlate,
            estimator,
            inverse_weighting,
            inverse_noise,
            omit_nan,
            ignore_inf,
            min_count,
        )
        for output, value in zip(outputs, result, strict=True):
            output[index] = value
    return outputs[0], outputs[1], outputs[2]


def _evaluate(
    array: np.ndarray,
    noise: np.ndarray,
    weights: np.ndarray,
    mode: Sequence[str],
    index: tuple[int, ...],
    centers: tuple[int, ...],
    constant: float,
    correlate: bool,
    estimator: str,
    inverse_weighting: bool,
    inverse_noise: bool,
    omit_nan: bool,
    ignore_inf: bool,
    min_count: int,
) -> tuple[float, float, float]:
    nan = (math.nan, math.nan, math.nan)
    retained = []
    noise_nan = False
    infinite_variance = False
    for position in np.ndindex(weights.shape):
        weight = float(weights[position])
        if weight == 0:
            continue
        coordinates = []
        for axis, (coordinate, tap, center) in enumerate(
            zip(index, position, centers, strict=True)
        ):
            delta = tap - center if correlate else center - tap
            coordinates.append(
                _map_index(coordinate + delta, array.shape[axis], mode[axis])
            )
        source = None if None in coordinates else tuple(coordinates)
        value = constant if source is None else float(array[source])
        if math.isnan(value):
            if omit_nan:
                continue
            return nan
        if ignore_inf and math.isinf(value):
            continue
        if source is None:
            kind, level = ("valid", 0.0) if not inverse_weighting else ("invalid", 0.0)
        else:
            raw = float(noise[source])
            if not raw >= 0:
                kind, level = "invalid", 0.0
            else:
                inverse = raw if inverse_noise else (math.inf if raw == 0 else 1 / raw)
                variance = (
                    raw if not inverse_noise else (math.inf if raw == 0 else 1 / raw)
                )
                level = inverse if inverse_weighting else variance
                if (not inverse_weighting and variance == math.inf) or (
                    inverse_weighting and inverse == 0
                ):
                    kind = "uninformative"
                elif inverse_weighting and inverse == math.inf:
                    kind = "invalid"
                else:
                    kind = "valid"
        if kind != "valid":
            if omit_nan or (kind == "uninformative" and inverse_weighting):
                continue
            if kind == "invalid" and inverse_weighting:
                return nan
            noise_nan |= kind == "invalid"
            infinite_variance |= kind == "uninformative"
            level = math.nan
        retained.append((source, weight, value, level))
    if (omit_nan and len(retained) < min_count) or (
        estimator != "sum" and not retained
    ):
        return nan
    signs = set()
    numerator = Fraction()
    denominator = Fraction(1) if estimator == "sum" else Fraction()
    groups: dict[tuple, Fraction] = defaultdict(Fraction)
    levels = {}
    for source, weight, value, level in retained:
        coefficient = Fraction(weight) * (Fraction(level) if inverse_weighting else 1)
        if math.isinf(value):
            signs.add(math.copysign(1, value) * math.copysign(1, coefficient))
        else:
            numerator += coefficient * Fraction(value)
        if estimator == "mean":
            denominator += Fraction(weight)
        elif estimator == "amplitude":
            denominator += Fraction(weight) * coefficient
        if source is not None and math.isfinite(level):
            groups[source] += Fraction(weight)
            levels[source] = Fraction(level)
    if len(signs) > 1:
        return nan
    noise_sum = sum(
        (total * total * levels[source] for source, total in groups.items()),
        Fraction(),
    )
    value = (
        math.copysign(math.inf, signs.pop())
        if signs
        else _ratio(numerator, denominator)
    )
    if noise_nan:
        return value, math.nan, math.nan
    if infinite_variance:
        return value, math.inf, value / math.inf
    variance = _ratio(noise_sum, denominator * denominator)
    if math.isinf(value):
        with np.errstate(divide="ignore"):
            return value, variance, float(np.float64(value) / np.sqrt(variance))
    if noise_sum == 0:
        return value, variance, _ratio(numerator, Fraction())
    return value, variance, float(numerator) / math.sqrt(noise_sum)
