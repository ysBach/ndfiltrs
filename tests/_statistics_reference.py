"""Literal windows with exact rational variance and signed-zero extrema."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tests._reference import _map_index

Statistic = Literal["min", "max", "var", "std"]
FloatArray = NDArray[np.float32] | NDArray[np.float64]


def _sqrt_fraction(value: Fraction) -> float:
    """Take a square root before converting an exact variance to binary64.

    Parameters
    ----------
    value : fractions.Fraction
        Nonnegative exact variance, including values outside binary64 range.

    Returns
    -------
    float
        High-precision square root rounded to binary64. Decimal precision
        includes every integer digit plus twice the binary64 decimal precision
        as guard digits for division and square root.
    """
    with localcontext() as context:
        context.prec = (
            max(len(str(value.numerator)), len(str(value.denominator)))
            + 2 * sys.float_info.dig
        )
        variance = Decimal(value.numerator) / Decimal(value.denominator)
        return float(variance.sqrt())


def statistics_filter_reference(
    array: FloatArray,
    size: Sequence[int],
    mode: Sequence[str],
    *,
    statistic: Statistic,
    cval: float = 0.0,
    omit_nan: bool = False,
    ignore_inf: bool = False,
    min_count: int = 1,
    ddof: int = 0,
) -> FloatArray:
    """Evaluate independent coordinate windows and exact finite statistics.

    Parameters
    ----------
    array : numpy.ndarray
        Small native floating input; its binary values are the exact samples.
    size, mode : sequence
        Per-axis odd window sizes and boundary modes.
    statistic : {'min', 'max', 'var', 'std'}
        Statistic to evaluate. Extrema order negative zero before positive zero.
    cval : float, optional
        Boundary value, cast to the input dtype before evaluating windows.
    omit_nan, ignore_inf : bool, optional
        Remove NaNs and infinities, respectively, before counting samples.
    min_count : int, optional
        Required retained count, including repeated boundary positions.
    ddof : int, optional
        Nonnegative correction to the variance denominator.

    Returns
    -------
    numpy.ndarray
        Same-shape, same-dtype reference. Finite variance uses exact rational
        centering and squared deviations. Standard deviation takes a Decimal
        square root before binary64 conversion. Results then cast to the input
        dtype. Unsupported support or retained nonfinite variance yields NaN.
    """
    if statistic not in ("min", "max", "var", "std"):
        raise ValueError(f"unsupported statistic: {statistic}")
    output = np.empty(array.shape, dtype=array.dtype)
    constant = float(array.dtype.type(cval))
    for center in np.ndindex(array.shape):
        values: list[float] = []
        for position in np.ndindex(tuple(size)):
            mapped = [
                _map_index(c + k - n // 2, length, boundary)
                for c, k, n, length, boundary in zip(
                    center, position, size, array.shape, mode, strict=True
                )
            ]
            value = constant if None in mapped else float(array[tuple(mapped)])
            if omit_nan and math.isnan(value):
                continue
            if ignore_inf and math.isinf(value):
                continue
            values.append(value)
        if len(values) < min_count or any(math.isnan(value) for value in values):
            result = math.nan
        elif statistic in ("min", "max"):
            # Numeric order resolves unequal values; the second key orders
            # equal zeros independently of input order, unlike Python min/max.
            select = min if statistic == "min" else max
            result = select(values, key=lambda value: (value, math.copysign(1, value)))
        elif len(values) <= ddof or any(math.isinf(value) for value in values):
            result = math.nan
        else:
            rational = [Fraction(value) for value in values]
            mean = sum(rational, Fraction()) / len(rational)
            variance = sum(((value - mean) ** 2 for value in rational), Fraction())
            variance /= len(rational) - ddof
            if statistic == "std":
                result = _sqrt_fraction(variance)
            else:
                try:
                    result = float(variance)
                except OverflowError:
                    result = math.inf
        # Expected IEEE output conversion may overflow/underflow; production
        # calls are outside this context and their warnings remain observable.
        with np.errstate(over="ignore", under="ignore"):
            output[center] = result
    return output
