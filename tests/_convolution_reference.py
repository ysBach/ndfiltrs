"""Exact weighted windows for independent convolution and Gaussian checks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from tests._reference import _map_index


def convolution_reference(
    array: np.ndarray,
    weights: np.ndarray,
    mode: Sequence[str],
    *,
    cval: float = 0.0,
    origin: Sequence[int] | None = None,
    correlate: bool = False,
    omit_nan: bool = False,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> np.ndarray:
    """Evaluate weighted windows using exact finite products and sums.

    Parameters
    ----------
    array, weights : numpy.ndarray
        Small test input and same-rank real kernel.
    mode : sequence of str
        Boundary mapping for each input axis.
    cval : float, optional
        Exterior value, cast to the input dtype.
    origin : sequence of int or None, optional
        Kernel-center offsets; `None` selects zeros.
    correlate : bool, optional
        Use correlation coordinates when `True`.
    omit_nan, ignore_inf : bool, optional
        Missing-value policies. NaN omission also normalizes retained weights.
    min_count : int, optional
        Required retained nonzero-weight positions for normalized filtering.

    Returns
    -------
    numpy.ndarray
        Same-shape, same-dtype result, rounded once after rational arithmetic.
    """
    output = np.empty_like(array)
    offsets = (0,) * array.ndim if origin is None else origin
    centers = tuple(
        n // 2 + shift for n, shift in zip(weights.shape, offsets, strict=True)
    )
    constant = float(array.dtype.type(cval))
    for index in np.ndindex(array.shape):
        numerator = Fraction()
        denominator = Fraction()
        infinities: set[float] = set()
        retained = 0
        propagate_nan = False
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
            value = (
                constant if None in coordinates else float(array[tuple(coordinates)])
            )
            if math.isnan(value):
                if omit_nan:
                    continue
                propagate_nan = True
                break
            if ignore_inf and math.isinf(value):
                continue
            retained += 1
            denominator += Fraction(weight)
            if math.isinf(value):
                infinities.add(math.copysign(1.0, value) * math.copysign(1.0, weight))
            else:
                numerator += Fraction(value) * Fraction(weight)
        if propagate_nan or (omit_nan and retained < min_count) or len(infinities) > 1:
            result = math.nan
        elif infinities:
            result = math.copysign(math.inf, next(iter(infinities)))
        elif omit_nan and denominator == 0:
            result = math.nan
        else:
            exact = numerator / denominator if omit_nan else numerator
            try:
                result = float(exact)
            except OverflowError:
                result = math.inf if exact > 0 else -math.inf
        with np.errstate(over="ignore", under="ignore"):
            output[index] = result
    return output
