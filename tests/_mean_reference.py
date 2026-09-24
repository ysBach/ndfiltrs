"""Exact rational window means, independent of native summation arithmetic."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from tests._reference import _map_index


def mean_filter_reference(
    array: np.ndarray,
    size: Sequence[int],
    mode: Sequence[str],
    *,
    cval: float = 0.0,
    omit_nan: bool = False,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> np.ndarray:
    """Evaluate windows using exact finite sums and rational division.

    Parameters
    ----------
    array : numpy.ndarray
        Small reference input with floating dtype.
    size, mode : sequence
        Per-axis window sizes and boundary modes.
    cval : float, optional
        Constant boundary sample, cast to the input dtype.
    omit_nan, ignore_inf : bool, optional
        Omission policies.
    min_count : int, optional
        Minimum retained samples, counting boundary repetitions.

    Returns
    -------
    numpy.ndarray
        Reference means with the input shape and dtype. The rational result
        is rounded to `float64` and then cast to the output dtype.
    """
    output = np.empty_like(array)
    constant = float(array.dtype.type(cval))
    for center in np.ndindex(array.shape):
        values = []
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
        if (
            len(values) < min_count
            or any(math.isnan(value) for value in values)
            or (math.inf in values and -math.inf in values)
        ):
            result = math.nan
        elif math.inf in values:
            result = math.inf
        elif -math.inf in values:
            result = -math.inf
        else:
            result = float(sum(map(Fraction, values), Fraction()) / len(values))
        output[center] = result
    return output
