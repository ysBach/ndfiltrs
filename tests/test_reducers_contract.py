"""Public consequences of the pure-Rust reducers median contract."""

import ndfiltrs as nf
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_even_retained_extreme_values_have_overflow_safe_midpoint(
    dtype: np.dtype,
) -> None:
    maximum = np.finfo(dtype).max
    previous = np.nextafter(maximum, dtype(0), dtype=dtype)
    values = np.asarray([maximum, previous, np.nan], dtype=dtype)
    actual = nf.nanmedian_filter(values, 3, mode="constant", cval=np.nan)
    assert np.isfinite(actual[1])
    assert actual[1] == dtype(previous + (maximum - previous) / dtype(2))


def test_infinity_pairings_and_signed_zero_follow_contract() -> None:
    values = np.asarray([-np.inf, np.inf, np.nan], dtype=np.float64)
    mixed = nf.nanmedian_filter(values, 3, mode="constant", cval=np.nan)
    assert np.isnan(mixed[1])
    zeros = np.asarray([-0.0, 0.0, np.nan], dtype=np.float64)
    midpoint = nf.nanmedian_filter(zeros, 3, mode="constant", cval=np.nan)
    assert midpoint.view(np.uint64)[1] == np.float64(0.0).view(np.uint64)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("infinity", [-np.inf, np.inf])
def test_equal_infinities_are_retained(dtype: np.dtype, infinity: float) -> None:
    values = np.asarray([infinity, infinity, np.nan], dtype=dtype)
    result = nf.nanmedian_filter(values, 3, mode="constant", cval=np.nan)
    assert result[1] == infinity


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_negative_extremes_and_repeated_values_follow_contract(
    dtype: np.dtype,
) -> None:
    minimum = -np.finfo(dtype).max
    next_value = np.nextafter(minimum, dtype(0), dtype=dtype)
    extremes = np.asarray([minimum, next_value, np.nan], dtype=dtype)
    result = nf.nanmedian_filter(extremes, 3, mode="constant", cval=np.nan)
    assert np.isfinite(result[1])
    repeated = np.asarray([2.0, 2.0, 4.0, 4.0, np.nan], dtype=dtype)
    result = nf.nanmedian_filter(repeated, 5, mode="constant", cval=np.nan)
    assert result[2] == dtype(3.0)
