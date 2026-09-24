"""Behavioral checks for percentiles, MAD, and finite-sample sigma clipping."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from ndfiltrs._robust import (
    mad_filter,
    nanmad_filter,
    nanpercentile_filter,
    percentile_filter,
    sigma_clipped_filter,
)
from scipy import ndimage

DTYPES = (np.float32, np.float64)
FUNCTIONS = (percentile_filter, nanpercentile_filter, mad_filter, nanmad_filter)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
@pytest.mark.parametrize("percentile", [0.0, 12.5, 50.0, 87.5, 100.0])
def test_percentiles_match_independent_linear_interpolation(
    dtype: type[np.floating], mode: str, percentile: float
) -> None:
    array = np.array([[3, -4, 8], [1, 12, -2]], dtype=dtype)
    expected = ndimage.generic_filter(
        array,
        lambda values: np.percentile(values, percentile, method="linear"),
        size=(3, 5),
        mode=mode,
        cval=1.25,
    )
    actual = percentile_filter(array, percentile, (3, 5), mode=mode, cval=1.25)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
def test_mad_matches_independent_two_medians(
    dtype: type[np.floating], mode: str
) -> None:
    array = np.array([[3, -4, 8], [1, 12, -2]], dtype=dtype)
    expected = ndimage.generic_filter(
        array,
        lambda values: 2 * np.median(np.abs(values - np.median(values))),
        size=(3, 5),
        mode=mode,
        cval=1.25,
    )
    actual = mad_filter(array, (3, 5), mode=mode, cval=1.25, scale=2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_even_retained_population_sets_percentile_ranks_and_both_medians(
    dtype: type[np.floating],
) -> None:
    values = np.array([np.nan, 1, 3, 9, 11], dtype=dtype)
    np.testing.assert_array_equal(nanpercentile_filter(values, 25, 5, mode="wrap"), 2.5)
    np.testing.assert_array_equal(nanmad_filter(values, 5, mode="wrap"), 4)
    np.testing.assert_array_equal(nanmad_filter(values, 5, scale=2, mode="wrap"), 8)
    assert np.isnan(nanpercentile_filter(values, 25, 5, mode="wrap", min_count=5)).all()
    assert np.isnan(nanmad_filter(values, 5, mode="wrap", min_count=5)).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_percentile_interpolation_does_not_overflow_finite_endpoints(
    dtype: type[np.floating],
) -> None:
    maximum = np.finfo(dtype).max
    values = np.array([-maximum, np.nan, maximum], dtype=dtype)
    for percentile, expected in [
        (0, -maximum),
        (25, -maximum / 2),
        (50, 0),
        (100, maximum),
    ]:
        np.testing.assert_array_equal(
            nanpercentile_filter(values, percentile, mode="wrap"), dtype(expected)
        )
    adjacent = np.nextafter(dtype(maximum), dtype(0))
    result = nanpercentile_filter(
        np.array([adjacent, np.nan, maximum], dtype=dtype), 50, mode="wrap"
    )
    assert np.isfinite(result).all()
    assert np.all((result >= adjacent) & (result <= maximum))


def test_subnormal_percentile_mad_and_clipping_keep_scaled_information() -> None:
    tiny = np.nextafter(0.0, 1.0)
    np.testing.assert_array_equal(
        nanpercentile_filter(np.array([-tiny, 2 * tiny, np.nan]), 50, mode="wrap"),
        0.0,
    )
    np.testing.assert_array_equal(
        nanmad_filter(np.array([0.0, tiny, np.nan]), scale=2, mode="wrap"),
        tiny,
    )
    np.testing.assert_array_equal(
        sigma_clipped_filter(
            np.array([0.0, tiny, tiny]),
            sigma=3,
            spread="std",
            center="median",
            min_count=3,
            mode="wrap",
        ),
        tiny,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_mad_extremes_keep_representable_deviations(dtype: type[np.floating]) -> None:
    maximum = np.finfo(dtype).max
    values = np.array([-maximum, 0, maximum], dtype=dtype)
    np.testing.assert_array_equal(mad_filter(values, mode="wrap"), maximum)
    np.testing.assert_array_equal(
        mad_filter(values, mode="wrap", scale=0.5), maximum / 2
    )
    # One overflowing deviation is an outlier, not a reason for a NaN MAD.
    values = np.array([-maximum, -maximum, maximum], dtype=dtype)
    np.testing.assert_array_equal(mad_filter(values, mode="wrap"), 0)
    smallest = np.nextafter(dtype(0), dtype(1))
    values = np.array([-smallest, 0, smallest], dtype=dtype)
    np.testing.assert_array_equal(mad_filter(values, mode="wrap"), smallest)


@pytest.mark.parametrize("dtype", DTYPES)
def test_percentile_endpoints_preserve_signed_zero_order(
    dtype: type[np.floating],
) -> None:
    values = np.array([0.0, -0.0, 0.0], dtype=dtype)
    assert np.signbit(percentile_filter(values, 0, mode="wrap")).all()
    assert not np.signbit(percentile_filter(values, 100, mode="wrap")).any()


def test_mixed_boundary_modes_and_size_one_axes() -> None:
    values = np.array([[[1.0, 2.0, 3.0]], [[10.0, 20.0, 30.0]]])
    size, mode = (3, 1, 1), ("constant", "nearest", "wrap")
    np.testing.assert_array_equal(
        percentile_filter(values, 25, size, mode=mode, cval=5),
        [[[3, 3.5, 4]], [[3, 3.5, 4]]],
    )
    np.testing.assert_array_equal(
        mad_filter(values, size, mode=mode, cval=5),
        [[[4, 3, 2]], [[4, 3, 2]]],
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_infinite_percentile_endpoints_and_undefined_interpolation(
    dtype: type[np.floating],
) -> None:
    values = np.array([-np.inf, np.nan, np.inf], dtype=dtype)
    np.testing.assert_array_equal(nanpercentile_filter(values, 0, mode="wrap"), -np.inf)
    np.testing.assert_array_equal(
        nanpercentile_filter(values, 100, mode="wrap"), np.inf
    )
    assert np.isnan(nanpercentile_filter(values, 50, mode="wrap")).all()
    np.testing.assert_array_equal(
        percentile_filter(np.full(3, np.inf, dtype=dtype), 25, mode="wrap"), np.inf
    )
    values = np.array([1, 3, np.inf], dtype=dtype)
    np.testing.assert_array_equal(
        nanpercentile_filter(values, 50, mode="wrap", ignore_inf=True), 2
    )
    np.testing.assert_array_equal(
        nanmad_filter(values, mode="wrap", ignore_inf=True), 1
    )
    assert np.isnan(mad_filter(np.full(3, np.inf, dtype=dtype))).all()
    assert np.isnan(nanmad_filter(np.full(3, np.inf, dtype=dtype))).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_nan_propagation_and_insufficient_support_are_canonical(
    dtype: type[np.floating],
) -> None:
    values = np.array([1, np.nan, 3], dtype=dtype)
    unsigned = np.uint32 if dtype is np.float32 else np.uint64
    canonical = 0x7FC00000 if dtype is np.float32 else 0x7FF8000000000000
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = (
            percentile_filter(values, 50),
            mad_filter(values),
            nanpercentile_filter(np.full(3, np.nan, dtype=dtype), 50),
            nanmad_filter(np.full(3, np.nan, dtype=dtype)),
            sigma_clipped_filter(np.full(3, np.nan, dtype=dtype)),
            mad_filter(
                np.array([-np.inf, 0, np.inf], dtype=dtype), scale=0, mode="wrap"
            ),
        )
    for result in results:
        np.testing.assert_array_equal(result.view(unsigned), canonical)


@pytest.mark.parametrize("dtype", DTYPES)
def test_sigma_clipping_rejects_monotonically_across_passes(
    dtype: type[np.floating],
) -> None:
    values = np.array([0, 0, 0, 0, 1, 10, 100], dtype=dtype)
    for passes, expected in [
        (0, 111 / 7),
        (1, 11 / 6),
        (2, 0.2),
        (3, 0),
        (10, 0),
        (None, 0),
    ]:
        actual = sigma_clipped_filter(values, 7, sigma=1, maxiters=passes, mode="wrap")
        np.testing.assert_allclose(
            actual, dtype(expected), rtol=np.finfo(dtype).eps, atol=0
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sigma_clipping_zero_spread_and_mad_normal_consistency(
    dtype: type[np.floating],
) -> None:
    np.testing.assert_array_equal(
        sigma_clipped_filter(
            np.array([1, 1, 1, 1, 100], dtype=dtype), 5, spread="mad", mode="wrap"
        ),
        1,
    )
    values = np.array([-10, -1, -1, 0, 1, 1.4, 10], dtype=dtype)
    # Median=0, raw MAD=1. Its normal-consistent spread retains 1.4 but not 10.
    expected = np.mean(values[1:6].astype(np.float64))
    np.testing.assert_allclose(
        sigma_clipped_filter(values, 7, spread="mad", sigma=1, maxiters=1, mode="wrap"),
        dtype(expected),
        rtol=np.finfo(dtype).eps,
        atol=0,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sigma_clipping_asymmetric_bounds_and_inclusive_endpoints(
    dtype: type[np.floating],
) -> None:
    values = np.array([-100, 0, 0, 0, 10], dtype=dtype)
    np.testing.assert_array_equal(
        sigma_clipped_filter(
            values, 5, sigma_lower=10, sigma_upper=0.1, maxiters=1, mode="wrap"
        ),
        -25,
    )
    np.testing.assert_array_equal(
        sigma_clipped_filter(
            values, 5, sigma_lower=0.1, sigma_upper=10, maxiters=1, mode="wrap"
        ),
        2.5,
    )
    values = np.array([-1, np.nan, 1], dtype=dtype)
    np.testing.assert_array_equal(
        sigma_clipped_filter(values, sigma=1, center="mean", min_count=2, mode="wrap"),
        0,
    )
    np.testing.assert_array_equal(
        sigma_clipped_filter(np.array([1, 1, 2], dtype=dtype), sigma=0, mode="wrap"),
        1,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sigma_clipping_omits_nonfinite_samples_and_enforces_final_count(
    dtype: type[np.floating],
) -> None:
    values = np.array([1, 3, np.nan, np.inf, -np.inf], dtype=dtype)
    np.testing.assert_array_equal(sigma_clipped_filter(values, 5, mode="wrap"), 2)
    assert np.isnan(sigma_clipped_filter(values, 5, mode="wrap", min_count=3)).all()
    values = np.array([1, 1, 1, 1, 100], dtype=dtype)
    assert np.isnan(
        sigma_clipped_filter(values, 5, spread="mad", mode="wrap", min_count=5)
    ).all()
    values = np.array([-np.finfo(dtype).max, np.nan, np.finfo(dtype).max], dtype=dtype)
    np.testing.assert_array_equal(
        sigma_clipped_filter(values, sigma=1, mode="wrap", min_count=2), 0
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_sigma_clipping_center_and_final_statistic_are_independent(
    dtype: type[np.floating],
) -> None:
    values = np.array([0, 1, 20], dtype=dtype)
    np.testing.assert_array_equal(
        sigma_clipped_filter(values, maxiters=0, statistic="median", mode="wrap"), 1
    )
    np.testing.assert_array_equal(
        sigma_clipped_filter(values, maxiters=0, mode="wrap"), 7
    )
    values = np.array([0, 0, 0, 0, 10], dtype=dtype)
    assert np.isnan(
        sigma_clipped_filter(values, 5, center="mean", spread="mad", mode="wrap")
    ).all()


@pytest.mark.parametrize("function", (*FUNCTIONS, sigma_clipped_filter))
@pytest.mark.parametrize("dtype", DTYPES)
def test_layout_dtype_empty_axes_and_independent_output(
    function: Callable, dtype: type[np.floating]
) -> None:
    source = np.arange(24, dtype=dtype).reshape(4, 6)
    positional = (50,) if "percentile" in function.__name__ else ()
    for values in (source.T, source[::-1, ::2], source[:0], np.asfortranarray(source)):
        values.flags.writeable = False
        actual = function(values, *positional, size=1)
        expected = np.zeros_like(values) if "mad" in function.__name__ else values
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == values.shape and actual.dtype == values.dtype
        assert actual.flags.owndata and actual.flags.c_contiguous
        assert actual.flags.aligned and actual.flags.writeable
        assert not np.shares_memory(actual, values)


@pytest.mark.parametrize(
    "function", (nanpercentile_filter, nanmad_filter, sigma_clipped_filter)
)
def test_constant_padding_and_repeated_boundary_samples_count(
    function: Callable,
) -> None:
    positional = (50,) if "percentile" in function.__name__ else ()
    options = {"maxiters": 0} if function is sigma_clipped_filter else {}
    result = function(np.array([5.0]), *positional, min_count=3, **options)
    np.testing.assert_array_equal(result, 0 if function is nanmad_filter else 5)
    result = function(
        np.array([np.nan]), *positional, mode="constant", cval=2, min_count=2, **options
    )
    np.testing.assert_array_equal(result, 0 if function is nanmad_filter else 2)


@pytest.mark.parametrize("function", (percentile_filter, nanpercentile_filter))
@pytest.mark.parametrize("percentile", [-1, 101, np.nan, np.inf, True, "50", [50], 1j])
def test_invalid_percentiles_rejected_on_empty_input(
    function: Callable, percentile: object
) -> None:
    with pytest.raises((ValueError, TypeError)):
        function(np.empty(0), percentile)


@pytest.mark.parametrize("function", (mad_filter, nanmad_filter))
@pytest.mark.parametrize("scale", [-1, np.nan, np.inf, True, "1", 1j])
def test_invalid_mad_scales_rejected_on_empty_input(
    function: Callable, scale: object
) -> None:
    with pytest.raises((ValueError, TypeError)):
        function(np.empty(0), scale=scale)


@pytest.mark.parametrize(
    "options",
    [
        {"sigma": -1},
        {"sigma": True},
        {"sigma": np.inf},
        {"sigma_lower": -1},
        {"sigma_upper": np.nan},
        {"maxiters": -1},
        {"maxiters": True},
        {"maxiters": 1.5},
        {"center": "mode"},
        {"spread": "variance"},
        {"statistic": "sum"},
        {"min_count": 0},
        {"min_count": True},
    ],
)
def test_invalid_clipping_parameters_rejected_on_empty_input(options: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        sigma_clipped_filter(np.empty(0), **options)


@pytest.mark.parametrize("function", (nanpercentile_filter, nanmad_filter))
@pytest.mark.parametrize(
    "options", [{"ignore_inf": 1}, {"min_count": 0}, {"min_count": True}]
)
def test_invalid_omission_options_rejected_on_empty_input(
    function: Callable, options: dict
) -> None:
    positional = (50,) if "percentile" in function.__name__ else ()
    with pytest.raises((ValueError, TypeError)):
        function(np.empty(0), *positional, **options)
