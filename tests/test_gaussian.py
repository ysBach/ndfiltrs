"""Gaussian kernels, derivatives, and missing support against dense references."""

from __future__ import annotations

import math
from collections.abc import Callable
from decimal import Decimal, localcontext
from fractions import Fraction

import ndfiltrs as nf
import numpy as np
import pytest
from scipy import ndimage

from tests._convolution_reference import convolution_reference
from tests._reference import MODES

FUNCTIONS = (
    nf.gaussian_filter,
    nf.nangaussian_filter,
    nf.gaussian_laplace,
    nf.gaussian_gradient_magnitude,
)
DTYPES = (np.float32, np.float64)


def _gaussian_product_kernel(
    ndim: int,
    axes: tuple[int, ...],
    sigma: tuple[float, ...],
    order: tuple[int, ...],
    radius: tuple[int, ...],
) -> np.ndarray:
    """Build a dense kernel from analytic derivatives through order four.

    Parameters
    ----------
    ndim : int
        Input rank.
    axes : tuple of int
        Selected nonnegative axes in parameter order.
    sigma, order, radius : tuple
        Gaussian widths, derivative orders, and radii for selected axes.

    Returns
    -------
    numpy.ndarray
        Dense `float64` product kernel, with unit length on unselected axes.
    """
    shape = [1] * ndim
    for axis, extent in zip(axes, radius, strict=True):
        shape[axis] = 2 * extent + 1
    kernel = np.ones(shape)
    for axis, width, derivative, extent in zip(axes, sigma, order, radius, strict=True):
        x = np.arange(-extent, extent + 1, dtype=float)
        if width == 0:
            factors = np.equal(x, 0).astype(float)
        else:
            z = x / width
            base = np.exp(-0.5 * z * z)
            base /= sum(base)
            if derivative == 0:
                polynomial = np.ones_like(z)
            elif derivative == 1:
                polynomial = -z
            elif derivative == 2:
                polynomial = z * z - 1
            elif derivative == 3:
                polynomial = 3 * z - z**3
            elif derivative == 4:
                polynomial = z**4 - 6 * z * z + 3
            else:
                raise ValueError(
                    "reference supports derivative orders zero through four"
                )
            factors = base * polynomial / width**derivative
        axis_shape = [1] * ndim
        axis_shape[axis] = len(factors)
        kernel *= factors.reshape(axis_shape)
    return kernel


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_one_axis_derivatives_match_analytic_kernels_and_scipy(
    dtype: type[np.floating], mode: str, order: int
) -> None:
    array = np.array([3, -1, 2, 7, 4], dtype=dtype)
    weights = _gaussian_product_kernel(1, (0,), (0.8,), (order,), (2,))
    expected = convolution_reference(array, weights, (mode,))
    actual = nf.gaussian_filter(array, 0.8, order=order, mode=mode, radius=2)
    tolerance = 64 * np.finfo(dtype).eps
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    scipy_result = ndimage.gaussian_filter(
        array.astype(float), 0.8, order=order, mode=mode, radius=2
    ).astype(dtype)
    np.testing.assert_allclose(actual, scipy_result, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("order", [(0, 0), (1, 0), (2, 1)])
def test_selected_axes_mixed_modes_and_nonzero_constant_match_dense_kernel(
    dtype: type[np.floating], order: tuple[int, int]
) -> None:
    array = (np.arange(24).reshape(2, 3, 4) % 7 - 2).astype(dtype)
    axes, sigma, radius = (2, 0), (0.7, 1.2), (2, 1)
    weights = _gaussian_product_kernel(3, axes, sigma, order, radius)
    expected = convolution_reference(
        array, weights, ("mirror", "reflect", "constant"), cval=3.25
    )
    actual = nf.gaussian_filter(
        array,
        sigma,
        order=order,
        radius=radius,
        axes=(-1, 0),
        mode=("constant", "mirror"),
        cval=3.25,
    )
    tolerance = 64 * np.finfo(dtype).eps
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ignore_inf", [False, True])
@pytest.mark.parametrize("min_count", [1, 5, 9])
def test_nan_gaussian_normalizes_dense_support_once(
    dtype: type[np.floating], mode: str, ignore_inf: bool, min_count: int
) -> None:
    # Unequal missing support in neighboring rows detects per-axis normalization.
    array = np.array(
        [[1, np.nan, np.nan, 8], [3, 7, np.inf, 2], [np.nan, 5, 9, -np.inf]],
        dtype=dtype,
    )
    weights = _gaussian_product_kernel(2, (0, 1), (0.7, 1.3), (0, 0), (1, 1))
    for cval in (np.nan, 4.0):
        expected = convolution_reference(
            array,
            weights,
            (mode, mode),
            cval=cval,
            omit_nan=True,
            ignore_inf=ignore_inf,
            min_count=min_count,
        )
        actual = nf.nangaussian_filter(
            array,
            (0.7, 1.3),
            radius=(1, 1),
            mode=mode,
            cval=cval,
            ignore_inf=ignore_inf,
            min_count=min_count,
        )
        tolerance = 64 * np.finfo(dtype).eps
        np.testing.assert_allclose(
            actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True
        )


def test_nan_gaussian_with_selected_axes_never_mixes_other_planes() -> None:
    array = np.array([[[1, np.nan, 3], [9, 7, 5]], [[100, 200, 300], [400, 500, 600]]])
    weights = _gaussian_product_kernel(3, (2,), (1.0,), (0,), (1,))
    expected = convolution_reference(array, weights, ("reflect",) * 3, omit_nan=True)
    actual = nf.nangaussian_filter(array, 1.0, radius=1, axes=(2,))
    np.testing.assert_allclose(actual, expected, rtol=16 * np.finfo(float).eps, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "function", [nf.gaussian_laplace, nf.gaussian_gradient_magnitude]
)
@pytest.mark.parametrize("mode", ["reflect", "constant", "wrap"])
def test_gaussian_differential_operators_match_dense_components_and_scipy(
    dtype: type[np.floating], function: Callable, mode: str
) -> None:
    array = (np.arange(20).reshape(4, 5) % 11 - 3).astype(dtype)
    sigma, radius = (0.8, 1.2), (2, 1)
    derivative = 2 if function is nf.gaussian_laplace else 1
    kernels = [
        _gaussian_product_kernel(2, (0, 1), sigma, (derivative, 0), radius),
        _gaussian_product_kernel(2, (0, 1), sigma, (0, derivative), radius),
    ]
    if function is nf.gaussian_laplace:
        expected = convolution_reference(array, kernels[0] + kernels[1], (mode, mode))
    else:
        components = [
            convolution_reference(array.astype(float), kernel, (mode, mode))
            for kernel in kernels
        ]
        expected = np.hypot(*components).astype(dtype)
    actual = function(array, sigma, radius=radius, mode=mode)
    tolerance = 128 * np.finfo(dtype).eps
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    scipy_function = getattr(ndimage, function.__name__)
    scipy_result = scipy_function(
        array.astype(float), sigma, radius=radius, mode=mode
    ).astype(dtype)
    np.testing.assert_allclose(actual, scipy_result, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "function", [nf.gaussian_laplace, nf.gaussian_gradient_magnitude]
)
def test_differential_operators_respect_selected_axes(function: Callable) -> None:
    array = np.arange(24.0).reshape(2, 3, 4)
    derivative = 2 if function is nf.gaussian_laplace else 1
    weights = _gaussian_product_kernel(3, (2,), (1.0,), (derivative,), (2,))
    expected = convolution_reference(
        array, weights, ("reflect", "reflect", "constant"), cval=2
    )
    if function is nf.gaussian_gradient_magnitude:
        expected = np.abs(expected)
    actual = function(array, 1.0, radius=2, axes=(2,), mode="constant", cval=2)
    np.testing.assert_allclose(
        actual, expected, rtol=64 * np.finfo(float).eps, atol=1e-14
    )


def test_derivative_zero_taps_do_not_propagate_neighboring_nan() -> None:
    side = math.exp(-0.5) / (1 + 2 * math.exp(-0.5))
    actual = nf.gaussian_filter(
        np.array([np.nan, 2, np.nan]), 1.0, order=1, radius=1, mode="constant"
    )
    np.testing.assert_allclose(
        actual, [2 * side, np.nan, -2 * side], rtol=1e-14, atol=0
    )


def test_gradient_magnitude_avoids_squaring_overflow_and_underflow() -> None:
    for scale in (1e200, 1e-200):
        array = np.array([-scale, 0, scale])
        weights = _gaussian_product_kernel(1, (0,), (1.0,), (1,), (1,))
        expected = np.abs(convolution_reference(array, weights, ("reflect",)))
        actual = nf.gaussian_gradient_magnitude(array, 1.0, radius=1)
        assert np.isfinite(actual).all()
        assert np.greater(actual, 0).all()
        np.testing.assert_allclose(
            actual, expected, rtol=16 * np.finfo(float).eps, atol=0
        )


@pytest.mark.parametrize("function", [nf.gaussian_filter, nf.nangaussian_filter])
def test_extreme_positive_sigma_smoothing_with_explicit_radius(
    function: Callable,
) -> None:
    array = np.array([1.0, 3.0, 8.0])
    np.testing.assert_array_equal(function(array, 1e-200, radius=1), array)
    actual = function(array, 1e200, radius=1)
    np.testing.assert_allclose(
        actual, [5 / 3, 4, 19 / 3], rtol=8 * np.finfo(float).eps, atol=0
    )


def test_high_order_derivative_recovers_a_representable_underflowed_tail() -> None:
    order, location = 101, 40
    # The closed polynomial sum uses exact integers rather than the native
    # floating recurrence: He_n(z) = n! sum_k (-1)^k z^(n-2k)/(2^k k! (n-2k)!).
    hermite = sum(
        (-1) ** k
        * (
            math.factorial(order)
            // (2**k * math.factorial(k) * math.factorial(order - 2 * k))
        )
        * location ** (order - 2 * k)
        for k in range(order // 2 + 1)
    )
    with localcontext() as context:
        # Exact integer evaluation removes polynomial cancellation. Eighty
        # decimal digits leave over sixty guard digits beyond float64 rounding.
        context.prec = 80
        tail = Decimal(-800).exp()
        expected = float(
            -Decimal(hermite) * Decimal(location) ** order * tail / (1 + 2 * tail)
        )
    actual = nf.gaussian_filter(
        np.array([1.0, 0.0, 0.0]), 0.025, order=order, radius=1, mode="constant"
    )
    # Include the tail exponent and order in the log-arithmetic roundoff budget;
    # float64(0.025) also differs slightly from the exact reference width 1/40.
    tolerance = 16 * np.finfo(float).eps * (800 + order + abs(math.log(abs(expected))))
    assert expected < 0 and actual[1] < 0
    np.testing.assert_allclose(actual[1], expected, rtol=tolerance, atol=0)


@pytest.mark.parametrize(("sigma", "order"), [(10, 302), (100, 1000)])
def test_even_center_derivatives_do_not_overflow_before_sigma_scaling(
    sigma: int, order: int
) -> None:
    # At zero, He_(2m)(0)=(-1)^m (2m-1)!!. Radius zero has unit normalization.
    expected = float(
        Fraction((-1) ** (order // 2) * math.prod(range(1, order, 2)), sigma**order)
    )
    actual = nf.gaussian_filter(np.ones(1), sigma, order=order, radius=0)
    if expected == 0:
        np.testing.assert_array_equal(actual, 0)
    else:
        tolerance = 16 * np.finfo(float).eps * (order + abs(math.log(abs(expected))))
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0)


@pytest.mark.parametrize("axes", [(0, 1, 2), (2, 1, 0), (1, 0, 2)])
def test_nd_derivative_products_preserve_scale_across_axis_orders(
    axes: tuple[int, ...],
) -> None:
    widths = (1e150, 1e150, 1e-150)
    # Each second-derivative center contributes -1/sigma**2. The full product
    # is representable even though multiplying the first two factors underflows.
    with localcontext() as context:
        context.prec = 80
        expected = float(-(Decimal(10) ** -300))
    actual = nf.gaussian_filter(
        np.ones((1, 1, 1)),
        tuple(widths[axis] for axis in axes),
        order=(2, 2, 2),
        radius=0,
        axes=axes,
    )
    # Log terms reach about 700 before cancellation; budget their summed
    # magnitudes rather than requiring identical last bits across axis orders.
    tolerance = (
        16
        * np.finfo(float).eps
        * (6 + 2 * sum(abs(math.log(width)) for width in widths))
    )
    assert np.less(actual, 0).all()
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0)


@pytest.mark.parametrize("sigma", [(1e-200,), (1e-100, 1e-100)])
def test_unrepresentable_final_derivative_coefficients_raise_value_error(
    sigma: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="representable"):
        nf.gaussian_filter(np.ones((1,) * len(sigma)), sigma, order=2, radius=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_plain_gaussian_smoothing_preserves_maximum_finite_constants(
    dtype: type[np.floating], mode: str
) -> None:
    maximum = np.finfo(dtype).max
    actual = nf.gaussian_filter(
        np.full(9, maximum, dtype=dtype), 0.8, radius=2, mode=mode, cval=maximum
    )
    np.testing.assert_array_equal(actual, maximum)


@pytest.mark.parametrize("dtype", DTYPES)
def test_plain_gaussian_nan_propagation_stays_local_at_extreme_scale(
    dtype: type[np.floating],
) -> None:
    maximum = np.finfo(dtype).max
    array = np.full(11, maximum, dtype=dtype)
    array[5] = np.nan
    actual = nf.gaussian_filter(array, 0.8, radius=2)
    np.testing.assert_array_equal(
        np.isnan(actual), [False] * 3 + [True] * 5 + [False] * 3
    )
    np.testing.assert_array_equal(actual[~np.isnan(actual)], maximum)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_default_truncation_and_explicit_radius_precedence(function: Callable) -> None:
    array = np.arange(20.0).reshape(4, 5)
    if function is nf.nangaussian_filter:
        array[1, 2] = np.nan
    sigma = (0.7, 1.2)
    # The accepted default truncate=4 rounds each radius as floor(4*sigma+0.5).
    expected = function(array, sigma, radius=(3, 5))
    np.testing.assert_array_equal(function(array, sigma), expected)
    np.testing.assert_array_equal(function(array, sigma, truncate=4), expected)
    expected_small = function(array, sigma, radius=(1, 2))
    np.testing.assert_array_equal(
        function(array, sigma, truncate=1e300, radius=(1, 2)), expected_small
    )


@pytest.mark.parametrize("function", [nf.gaussian_filter, nf.nangaussian_filter])
@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_sigma_identity_with_invalid_value_policy(
    function: Callable, dtype: type[np.floating]
) -> None:
    array = np.array([1, np.nan, np.inf], dtype=dtype)
    np.testing.assert_array_equal(function(array, 0.0, radius=1), array)
    if function is nf.nangaussian_filter:
        np.testing.assert_array_equal(
            function(array, 0.0, radius=1, ignore_inf=True), [1, np.nan, np.nan]
        )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gaussian_layouts_empty_axes_and_ownership(
    function: Callable, dtype: type[np.floating]
) -> None:
    source = np.arange(24, dtype=dtype).reshape(4, 6)
    for array in (
        source,
        source.T,
        source[::-1, ::2],
        np.asfortranarray(source),
        source[:0],
    ):
        original = array.copy()
        array.flags.writeable = False
        actual = function(array, 1.0, radius=1)
        expected = function(np.array(array, order="C"), 1.0, radius=1)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(array, original)
        assert actual.dtype == array.dtype and actual.shape == array.shape
        assert actual.flags.owndata and actual.flags.c_contiguous
        assert actual.flags.aligned and actual.flags.writeable
        assert not np.shares_memory(actual, array)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("sigma", [-1, np.nan, np.inf, True, 1j, (1, 2)])
def test_invalid_sigma_rejected_on_empty_input(
    function: Callable, sigma: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        function(np.empty(0), sigma, radius=1)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    "options",
    [
        {"radius": -1},
        {"radius": True},
        {"radius": 1.5},
        {"radius": (1, 2)},
        {"axes": (0, 0)},
        {"axes": (1,)},
        {"axes": (True,)},
        {"truncate": -1},
        {"truncate": np.nan},
        {"truncate": np.inf},
    ],
)
def test_invalid_axis_and_kernel_options_rejected_on_empty_input(
    function: Callable, options: dict
) -> None:
    with pytest.raises((TypeError, ValueError)):
        function(np.empty(0), 1.0, **options)


@pytest.mark.parametrize("order", [-1, True, 1.5, (0, 1)])
def test_invalid_derivative_orders_are_rejected(order: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        nf.gaussian_filter(np.empty(0), 1.0, order=order, radius=1)


def test_zero_sigma_cannot_define_a_derivative() -> None:
    with pytest.raises(ValueError):
        nf.gaussian_filter(np.ones(3), 0.0, order=1, radius=1)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_gaussian_array_contract(function: Callable) -> None:
    for array in (
        np.ones(3, dtype=int),
        np.ones(3, dtype=np.float16),
        np.ones(3, dtype=complex),
        np.ma.array([1.0], mask=[True]),
        np.ones(3, dtype=">f8" if np.little_endian else "<f8"),
    ):
        with pytest.raises(TypeError):
            function(array, 1.0, radius=1)
    with pytest.raises(ValueError):
        function(np.array(1.0), 1.0, radius=1)
    unaligned = np.ndarray(3, dtype=np.float64, buffer=bytearray(25), offset=1)
    with pytest.raises(ValueError, match="aligned"):
        function(unaligned, 1.0, radius=1)
