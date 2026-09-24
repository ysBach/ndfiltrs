"""Weighted filtering geometry, omission semantics, and exact arithmetic."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import ndfiltrs as nf
import numpy as np
import pytest
from scipy import ndimage

from tests._convolution_reference import convolution_reference
from tests._reference import MODES

RAW = (nf.convolve, nf.correlate)
NORMALIZED = (nf.nanconvolve, nf.nancorrelate)
FUNCTIONS = RAW + NORMALIZED
DTYPES = (np.float32, np.float64)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_explicit_weighted_filter_signatures(function: Callable) -> None:
    signature = inspect.signature(function)
    assert signature.parameters["mode"].default == "reflect"
    assert signature.parameters["cval"].default == 0.0
    assert signature.parameters["origin"].default == 0
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in signature.parameters.items()
        if name not in ("array", "weights")
    )
    if function in NORMALIZED:
        assert signature.parameters["min_count"].default == 1
        assert signature.parameters["ignore_inf"].default is False


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("function", "weights", "expected"),
    [
        (nf.correlate, [1, 2, 4], [10, 17, 24, 11]),
        (nf.convolve, [1, 2, 4], [4, 11, 18, 20]),
        (nf.correlate, [1, 3], [3, 7, 11, 15]),
        (nf.convolve, [1, 3], [5, 9, 13, 12]),
    ],
)
def test_asymmetric_and_even_kernel_orientation(
    dtype: type[np.floating], function: Callable, weights: list, expected: list
) -> None:
    actual = function(np.array([1, 2, 3, 4], dtype=dtype), weights, mode="constant")
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("function", RAW)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 1, 3)])
def test_raw_windows_match_exact_reference_and_scipy(
    function: Callable, dtype: type[np.floating], mode: str, shape: tuple[int, ...]
) -> None:
    array = ((np.arange(np.prod(shape)).reshape(shape) * 7) % 19 - 9).astype(dtype)
    kernel_shape = (4,) + (3,) * (array.ndim - 1)
    weights = (np.arange(np.prod(kernel_shape)).reshape(kernel_shape) % 7 - 3) / 4
    origin = (-1,) + (1,) * (array.ndim - 1)
    expected = convolution_reference(
        array,
        weights,
        (mode,) * array.ndim,
        cval=1.25,
        origin=origin,
        correlate=function is nf.correlate,
    )
    actual = function(array, weights, mode=mode, cval=1.25, origin=origin)
    np.testing.assert_array_equal(actual, expected)
    scipy_function = getattr(ndimage, function.__name__)
    np.testing.assert_array_equal(
        actual, scipy_function(array, weights, mode=mode, cval=1.25, origin=origin)
    )


@pytest.mark.parametrize("function", FUNCTIONS)
def test_mixed_modes_and_numpy_integer_origins(function: Callable) -> None:
    array = np.arange(12.0).reshape(2, 1, 2, 3)
    weights = np.arange(1.0, 25.0).reshape(2, 1, 3, 4)
    mode = ("constant", "nearest", "mirror", "wrap")
    origin = (np.int64(-1), np.int64(0), np.int64(1), np.int64(1))
    expected = convolution_reference(
        array,
        weights,
        mode,
        origin=origin,
        cval=-2,
        correlate="correlate" in function.__name__,
        omit_nan=function in NORMALIZED,
    )
    actual = function(array, weights, mode=list(mode), cval=-2, origin=list(origin))
    np.testing.assert_allclose(actual, expected, rtol=4 * np.finfo(float).eps, atol=0)


@pytest.mark.parametrize("function", NORMALIZED)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ignore_inf", [False, True])
@pytest.mark.parametrize("min_count", [1, 3, 5])
def test_normalized_omission_matches_exact_weighted_average(
    function: Callable, dtype: type[np.floating], ignore_inf: bool, min_count: int
) -> None:
    array = np.array([[1, np.nan, np.inf], [-np.inf, 7, 11]], dtype=dtype)
    weights = np.array([[0, 1, 2], [3, 4, 8]], dtype=float)
    mode = ("constant", "mirror")
    for cval in (np.nan, np.inf, -np.inf, 2.0):
        options = dict(cval=cval, ignore_inf=ignore_inf, min_count=min_count)
        expected = convolution_reference(
            array,
            weights,
            mode,
            correlate=function is nf.nancorrelate,
            omit_nan=True,
            **options,
        )
        actual = function(array, weights, mode=mode, **options)
        np.testing.assert_allclose(
            actual, expected, rtol=8 * np.finfo(dtype).eps, atol=0, equal_nan=True
        )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_taps_skip_nonfinite_samples_and_support_counts(
    function: Callable, dtype: type[np.floating]
) -> None:
    array = np.array([np.nan, 3, np.inf], dtype=dtype)
    np.testing.assert_array_equal(function(array, [0, 1, 0]), array)
    if function in NORMALIZED:
        with pytest.raises(ValueError):
            function(array, [0, 1, 0], min_count=2)
        np.testing.assert_array_equal(
            function(np.array([5], dtype=dtype), [1, 2, 3], min_count=3), [5]
        )
        actual = function(
            np.array([np.nan], dtype=dtype),
            [1, 2, 3],
            mode="constant",
            cval=7,
            min_count=2,
        )
        np.testing.assert_array_equal(actual, [7])
        assert np.isnan(
            function(
                np.array([np.nan], dtype=dtype),
                [1, 2, 3],
                mode="constant",
                cval=7,
                min_count=3,
            )
        ).all()
    else:
        actual = function(array, [-0.0, 0.0, 0.0])
        np.testing.assert_array_equal(actual, 0)
        assert not np.signbit(actual).any()


@pytest.mark.parametrize("function", RAW)
@pytest.mark.parametrize("dtype", DTYPES)
def test_signed_weights_transform_infinities_locally(
    function: Callable, dtype: type[np.floating]
) -> None:
    array = np.array([1, np.inf, np.nan, -np.inf, 2], dtype=dtype)
    actual = function(array, [-2])
    np.testing.assert_array_equal(actual, [-2, -np.inf, np.nan, np.inf, -4])
    bits = np.uint32 if dtype is np.float32 else np.uint64
    canonical = 0x7FC00000 if dtype is np.float32 else 0x7FF8000000000000
    np.testing.assert_array_equal(actual[np.isnan(actual)].view(bits), canonical)


@pytest.mark.parametrize("function", RAW)
@pytest.mark.parametrize("dtype", DTYPES)
def test_raw_compensation_product_overflow_cancellation_and_subnormals(
    function: Callable, dtype: type[np.floating]
) -> None:
    maximum = np.finfo(dtype).max
    tiny = np.nextafter(dtype(0), dtype(1))
    cases = [
        ([1e16, 1, -1e16], [1, 1, 1]),
        ([maximum, maximum, -maximum], [0.75, 0.75, 0.75]),
        ([maximum, maximum, -maximum, -maximum, 1], [2, 2, 2, 2, 1]),
        ([tiny, tiny, tiny], [0.5, 0.5, 1]),
        ([maximum], [2]),
    ]
    for values, taps in cases:
        array = np.array(values, dtype=dtype)
        weights = np.array(taps, dtype=float)
        expected = convolution_reference(
            array, weights, ("wrap",), correlate=function is nf.correlate
        )
        actual = function(array, weights, mode="wrap")
        np.testing.assert_allclose(
            actual, expected, rtol=8 * np.finfo(dtype).eps, atol=0, equal_nan=True
        )


@pytest.mark.parametrize("function", NORMALIZED)
@pytest.mark.parametrize("dtype", DTYPES)
def test_normalized_extreme_weights_preserve_representable_results(
    function: Callable, dtype: type[np.floating]
) -> None:
    maximum = np.finfo(dtype).max
    tiny = np.nextafter(dtype(0), dtype(1))
    for values, taps in (
        ([maximum, maximum, maximum], [1e308, 1e308, 1e308]),
        ([maximum, -maximum, maximum], [1e308, 1e308, 1e308]),
        ([tiny, tiny, tiny], [1e-300, 1e-300, 1e-300]),
    ):
        array = np.array(values, dtype=dtype)
        weights = np.array(taps)
        expected = convolution_reference(
            array,
            weights,
            ("wrap",),
            correlate=function is nf.nancorrelate,
            omit_nan=True,
        )
        actual = function(array, weights, mode="wrap")
        np.testing.assert_allclose(
            actual, expected, rtol=8 * np.finfo(dtype).eps, atol=0
        )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_layouts_empty_axes_ownership_and_kernel_views(
    function: Callable, dtype: type[np.floating]
) -> None:
    source = np.arange(24, dtype=dtype).reshape(4, 6)
    weights = np.array([[1.0]])[:, ::-1]
    weights.flags.writeable = False
    for array in (
        source,
        source.T,
        source[::-1, ::2],
        np.asfortranarray(source),
        source[:0],
    ):
        array.flags.writeable = False
        actual = function(array, weights)
        np.testing.assert_array_equal(actual, array)
        assert actual.dtype == array.dtype and actual.shape == array.shape
        assert actual.flags.owndata and actual.flags.c_contiguous
        assert actual.flags.aligned and actual.flags.writeable
        assert not np.shares_memory(actual, array)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    "weights",
    [[], [np.nan], [np.inf], [True], [1j], [[1.0]], np.ma.array([1.0], mask=[True])],
)
def test_invalid_kernels_are_rejected_on_empty_arrays(
    function: Callable, weights: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        function(np.empty(0), weights)


@pytest.mark.parametrize("function", NORMALIZED)
@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [1, -1]])
def test_normalized_kernels_require_nonnegative_positive_support(
    function: Callable, weights: list
) -> None:
    with pytest.raises(ValueError):
        function(np.ones(3), weights)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("origin", [True, 0.5, -2, 1, (0, 0)])
def test_invalid_even_kernel_origins_are_rejected(
    function: Callable, origin: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        function(np.empty(0), [1, 1], origin=origin)


@pytest.mark.parametrize("function", NORMALIZED)
@pytest.mark.parametrize(
    "options", [{"min_count": 0}, {"min_count": True}, {"ignore_inf": 1}]
)
def test_normalized_options_are_validated_on_empty_arrays(
    function: Callable, options: dict
) -> None:
    with pytest.raises((TypeError, ValueError)):
        function(np.empty(0), [1], **options)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_array_contract_rejects_unsupported_inputs(function: Callable) -> None:
    for array in (
        np.ones(3, dtype=int),
        np.ones(3, dtype=np.float16),
        np.ones(3, dtype=complex),
        np.ma.array([1.0], mask=[True]),
        np.ones(3, dtype=">f8" if np.little_endian else "<f8"),
    ):
        with pytest.raises(TypeError):
            function(array, [1])
    with pytest.raises(ValueError):
        function(np.array(1.0), [1])
    unaligned = np.ndarray(3, dtype=np.float64, buffer=bytearray(25), offset=1)
    with pytest.raises(ValueError, match="aligned"):
        function(unaligned, [1])
