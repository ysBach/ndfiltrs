"""Mean geometry, numerical semantics, ownership, and independent parity."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from fractions import Fraction

import ndfiltrs as nf
import numpy as np
import pytest
from scipy.ndimage import uniform_filter

from tests._mean_reference import mean_filter_reference
from tests._reference import MODES


@pytest.mark.parametrize("function", [nf.mean_filter, nf.nanmean_filter])
def test_mean_signatures(function: Callable) -> None:
    signature = inspect.signature(function)
    assert signature.parameters["size"].default is None
    assert signature.parameters["footprint"].default is None
    assert signature.parameters["origin"].default == 0
    assert signature.parameters["mode"].default == "reflect"
    assert signature.parameters["cval"].default == 0.0
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in signature.parameters.items()
        if name not in ("array", "size")
    )
    if function is nf.nanmean_filter:
        assert signature.parameters["ignore_inf"].default is False
        assert signature.parameters["min_count"].default == 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 1, 3), (2, 1, 2, 3)])
def test_finite_means_match_exact_windows_and_scipy(
    dtype: type[np.floating], mode: str, shape: tuple[int, ...]
) -> None:
    size = (5,) + (3,) * (len(shape) - 1)
    volume = int(np.prod(size))
    # Small integer multiples of the volume make every intermediate SciPy
    # axis division, and the final direct-window mean, exactly representable.
    array = (
        np.random.default_rng(20260906).integers(-32, 33, size=shape) * volume
    ).astype(dtype)
    cval = 2.0 * volume
    actual = nf.mean_filter(array, size, mode=mode, cval=cval)
    expected = mean_filter_reference(array, size, (mode,) * array.ndim, cval=cval)
    np.testing.assert_array_equal(actual, expected)
    scipy_result = uniform_filter(
        array.astype(np.float64), size, mode=mode, cval=cval
    ).astype(dtype)
    np.testing.assert_array_equal(actual, scipy_result)
    np.testing.assert_array_equal(
        nf.nanmean_filter(array, size, mode=mode, cval=cval), actual
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ignore_inf", [False, True])
@pytest.mark.parametrize("min_count", [1, 5, 15])
def test_nanmean_mixed_boundaries_and_retained_counts(
    dtype: type[np.floating], ignore_inf: bool, min_count: int
) -> None:
    array = np.array([[1, np.nan, np.inf], [-np.inf, 7, 11]], dtype=dtype)
    size, mode = (3, 5), ("constant", "mirror")
    for cval in (np.nan, np.inf, -np.inf, 2.0):
        options = dict(cval=cval, ignore_inf=ignore_inf, min_count=min_count)
        actual = nf.nanmean_filter(array, size, mode=mode, **options)
        expected = mean_filter_reference(array, size, mode, omit_nan=True, **options)
        # Small integer sums are exact in f64; both paths divide and cast once.
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nonfinite_effects_are_local_and_nan_bits_canonical(
    dtype: type[np.floating],
) -> None:
    array = np.ones(9, dtype=dtype)
    array[4] = np.nan
    actual = nf.mean_filter(array)
    np.testing.assert_array_equal(
        np.isnan(actual), [False] * 3 + [True] * 3 + [False] * 3
    )
    np.testing.assert_array_equal(nf.nanmean_filter(array), np.ones(9))
    array[4] = np.inf
    actual = nf.mean_filter(array)
    np.testing.assert_array_equal(
        np.isposinf(actual), [False] * 3 + [True] * 3 + [False] * 3
    )
    array[5] = -np.inf
    actual = nf.mean_filter(array)
    assert np.isnan(actual[4:6]).all()
    bits = np.uint32 if dtype is np.float32 else np.uint64
    canonical = 0x7FC00000 if dtype is np.float32 else 0x7FF8000000000000
    np.testing.assert_array_equal(actual[np.isnan(actual)].view(bits), canonical)
    np.testing.assert_array_equal(nf.nanmean_filter(array, ignore_inf=True), np.ones(9))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_empty_support_thresholds_and_boundary_multiplicity(
    dtype: type[np.floating],
) -> None:
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        assert np.isnan(nf.nanmean_filter(np.full(3, np.nan, dtype=dtype))).all()
        assert np.isnan(
            nf.nanmean_filter(np.full(3, np.inf, dtype=dtype), ignore_inf=True)
        ).all()
        np.testing.assert_array_equal(
            nf.nanmean_filter(np.array([5.0], dtype=dtype), min_count=3), [5.0]
        )
        np.testing.assert_array_equal(
            nf.nanmean_filter(
                np.array([np.nan], dtype=dtype), mode="constant", cval=7, min_count=2
            ),
            [7.0],
        )
        assert np.isnan(
            nf.nanmean_filter(
                np.array([np.nan], dtype=dtype), mode="constant", cval=7, min_count=3
            )
        ).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compensation_overflow_and_subnormal_means(dtype: type[np.floating]) -> None:
    maximum = np.finfo(dtype).max
    tiny = np.nextafter(dtype(0), dtype(1))
    cases = [
        [1e16, 1, -1e16],
        [maximum, maximum, maximum],
        [-maximum, -maximum, -maximum],
        [maximum, maximum, -maximum],
        [maximum, maximum, -maximum, -maximum, 1e-100 if dtype is np.float64 else 1],
        [tiny, tiny, tiny],
    ]
    for values in cases:
        array = np.array(values, dtype=dtype)
        expected = dtype(
            float(sum(map(Fraction, map(float, array)), Fraction()) / len(array))
        )
        actual = nf.mean_filter(array, size=len(array), mode="wrap")
        np.testing.assert_allclose(
            actual, expected, rtol=4 * np.finfo(dtype).eps, atol=0
        )
    finite_overflow_with_inf = np.array([maximum, maximum, -np.inf], dtype=dtype)
    assert np.isneginf(nf.mean_filter(finite_overflow_with_inf, mode="wrap")).all()
    np.testing.assert_array_equal(
        nf.nanmean_filter(finite_overflow_with_inf, mode="wrap", ignore_inf=True),
        maximum,
    )


def test_overflow_recovery_uses_retained_count_and_stays_local() -> None:
    maximum = np.finfo(np.float64).max
    array = np.array([maximum, maximum, -maximum, np.nan, np.nan])
    np.testing.assert_array_equal(
        nf.nanmean_filter(array, 5, mode="wrap", min_count=3), maximum / 3
    )
    assert np.isnan(nf.nanmean_filter(array, 5, mode="wrap", min_count=4)).all()
    line = np.ones(15)
    line[5:8] = maximum
    actual = nf.mean_filter(line, 3)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[:4], 1.0)
    np.testing.assert_array_equal(actual[9:], 1.0)


@pytest.mark.parametrize("ignore_inf", [False, True])
@pytest.mark.parametrize("min_count", [1, 2, 3])
def test_overlapping_overflow_windows_keep_omission_and_infinity_state_local(
    ignore_inf: bool, min_count: int
) -> None:
    maximum = np.finfo(np.float64).max
    array = np.array(
        [maximum, maximum, np.nan, maximum, -np.inf, np.inf, np.nan, 3, 6, 9]
    )
    # Neighboring windows cross overflow, missing support, both infinity signs,
    # and ordinary finite means. Infinity omission also changes retained counts.
    options = dict(cval=np.nan, ignore_inf=ignore_inf, min_count=min_count)
    expected = mean_filter_reference(
        array, (3,), ("constant",), omit_nan=True, **options
    )
    actual = nf.nanmean_filter(array, 3, mode="constant", **options)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        actual[np.isnan(actual)].view(np.uint64), 0x7FF8000000000000
    )


@pytest.mark.parametrize("function", [nf.mean_filter, nf.nanmean_filter])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mean_layouts_empty_arrays_and_signed_zero(
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
        array.flags.writeable = False
        actual = function(array, 1)
        np.testing.assert_array_equal(actual, array)
        assert actual.dtype == array.dtype and actual.shape == array.shape
        assert actual.flags.owndata and actual.flags.c_contiguous
        assert actual.flags.aligned and actual.flags.writeable
        assert not np.shares_memory(actual, array)
    zeros = function(np.array([-0.0, 0.0, -0.0], dtype=dtype), 1)
    assert not np.signbit(zeros).any()


@pytest.mark.parametrize("function", [nf.mean_filter, nf.nanmean_filter])
@pytest.mark.parametrize(
    "kwargs",
    [dict(size=0), dict(size=True), dict(size=(3,)), dict(mode="bad"), dict(cval=True)],
)
def test_mean_reuses_geometry_validation(function: Callable, kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        function(np.ones((2, 3)), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [dict(ignore_inf=1), dict(min_count=0), dict(min_count=4), dict(min_count=True)],
)
def test_nanmean_validates_omission_options_on_empty_input(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        nf.nanmean_filter(np.empty(0), **kwargs)


@pytest.mark.parametrize("function", [nf.mean_filter, nf.nanmean_filter])
def test_mean_reuses_array_validation(function: Callable) -> None:
    for array in (
        np.ones(3, dtype=int),
        np.ones(3, dtype=np.float16),
        np.ma.array([1.0], mask=[True]),
    ):
        with pytest.raises(TypeError):
            function(array)
    with pytest.raises(ValueError):
        function(np.array(1.0))
    unaligned = np.ndarray(3, dtype=np.float64, buffer=bytearray(25), offset=1)
    with pytest.raises(ValueError, match="aligned"):
        function(unaligned)
    swapped = np.ones(3, dtype=np.dtype(">f8" if np.little_endian else "<f8"))
    with pytest.raises(TypeError, match="native-endian"):
        function(swapped)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("size", [1, (3, 5), (7, 1, 3)])
def test_separable_mean_matches_direct_mean(
    dtype: type[np.floating], mode: str, size: int | tuple[int, ...]
) -> None:
    ndim = 3 if isinstance(size, tuple) and len(size) == 3 else 2
    shape = (6, 7, 5)[:ndim]
    rng = np.random.default_rng(31)
    array = rng.normal(size=shape).astype(dtype)
    array.flat[::11] = np.nan
    array.flat[5] = np.inf
    for function, options in (
        (nf.mean_filter, {}),
        (nf.nanmean_filter, {"min_count": 1 if size == 1 else 2, "ignore_inf": True}),
    ):
        direct = function(array, size, mode=mode, cval=0.5, **options)
        separable = function(
            array, size, mode=mode, cval=0.5, method="separable", **options
        )
        assert separable.dtype == array.dtype
        np.testing.assert_array_equal(np.isnan(separable), np.isnan(direct))
        np.testing.assert_array_equal(
            separable[np.isinf(direct)], direct[np.isinf(direct)]
        )
        tolerance = 1e-5 if dtype == np.float32 else 1e-13
        finite = np.isfinite(direct)
        np.testing.assert_allclose(
            separable[finite], direct[finite], rtol=0, atol=tolerance
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": 4},
        {"size": 3, "origin": 1},
        {"footprint": np.ones((3, 3))},
        {"size": 3, "method": "fast"},
    ],
)
def test_separable_mean_rejects_windows_it_cannot_express(kwargs: dict) -> None:
    kwargs = {"method": "separable", **kwargs}
    with pytest.raises(ValueError):
        nf.mean_filter(np.ones((5, 5)), **kwargs)
