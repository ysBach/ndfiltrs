"""Exact rolling means against literal windows and rational rounding."""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import ndfiltrs as nf
import numpy as np
import pytest
from numpy.typing import NDArray

from tests._reference import MODES, _map_index

FloatArray = NDArray[np.float32] | NDArray[np.float64]
DTYPES = (np.float32, np.float64)


def _round_exact(value: Fraction, dtype: np.dtype) -> np.floating:
    """Round a rational directly to the target dtype, breaking ties to even.

    Notes
    -----
    Conversion through Python float supplies a candidate, not the answer.
    Compare that candidate and its adjacent target floats using exact rational
    distances, avoiding double rounding for `float32`. Means are bounded by
    finite input samples, so an infinite candidate is never needed.
    """
    if value == 0:
        return dtype.type(0)
    candidate = dtype.type(float(value))
    candidates = (
        candidate,
        np.nextafter(candidate, dtype.type(-np.inf)),
        np.nextafter(candidate, dtype.type(np.inf)),
    )
    bits = np.uint32 if dtype == np.dtype(np.float32) else np.uint64
    return min(
        (item for item in candidates if np.isfinite(item)),
        key=lambda item: (
            abs(Fraction(float(item)) - value),
            int(item.view(bits)) & 1,
        ),
    )


def _reference(
    array: FloatArray,
    size: tuple[int, ...],
    origin: tuple[int, ...],
    modes: tuple[str, ...],
    cval: float,
    omit: bool,
    ignore_inf: bool,
    min_count: int,
) -> FloatArray:
    """Average literal shifted windows with exact sums and one final rounding."""
    output = np.empty(array.shape, dtype=array.dtype)
    constant = float(array.dtype.type(cval))
    for center in np.ndindex(array.shape):
        values = []
        for tap in np.ndindex(size):
            mapped = [
                _map_index(c + k - n // 2 - shift, length, mode)
                for c, k, n, shift, length, mode in zip(
                    center, tap, size, origin, array.shape, modes, strict=True
                )
            ]
            value = constant if None in mapped else float(array[tuple(mapped)])
            if omit and (math.isnan(value) or (ignore_inf and math.isinf(value))):
                continue
            values.append(value)
        if (
            not values
            or (omit and len(values) < min_count)
            or any(math.isnan(value) for value in values)
            or (math.inf in values and -math.inf in values)
        ):
            output[center] = np.nan
        elif math.inf in values:
            output[center] = math.inf
        elif -math.inf in values:
            output[center] = -math.inf
        else:
            output[center] = _round_exact(
                sum(map(Fraction, values), Fraction()) / len(values), array.dtype
            )
    return output


def _check(
    array: FloatArray,
    size: tuple[int, ...],
    origin: tuple[int, ...],
    modes: tuple[str, ...],
    *,
    cval: float = 0.0,
    omit: bool = False,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> FloatArray:
    """Check public results bitwise, including NaNs, zeros, and source ownership."""
    original = array.copy()
    expected = _reference(array, size, origin, modes, cval, omit, ignore_inf, min_count)
    function = nf.nanmean_filter if omit else nf.mean_filter
    options = {"ignore_inf": ignore_inf, "min_count": min_count} if omit else {}
    actual = function(
        array, size, origin=origin, mode=modes, cval=cval, method="boxcar", **options
    )
    assert actual.shape == array.shape and actual.dtype == array.dtype
    assert actual.flags.c_contiguous and actual.flags.owndata
    assert actual.flags.writeable and not np.shares_memory(actual, array)
    bits = np.uint32 if array.dtype == np.dtype(np.float32) else np.uint64
    np.testing.assert_array_equal(actual.view(bits), expected.view(bits))
    np.testing.assert_array_equal(array.view(bits), original.view(bits))
    return actual


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("odd_lower", [False, True])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_rounds_once_at_and_beside_midpoints(
    dtype: type[np.floating], odd_lower: bool, direction: int
) -> None:
    lower = dtype(1)
    if odd_lower:
        lower = np.nextafter(lower, dtype(np.inf))
    upper = np.nextafter(lower, dtype(np.inf))
    tiny = np.nextafter(dtype(0), dtype(1))
    # Mean = (lower + upper) / 2 + direction * tiny / 4.
    array = np.array([2 * lower, 2 * upper, direction * tiny, 0], dtype=dtype)
    expected = upper if direction > 0 or (direction == 0 and odd_lower) else lower
    actual = _check(array, (4,), (0,), ("wrap",))
    np.testing.assert_array_equal(actual, expected)
    if dtype is np.float32 and direction > 0 and not odd_lower:
        rational = sum(map(Fraction, map(float, array)), Fraction()) / 4
        assert dtype(float(rational)) == lower  # Double rounding loses the tiny term.


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size", [3, 4])
def test_extreme_values_leave_without_erasing_tiny_neighbors(
    dtype: type[np.floating], size: int
) -> None:
    maximum = np.finfo(dtype).max
    tiny = np.nextafter(dtype(0), dtype(1))
    array = np.array(
        [maximum, maximum, -maximum, tiny, tiny, tiny, -tiny, -tiny, -tiny, 0, 1, -1],
        dtype=dtype,
    )
    _check(array, (size,), (0,), ("constant",))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("shape", "size"),
    [
        ((5,), (4,)),
        ((3, 4), (4, 3)),
        ((2, 3, 4), (3, 4, 2)),
        ((2, 1, 3, 2), (2, 3, 2, 3)),
    ],
)
def test_nd_layout_boundaries_and_extreme_origins(
    dtype: type[np.floating], mode: str, shape: tuple[int, ...], size: tuple[int, ...]
) -> None:
    positions = np.arange(math.prod(shape)).reshape(shape)
    array = np.ldexp((positions % 23 - 11).astype(dtype), positions % 3 * 20 - 20)
    for origin in (
        tuple(-(n // 2) for n in size),
        (0,) * len(size),
        tuple((n - 1) // 2 for n in size),
    ):
        _check(array, size, origin, (mode,) * len(size), cval=-1.25)
    # Axis-dependent boundaries also exercise constant padding after earlier passes.
    mixed = tuple(
        MODES[(MODES.index(mode) + axis) % len(MODES)] for axis in range(array.ndim)
    )
    _check(array, size, (0,) * array.ndim, mixed, cval=0.1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cval", [1.25, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    ("omit", "ignore_inf", "min_count"),
    [(False, False, 1), (True, False, 1), (True, True, 3), (True, False, 12)],
)
def test_mixed_boundaries_omission_counts_and_noncanonical_nans(
    dtype: type[np.floating], cval: float, omit: bool, ignore_inf: bool, min_count: int
) -> None:
    bits = np.uint32 if dtype is np.float32 else np.uint64
    payload = 0xFFC01234 if dtype is np.float32 else 0xFFF8000000001234
    noncanonical = np.array([payload], dtype=bits).view(dtype)[0]
    array = np.array(
        [
            [noncanonical, np.inf, -np.inf, 2],
            [-0.0, 0.0, np.finfo(dtype).max, np.nextafter(dtype(0), dtype(1))],
        ],
        dtype=dtype,
    )
    for modes in (("constant", "mirror"), ("wrap", "constant")):
        _check(
            array,
            (3, 4),
            (1, -2),
            modes,
            cval=cval,
            omit=omit,
            ignore_inf=ignore_inf,
            min_count=min_count,
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_support_and_signed_zero(dtype: type[np.floating]) -> None:
    for invalid in (np.nan, np.inf, -np.inf):
        _check(
            np.full((2, 3), invalid, dtype=dtype),
            (3, 4),
            (0, 0),
            ("reflect", "wrap"),
            omit=True,
            ignore_inf=True,
        )
    _check(np.array([-0.0, 0.0, -0.0], dtype=dtype), (2,), (0,), ("nearest",))
    tiny = np.nextafter(dtype(0), dtype(1))
    # Negative half a subnormal rounds to negative zero; exact zero is positive.
    _check(np.array([-tiny, 0, tiny, 0], dtype=dtype), (2,), (0,), ("wrap",))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(0,), (0, 2), (2, 0, 3)])
def test_empty_axes(dtype: type[np.floating], shape: tuple[int, ...]) -> None:
    _check(
        np.empty(shape, dtype=dtype),
        (2,) * len(shape),
        (0,) * len(shape),
        ("constant",) * len(shape),
        omit=True,
        min_count=2,
    )


@pytest.mark.parametrize("shape", [(2, 3), (0, 3)])
@pytest.mark.parametrize("min_count", [0, 7])
def test_invalid_min_count_is_rejected_even_for_empty_arrays(
    shape: tuple[int, ...], min_count: int
) -> None:
    with pytest.raises(ValueError):
        nf.nanmean_filter(
            np.ones(shape),
            (2, 3),
            mode=("reflect", "wrap"),
            min_count=min_count,
            method="boxcar",
        )


@pytest.mark.parametrize("function", [nf.mean_filter, nf.nanmean_filter])
@pytest.mark.parametrize("shape", [(0, 3), (4, 3)])
@pytest.mark.parametrize("method", [None, True, 0, [], "invalid", "sortnet"])
def test_invalid_method_is_rejected_on_empty_and_nonempty_input(
    function: Callable, shape: tuple[int, ...], method: object
) -> None:
    with pytest.raises(ValueError, match="method"):
        function(np.ones(shape), method=method)


@pytest.mark.parametrize("function", [nf.mean_filter, nf.nanmean_filter])
@pytest.mark.parametrize("shape", [(0, 3), (4, 3)])
def test_boxcar_rejects_footprints(function: Callable, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="footprint"):
        function(np.ones(shape), footprint=np.ones((2, 3)), method="boxcar")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("omit", [False, True])
def test_boxcar_strided_inputs_and_default_direct_ownership(
    dtype: type[np.floating], omit: bool
) -> None:
    source = np.arange(48, dtype=dtype).reshape(6, 8) / 8
    source[2, :4] = [np.nan, np.inf, -0.0, -np.inf]
    for array in (source.T, source[::-1, ::2], np.asfortranarray(source)):
        array.flags.writeable = False
        _check(array, (2, 3), (-1, 1), ("mirror", "constant"), omit=omit)
        function = nf.nanmean_filter if omit else nf.mean_filter
        options = dict(size=(2, 3), origin=(-1, 1), mode=("mirror", "constant"))
        direct = function(array, method="direct", **options)
        default = function(array, **options)
        bits = f"u{array.dtype.itemsize}"
        np.testing.assert_array_equal(default.view(bits), direct.view(bits))


@pytest.mark.parametrize("function", [nf.gaussian_filter, nf.nangaussian_filter])
@pytest.mark.parametrize("shape", [(0, 3), (4, 3)])
@pytest.mark.parametrize("method", ["sortnet", "boxcar"])
def test_gaussian_rejects_unrelated_algorithms(
    function: Callable, shape: tuple[int, ...], method: str
) -> None:
    with pytest.raises(ValueError, match="method"):
        function(np.ones(shape), 1.0, method=method)
