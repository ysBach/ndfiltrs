"""Exact network selection across invalid values, shifted taps, and row edges."""

from __future__ import annotations

import math
from collections.abc import Callable

import ndfiltrs as nf
import numpy as np
import pytest

from tests._reference import MODES, _map_index, _median, median_filter_reference


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("count", [13, 17, 25, 33, 49, 65])
@pytest.mark.parametrize("ignore_inf", [None, False, True])
def test_padding_transitions_with_changing_ranks_and_signed_zeros(
    dtype: type[np.floating], count: int, ignore_inf: bool | None
) -> None:
    values = np.random.default_rng(129).normal(size=5 * count).astype(dtype)
    bits = np.uint32 if dtype is np.float32 else np.uint64
    payload = 0xFFC01234 if dtype is np.float32 else 0xFFF8000000001234
    negative_nan = np.array([payload], dtype=bits).view(dtype)[0]
    values[count // 3 : 2 * count // 3] = negative_nan
    values[count + 2] = np.inf
    values[2 * count - 3] = -np.inf
    values[3 * count :] = np.resize(np.array([-0.0, 0.0], dtype=dtype), 2 * count)
    expected = median_filter_reference(
        values,
        (count,),
        ("reflect",),
        cval=0.0,
        omit_nan=ignore_inf is not None,
        ignore_inf=bool(ignore_inf),
        min_count=1,
    )
    if ignore_inf is None:
        actual = nf.median_filter(values, count, method="sortnet")
    else:
        actual = nf.nanmedian_filter(
            values, count, ignore_inf=ignore_inf, method="sortnet"
        )
    np.testing.assert_array_equal(actual.view(bits), expected.view(bits))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("policy", ["plain", "omit", "finite", "minimum"])
@pytest.mark.parametrize(
    ("shape", "size", "origin"),
    [
        ((17,), (4,), (-2,)),
        ((7, 9), (3, 4), (1, -2)),
        ((4, 5, 7), (2, 3, 2), (-1, 0, 0)),
    ],
)
def test_network_matches_literal_sorted_footprints(
    dtype: type[np.floating],
    mode: str,
    policy: str,
    shape: tuple[int, ...],
    size: tuple[int, ...],
    origin: tuple[int, ...],
) -> None:
    values = np.random.default_rng(712).normal(size=shape).astype(dtype)
    special = [
        np.nan,
        np.inf,
        -np.inf,
        -0.0,
        0.0,
        np.finfo(dtype).max,
        -np.finfo(dtype).max,
    ]
    values.flat[::7] = np.resize(
        np.asarray(special, dtype=dtype), values.flat[::7].size
    )
    original = values.copy()
    footprint = np.indices(size).sum(axis=0) % 3 != 0
    taps = np.argwhere(footprint)
    omit = policy != "plain"
    ignore_inf = policy == "finite"
    min_count = len(taps) if policy == "minimum" else 1
    cval = dtype(-0.0)
    expected = np.empty_like(values)
    for center in np.ndindex(shape):
        samples = []
        for tap in taps:
            mapped = [
                _map_index(c + int(k) - n // 2 - shift, length, mode)
                for c, k, n, shift, length in zip(
                    center, tap, size, origin, shape, strict=True
                )
            ]
            value = float(cval if None in mapped else values[tuple(mapped)])
            if (omit and math.isnan(value)) or (ignore_inf and math.isinf(value)):
                continue
            samples.append(value)
        expected[center] = (
            np.nan
            if len(samples) < min_count or any(map(math.isnan, samples))
            else _median(samples)
        )
    function = nf.nanmedian_filter if omit else nf.median_filter
    options = {"ignore_inf": ignore_inf, "min_count": min_count} if omit else {}
    actual = function(
        values,
        footprint=footprint,
        origin=origin,
        mode=mode,
        cval=float(cval),
        method="sortnet",
        **options,
    )
    bits = f"u{values.dtype.itemsize}"
    np.testing.assert_array_equal(actual.view(bits), expected.view(bits))
    np.testing.assert_array_equal(values.view(bits), original.view(bits))


@pytest.mark.parametrize("function", [nf.median_filter, nf.nanmedian_filter])
@pytest.mark.parametrize("shape", [(0, 3), (4, 3)])
@pytest.mark.parametrize("method", [None, True, 0, [], "invalid", "boxcar"])
def test_invalid_method_is_rejected_on_empty_and_nonempty_input(
    function: Callable, shape: tuple[int, ...], method: object
) -> None:
    with pytest.raises(ValueError, match="method"):
        function(np.ones(shape), method=method)


@pytest.mark.parametrize("function", [nf.median_filter, nf.nanmedian_filter])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sortnet_strided_inputs_preserve_source_and_direct_results(
    function: Callable, dtype: type[np.floating]
) -> None:
    source = np.arange(80, dtype=dtype).reshape(8, 10) - 40
    source[2, :5] = [np.nan, np.inf, -np.inf, -0.0, 0.0]
    for array in (source.T, source[::-1, ::2], np.asfortranarray(source)):
        array.flags.writeable = False
        original = array.copy()
        options = dict(size=(2, 3), origin=(-1, 1), mode=("mirror", "constant"))
        direct = function(array, method="direct", **options)
        actual = function(array, method="sortnet", **options)
        bits = f"u{array.dtype.itemsize}"
        np.testing.assert_array_equal(actual.view(bits), direct.view(bits))
        np.testing.assert_array_equal(
            function(array, **options).view(bits), direct.view(bits)
        )
        np.testing.assert_array_equal(array.view(bits), original.view(bits))
        assert actual.shape == array.shape and actual.dtype == array.dtype
        assert actual.flags.owndata and actual.flags.c_contiguous
        assert actual.flags.writeable and not np.shares_memory(actual, array)


@pytest.mark.parametrize("function", [nf.median_filter, nf.nanmedian_filter])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sortnet_empty_axes(function: Callable, dtype: type[np.floating]) -> None:
    array = np.empty((2, 0, 3), dtype=dtype)
    actual = function(array, size=(2, 3, 4), origin=(-1, 0, 1), method="sortnet")
    assert actual.shape == array.shape and actual.dtype == array.dtype
    assert actual.flags.owndata and actual.flags.writeable
    assert actual.flags.c_contiguous and not np.shares_memory(actual, array)
