"""Histogram median and rank results equal direct selection bit for bit."""

from __future__ import annotations

import ndfiltrs as nf
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._reference import MODES

POLICIES = {
    "plain": None,
    "omit": {"ignore_inf": False, "min_count": 1},
    "finite": {"ignore_inf": True, "min_count": 1},
    "all": {"ignore_inf": False, "min_count": -1},
}

GEOMETRIES = [
    ((150,), (1,), (0,)),
    ((150,), (8,), (-4,)),
    ((150,), (33,), (16,)),
    ((70,), (101,), (0,)),
    ((70, 66), (3, 3), (0, 0)),
    ((70, 66), (6, 5), (-3, 2)),
    ((70, 66), (2, 41), (0, -20)),
    ((1, 90), (4, 7), (1, 0)),
    ((9, 40, 37), (3, 2, 5), (-1, 0, 2)),
    ((9, 40, 37), (12, 1, 4), (5, 0, -2)),
]


def _sample(shape: tuple[int, ...], dtype: type[np.floating], seed: int) -> np.ndarray:
    """Return values with ties, signed zeros, infinities, and NaN payloads."""
    rng = np.random.default_rng(seed)
    values = (rng.integers(-4, 5, size=shape) / 4).astype(dtype)
    bits = np.uint32 if dtype is np.float32 else np.uint64
    payload = 0xFFC01234 if dtype is np.float32 else 0xFFF8000000001234
    special = np.array(
        [np.nan, np.inf, -np.inf, -0.0, 0.0, np.finfo(dtype).max], dtype=dtype
    )
    special = np.append(special, np.array([payload], dtype=bits).view(dtype))
    chosen = rng.random(shape) < 0.15
    values[chosen] = rng.choice(special, size=int(chosen.sum()))
    return values


def _bits(array: np.ndarray) -> np.ndarray:
    return array.view(np.uint32 if array.dtype == np.float32 else np.uint64)


def _median(
    values: np.ndarray, method: str, policy: str, **kwargs: object
) -> np.ndarray:
    options = POLICIES[policy]
    if options is None:
        return nf.median_filter(values, method=method, **kwargs)
    min_count = options["min_count"]
    if min_count == -1:
        min_count = int(np.prod(kwargs["size"]))
    return nf.nanmedian_filter(
        values,
        method=method,
        ignore_inf=options["ignore_inf"],
        min_count=min_count,
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("policy", list(POLICIES))
@pytest.mark.parametrize(("shape", "size", "origin"), GEOMETRIES)
def test_histogram_median_matches_direct(
    dtype: type[np.floating],
    mode: str,
    policy: str,
    shape: tuple[int, ...],
    size: tuple[int, ...],
    origin: tuple[int, ...],
) -> None:
    values = _sample(shape, dtype, seed=len(shape) * 100 + size[0])
    kwargs = {"size": size, "origin": origin, "mode": mode, "cval": 0.5}
    expected = _median(values, "direct", policy, **kwargs)
    actual = _median(values, "histogram", policy, **kwargs)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


@pytest.mark.parametrize("cval", [np.nan, np.inf, -np.inf, -0.0, 0.0, 3.0])
@pytest.mark.parametrize("policy", list(POLICIES))
def test_constant_values_follow_the_sample_policy(cval: float, policy: str) -> None:
    values = _sample((40, 45), np.float64, seed=7)
    kwargs = {
        "size": (5, 4),
        "origin": (2, -2),
        "mode": ("constant", "wrap"),
        "cval": cval,
    }
    expected = _median(values, "direct", policy, **kwargs)
    actual = _median(values, "histogram", policy, **kwargs)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("shape", "size", "origin"), GEOMETRIES)
def test_histogram_rank_matches_direct(
    dtype: type[np.floating],
    mode: str,
    shape: tuple[int, ...],
    size: tuple[int, ...],
    origin: tuple[int, ...],
) -> None:
    values = _sample(shape, dtype, seed=31)
    count = int(np.prod(size))
    for rank in sorted({0, count // 3, -1}):
        kwargs = {"size": size, "origin": origin, "mode": mode, "cval": -0.0}
        expected = nf.rank_filter(values, rank, **kwargs)
        actual = nf.rank_filter(values, rank, method="histogram", **kwargs)
        np.testing.assert_array_equal(_bits(actual), _bits(expected))


@settings(max_examples=150, deadline=None)
@given(
    data=st.data(),
    rank=st.integers(min_value=1, max_value=3),
    dtype=st.sampled_from([np.float32, np.float64]),
    omit_nan=st.booleans(),
    ignore_inf=st.booleans(),
)
def test_generated_geometries_match_direct(
    data: st.DataObject,
    rank: int,
    dtype: type[np.floating],
    omit_nan: bool,
    ignore_inf: bool,
) -> None:
    longest = {1: 120, 2: 50, 3: 12}[rank]
    shape = data.draw(
        st.tuples(*(st.integers(min_value=1, max_value=longest) for _ in range(rank)))
    )
    size = data.draw(st.tuples(*(st.integers(1, 9) for _ in range(rank))))
    origin = tuple(
        data.draw(st.integers(-(length // 2), (length - 1) // 2)) for length in size
    )
    mode = data.draw(st.tuples(*(st.sampled_from(MODES) for _ in range(rank))))
    cval = data.draw(st.sampled_from([-np.inf, -0.0, 0.0, 1.0, np.inf, np.nan]))
    values = _sample(shape, dtype, seed=data.draw(st.integers(0, 2**32 - 1)))
    kwargs = {"size": size, "origin": origin, "mode": mode, "cval": cval}
    if omit_nan:
        min_count = data.draw(st.integers(1, int(np.prod(size))))
        kwargs |= {"ignore_inf": ignore_inf, "min_count": min_count}
        call = nf.nanmedian_filter
    else:
        call = nf.median_filter
    expected = call(values, **kwargs)
    actual = call(values, method="histogram", **kwargs)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


def test_large_window_matches_direct_across_blocks() -> None:
    values = _sample((130, 140), np.float32, seed=3)
    expected = nf.nanmedian_filter(values, (45, 61), mode="mirror")
    actual = nf.nanmedian_filter(values, (45, 61), mode="mirror", method="histogram")
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


@pytest.mark.parametrize(
    "call",
    [
        lambda a, f: nf.median_filter(a, footprint=f, method="histogram"),
        lambda a, f: nf.nanmedian_filter(a, footprint=f, method="histogram"),
        lambda a, f: nf.rank_filter(a, 0, footprint=f, method="histogram"),
    ],
)
def test_histogram_methods_reject_footprints(call) -> None:
    with pytest.raises(ValueError, match="requires size, not footprint"):
        call(np.zeros((5, 5)), np.ones((3, 3), dtype=bool))


def test_rank_filter_rejects_unknown_methods() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        nf.rank_filter(np.zeros(5), 0, size=3, method="sortnet")
