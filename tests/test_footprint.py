"""Footprint and origin behavior of the window filters, checked against SciPy."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from typing import Any

import ndfiltrs as nf
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import ndimage

MODES = ("reflect", "constant", "nearest", "mirror", "wrap")
DTYPES = (np.float32, np.float64)
CVAL = 1.5

# Array shape and footprint; "odd"/"even" refers to the footprint axis lengths.
FOOTPRINTS: dict[str, tuple[tuple[int, ...], list[Any]]] = {
    "1d-odd": ((23,), [1, 0, 1, 1, 1]),
    "1d-even": ((23,), [1, 1, 0, 1]),
    "2d-odd": ((9, 11), [[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    "2d-even": ((9, 11), [[1, 0, 1, 1], [0, 1, 1, 0]]),
    "3d-odd": (
        (5, 6, 7),
        [
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 0], [1, 1, 1], [0, 1, 0]],
        ],
    ),
    "3d-even": ((5, 6, 7), [[[1, 0], [1, 1], [0, 1]], [[0, 1], [1, 0], [1, 1]]]),
}
ORIGINS = ("zero", "low", "high", "mixed")


def _origin(shape: tuple[int, ...], kind: str) -> tuple[int, ...]:
    """Return a valid SciPy origin, including the extreme negative/positive shifts."""
    low = tuple(-(length // 2) for length in shape)
    high = tuple((length - 1) // 2 for length in shape)
    if kind == "zero":
        return (0,) * len(shape)
    if kind == "low":
        return low
    if kind == "high":
        return high
    return tuple(low[axis] if axis % 2 else high[axis] for axis in range(len(shape)))


def _case(
    name: str, origin_kind: str, dtype: type[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.bool_], tuple[int, ...]]:
    shape, footprint = FOOTPRINTS[name]
    selected = np.asarray(footprint, dtype=bool)
    rng = np.random.default_rng(sum(map(ord, name + origin_kind)))
    values = rng.normal(size=shape).astype(dtype)
    return values, selected, _origin(selected.shape, origin_kind)


def _tolerance(dtype: type[np.floating]) -> dict[str, float]:
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    return {"rtol": rtol, "atol": rtol}


def _generic(
    values: NDArray[np.floating],
    function: Callable[[NDArray[np.float64]], float],
    footprint: NDArray[np.bool_],
    origin: tuple[int, ...],
    mode: str,
) -> NDArray[np.floating]:
    return ndimage.generic_filter(
        values, function, footprint=footprint, mode=mode, cval=CVAL, origin=origin
    )


def _mad(samples: NDArray[np.float64]) -> float:
    return float(np.median(np.abs(samples - np.median(samples))))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("origin_kind", ORIGINS)
@pytest.mark.parametrize("name", FOOTPRINTS)
def test_order_statistics_match_scipy_exactly(
    name: str, origin_kind: str, mode: str, dtype: type[np.floating]
) -> None:
    values, footprint, origin = _case(name, origin_kind, dtype)
    count = int(footprint.sum())
    window = {"footprint": footprint, "origin": origin, "mode": mode, "cval": CVAL}
    for actual, expected in (
        (nf.min_filter(values, **window), ndimage.minimum_filter(values, **window)),
        (nf.max_filter(values, **window), ndimage.maximum_filter(values, **window)),
    ):
        assert actual.dtype == dtype
        np.testing.assert_array_equal(actual, expected)

    # (ndfiltrs result, zero-based sorted position, SciPy result)
    ranked = [
        (
            nf.percentile_filter(values, 0, **window),
            0,
            ndimage.percentile_filter(values, 0, **window),
        ),
        (
            nf.percentile_filter(values, 100, **window),
            count - 1,
            ndimage.percentile_filter(values, 100, **window),
        ),
    ]
    for rank in (0, 1, count // 2, -1, -count):
        ranked.append(
            (
                nf.rank_filter(values, rank, **window),
                rank % count,
                ndimage.rank_filter(values, rank, **window),
            )
        )
    if count % 2:
        ranked.append(
            (
                nf.median_filter(values, **window),
                count // 2,
                ndimage.median_filter(values, **window),
            )
        )
        ranked.append(
            (
                nf.percentile_filter(values, 50, **window),
                count // 2,
                ndimage.percentile_filter(values, 50, **window),
            )
        )
    # SciPy 1.18.1 rank-type filters are wrong for 1-D footprints with holes:
    # rank_filter(x, 1, footprint=[1, 0, 1, 1, 1]) differs from sorting each
    # window. Sorting is the reference everywhere; SciPy where it is correct.
    scipy_is_correct = values.ndim > 1 or bool(footprint.all())
    for actual, position, scipy_result in ranked:
        assert actual.dtype == dtype
        expected = _generic(
            values, lambda s, k=position: np.sort(s)[k], footprint, origin, mode
        )
        np.testing.assert_array_equal(actual, expected)
        if scipy_is_correct:
            np.testing.assert_array_equal(actual, scipy_result)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("origin_kind", ORIGINS)
@pytest.mark.parametrize("name", FOOTPRINTS)
def test_moments_and_interpolated_statistics_match_generic_filter(
    name: str, origin_kind: str, mode: str, dtype: type[np.floating]
) -> None:
    values, footprint, origin = _case(name, origin_kind, dtype)
    window = {"footprint": footprint, "origin": origin, "mode": mode, "cval": CVAL}
    cases: list[tuple[NDArray[np.floating], Callable[[NDArray[np.float64]], float]]] = [
        (nf.mean_filter(values, **window), np.mean),
        (nf.var_filter(values, **window), np.var),
        (nf.var_filter(values, ddof=1, **window), lambda v: np.var(v, ddof=1)),
        (nf.std_filter(values, **window), np.std),
        (nf.std_filter(values, ddof=1, **window), lambda v: np.std(v, ddof=1)),
        (nf.mad_filter(values, **window), _mad),
        (nf.median_filter(values, **window), np.median),
        (
            nf.percentile_filter(values, 37.5, **window),
            lambda v: np.percentile(v, 37.5),
        ),
    ]
    for actual, function in cases:
        expected = _generic(values, function, footprint, origin, mode)
        np.testing.assert_allclose(actual, expected, **_tolerance(dtype))


NAN_FILTERS: list[
    tuple[str, dict[str, Any], Callable[[NDArray[np.float64]], float]]
] = [
    ("nanmedian_filter", {}, np.median),
    ("nanmean_filter", {}, np.mean),
    ("nanmin_filter", {}, np.min),
    ("nanmax_filter", {}, np.max),
    ("nanvar_filter", {"ddof": 1}, lambda v: np.var(v, ddof=1)),
    ("nanstd_filter", {}, np.std),
    ("nanpercentile_filter", {"percentile": 80.0}, lambda v: np.percentile(v, 80.0)),
    ("nanmad_filter", {}, _mad),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", FOOTPRINTS)
@pytest.mark.parametrize(("function_name", "extra", "reducer"), NAN_FILTERS)
def test_nan_filters_omit_nan_inside_footprint(
    function_name: str,
    extra: dict[str, Any],
    reducer: Callable[[NDArray[np.float64]], float],
    name: str,
    mode: str,
    dtype: type[np.floating],
) -> None:
    values, footprint, origin = _case(name, "mixed", dtype)
    values.ravel()[::3] = np.nan
    min_count = 2

    def reference(samples: NDArray[np.float64]) -> float:
        kept = samples[~np.isnan(samples)]
        if kept.size < min_count or kept.size <= extra.get("ddof", 0):
            return np.nan
        return reducer(kept)

    actual = getattr(nf, function_name)(
        values,
        footprint=footprint,
        origin=origin,
        mode=mode,
        cval=CVAL,
        min_count=min_count,
        **extra,
    )
    expected = _generic(values, reference, footprint, origin, mode)
    np.testing.assert_allclose(actual, expected, **_tolerance(dtype))


# Every window filter, called as f(array, footprint=..., origin=..., ...).
ALL_FILTERS: list[tuple[str, dict[str, Any]]] = [
    ("median_filter", {}),
    ("mean_filter", {}),
    ("min_filter", {}),
    ("max_filter", {}),
    ("var_filter", {"ddof": 1}),
    ("std_filter", {}),
    ("mad_filter", {"scale": 1.4826}),
    ("mode_filter", {"mode_estimator": "sex"}),
    ("biweight_location_filter", {"c": 4.0}),
    ("biweight_scale_filter", {"modify_sample_size": True}),
    ("nanmedian_filter", {"min_count": 2}),
    ("nanmean_filter", {"min_count": 2}),
    ("nanmin_filter", {"min_count": 2}),
    ("nanmax_filter", {"min_count": 2}),
    ("nanvar_filter", {"min_count": 2}),
    ("nanstd_filter", {"ddof": 1}),
    ("nanmad_filter", {"min_count": 2}),
    ("nanmode_filter", {"mode_estimator": "iraf", "min_count": 2}),
    ("nanbiweight_location_filter", {"min_count": 2}),
    ("nanbiweight_scale_filter", {"c": 6.0, "min_count": 2}),
    ("sigma_clipped_filter", {"sigma": 1.5, "min_count": 2}),
    ("sigma_clipped_filter", {"sigma": 1.0, "statistic": "mode"}),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", FOOTPRINTS)
@pytest.mark.parametrize(("function_name", "extra"), ALL_FILTERS)
def test_every_filter_reduces_exactly_the_footprint_samples(
    function_name: str, extra: dict[str, Any], name: str, mode: str
) -> None:
    # Oracle: SciPy gathers the footprint samples (row-major, like ndfiltrs);
    # the same filter over exactly that sample vector must give the same value.
    function = getattr(nf, function_name)
    values, footprint, origin = _case(name, "mixed", np.float64)
    if function_name.startswith(("nan", "sigma")):
        values.ravel()[::4] = np.nan

    def reduce_samples(samples: NDArray[np.float64]) -> float:
        # A size-n window centered in a length-n array covers it exactly.
        return function(samples, samples.size, **extra)[samples.size // 2]

    actual = function(
        values, footprint=footprint, origin=origin, mode=mode, cval=CVAL, **extra
    )
    expected = _generic(values, reduce_samples, footprint, origin, mode)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", DTYPES)
def test_even_count_median_is_the_midpoint_where_scipy_takes_the_upper(
    dtype: type[np.floating],
) -> None:
    values = np.array([0, 4, 1, 7, 2, 9, 3], dtype=dtype)
    # Taps at offsets -1 and 0: each output is the median of (x[i-1], x[i]).
    actual = nf.median_filter(values, footprint=[1, 1], mode="nearest")
    previous = np.concatenate([values[:1], values[:-1]])
    np.testing.assert_array_equal(actual, (previous + values) / 2)
    upper = ndimage.median_filter(values, footprint=[1, 1], mode="nearest")
    np.testing.assert_array_equal(upper, np.maximum(previous, values))
    np.testing.assert_array_equal(
        nf.rank_filter(values, 1, footprint=[1, 1], mode="nearest"), upper
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("size", "origin"), [(4, 0), (4, -2), (4, 1), ((2, 4), (-1, 1))]
)
def test_even_sizes_match_scipy(
    size: int | tuple[int, int],
    origin: int | tuple[int, int],
    mode: str,
    dtype: type[np.floating],
) -> None:
    values = np.random.default_rng(5).normal(size=(8, 9)).astype(dtype)
    window = {"size": size, "origin": origin, "mode": mode, "cval": CVAL}
    count = int(np.prod(np.broadcast_to(size, (2,))))
    np.testing.assert_array_equal(
        nf.min_filter(values, **window), ndimage.minimum_filter(values, **window)
    )
    np.testing.assert_array_equal(
        nf.max_filter(values, **window), ndimage.maximum_filter(values, **window)
    )
    np.testing.assert_array_equal(
        nf.rank_filter(values, 3, **window), ndimage.rank_filter(values, 3, **window)
    )
    lower = ndimage.rank_filter(values, count // 2 - 1, **window).astype(np.float64)
    upper = ndimage.rank_filter(values, count // 2, **window).astype(np.float64)
    np.testing.assert_allclose(
        nf.median_filter(values, **window),
        ((lower + upper) / 2).astype(dtype),
        **_tolerance(dtype),
    )


def test_numeric_footprint_selects_nonzero_entries() -> None:
    values = np.random.default_rng(8).normal(size=(7, 8))
    numeric = np.array([[0.0, 2.5, -1.0], [0.0, 1e-300, 0.0]])
    boolean = numeric != 0
    for function in (nf.median_filter, nf.mean_filter, nf.mad_filter, nf.mode_filter):
        np.testing.assert_array_equal(
            function(values, footprint=numeric, origin=(0, -1)),
            function(values, footprint=boolean, origin=(0, -1)),
        )


@pytest.mark.parametrize(
    "function",
    [nf.median_filter, nf.mean_filter, nf.max_filter, nf.mad_filter, nf.mode_filter],
)
def test_full_footprint_equals_size(
    function: Callable[..., NDArray[np.floating]],
) -> None:
    values = np.random.default_rng(9).normal(size=(6, 7)).astype(np.float32)
    np.testing.assert_array_equal(
        function(values, footprint=np.ones((3, 2)), origin=(1, -1), mode="wrap"),
        function(values, (3, 2), origin=(1, -1), mode="wrap"),
    )


VALIDATED = [
    nf.median_filter,
    nf.nanmean_filter,
    nf.min_filter,
    nf.nanstd_filter,
    partial(nf.percentile_filter, percentile=30.0),
    nf.mad_filter,
    nf.sigma_clipped_filter,
    nf.mode_filter,
    nf.nanbiweight_scale_filter,
]


@pytest.mark.parametrize("function", VALIDATED)
@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"size": 3, "footprint": np.ones(3)}, ValueError),
        ({"footprint": np.zeros(3)}, ValueError),
        ({"footprint": np.ones((3, 3))}, ValueError),
        ({"footprint": np.ones(0)}, ValueError),
        ({"footprint": np.ma.masked_array(np.ones(3))}, TypeError),
        ({"size": 3, "origin": 2}, ValueError),
        ({"size": 3, "origin": -2}, ValueError),
        ({"footprint": np.ones(4), "origin": 2}, ValueError),
        ({"size": 2, "origin": 1}, ValueError),
        ({"size": 3, "origin": True}, TypeError),
        ({"size": 3, "origin": (np.True_,)}, TypeError),
        ({"size": 3, "origin": 0.0}, TypeError),
        ({"size": 3, "origin": (0, 0)}, ValueError),
    ],
)
def test_invalid_windows_are_rejected(
    function: Callable[..., Any], kwargs: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        function(np.arange(6.0), **kwargs)


@pytest.mark.parametrize("origin", [-2, 1])
def test_extreme_origins_of_even_footprint_are_accepted(origin: int) -> None:
    values = np.arange(10.0)
    expected = ndimage.maximum_filter(values, footprint=np.ones(4), origin=origin)
    np.testing.assert_array_equal(
        nf.max_filter(values, footprint=np.ones(4), origin=origin), expected
    )


@pytest.mark.parametrize(
    "function_name",
    [
        "nanmedian_filter",
        "nanmean_filter",
        "nanmin_filter",
        "nanvar_filter",
        "nanpercentile_filter",
        "nanmad_filter",
        "nanmode_filter",
        "nanbiweight_location_filter",
        "sigma_clipped_filter",
    ],
)
def test_min_count_is_bounded_by_the_footprint_count(function_name: str) -> None:
    function = getattr(nf, function_name)
    extra = {"percentile": 50.0} if function_name == "nanpercentile_filter" else {}
    values = np.arange(12.0).reshape(3, 4)
    footprint = np.array([[1, 0, 1], [0, 1, 0]])  # three samples in a 2x3 box
    output = function(values, footprint=footprint, min_count=3, **extra)
    assert np.isfinite(output).all()
    with pytest.raises(ValueError, match="min_count"):
        function(values, footprint=footprint, min_count=4, **extra)


def test_rank_is_bounded_by_the_footprint_count() -> None:
    values = np.arange(12.0).reshape(3, 4)
    footprint = np.array([[1, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(
        nf.rank_filter(values, -3, footprint=footprint),
        nf.min_filter(values, footprint=footprint),
    )
    for rank in (3, -4):
        with pytest.raises(ValueError, match="rank"):
            nf.rank_filter(values, rank, footprint=footprint)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("origin_kind", ORIGINS)
@pytest.mark.parametrize("name", ["2d-odd", "2d-even", "3d-even"])
def test_fixpix_median_footprint_uses_unmasked_samples(
    name: str, origin_kind: str, mode: str, dtype: type[np.floating]
) -> None:
    values, footprint, origin = _case(name, origin_kind, dtype)
    mask = np.random.default_rng(11).random(values.shape) < 0.3
    values.ravel()[5] = np.nan
    window = {"footprint": footprint, "origin": origin, "mode": mode}
    actual = nf.fixpix(values, mask, method="median", **window)
    masked = np.where(mask, np.nan, values).astype(dtype)
    expected = np.where(
        mask, nf.nanmedian_filter(masked, cval=np.nan, **window), values
    )
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="either size or footprint"):
        nf.fixpix(values, mask, method="median", size=3, footprint=footprint)


@pytest.mark.parametrize(
    "function_name",
    [
        "median_filter",
        "nanmean_filter",
        "max_filter",
        "nanvar_filter",
        "percentile_filter",
        "nanmad_filter",
        "mode_filter",
        "nanbiweight_location_filter",
        "biweight_scale_filter",
        "sigma_clipped_filter",
    ],
)
def test_footprint_output_is_bitwise_identical_across_thread_counts(
    function_name: str,
) -> None:
    program = """
import hashlib
import sys
import numpy as np
import ndfiltrs

rng = np.random.default_rng(71)
values = rng.normal(size=(29, 31))
values[::5, ::4] = np.nan
footprint = rng.random((4, 5)) < 0.6
footprint[0, 0] = True
function = getattr(ndfiltrs, sys.argv[1])
args = (values, 40.0) if sys.argv[1] == "percentile_filter" else (values,)
output = function(*args, footprint=footprint, origin=(-2, 1), mode=("mirror", "wrap"))
print(hashlib.sha256(output.tobytes()).hexdigest())
"""
    digests: set[str] = set()
    for threads in (1, 2, 4):
        environment = os.environ.copy()
        environment["RAYON_NUM_THREADS"] = str(threads)
        result = subprocess.run(
            [sys.executable, "-c", program, function_name],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1
