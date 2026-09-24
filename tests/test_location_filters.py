"""Local mode, biweight, clipped-mode, and kernel-repair filters versus references.

The MMM and IRAF mode references re-implement astroimred's ``_mode_from_stats``;
the SEx reference follows SExtractor's ``backguess`` in ``src/back.c``
(population standard deviation, ``abs(mean - median) / std >= 0.3`` selects the
median, mean when the standard deviation is zero). The biweight references implement
``astropy.stats.biweight_location``/``biweight_scale`` about the median. Both were
checked once against astroimred and astropy 8.0.1 (``~/.venvs/ysbpy``) on 300
random heavy-tailed samples, c in {1.5, 4, 6, 9}, both ``modify_sample_size``
values: mode identical, biweight within 1e-12 relative.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import ndfiltrs as nf
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import ndimage

MODES = ("reflect", "constant", "nearest", "mirror", "wrap")
DTYPES = (np.float32, np.float64)
ESTIMATORS = ("sex", "mmm", "iraf")
NORMAL_UPPER_QUARTILE = 0.6744897501960817
Samples = NDArray[np.float64]

WINDOWS: dict[str, tuple[tuple[int, ...], dict[str, Any]]] = {
    "1d": ((31,), {"size": 5}),
    "2d": ((10, 11), {"size": 3}),
    "2d-even": ((10, 11), {"size": (2, 4), "origin": (-1, 1)}),
    "2d-footprint": (
        (10, 11),
        {
            "footprint": np.array([[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0]]),
            "origin": (1, -2),
        },
    ),
    "3d": ((5, 6, 7), {"size": (3, 1, 3), "origin": (0, 0, -1)}),
}


def _tolerance(dtype: type[np.floating]) -> dict[str, float]:
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    return {"rtol": rtol, "atol": rtol}


def mode_reference(samples: Samples, estimator: str) -> float:
    mean, median = np.mean(samples), np.median(samples)
    difference = median - mean
    if estimator == "mmm":
        return median + 2 * difference
    if estimator == "iraf":
        return mean if mean < median else median + 2 * difference
    std = np.std(samples)
    if std == 0:
        return mean
    if abs(difference) / std >= 0.3:
        return median
    return median + 1.5 * difference


def biweight_location_reference(samples: Samples, c: float) -> float:
    median = np.median(samples)
    deviation = samples - median
    mad = np.median(np.abs(deviation))
    if mad == 0:
        return median
    u = deviation / (c * mad)
    weight = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
    return median + np.sum(deviation * weight) / np.sum(weight)


def biweight_scale_reference(
    samples: Samples, c: float, modify_sample_size: bool
) -> float:
    median = np.median(samples)
    deviation = samples - median
    mad = np.median(np.abs(deviation))
    if mad == 0:
        return 0.0
    u2 = (deviation / (c * mad)) ** 2
    keep = u2 < 1
    count = np.count_nonzero(keep) if modify_sample_size else samples.size
    numerator = np.sum(np.where(keep, deviation**2 * (1 - u2) ** 4, 0.0))
    denominator = np.sum(np.where(keep, (1 - u2) * (1 - 5 * u2), 0.0))
    return np.sqrt(count * numerator) / abs(denominator)


# (filter name, filter keyword arguments, sample reducer)
STATISTICS: list[tuple[str, dict[str, Any], Callable[[Samples], float]]] = [
    *(
        (
            "mode_filter",
            {"mode_estimator": name},
            lambda s, e=name: mode_reference(s, e),
        )
        for name in ESTIMATORS
    ),
    *(
        (
            "biweight_location_filter",
            {"c": c},
            lambda s, c=c: biweight_location_reference(s, c),
        )
        for c in (1.5, 6.0)
    ),
    *(
        (
            "biweight_scale_filter",
            {"c": c, "modify_sample_size": modify},
            lambda s, c=c, m=modify: biweight_scale_reference(s, c, m),
        )
        for c, modify in ((3.0, True), (9.0, False))
    ),
]
STATISTIC_IDS = [f"{name}-{list(kwargs.values())}" for name, kwargs, _ in STATISTICS]


def _data(
    shape: tuple[int, ...], dtype: type[np.floating], seed: int
) -> NDArray[np.floating]:
    # Heavy tails give zero biweight weights and skewed mode windows.
    return np.random.default_rng(seed).standard_t(2, size=shape).astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize(("name", "kwargs", "reducer"), STATISTICS, ids=STATISTIC_IDS)
def test_plain_and_nan_filters_match_references(
    name: str,
    kwargs: dict[str, Any],
    reducer: Callable[[Samples], float],
    window: str,
    mode: str,
    dtype: type[np.floating],
) -> None:
    shape, geometry = WINDOWS[window]
    values = _data(shape, dtype, seed=len(window))
    call = {**geometry, "mode": mode, "cval": 0.25}
    actual = getattr(nf, name)(values, **call, **kwargs)
    expected = ndimage.generic_filter(values, reducer, **call)
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected, **_tolerance(dtype))
    # Without NaN, the omitting variant sees the same samples.
    np.testing.assert_array_equal(
        getattr(nf, "nan" + name)(values, **call, **kwargs), actual
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", ["constant", "reflect", "wrap"])
@pytest.mark.parametrize("window", ["1d", "2d-footprint", "3d"])
@pytest.mark.parametrize(("name", "kwargs", "reducer"), STATISTICS, ids=STATISTIC_IDS)
def test_nan_and_infinity_policies(
    name: str,
    kwargs: dict[str, Any],
    reducer: Callable[[Samples], float],
    window: str,
    mode: str,
    dtype: type[np.floating],
) -> None:
    shape, geometry = WINDOWS[window]
    values = _data(shape, dtype, seed=7)
    values.flat[::5] = np.nan
    values.flat[3::11] = np.inf
    values.flat[7::13] = -np.inf
    call = {**geometry, "mode": mode, "cval": np.nan}
    min_count = 3

    def plain(samples: Samples) -> float:
        return reducer(samples) if np.isfinite(samples).all() else np.nan

    def omitting(samples: Samples, ignore_inf: bool) -> float:
        kept = samples[~np.isnan(samples)]
        if ignore_inf:
            kept = kept[np.isfinite(kept)]
        if kept.size < min_count or not np.isfinite(kept).all():
            return np.nan
        return reducer(kept)

    np.testing.assert_allclose(
        getattr(nf, name)(values, **call, **kwargs),
        ndimage.generic_filter(values, plain, **call),
        **_tolerance(dtype),
    )
    for ignore_inf in (False, True):
        actual = getattr(nf, "nan" + name)(
            values, **call, **kwargs, ignore_inf=ignore_inf, min_count=min_count
        )
        expected = ndimage.generic_filter(
            values, omitting, extra_arguments=(ignore_inf,), **call
        )
        np.testing.assert_allclose(actual, expected, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("prefix", ["", "nan"])
def test_zero_mad_returns_median_location_and_zero_scale(
    prefix: str, dtype: type[np.floating]
) -> None:
    # Every size-5 window holds at most one outlier, so its MAD is zero.
    values = np.full(20, 2.5, dtype=dtype)
    values[[4, 12]] = [9.0, -40.0]
    location = getattr(nf, prefix + "biweight_location_filter")(values, 5)
    scale = getattr(nf, prefix + "biweight_scale_filter")(
        values, 5, modify_sample_size=True
    )
    np.testing.assert_array_equal(location, np.full(20, 2.5, dtype=dtype))
    np.testing.assert_array_equal(scale, np.zeros(20, dtype=dtype))


@pytest.mark.parametrize(
    ("samples", "estimator", "expected"),
    [
        # mean 2, median 1, population std sqrt(14 / 3): skew 0.46 selects the median.
        ([0.0, 1.0, 5.0], "sex", 1.0),
        ([0.0, 1.0, 5.0], "mmm", -1.0),
        ([0.0, 1.0, 5.0], "iraf", -1.0),
        # A low or high tail both select the median: |1 - 0| / sqrt(26 / 3) >= 0.3.
        ([-2.0, 0.0, 5.0], "sex", 0.0),
        # Skew 0.13 < 0.3 uses 2.5 median - 1.5 mean.
        ([1.0, 2.0, 3.0, 3.5], "sex", 2.6875),
        # A tight core with a low outlier stays inside the sample range.
        ([10.0] * 6 + [0.0, 0.0, -50.0], "sex", 10.0),
        # Mean below median: IRAF returns the mean.
        ([-9.0, 1.0, 2.0], "iraf", -2.0),
    ],
)
def test_mode_estimators_have_literal_results(
    samples: list[float], estimator: str, expected: float
) -> None:
    # A full-length wrapped window centered at len // 2 holds every sample once.
    actual = nf.mode_filter(
        np.array(samples), len(samples), mode="wrap", mode_estimator=estimator
    )
    assert actual[len(samples) // 2] == expected


def _clip_then_mode(
    samples: Samples,
    maxiters: int | None,
    spread: str,
    estimator: str,
    sigma: float = 1.5,
    min_count: int = 2,
) -> float:
    kept = samples[np.isfinite(samples)]
    iteration = 0
    while kept.size >= min_count and (maxiters is None or iteration < maxiters):
        center = np.median(kept)
        if spread == "std":
            width = np.std(kept)
        else:
            width = np.median(np.abs(kept - center)) / NORMAL_UPPER_QUARTILE
        inside = (kept >= center - sigma * width) & (kept <= center + sigma * width)
        iteration += 1
        if inside.all():
            break
        kept = kept[inside]
    if kept.size < min_count:
        return np.nan
    return mode_reference(kept, estimator)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("maxiters", [0, 1, None])
@pytest.mark.parametrize("spread", ["std", "mad"])
@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("window", ["2d", "2d-footprint"])
def test_sigma_clipped_mode_is_the_mode_of_retained_samples(
    window: str,
    estimator: str,
    spread: str,
    maxiters: int | None,
    dtype: type[np.floating],
) -> None:
    shape, geometry = WINDOWS[window]
    values = np.random.default_rng(4).normal(size=shape).astype(dtype)
    values.flat[::7] += 12.0
    values.flat[5::17] = np.nan
    values.flat[2::19] = -np.inf
    call = {**geometry, "mode": "mirror"}
    actual = nf.sigma_clipped_filter(
        values,
        **call,
        sigma=1.5,
        maxiters=maxiters,
        spread=spread,
        statistic="mode",
        mode_estimator=estimator,
        min_count=2,
    )
    expected = ndimage.generic_filter(
        values, _clip_then_mode, extra_arguments=(maxiters, spread, estimator), **call
    )
    np.testing.assert_allclose(actual, expected, **_tolerance(dtype))
    if maxiters == 0:
        unclipped = nf.nanmode_filter(
            values, **call, mode_estimator=estimator, ignore_inf=True, min_count=2
        )
        np.testing.assert_array_equal(actual, unclipped)


@pytest.mark.parametrize(
    ("function", "kwargs", "error"),
    [
        (nf.mode_filter, {"mode_estimator": "median"}, ValueError),
        (nf.mode_filter, {"mode_estimator": None}, ValueError),
        (nf.nanmode_filter, {"mode_estimator": "SEX"}, ValueError),
        (nf.nanmode_filter, {"ignore_inf": 1}, TypeError),
        (nf.nanmode_filter, {"min_count": 4}, ValueError),
        (nf.biweight_location_filter, {"c": 0.0}, ValueError),
        (nf.biweight_location_filter, {"c": -6.0}, ValueError),
        (nf.biweight_location_filter, {"c": np.inf}, ValueError),
        (nf.nanbiweight_location_filter, {"c": np.nan}, ValueError),
        (nf.nanbiweight_location_filter, {"c": True}, TypeError),
        (nf.biweight_scale_filter, {"c": "9"}, TypeError),
        (nf.biweight_scale_filter, {"modify_sample_size": 1}, TypeError),
        (nf.nanbiweight_scale_filter, {"modify_sample_size": None}, TypeError),
        (nf.nanbiweight_scale_filter, {"c": -1.0}, ValueError),
        (nf.nanbiweight_scale_filter, {"min_count": 0}, ValueError),
        (
            nf.sigma_clipped_filter,
            {"statistic": "mode", "mode_estimator": "mean"},
            ValueError,
        ),
        (nf.sigma_clipped_filter, {"statistic": "modes"}, ValueError),
    ],
)
def test_invalid_estimator_options_are_rejected(
    function: Callable[..., Any], kwargs: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        function(np.arange(8.0), **kwargs)


@pytest.mark.parametrize(
    "function",
    [nf.nanmode_filter, nf.nanbiweight_location_filter, nf.nanbiweight_scale_filter],
)
def test_nan_estimators_reject_none_ignore_inf(function: Callable[..., Any]) -> None:
    with pytest.raises(TypeError, match="ignore_inf"):
        function(np.array([1.0, np.nan, 2.0, 3.0]), ignore_inf=None)


KERNELS = {
    "odd": np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]),
    "even": np.array([[0.5, 1.0, 0.0], [1.0, 3.0, 2.0]]),
}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("origin", [0, (-1, 1)])
@pytest.mark.parametrize("kernel", KERNELS)
def test_fixpix_mean_is_nancorrelate_of_masked_data(
    kernel: str, origin: int | tuple[int, int], mode: str, dtype: type[np.floating]
) -> None:
    rng = np.random.default_rng(12)
    values = rng.normal(size=(9, 10)).astype(dtype)
    values[0, 0] = np.nan
    values[1, 1] = -0.0
    mask = rng.random(values.shape) < 0.25
    mask[:2, :2] = False
    mask[5, :] = True  # a fully masked row
    original = values.copy()
    weights = KERNELS[kernel]
    actual = nf.fixpix(
        values, mask, method="mean", kernel=weights, mode=mode, origin=origin
    )
    masked = np.where(mask, np.nan, values).astype(dtype)
    repaired = nf.nancorrelate(masked, weights, mode=mode, cval=np.nan, origin=origin)
    np.testing.assert_array_equal(actual, np.where(mask, repaired, values))
    assert actual.dtype == dtype
    assert actual.flags.c_contiguous
    # Unmasked pixels, including NaN and -0.0, are bitwise unchanged.
    np.testing.assert_array_equal(
        actual[~mask].view(np.uint8), values[~mask].view(np.uint8)
    )
    np.testing.assert_array_equal(values.view(np.uint8), original.view(np.uint8))


def test_fixpix_mean_forwards_boundary_and_omission_options() -> None:
    values = np.random.default_rng(2).normal(size=(7, 8))
    values[3, 3] = np.inf
    mask = np.zeros(values.shape, dtype=bool)
    mask[[0, 2, 3, 6], [0, 3, 4, 7]] = True
    options = {"mode": "constant", "cval": 0.5, "ignore_inf": True, "min_count": 6}
    weights = KERNELS["odd"]
    actual = nf.fixpix(values, mask, method="mean", kernel=weights, **options)
    expected = nf.nancorrelate(np.where(mask, np.nan, values), weights, **options)
    np.testing.assert_array_equal(actual, np.where(mask, expected, values))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"mode": "constant", "cval": 0.25, "ignore_inf": True, "min_count": 5},
        {"mode": ("wrap", "mirror"), "origin": (1, -2)},
    ],
)
def test_fixpix_mean_sparse_mask_matches_dense_nancorrelate_bitwise(
    options: dict[str, Any], dtype: type[np.floating]
) -> None:
    rng = np.random.default_rng(31)
    values = rng.normal(size=(301, 257)).astype(dtype)
    values[rng.random(values.shape) < 0.001] = np.inf
    values[rng.random(values.shape) < 0.001] = np.nan
    mask = rng.random(values.shape) < 0.001
    mask[100:106, 50:58] = True  # a cluster whose members share windows
    mask[0, :] = mask[:, -1] = True  # edges exercise every boundary mode
    weights = nf.kernels.gaussian(1.5, ndim=2)
    actual = nf.fixpix(values, mask, kernel=weights, **options)
    dense = nf.nancorrelate(
        np.where(mask, np.nan, values).astype(dtype),
        weights,
        **{"cval": np.nan, **options},
    )
    expected = np.where(mask, dense, values)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


@pytest.mark.parametrize(
    "window",
    [{}, {"size": 5}, {"size": (3, 4), "origin": (0, -1)}, {"footprint": np.eye(3)}],
)
def test_fixpix_mean_without_kernel_is_the_local_nanmean(
    window: dict[str, Any],
) -> None:
    values = np.random.default_rng(5).normal(size=(8, 9))
    mask = np.random.default_rng(6).random(values.shape) < 0.3
    actual = nf.fixpix(values, mask, method="mean", mode="nearest", **window)
    expected = nf.nanmean_filter(
        np.where(mask, np.nan, values), mode="nearest", **window
    )
    np.testing.assert_allclose(actual, np.where(mask, expected, values), rtol=1e-13)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "mean", "kernel": KERNELS["odd"], "size": 3}, "at most one"),
        (
            {
                "method": "mean",
                "kernel": KERNELS["odd"],
                "footprint": np.ones((3, 3)),
            },
            "at most one",
        ),
        ({"method": "mean", "size": 3, "footprint": np.ones((3, 3))}, "not both"),
        (
            {"method": "mean", "kernel": KERNELS["odd"], "priority": (0, 1)},
            "priority",
        ),
        ({"method": "median", "kernel": KERNELS["odd"]}, "kernel"),
        ({"method": "linear", "kernel": KERNELS["odd"]}, "kernel"),
        ({"method": "linear", "origin": 0}, "origin"),
        ({"method": "mean", "kernel": -KERNELS["odd"]}, None),
        ({"method": "mean", "kernel": np.ones(3)}, None),
    ],
)
@pytest.mark.parametrize("has_bad_pixels", [False, True])
def test_fixpix_option_misuse_is_rejected_even_without_bad_pixels(
    kwargs: dict[str, Any], match: str | None, has_bad_pixels: bool
) -> None:
    values = np.arange(20.0).reshape(4, 5)
    mask = np.zeros(values.shape, dtype=bool)
    mask[2, 2] = has_bad_pixels
    with pytest.raises(ValueError, match=match):
        nf.fixpix(values, mask, **kwargs)


def test_fixpix_kernel_selects_mean_repair_by_default() -> None:
    values = np.random.default_rng(8).normal(size=(6, 7))
    mask = np.zeros(values.shape, dtype=bool)
    mask[2, 3] = mask[4, 1] = True
    weights = KERNELS["odd"]
    np.testing.assert_array_equal(
        nf.fixpix(values, mask, kernel=weights),
        nf.fixpix(values, mask, method="mean", kernel=weights),
    )
    np.testing.assert_array_equal(
        nf.fixpix(values, mask), nf.fixpix(values, mask, method="linear")
    )


@pytest.mark.parametrize("mask", [None, np.zeros((4, 5), dtype=bool)])
def test_fixpix_mean_without_bad_pixels_returns_a_copy(
    mask: NDArray[np.bool_] | None,
) -> None:
    values = np.asfortranarray(np.arange(20.0).reshape(4, 5))
    values[1, 1] = np.nan
    actual = nf.fixpix(values, mask, method="mean", kernel=KERNELS["even"])
    assert not np.shares_memory(actual, values)
    assert actual.flags.c_contiguous
    assert actual.flags.writeable
    np.testing.assert_array_equal(
        actual.view(np.uint8), np.ascontiguousarray(values).view(np.uint8)
    )


def test_pearson_factor_sets_the_mode_formula() -> None:
    # mean 2.375, median 2.5, skew 0.13 < 0.3: median + (pearson - 1) * 0.125.
    samples = np.array([1.0, 2.0, 3.0, 3.5])
    center = len(samples) // 2

    def mode(**options: Any) -> float:
        return float(
            nf.mode_filter(samples, len(samples), mode="wrap", **options)[center]
        )

    assert mode() == 2.6875
    assert mode(pearson=3.5) == 2.8125
    assert mode(mode_estimator="mmm") == 2.75
    assert mode(mode_estimator="mmm", pearson=1) == 2.5
    clipped = nf.sigma_clipped_filter(
        samples, 4, mode="wrap", maxiters=0, statistic="mode", pearson=3.5
    )
    assert clipped[center] == 2.8125


@pytest.mark.parametrize("pearson", [0.5, np.nan, True, "3"])
def test_pearson_factor_is_validated(pearson: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        nf.nanmode_filter(np.arange(5.0), pearson=pearson)
    with pytest.raises((TypeError, ValueError)):
        nf.sigma_clipped_filter(np.arange(5.0), statistic="mode", pearson=pearson)
