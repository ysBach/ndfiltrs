"""Variance propagation and matched filtering against exact and Monte Carlo checks."""

from __future__ import annotations

import ndfiltrs as nf
import numpy as np
import pytest
from scipy import stats

from tests._noise_reference import noise_reference
from tests._reference import MODES

DTYPES = (np.float32, np.float64)
SHAPES = ((7,), (4, 5), (3, 2, 4))


def _inputs(shape: tuple[int, ...], dtype: type[np.floating]) -> tuple:
    """Signed data with NaN and infinity, and noise with every special class."""
    count = int(np.prod(shape))
    data = (((np.arange(count) * 7) % 19 - 9) / 4).astype(dtype).reshape(shape)
    noise = (((np.arange(count) * 5) % 11 + 1) / 8).astype(dtype).reshape(shape)
    flat_data, flat_noise = data.reshape(-1), noise.reshape(-1)
    flat_data[1] = np.nan
    flat_data[-2] = np.inf
    flat_noise[2] = np.nan
    flat_noise[3] = -1.0
    flat_noise[4] = np.inf
    flat_noise[5] = 0.0
    kernel_shape = (3,) + (2,) * (len(shape) - 1)
    weights = (np.arange(np.prod(kernel_shape)).reshape(kernel_shape) % 5 + 1) / 4
    return data, noise, weights


def _assert_close(actual: np.ndarray, expected: np.ndarray) -> None:
    """Allow a few rounding steps, relative to each value and to the array scale.

    Signed weights cancel, so a small result can carry rounding error from
    larger terms. The absolute part bounds that error by the array scale.
    """
    eps = np.finfo(actual.dtype).eps
    finite = expected[np.isfinite(expected)]
    scale = np.abs(finite).max(initial=0.0)
    np.testing.assert_allclose(
        actual, expected.astype(actual.dtype), rtol=4 * eps, atol=16 * eps * scale
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("correlate", [False, True])
def test_raw_sum_variance_matches_exact_reference(
    dtype: type[np.floating], mode: str, shape: tuple[int, ...], correlate: bool
) -> None:
    data, noise, weights = _inputs(shape, dtype)
    weights = weights - 0.5  # Signed coefficients.
    value, variance = nf.convolve_variance(
        data, noise, weights, mode=mode, cval=1.25, correlate=correlate
    )
    expected = noise_reference(
        data, noise, weights, (mode,) * data.ndim, cval=1.25, correlate=correlate
    )
    _assert_close(value, expected[0])
    _assert_close(variance, expected[1])
    plain = nf.correlate if correlate else nf.convolve
    np.testing.assert_array_equal(value, plain(data, weights, mode=mode, cval=1.25))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("noise_type", ["variance", "inverse_variance"])
def test_normalized_variance_matches_exact_reference(
    dtype: type[np.floating], mode: str, shape: tuple[int, ...], noise_type: str
) -> None:
    data, noise, weights = _inputs(shape, dtype)
    inverse = noise_type == "inverse_variance"
    if inverse:
        with np.errstate(divide="ignore", invalid="ignore"):
            noise = np.where(noise >= 0, 1 / noise, noise).astype(dtype)
    for cval in (np.nan, 0.5):
        value, variance = nf.nanconvolve_variance(
            data,
            noise,
            weights,
            mode=mode,
            cval=cval,
            min_count=2,
            noise_type=noise_type,
        )
        expected = noise_reference(
            data,
            noise,
            weights,
            (mode,) * data.ndim,
            cval=cval,
            estimator="mean",
            inverse_noise=inverse,
            omit_nan=True,
            min_count=2,
        )
        _assert_close(value, expected[0])
        _assert_close(variance, expected[1])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("weighting", ["inverse_variance", "kernel"])
def test_matched_filter_matches_exact_reference(
    dtype: type[np.floating], mode: str, shape: tuple[int, ...], weighting: str
) -> None:
    data, noise, weights = _inputs(shape, dtype)
    weights = weights - 0.5
    result = nf.matched_filter(
        data,
        noise,
        weights,
        mode=mode,
        cval=0.75,
        weighting=weighting,
        ignore_inf=True,
    )
    expected = noise_reference(
        data,
        noise,
        weights,
        (mode,) * data.ndim,
        cval=0.75,
        correlate=True,
        estimator="amplitude",
        inverse_weighting=weighting == "inverse_variance",
        omit_nan=True,
        ignore_inf=True,
    )
    _assert_close(result.amplitude, expected[0])
    _assert_close(result.variance, expected[1])
    _assert_close(result.snr, expected[2])


def test_noise_that_omits_nothing_keeps_nanconvolve_bitwise() -> None:
    rng = np.random.default_rng(5)
    data = rng.normal(size=(9, 11))
    data[rng.random(data.shape) < 0.2] = np.nan
    weights = nf.kernels.gaussian(1.2, ndim=2, radius=2)
    value, _ = nf.nanconvolve_variance(data, np.ones_like(data), weights)
    np.testing.assert_array_equal(value, nf.nanconvolve(data, weights))


def test_boundary_repeats_add_weights_before_squaring() -> None:
    # Reflect reads pixel 0 through two taps at output 0.
    value, variance = nf.convolve_variance(
        np.array([1.0, 2.0, 4.0]), np.array([1.0, 10.0, 100.0]), [3.0, 2.0, 1.0]
    )
    np.testing.assert_array_equal(value, [9.0, 17.0, 22.0])
    np.testing.assert_array_equal(variance, [99.0, 941.0, 2510.0])


def test_uniform_noise_gives_equal_snr_for_both_weightings() -> None:
    rng = np.random.default_rng(11)
    data = rng.normal(size=(16, 16))
    template = nf.kernels.gaussian(1.5, ndim=2, radius=3)
    noise = np.full(data.shape, 2.25)
    ivar = nf.matched_filter(data, noise, template)
    kernel = nf.matched_filter(data, noise, template, weighting="kernel")
    np.testing.assert_allclose(ivar.snr, kernel.snr, rtol=1e-12)
    np.testing.assert_allclose(ivar.amplitude, kernel.amplitude, rtol=1e-12)
    # S/N of the raw sum is the same ratio, because normalization cancels.
    value, variance = nf.convolve_variance(data, noise, template, correlate=True)
    np.testing.assert_allclose(kernel.snr, value / np.sqrt(variance), rtol=1e-12)


def test_constant_boundary_samples_are_exact_or_omitted() -> None:
    data = np.array([2.0, 4.0])
    noise = np.array([1.0, 3.0])
    value, variance = nf.convolve_variance(
        data, noise, [1.0, 1.0, 1.0], mode="constant", cval=5.0
    )
    np.testing.assert_array_equal(value, [11.0, 11.0])
    np.testing.assert_array_equal(variance, [4.0, 4.0])
    # Inverse-variance weighting gives zero-variance padding infinite weight, so
    # it is omitted: the output equals the unpadded fit.
    padded = nf.matched_filter(data, noise, [1.0, 1.0, 1.0], mode="constant", cval=5.0)
    exterior = nf.matched_filter(
        data, noise, [1.0, 1.0, 1.0], mode="constant", cval=np.nan
    )
    np.testing.assert_array_equal(padded.amplitude, exterior.amplitude)


@pytest.mark.parametrize("dtype", DTYPES)
def test_matched_filter_recovers_point_source(dtype: type[np.floating]) -> None:
    template = nf.kernels.gaussian(1.3, ndim=2, radius=4)
    image = np.zeros((21, 21), dtype=dtype)
    image[6:15, 6:15] = 7.5 * template
    variance = np.linspace(0.5, 2.0, image.size).reshape(image.shape).astype(dtype)
    result = nf.matched_filter(image, variance, template)
    tolerance = 10 * float(np.finfo(dtype).eps)
    assert result.amplitude[10, 10] == pytest.approx(7.5, rel=tolerance)
    assert np.argmax(result.snr) == np.ravel_multi_index((10, 10), image.shape)


def _chi2_bounds(realizations: int, tests: int, alpha: float) -> tuple[float, float]:
    """Bonferroni-corrected two-sided bounds on the sample/true variance ratio."""
    dof = realizations - 1
    tail = alpha / (2 * tests)
    return (
        stats.chi2.ppf(tail, dof) / dof,
        stats.chi2.ppf(1 - tail, dof) / dof,
    )


def test_propagated_variance_matches_monte_carlo() -> None:
    """Compare predicted and empirical variance over Gaussian noise realizations.

    Each output is a fixed linear combination of independent Gaussian samples,
    so ``(R - 1) s**2 / sigma**2`` follows a chi-squared law with ``R - 1``
    degrees of freedom. The bounds are its quantiles at a total false-failure
    probability ``alpha = 1e-6``, Bonferroni-split over every tested pixel and
    output. Bonferroni stays valid although neighboring outputs correlate. With
    ``R = 4000`` and 1080 tests (three maps of 360 pixels), the accepted ratio
    range is about ``[0.869, 1.143]``. The seed
    is fixed, so the test is deterministic.
    """
    rng = np.random.default_rng(20261112)
    realizations, shape = 4000, (18, 20)
    variance = rng.uniform(0.2, 5.0, size=shape)
    missing = rng.random(shape) < 0.15
    missing[:4, :4] = False  # Keep the corner check below free of omissions.
    template = nf.kernels.gaussian(1.4, ndim=2, radius=3)
    weights = template + 0.01  # Unequal weights on every tap.
    signal = 3.0 * np.pad(template, ((5, 6), (6, 7)))
    data = signal + np.sqrt(variance) * rng.standard_normal((realizations, *shape))
    data[:, missing] = np.nan
    stacked_noise = np.broadcast_to(variance, data.shape)
    cube = (1, *weights.shape)

    means, mean_variance = nf.nanconvolve_variance(
        data, stacked_noise, weights.reshape(cube), mode="reflect", min_count=5
    )
    fit = nf.matched_filter(data, stacked_noise, template.reshape(cube), mode="reflect")
    checks = [
        (means, mean_variance[0]),
        (fit.amplitude, fit.variance[0]),
    ]
    valid = np.isfinite(mean_variance[0]) & np.isfinite(fit.variance[0])
    lower, upper = _chi2_bounds(realizations, 3 * int(valid.sum()), 1e-6)
    for samples, predicted in checks:
        ratio = samples.var(axis=0, ddof=1)[valid] / predicted[valid]
        assert lower < ratio.min() and ratio.max() < upper, (ratio.min(), ratio.max())
    # The amplitude is unbiased at the source position (Bonferroni over one test).
    center = (8, 9)
    error = fit.amplitude[:, center[0], center[1]].mean() - 3.0
    z = error / np.sqrt(fit.variance[0][center] / realizations)
    assert abs(z) < stats.norm.ppf(1 - 0.5e-6)
    # Under pure noise, the signal-to-noise ratio has unit variance.
    noise_only = nf.matched_filter(
        data - signal, stacked_noise, template.reshape(cube), mode="reflect"
    )
    ratio = noise_only.snr.var(axis=0, ddof=1)[valid]
    assert lower < ratio.min() and ratio.max() < upper
    # The squared-weight formula without grouping repeated reflect pixels is
    # rejected at the corner, so the test can detect that error.
    taps = np.arange(-3, 4)
    reflected = np.where(taps < 0, -taps - 1, taps)
    naive = (weights**2 * variance[np.ix_(reflected, reflected)]).sum()
    naive /= weights.sum() ** 2
    ratio = means[:, 0, 0].var(ddof=1) / naive
    assert not lower < ratio < upper, ratio


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda: nf.convolve_variance([1.0, 2.0], [1.0], [1.0]), ValueError, "shape"),
        (
            lambda: nf.convolve_variance([1.0], [1.0], [1.0], noise_type="rms"),
            ValueError,
            "noise_type",
        ),
        (
            lambda: nf.convolve_variance([1.0], np.ma.masked_array([1.0]), [1.0]),
            TypeError,
            "masked",
        ),
        (
            lambda: nf.nanconvolve_variance([1.0], [1.0], [-1.0]),
            ValueError,
            "nonnegative",
        ),
        (
            lambda: nf.matched_filter([1.0], [1.0], [0.0]),
            ValueError,
            "nonzero",
        ),
        (
            lambda: nf.matched_filter([1.0], [1.0], [1.0], weighting="optimal"),
            ValueError,
            "weighting",
        ),
        (
            lambda: nf.matched_filter([1.0], [1.0], [1.0], min_count=2),
            ValueError,
            "min_count",
        ),
        (
            lambda: nf.convolve_variance([1.0], [True], [1.0]),
            TypeError,
            "real",
        ),
    ],
)
def test_invalid_arguments_are_rejected(call, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        call()


def test_empty_arrays_return_empty_outputs() -> None:
    data = np.empty((0, 3))
    value, variance = nf.convolve_variance(data, data, np.ones((1, 3)))
    assert value.shape == variance.shape == (0, 3)
