"""Tests for sampled N-dimensional kernel factories."""

from __future__ import annotations

import math
from collections.abc import Callable

import ndfiltrs as nf
import ndfiltrs.kernels as kernels
import numpy as np
import pytest
from scipy.stats import multivariate_normal, multivariate_t


@pytest.mark.parametrize(
    ("factory", "shape"),
    [
        (lambda: kernels.box((2, 3)), (2, 3)),
        (lambda: kernels.box(3, ndim=2), (3, 3)),
        (lambda: kernels.tophat(1.5, 2), (3, 3)),
        (lambda: kernels.ring(1.0, 2.0, 2), (5, 5)),
        (lambda: kernels.gaussian((0.7, 1.2)), (7, 11)),
        (lambda: kernels.moffat(1.2, 2.5, ndim=2, radius=3), (7, 7)),
        (lambda: kernels.student_t(1.2, 4, ndim=2, radius=3), (7, 7)),
    ],
)
def test_shapes_dtype_contiguity_and_normalization(
    factory: Callable[[], np.ndarray], shape: tuple[int, ...]
) -> None:
    result = factory()
    assert result.shape == shape
    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    assert result.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: kernels.tophat(2.1, 3),
        lambda: kernels.ring(1.0, 2.5, 3),
        lambda: kernels.gaussian((0.7, 1.1, 1.5)),
        lambda: kernels.moffat(1.2, 2.5, ndim=3, radius=3),
        lambda: kernels.student_t((0.8, 1.2, 1.5), 5, radius=(3, 4, 5)),
    ],
)
def test_axis_aligned_profiles_are_symmetric(
    factory: Callable[[], np.ndarray],
) -> None:
    result = factory()
    for axis in range(result.ndim):
        np.testing.assert_allclose(result, np.flip(result, axis=axis), atol=0.0)


def test_box_even_size_and_raw_weights() -> None:
    result = kernels.box((2, 4), normalize=False)
    assert result.shape == (2, 4)
    np.testing.assert_array_equal(result, np.ones((2, 4)))


def test_generated_weights_work_with_convolution_functions() -> None:
    image = np.ones((5, 5), dtype=np.float64)
    weights = kernels.gaussian(0.8, ndim=2)
    np.testing.assert_allclose(nf.convolve(image, weights), image)
    image[2, 2] = np.nan
    np.testing.assert_allclose(nf.nanconvolve(image, weights), 1.0)


def test_tophat_extent_is_smallest_for_each_oversampling_factor() -> None:
    for factor in (1, 2, 5, 10):
        result = kernels.tophat(1.0, 2, oversample=factor, normalize=False)
        assert result.shape == (3, 3)


def test_ring_extent_crops_only_zero_outer_layers() -> None:
    result = kernels.ring(1.2, 2.0, 2, normalize=False)
    assert result.shape == (5, 5)
    assert np.any(result != 0)
    assert np.any(result[[0, -1], :] != 0) or np.any(result[:, [0, -1]] != 0)


def test_oversampling_averages_subpixel_samples() -> None:
    center = kernels.tophat(0.6, 1, normalize=False)
    sampled = kernels.tophat(0.6, 1, oversample=10, normalize=False)
    assert center[0] == 1.0
    assert 0.0 < sampled[0] < center[0]
    assert sampled.sum() > center.sum()


def test_rotation_preserves_isotropic_gaussian() -> None:
    angle = 0.37
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    expected = kernels.gaussian(1.1, ndim=2, radius=4)
    actual = kernels.gaussian(1.1, rotation=rotation, radius=4)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-16)


def test_rotated_gaussian_default_radius_uses_marginal_covariance() -> None:
    angle = 0.43
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    widths = (0.7, 1.4)
    covariance = rotation @ np.diag(np.square(widths)) @ rotation.T
    radii = np.ceil(4.0 * np.sqrt(np.diag(covariance))).astype(int)
    result = kernels.gaussian(widths, rotation=rotation)
    assert result.shape == tuple(2 * radii + 1)


def test_rotated_gaussian_matches_multivariate_normal_shape() -> None:
    angle = 0.43
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    widths = (0.7, 1.4)
    result = kernels.gaussian(widths, rotation=rotation, radius=5)
    y, x = np.indices(result.shape, dtype=np.float64)
    coordinates = np.stack((y - 5, x - 5), axis=-1)
    covariance = rotation @ np.diag(np.square(widths)) @ rotation.T
    expected = multivariate_normal.pdf(coordinates, mean=(0.0, 0.0), cov=covariance)
    expected /= expected.sum()
    np.testing.assert_allclose(result, expected, rtol=2e-14, atol=1e-16)


def test_student_t_matches_multivariate_t_shape() -> None:
    angle = 0.31
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    widths = (0.8, 1.3)
    degrees = 4.5
    result = kernels.student_t(widths, degrees, rotation=rotation, radius=5)
    y, x = np.indices(result.shape, dtype=np.float64)
    coordinates = np.stack((y - 5, x - 5), axis=-1)
    scale = rotation @ np.diag(np.square(widths)) @ rotation.T
    expected = multivariate_t.pdf(coordinates, loc=(0.0, 0.0), shape=scale, df=degrees)
    expected /= expected.sum()
    np.testing.assert_allclose(result, expected, rtol=2e-14, atol=1e-16)


def test_moffat_student_t_relation() -> None:
    dimension = 3
    degrees = 2.5
    scale = 1.3
    radius = (4, 3, 5)
    moffat_weights = kernels.moffat(
        math.sqrt(degrees) * scale,
        (degrees + dimension) / 2,
        ndim=dimension,
        radius=radius,
    )
    student_weights = kernels.student_t(scale, degrees, ndim=dimension, radius=radius)
    np.testing.assert_allclose(moffat_weights, student_weights, rtol=2e-15, atol=0)


def test_anisotropic_moffat_student_t_relation_with_rotation() -> None:
    angle = 0.28
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    degrees = 3.0
    scale = (0.8, 1.4)
    radius = (4, 6)
    moffat_weights = kernels.moffat(
        tuple(math.sqrt(degrees) * value for value in scale),
        (degrees + 2) / 2,
        rotation=rotation,
        radius=radius,
    )
    student_weights = kernels.student_t(
        scale, degrees, rotation=rotation, radius=radius
    )
    np.testing.assert_allclose(moffat_weights, student_weights, rtol=2e-15, atol=0)


@pytest.mark.parametrize(
    ("factory", "constructor"),
    [
        ("box", "Box2DKernel"),
        ("tophat", "Tophat2DKernel"),
        ("ring", "Ring2DKernel"),
        ("moffat", "Moffat2DKernel"),
    ],
)
@pytest.mark.parametrize(("mode", "factor"), [("center", 1), ("oversample", 10)])
def test_astropy_kernel_parity(
    factory: str, constructor: str, mode: str, factor: int
) -> None:
    astropy_convolution = pytest.importorskip("astropy.convolution")
    astropy_type = getattr(astropy_convolution, constructor)
    common = {"mode": mode, "factor": factor}
    if factory == "box":
        actual = kernels.box((5, 5))
        reference = astropy_type(5, **common).array
    elif factory == "tophat":
        actual = kernels.tophat(2.0, 2, oversample=factor)
        reference = astropy_type(2.0, **common).array
    elif factory == "ring":
        actual = kernels.ring(1.0, 3.0, 2, oversample=factor)
        reference = astropy_type(1.0, 2.0, **common).array
    else:
        reference = astropy_type(1.2, 2.5, **common).array
        radius = (reference.shape[0] - 1) // 2
        actual = kernels.moffat(1.2, 2.5, ndim=2, radius=radius, oversample=factor)
    assert actual.shape == reference.shape
    np.testing.assert_allclose(actual, reference, rtol=2e-13, atol=2e-15)


@pytest.mark.parametrize(("mode", "factor"), [("center", 1), ("oversample", 10)])
def test_astropy_rotated_gaussian_parity(mode: str, factor: int) -> None:
    astropy_convolution = pytest.importorskip("astropy.convolution")
    x_sigma, y_sigma, theta = 1.3, 0.7, 0.37
    # In array coordinates, columns are the rotated y-principal then x-principal axes.
    rotation = np.array(
        [
            [math.cos(theta), math.sin(theta)],
            [-math.sin(theta), math.cos(theta)],
        ]
    )
    actual = kernels.gaussian(
        (y_sigma, x_sigma),
        rotation=rotation,
        radius=(4, 7),
        oversample=factor,
    )
    reference = astropy_convolution.Gaussian2DKernel(
        x_sigma,
        y_sigma,
        theta=theta,
        x_size=15,
        y_size=9,
        mode=mode,
        factor=factor,
    ).array
    np.testing.assert_allclose(actual, reference, rtol=2e-13, atol=2e-15)


@pytest.mark.parametrize(
    ("call", "exception"),
    [
        (lambda: kernels.box(3), TypeError),
        (lambda: kernels.box(0, ndim=2), ValueError),
        (lambda: kernels.box((2, 3), ndim=3), ValueError),
        (lambda: kernels.tophat(1.0, 0), ValueError),
        (lambda: kernels.ring(2.0, 1.0, 2), ValueError),
        (lambda: kernels.gaussian(1.0), TypeError),
        (lambda: kernels.gaussian((1.0, 2.0), ndim=3), ValueError),
        (lambda: kernels.gaussian(1.0, ndim=2, rotation=np.ones((2, 2))), ValueError),
        (lambda: kernels.gaussian(1.0, ndim=2, oversample=1.5), TypeError),
        (lambda: kernels.moffat(1.0, 2.0, ndim=2), TypeError),
        (lambda: kernels.student_t(1.0, 0.0, ndim=2, radius=2), ValueError),
    ],
)
def test_invalid_arguments_raise_specific_exceptions(
    call: Callable[[], object], exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        call()


@pytest.mark.parametrize("normalize", [True, False])
def test_empty_sampled_region_is_rejected(normalize: bool) -> None:
    with pytest.raises(ValueError, match="no subpixel sample"):
        kernels.tophat(0.0, 1, oversample=2, normalize=normalize)
    with pytest.raises(ValueError, match="no subpixel sample"):
        kernels.ring(1.5, 1.5, 2, normalize=normalize)


def test_sample_reproduces_builtin_profiles() -> None:
    gaussian = kernels.gaussian((1.3, 0.8), radius=(5, 4), oversample=3)
    sampled = kernels.sample(
        lambda y, x: np.exp(-0.5 * ((y / 1.3) ** 2 + (x / 0.8) ** 2)),
        (5, 4),
        oversample=3,
    )
    np.testing.assert_allclose(sampled, gaussian, rtol=1e-14, atol=0)
    moffat = kernels.moffat(1.5, 2.5, ndim=3, radius=3, normalize=False)
    profile = kernels.sample(
        lambda *x: (1 + sum(c * c for c in x) / 1.5**2) ** -2.5,
        3,
        ndim=3,
        normalize=False,
    )
    np.testing.assert_allclose(profile, moffat, rtol=1e-14, atol=0)


def test_sample_broadcasts_scalars_and_keeps_zero_sum_weights() -> None:
    np.testing.assert_array_equal(kernels.sample(lambda x: 1.0, 2, ndim=1), 0.2)
    ricker = kernels.sample(
        lambda y, x: (1 - (y * y + x * x) / 2) * np.exp(-(y * y + x * x) / 2),
        radius=6,
        ndim=2,
        normalize=False,
    )
    assert ricker.shape == (13, 13) and ricker[6, 6] == 1.0
    with pytest.raises(ValueError, match="positive finite sum"):
        kernels.sample(lambda y, x: -np.ones_like(y * x), radius=(1, 1))


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda: kernels.sample(3.0, 1, ndim=1), TypeError, "callable"),
        (lambda: kernels.sample(lambda x: x, 1.5, ndim=1), TypeError, "radius"),
        (lambda: kernels.sample(lambda x: x, 1), TypeError, "ndim"),
        (lambda: kernels.sample(lambda x: x / 0.0, 1, ndim=1), ValueError, "finite"),
        (
            lambda: kernels.sample(lambda y, x: np.ones(4), (1, 1)),
            ValueError,
            "broadcast",
        ),
        (lambda: kernels.sample(lambda y: y, (1, 1)), TypeError, None),
    ],
)
def test_sample_rejects_invalid_input(
    call: Callable[[], object], error: type[Exception], match: str | None
) -> None:
    with pytest.raises(error, match=match), np.errstate(all="ignore"):
        call()


def test_sample_accepts_astropy_models() -> None:
    models = pytest.importorskip("astropy.modeling.models")
    sersic = models.Sersic2D(amplitude=1, r_eff=3, n=2, x_0=0, y_0=0, ellip=0.3)
    weights = kernels.sample(lambda y, x: sersic(x, y), radius=(4, 6), normalize=False)
    y, x = np.mgrid[-4:5, -6:7]
    np.testing.assert_allclose(weights, sersic(x, y), rtol=1e-14)
