"""Mesh background: reference parity, API checks, and sep/photutils parity."""

from __future__ import annotations

import warnings

import ndfiltrs as nf
import numpy as np
import pytest

from tests._background_reference import mesh_background as reference

PHOTUTILS = {
    "estimator": "sigma_clip",
    "maxiters": 10,
    "statistic": "mode",
    "mode_estimator": "sex",
    "rms": "std",
    "min_valid_fraction": 0.9,
    "valid_count": "retained",
    "valid_base": "box",
    "fill": "idw",
    "filter_edge": "truncate",
}

FIELDS = (
    "background",
    "rms",
    "mesh_background",
    "mesh_rms",
    "mesh_valid",
    "mesh_finite_count",
    "mesh_retained_count",
    "global_background",
    "global_rms",
)


def _image(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Sloped Gaussian noise (sigma 5) with a few bright sources and NaNs."""
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    image = 100.0 + sum(0.02 * (axis + 1) * g for axis, g in enumerate(grids))
    image = image + rng.normal(0.0, 5.0, shape)
    flat = image.reshape(-1)
    flat[rng.choice(flat.size, flat.size // 200, replace=False)] += 800.0
    flat[rng.choice(flat.size, flat.size // 300, replace=False)] = np.nan
    return image


def _assert_matches_reference(result: nf.MeshBackground, expected: dict) -> None:
    # Summation order (Neumaier vs NumPy pairwise) and the tridiagonal solver
    # differ, so float64 results agree to rounding: values near 100 within
    # 1e-11 absolute. Counts and validity are exact.
    for name in FIELDS:
        actual = np.asarray(getattr(result, name))
        wanted = np.asarray(expected[name])
        if actual.dtype.kind in "bi":
            np.testing.assert_array_equal(actual, wanted, err_msg=name)
        else:
            np.testing.assert_allclose(
                actual, wanted, rtol=0, atol=1e-11, equal_nan=True, err_msg=name
            )


@pytest.mark.parametrize(
    ("shape", "box", "options"),
    [
        ((150, 230), (32, 40), {}),
        ((150, 230), (32, 40), {"mode_estimator": "mmm"}),
        ((150, 230), (25, 25), PHOTUTILS),
        ((120, 90), (16, 30), {"filter_size": (3, 5), "filter_threshold": 0.5}),
        (
            (120, 90),
            (16, 30),
            {**PHOTUTILS, "filter_threshold": 101.0, "threshold_kind": "above"},
        ),
        ((97,), (10,), {"filter_size": 5}),
        ((20, 18, 15), (6, 5, 4), {}),
        ((20, 18, 15), (6, 5, 4), {**PHOTUTILS, "spread": "mad", "rms": "mad"}),
        ((64, 64), (64, 64), {}),
    ],
)
def test_matches_numpy_reference(shape, box, options) -> None:
    image = _image(shape, seed=len(shape) + sum(box))
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(slice(0, n // 3) for n in shape)] = True
    result = nf.mesh_background(image, box, mask=mask, **options)
    _assert_matches_reference(result, reference(image, box, mask=mask, **options))


def test_float32_maps_are_cast_float64_maps() -> None:
    image = _image((64, 80), seed=3)
    single = nf.mesh_background(image.astype(np.float32), 16)
    double = nf.mesh_background(image.astype(np.float32).astype(np.float64), 16)
    assert single.background.dtype == np.float32
    assert single.mesh_background.dtype == np.float64
    np.testing.assert_array_equal(single.mesh_background, double.mesh_background)
    np.testing.assert_array_equal(
        single.background, double.background.astype(np.float32)
    )


def test_noncontiguous_input_and_numeric_mask() -> None:
    image = _image((80, 60), seed=4)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[10:30, 5:25] = 7
    view = np.asfortranarray(image)[::-1]
    result = nf.mesh_background(view, 20, mask=mask[::-1])
    expected = reference(np.ascontiguousarray(view), 20, mask=mask[::-1] != 0)
    _assert_matches_reference(result, expected)


@pytest.mark.parametrize("interpolation", ["spline", "sextractor"])
@pytest.mark.parametrize("box_size", [4, 8])
def test_all_invalid_gives_nan(interpolation, box_size) -> None:
    result = nf.mesh_background(
        np.full((8, 8), np.nan), box_size, interpolation=interpolation
    )
    assert np.isnan(result.background).all()
    assert np.isnan(result.global_rms)
    assert not result.mesh_valid.any()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"box_size": 0}, ValueError),
        ({"box_size": (4,)}, ValueError),
        ({"box_size": 2.5}, TypeError),
        ({"filter_size": 2}, ValueError),
        ({"estimator": "histogram"}, ValueError),
        ({"min_valid_fraction": 1.5}, ValueError),
        ({"sigma": True}, TypeError),
        ({"maxiters": -1}, ValueError),
        ({"fill": "idw", "idw_neighbors": 0}, ValueError),
        ({"filter_threshold": np.nan}, ValueError),
        ({"mask": np.zeros((4, 4), dtype=bool)}, ValueError),
        ({"mask": np.zeros((8, 8), dtype=complex)}, TypeError),
        ({"interpolation": "linear"}, ValueError),
    ],
)
def test_invalid_arguments(kwargs, error) -> None:
    arguments = {"box_size": 4, **kwargs}
    with pytest.raises(error):
        nf.mesh_background(np.ones((8, 8)), **arguments)


def test_sextractor_interpolation_needs_two_dimensions() -> None:
    with pytest.raises(ValueError, match="2-D"):
        nf.mesh_background(np.ones((8, 8, 8)), 4, interpolation="sextractor")


# --- sep parity -----------------------------------------------------------


def _mesh_constant(shape, box, seed, bad_fraction):
    """Image constant inside each mesh, with whole meshes masked."""
    rng = np.random.default_rng(seed)
    mesh_shape = tuple(-(-n // b) for n, b in zip(shape, box, strict=True))
    levels = rng.integers(0, 2000, mesh_shape).astype(np.float32) / 8
    ones = np.ones(box, dtype=np.float32)
    image = np.kron(levels, ones)[: shape[0], : shape[1]].copy()
    bad = rng.random(mesh_shape) < bad_fraction
    mask = np.kron(bad, np.ones(box, dtype=bool))[: shape[0], : shape[1]].copy()
    return image, mask


@pytest.mark.parametrize("interpolation", ["spline", "sextractor"])
def test_single_sample_mesh_preserves_level(interpolation) -> None:
    image = np.full((3, 8), 3, dtype=np.float32)
    result = nf.mesh_background(image, 1, filter_size=1, interpolation=interpolation)
    assert result.global_background == 3
    np.testing.assert_array_equal(result.background, image)
    np.testing.assert_array_equal(result.rms, 0)


@pytest.mark.parametrize("interpolation", ["spline", "sextractor"])
def test_large_finite_constant_background(interpolation) -> None:
    image = np.full((10, 15), np.finfo(np.float64).max)
    result = nf.mesh_background(
        image, (3, 4), filter_size=1, interpolation=interpolation
    )
    assert result.mesh_valid.all()
    np.testing.assert_array_equal(result.background, image)
    np.testing.assert_array_equal(result.rms, 0)


def test_one_pixel_box_width_uses_requested_interpolation() -> None:
    # One horizontal mesh row. Each mesh center is exactly on its pixel.
    levels = np.array([[0, 3, 1, 8, 4, 2]], dtype=np.float64)
    image = levels.repeat(8, axis=0)
    natural = nf.mesh_background(image, (8, 1), filter_size=1)
    np.testing.assert_allclose(natural.background, image, rtol=0, atol=1e-13)

    # sep.Background(image, bh=8, bw=1, fh=1, fw=1).back()[4]
    legacy = nf.mesh_background(
        image, (8, 1), filter_size=1, interpolation="sextractor"
    )
    np.testing.assert_allclose(legacy.background[4], [0, 3, 6, 9, 12, 15])


def test_sep_interpolation_on_partial_meshes() -> None:
    """Sample a map measured with the public sep 1.4.1 binary."""
    grid = (np.arange(30).reshape(5, 6) ** 2 % 17).astype(np.float32)
    image = grid.repeat(3, axis=0).repeat(4, axis=1)[:-1, :-1].copy()
    pixels = np.ix_([0, 3, 7, 13], [0, 5, 9, 16, 22])
    # sep.Background(image, bh=3, bw=4, fh=1, fw=1).back()[pixels]
    expected = np.array(
        [
            [2.89116192, -6.96700048, -2.73611975, 12.78070450, 10.42677021],
            [-3.32890677, 8.51829243, 9.71400928, 16.72304153, 2.55588579],
            [3.01372504, 16.78385735, 10.82570934, 2.92412043, -0.31003678],
            [12.23737526, 10.47210693, 11.82488537, 7.89339352, 7.88244104],
        ],
        dtype=np.float32,
    )
    result = nf.mesh_background(
        image, (3, 4), filter_size=1, interpolation="sextractor"
    )
    np.testing.assert_allclose(
        result.background[pixels],
        expected,
        rtol=0,
        atol=4 * np.finfo(np.float32).eps * np.abs(expected).max(),
    )


@pytest.mark.parametrize(
    ("shape", "box", "filter_size", "bad"),
    [
        ((200, 300), (32, 40), (3, 3), 0.0),
        ((200, 300), (32, 40), (3, 5), 0.2),
        ((128, 96), (16, 16), (5, 3), 0.3),
        ((30, 90), (32, 20), (1, 1), 0.0),
        ((20, 7), (5, 1), (1, 1), 0.0),
    ],
)
def test_sep_fill_filter_interpolation(shape, box, filter_size, bad) -> None:
    """Exact mesh values isolate sep's fill, filter, and spline stages."""
    sep = pytest.importorskip("sep")
    image, mask = _mesh_constant(shape, box, seed=sum(shape), bad_fraction=bad)
    expected = sep.Background(
        image, mask=mask, bw=box[1], bh=box[0], fw=filter_size[1], fh=filter_size[0]
    )
    result = nf.mesh_background(
        image, box, mask=mask, filter_size=filter_size, interpolation="sextractor"
    )
    # sep evaluates everything in float32; 4 float32 ulps of the map scale.
    scale = np.abs(expected.back()).max()
    tolerance = 4 * np.finfo(np.float32).eps * scale
    np.testing.assert_allclose(result.background, expected.back(), atol=tolerance)
    assert result.global_background == expected.globalback


def test_sep_noise_meshes() -> None:
    """Exact-sample statistics agree with sep's histogram within its bins."""
    sep = pytest.importorskip("sep")
    sigma = 5.0
    image = _image((256, 279), seed=11).astype(np.float32)
    mask = ~np.isfinite(image)
    box = (32, 31)
    expected = sep.Background(image, mask=mask, bw=box[1], bh=box[0])
    result = nf.mesh_background(image, box, mask=mask, interpolation="sextractor")
    # With an even bh and odd bw, pixel (k*bh + bh/2, k*bw + bw//2) is a sep
    # spline node, where the map equals the (filtered) mesh value.
    rows = np.arange(result.mesh_background.shape[0] - 1) * box[0] + box[0] // 2
    cols = np.arange(result.mesh_background.shape[1] - 1) * box[1] + box[1] // 2
    sep_mesh = expected.back()[np.ix_(rows, cols)]
    sep_rms = expected.rms()[np.ix_(rows, cols)]
    ours = result.mesh_background[:-1, :-1]
    # sep's histogram bin is 10 sigma / nlevels with nlevels ~ 0.95 N for N
    # pixels per mesh. The SExtractor mode 2.5 median - 1.5 mean can move by
    # 2.5 + 1.5 bins when the median and mean each move by one bin.
    bin_width = 10 * sigma / (0.95 * box[0] * box[1])
    np.testing.assert_allclose(ours, sep_mesh, atol=4 * bin_width)
    np.testing.assert_allclose(result.mesh_rms[:-1, :-1], sep_rms, atol=4 * bin_width)


# --- photutils parity -----------------------------------------------------


def _photutils(image, box, **kwargs):
    background = pytest.importorskip("photutils.background")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return background.Background2D(image, box, **kwargs)


def test_photutils_meshes_without_fill() -> None:
    """Clipping, SExtractor mode, std, and the truncated median filter."""
    image = _image((256, 320), seed=5)
    expected = _photutils(image, (32, 32), filter_size=(3, 3))
    result = nf.mesh_background(image, 32, **PHOTUTILS)
    assert result.mesh_valid.all()
    np.testing.assert_array_equal(result.mesh_retained_count, expected.n_pixels_mesh)
    # Both compute in float64 from the same survivors; only summation order
    # differs.
    np.testing.assert_allclose(
        result.mesh_background, expected.background_mesh, rtol=1e-13
    )
    np.testing.assert_allclose(
        result.mesh_rms, expected.background_rms_mesh, rtol=1e-12
    )


def test_photutils_fill_away_from_ties() -> None:
    """IDW fill agrees except where photutils' k-d tree breaks distance ties."""
    image = _image((256, 320), seed=6)
    mask = np.zeros(image.shape, dtype=bool)
    mask[60:150, 100:170] = True
    expected = _photutils(image, (32, 32), mask=mask, filter_size=1)
    result = nf.mesh_background(image, 32, mask=mask, filter_size=1, **PHOTUTILS)
    valid = result.mesh_valid
    # photutils' validity rule on its own survivor counts.
    np.testing.assert_array_equal(valid, expected.n_pixels_mesh >= 0.9 * 32 * 32)
    good = np.argwhere(valid)
    untied = valid.copy()
    for index in np.argwhere(~valid):
        d2 = np.sort(((good - index) ** 2).sum(axis=1))
        untied[tuple(index)] = d2.size <= 10 or d2[9] != d2[10]
    assert untied.sum() > valid.sum()
    np.testing.assert_allclose(
        result.mesh_background[untied], expected.background_mesh[untied], rtol=1e-12
    )
