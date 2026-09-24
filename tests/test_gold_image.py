"""Persistent scene integrity and independent filtering/repair regression checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ndfiltrs as nf
import numpy as np
import pytest

from tests._fixpix_reference import fixpix_reference
from tests._mean_reference import mean_filter_reference
from tests._reference import _median, median_filter_reference

GOLD = Path(__file__).parent / "data" / "gold_v1"


@pytest.fixture(scope="module")
def scene() -> dict[str, np.ndarray]:
    """Load the frozen fixture without executable pickle content."""
    with np.load(GOLD / "scene.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def test_gold_arrays_have_manifest_identity_and_physical_layers(
    scene: dict[str, np.ndarray],
) -> None:
    manifest = json.loads((GOLD / "manifest.json").read_text())
    assert manifest["scene_id"] == "ndfiltrs-gold-v1"
    assert (
        hashlib.sha256((GOLD / "scene.npz").read_bytes()).hexdigest()
        == manifest["archive_sha256"]
    )
    assert set(scene) == set(manifest["arrays"])
    for name, expected in manifest["arrays"].items():
        value = scene[name]
        assert list(value.shape) == expected["shape"]
        assert value.dtype.str == expected["dtype"]
        assert (
            hashlib.sha256(value.tobytes(order="C")).hexdigest() == expected["sha256"]
        )
    np.testing.assert_array_equal(scene["model"] + scene["noise"], scene["clean"])
    assert np.isfinite(scene["model"]).all() and np.isfinite(scene["clean"]).all()
    assert (scene["noise_sigma"] > 0).all()
    for case in ("sparse", "stress"):
        mask = scene[f"mask_{case}"]
        assert mask.dtype == np.bool_
        assert int(mask.sum()) == manifest["cases"][case]["count"]
        np.testing.assert_array_equal(mask, scene[f"defect_kind_{case}"] != 0)
        np.testing.assert_array_equal(
            scene[f"observed_{case}"][~mask], scene["clean"][~mask]
        )
        assert np.all(mask | np.isfinite(scene[f"observed_{case}"]))
        for kind, value in ((2, -100), (3, np.nan), (4, np.inf), (5, -np.inf)):
            np.testing.assert_array_equal(
                scene[f"observed_{case}"][scene[f"defect_kind_{case}"] == kind], value
            )
        hot = scene[f"defect_kind_{case}"] == 1
        np.testing.assert_array_equal(
            scene[f"observed_{case}"][hot], scene["clean"][hot] + 500
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", ["sparse", "stress"])
def test_gold_linear_matches_independent_run_reference(
    scene: dict[str, np.ndarray],
    dtype: type[np.floating],
    case: str,
) -> None:
    array = scene[f"observed_{case}"].astype(dtype)
    mask = scene[f"mask_{case}"]
    original = array.copy()
    # The contract fixes arithmetic order; SciPy interp1d can round differently.
    expected = fixpix_reference(array, mask, use_scipy=False)
    actual = nf.fixpix(array, mask)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        actual[~mask].view(np.uint8), array[~mask].view(np.uint8)
    )
    np.testing.assert_array_equal(array.view(np.uint8), original.view(np.uint8))
    assert not np.shares_memory(actual, array)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", ["sparse", "stress"])
def test_gold_median_matches_independent_clipped_window_reference(
    scene: dict[str, np.ndarray],
    dtype: type[np.floating],
    case: str,
) -> None:
    array = scene[f"observed_{case}"].astype(dtype)
    mask = scene[f"mask_{case}"]
    original = array.copy()
    actual = nf.fixpix(
        array,
        mask,
        method="median",
        size=3,
        mode="constant",
        min_count=3,
        ignore_inf=True,
    )
    for y, x in np.argwhere(mask):
        rows = slice(max(0, y - 1), min(array.shape[0], y + 2))
        cols = slice(max(0, x - 1), min(array.shape[1], x + 2))
        patch = array[rows, cols]
        usable = (~mask[rows, cols]) & np.isfinite(patch)
        values = [float(value) for value in patch[usable]]
        expected = dtype(_median(values)) if len(values) >= 3 else dtype(np.nan)
        np.testing.assert_array_equal(actual[y, x], expected)
    np.testing.assert_array_equal(
        actual[~mask].view(np.uint8), array[~mask].view(np.uint8)
    )
    np.testing.assert_array_equal(array.view(np.uint8), original.view(np.uint8))
    assert actual.dtype == dtype and not np.shares_memory(actual, array)


@pytest.mark.parametrize("name", ["control_flat", "control_ramp"])
@pytest.mark.parametrize("method", ["linear", "median"])
def test_gold_controls_reconstruct_exact_interior_values(
    scene: dict[str, np.ndarray],
    name: str,
    method: str,
) -> None:
    truth = scene[name]
    mask = scene["control_mask"]
    corrupted = truth.copy()
    corrupted[mask] = np.nan
    np.testing.assert_array_equal(nf.fixpix(corrupted, mask, method=method), truth)
    np.testing.assert_array_equal(
        nf.median_filter(truth)[1:-1, 1:-1], truth[1:-1, 1:-1]
    )


def test_gold_filter_windows_match_literal_reference(
    scene: dict[str, np.ndarray],
) -> None:
    # Independent windows around a source, the step, and the bad-pixel cluster.
    for rows, cols in (
        (slice(44, 53), slice(53, 62)),
        (slice(117, 126), slice(155, 164)),
        (slice(173, 186), slice(38, 51)),
    ):
        clean = scene["clean"][rows, cols]
        noisy = scene["observed_stress"][rows, cols]
        for array, omit_nan in ((clean, False), (noisy, True)):
            expected = median_filter_reference(
                array,
                (3, 3),
                ("reflect", "reflect"),
                cval=0.0,
                omit_nan=omit_nan,
                ignore_inf=False,
                min_count=1,
            )
            actual = nf.nanmedian_filter(array) if omit_nan else nf.median_filter(array)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layer", ["clean", "observed_sparse", "observed_stress"])
def test_gold_mean_windows_match_exact_rational_reference(
    scene: dict[str, np.ndarray], dtype: type[np.floating], layer: str
) -> None:
    for rows, cols in (
        (slice(44, 53), slice(53, 62)),
        (slice(117, 126), slice(155, 164)),
        (slice(173, 186), slice(38, 51)),
    ):
        array = scene[layer][rows, cols].astype(dtype)
        original = array.copy()
        for function, omit_nan, ignore_inf in (
            (nf.mean_filter, False, False),
            (nf.nanmean_filter, True, False),
            (nf.nanmean_filter, True, True),
        ):
            options = dict(ignore_inf=ignore_inf, min_count=3) if omit_nan else {}
            expected = mean_filter_reference(
                array, (3, 3), ("reflect", "reflect"), omit_nan=omit_nan, **options
            )
            actual = function(array, **options)
            # Retained finite gold samples are positive apart from -100 dead
            # pixels, so these windows are well-conditioned. Four eps allow
            # compensated f64 sum rounding (~2u), division (u), and the final
            # dtype cast (u), with eps=2u; no fixed absolute floor hides error.
            np.testing.assert_allclose(
                actual, expected, rtol=4 * np.finfo(dtype).eps, atol=0
            )
            np.testing.assert_array_equal(array.view(np.uint8), original.view(np.uint8))


@pytest.mark.parametrize("name", ["control_flat", "control_ramp"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gold_mean_controls_preserve_exact_interior_values(
    scene: dict[str, np.ndarray], name: str, dtype: type[np.floating]
) -> None:
    array = scene[name].astype(dtype)
    for function in (nf.mean_filter, nf.nanmean_filter):
        np.testing.assert_array_equal(function(array)[1:-1, 1:-1], array[1:-1, 1:-1])
