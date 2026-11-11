"""Savitzky-Golay filters against SciPy and a per-window NumPy reference."""

from __future__ import annotations

import ndfiltrs as nf
import numpy as np
import pytest
from scipy import ndimage
from scipy.signal import savgol_filter as scipy_savgol

from tests._savgol_reference import savgol_reference

MODES = ("interp", "mirror", "nearest", "wrap", "constant")


def assert_close(actual: np.ndarray, expected: np.ndarray, rtol: float = 1e-11) -> None:
    """Compare with a tolerance relative to the largest expected magnitude."""
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = ~np.isnan(expected)
    scale = max(1.0, float(np.max(np.abs(expected[finite]), initial=0.0)))
    np.testing.assert_allclose(
        actual[finite], expected[finite], rtol=0, atol=rtol * scale
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("window", [5, 7, 11])
def test_complete_1d_data_matches_scipy_for_every_derivative(
    mode: str, window: int
) -> None:
    line = np.random.default_rng(window).normal(size=37)
    for polyorder in range(min(window, 5)):
        for deriv in range(polyorder + 1):
            expected = scipy_savgol(
                line, window, polyorder, deriv, 0.7, mode=mode, cval=1.5
            )
            actual = nf.savgol_filter(
                line, window, polyorder, deriv, 0.7, mode=mode, cval=1.5
            )
            masked = nf.nansavgol_filter(
                line, window, polyorder, deriv, 0.7, mode=mode, cval=1.5
            )
            assert_close(actual, expected)
            np.testing.assert_array_equal(masked, actual)


@pytest.mark.parametrize("mode", ["interp", "mirror"])
def test_per_axis_basis_equals_scipy_applied_axis_by_axis(mode: str) -> None:
    image = np.random.default_rng(1).normal(size=(13, 16))
    actual = nf.savgol_filter(
        image,
        (7, 5),
        (3, 2),
        deriv=(1, 2),
        delta=(0.5, 2.0),
        basis="per_axis",
        mode=mode,
    )
    expected = scipy_savgol(image, 7, 3, 1, 0.5, axis=0, mode=mode)
    expected = scipy_savgol(expected, 5, 2, 2, 2.0, axis=1, mode=mode)
    assert_close(actual, expected)


def test_selected_axis_matches_scipy_default_mode() -> None:
    cube = np.random.default_rng(2).normal(size=(3, 4, 15))
    actual = nf.savgol_filter(cube, 9, 3, 1, axes=(-1,))
    assert_close(actual, scipy_savgol(cube, 9, 3, 1, axis=-1))


@pytest.mark.parametrize(
    ("shape", "window", "deriv", "modes"),
    [
        ((9, 11), (5, 7), (0, 0), ("interp", "reflect")),
        ((9, 11), (5, 5), (1, 1), ("mirror", "interp")),
        ((9, 11), (5, 3), (0, 2), ("constant", "wrap")),
        ((5, 6, 7), (3, 5, 5), (1, 0, 0), ("interp", "nearest", "interp")),
    ],
)
def test_complete_total_degree_matches_bruteforce(
    shape: tuple[int, ...],
    window: tuple[int, ...],
    deriv: tuple[int, ...],
    modes: tuple[str, ...],
) -> None:
    data = np.random.default_rng(3).normal(size=shape)
    delta = tuple(0.5 + axis for axis in range(len(shape)))
    actual = nf.savgol_filter(data, window, 2, deriv, delta, mode=modes, cval=0.25)
    expected = savgol_reference(data, window, "total", 2, deriv, delta, modes, 0.25)
    assert_close(actual, expected, 1e-10)


@pytest.mark.parametrize("basis", ["total", "per_axis"])
@pytest.mark.parametrize(
    ("modes", "cval"),
    [
        (("interp", "interp"), 0.0),
        (("reflect", "wrap"), 0.0),
        (("constant", "constant"), np.nan),
    ],
)
def test_masked_2d_matches_bruteforce(
    basis: str, modes: tuple[str, str], cval: float
) -> None:
    rng = np.random.default_rng(4)
    data = rng.normal(size=(10, 12))
    data[rng.random(data.shape) < 0.25] = np.nan
    weights = rng.uniform(0.2, 2.0, size=(5, 5))
    weights[0, 4] = 0.0
    for deriv in [(0, 0), (1, 0), (1, 1)]:
        actual = nf.nansavgol_filter(
            data,
            5,
            2,
            deriv,
            (1.0, 0.5),
            basis=basis,
            weights=weights,
            mode=modes,
            cval=cval,
        )
        expected = savgol_reference(
            data,
            (5, 5),
            basis,
            2 if basis == "total" else (2, 2),
            deriv,
            (1.0, 0.5),
            modes,
            cval,
            min_count=6 if basis == "total" else 9,
            weights=weights,
        )
        propagated = nf.savgol_filter(
            data,
            5,
            2,
            deriv,
            (1.0, 0.5),
            basis=basis,
            weights=weights,
            mode=modes,
            cval=cval,
        )
        # Most outputs need a per-window solve.
        assert np.isfinite(actual[np.isnan(propagated)]).mean() > 0.5
        assert_close(actual, expected, 1e-9)


def test_masked_3d_matches_bruteforce() -> None:
    rng = np.random.default_rng(5)
    data = rng.normal(size=(6, 7, 8))
    data[rng.random(data.shape) < 0.2] = np.nan
    actual = nf.nansavgol_filter(data, (3, 5, 5), 2, (0, 0, 1), min_count=12)
    expected = savgol_reference(
        data,
        (3, 5, 5),
        "total",
        2,
        (0, 0, 1),
        (1.0,) * 3,
        ("interp",) * 3,
        min_count=12,
    )
    assert_close(actual, expected, 1e-9)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_masked_fit_reproduces_cubic_surfaces_and_derivatives(dtype: type) -> None:
    y, x = np.mgrid[0:14, 0:15].astype(np.float64)
    y, x = (y - 7) / 7, (x - 7) / 7
    surface = 1 + 2 * x - y + 0.5 * x * y - 0.3 * x**3 + 0.2 * y**2 * x
    derivatives = {
        (0, 0): surface,
        (0, 1): 2 + 0.5 * y - 0.9 * x**2 + 0.2 * y**2,
        (1, 1): 0.5 + 0.4 * y,
        (0, 2): -1.8 * x,
    }
    data = surface.astype(dtype)
    data[np.random.default_rng(6).random(data.shape) < 0.3] = np.nan
    rtol = 1e-9 if dtype == np.float64 else 2e-4
    for deriv, expected in derivatives.items():
        actual = nf.nansavgol_filter(data, 7, 3, deriv, delta=1 / 7)
        assert actual.dtype == dtype
        finite = np.isfinite(actual)
        assert finite.mean() > 0.9
        assert_close(actual[finite].astype(np.float64), expected[finite], rtol)


def test_too_few_or_degenerate_samples_give_nan() -> None:
    line = np.arange(9.0) ** 2
    line[3:6] = np.nan
    # Window 5 around index 4 keeps two samples: fewer than three terms.
    result = nf.nansavgol_filter(line, 5, 2, mode="mirror")
    assert np.isnan(result[4]) and np.isfinite(result[[0, 1, 7, 8]]).all()
    assert np.isnan(nf.nansavgol_filter(line, 5, 1, mode="mirror", min_count=3)[4])
    assert np.isfinite(nf.nansavgol_filter(line, 5, 1, mode="mirror")[4])

    image = np.full((5, 5), np.nan)
    image[2] = np.arange(5.0)
    # All retained samples lie on one row: the y slope is undetermined.
    collinear = nf.nansavgol_filter(image, 3, 1, mode="constant", cval=np.nan)
    assert np.isnan(collinear).all()
    image[1, 2] = 0.0
    determined = nf.nansavgol_filter(image, 3, 1, mode="constant", cval=np.nan)
    assert determined[2, 2] == pytest.approx(2.0)


def test_nan_propagates_through_the_window_footprint() -> None:
    image = np.random.default_rng(7).normal(size=(9, 9))
    image[4, 5] = np.nan
    result = nf.savgol_filter(image, (3, 5), 2, mode="mirror")
    footprint = np.zeros_like(image, dtype=bool)
    footprint[3:6, 3:8] = True
    np.testing.assert_array_equal(np.isnan(result), footprint)
    assert np.isfinite(nf.nansavgol_filter(image, (3, 5), 2, mode="mirror")).all()


def test_mask_and_infinity_omission_match_nan_input() -> None:
    rng = np.random.default_rng(8)
    image = rng.normal(size=(8, 9))
    mask = rng.random(image.shape) < 0.2
    expected = nf.nansavgol_filter(np.where(mask, np.nan, image), 5, 2)
    np.testing.assert_array_equal(nf.nansavgol_filter(image, 5, 2, mask=mask), expected)
    infinite = np.where(mask, np.inf, image)
    np.testing.assert_array_equal(
        nf.nansavgol_filter(infinite, 5, 2, ignore_inf=True), expected
    )
    assert np.isinf(nf.nansavgol_filter(infinite, 5, 0)).any()


def test_weights_follow_the_order_of_axes() -> None:
    rng = np.random.default_rng(9)
    image = rng.normal(size=(9, 10))
    weights = rng.uniform(0.5, 1.5, size=(5, 3))
    swapped = nf.savgol_filter(image, (5, 3), 1, axes=(1, 0), weights=weights)
    ordered = nf.savgol_filter(image, (3, 5), 1, axes=(0, 1), weights=weights.T)
    np.testing.assert_array_equal(swapped, ordered)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"window_length": 4, "polyorder": 2}, ValueError),
        ({"window_length": 5, "polyorder": 5}, ValueError),
        ({"window_length": 5, "polyorder": 2, "deriv": 3}, ValueError),
        ({"window_length": 5, "polyorder": (2, 2)}, TypeError),
        ({"window_length": 5, "polyorder": 2, "basis": "tensor"}, ValueError),
        ({"window_length": 5, "polyorder": 2, "delta": 0.0}, ValueError),
        ({"window_length": 5, "polyorder": 2, "mode": "grid-wrap"}, ValueError),
        ({"window_length": 21, "polyorder": 2}, ValueError),
        ({"window_length": 5, "polyorder": 2, "weights": np.ones(4)}, ValueError),
        ({"window_length": 5, "polyorder": 2, "weights": -np.ones(5)}, ValueError),
        ({"window_length": 5, "polyorder": 2, "min_count": 6}, ValueError),
        ({"window_length": 5, "polyorder": 2, "min_count": 0}, ValueError),
        ({"window_length": 5, "polyorder": 2, "mask": np.zeros(9)}, TypeError),
        ({"window_length": 5, "polyorder": 2, "ignore_inf": 1}, TypeError),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict[str, object], error: type) -> None:
    with pytest.raises(error):
        nf.nansavgol_filter(np.ones(9), **kwargs)


def test_empty_and_noncontiguous_inputs() -> None:
    empty = nf.savgol_filter(np.empty((0, 5), dtype=np.float32), 3, 1)
    assert empty.shape == (0, 5) and empty.dtype == np.float32
    image = np.random.default_rng(10).normal(size=(12, 14))
    view = image[::2, ::-1]
    np.testing.assert_array_equal(
        nf.savgol_filter(view, 3, 1), nf.savgol_filter(view.copy(), 3, 1)
    )


@pytest.mark.parametrize(("basis", "terms"), [("total", 6), ("per_axis", 9)])
def test_sparse_windows_match_bruteforce_rank_decisions(basis: str, terms: int) -> None:
    rng = np.random.default_rng(11)
    data = rng.normal(size=(16, 16))
    data[rng.random(data.shape) < 0.7] = np.nan
    degree = 2 if basis == "total" else (2, 2)
    actual = nf.nansavgol_filter(
        data, 5, 2, basis=basis, mode="constant", cval=np.nan, min_count=1
    )
    expected = savgol_reference(
        data, (5, 5), basis, degree, (0, 0), (1.0, 1.0), ("constant",) * 2, np.nan
    )
    retained = ndimage.correlate(
        np.isfinite(data).astype(float), np.ones((5, 5)), mode="constant"
    )
    # Windows with enough samples whose geometry leaves the fit undetermined.
    assert np.count_nonzero(np.isnan(expected) & (retained >= terms)) > 0
    assert np.isfinite(expected).any()
    assert_close(actual, expected, 1e-8)
