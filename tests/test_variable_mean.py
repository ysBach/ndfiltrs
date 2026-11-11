"""Per-pixel window means against literal windows with rational sums."""

from __future__ import annotations

import math
from fractions import Fraction

import ndfiltrs as nf
import numpy as np
import pytest

from tests._reference import MODES, _map_index

DTYPES = (np.float32, np.float64)


def _reference(
    array: np.ndarray,
    sizes: np.ndarray,
    modes: tuple[str, ...],
    *,
    cval: float = 0.0,
    omit_nan: bool = False,
    ignore_inf: bool = False,
    min_count: int = 1,
) -> np.ndarray:
    """Average every pixel's own window with exact rational sums.

    `sizes` has shape ``(ndim, *array.shape)``. The rational mean is rounded
    to `float64` and then cast to the array dtype.
    """
    output = np.empty(array.shape, dtype=np.float64)
    constant = float(array.dtype.type(cval))
    for center in np.ndindex(array.shape):
        window = tuple(int(sizes[(axis, *center)]) for axis in range(array.ndim))
        values = []
        for position in np.ndindex(window):
            mapped = [
                _map_index(c + k - n // 2, length, boundary)
                for c, k, n, length, boundary in zip(
                    center, position, window, array.shape, modes, strict=True
                )
            ]
            value = constant if None in mapped else float(array[tuple(mapped)])
            if (omit_nan and math.isnan(value)) or (ignore_inf and math.isinf(value)):
                continue
            values.append(value)
        if (
            len(values) < min_count
            or any(math.isnan(value) for value in values)
            or (math.inf in values and -math.inf in values)
        ):
            output[center] = math.nan
        elif math.inf in values or -math.inf in values:
            output[center] = math.inf if math.inf in values else -math.inf
        else:
            output[center] = float(sum(map(Fraction, values), Fraction()) / len(values))
    return output.astype(array.dtype)


def _random_array(
    rng: np.random.Generator, shape: tuple[int, ...], dtype: type, special: bool
) -> np.ndarray:
    array = rng.uniform(-100.0, 100.0, shape).astype(dtype)
    if special:
        flat = array.reshape(-1)
        picks = rng.choice(flat.size, size=min(3, flat.size), replace=False)
        for pick, value in zip(picks, (np.nan, np.inf, -np.inf), strict=False):
            flat[pick] = value
    return array


def _assert_close(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    rtol = 1e-6 if actual.dtype == np.float32 else 1e-12
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol * 100)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1,), (7,), (4, 5), (1, 6), (3, 2, 4)])
@pytest.mark.parametrize("per_axis", [False, True])
@pytest.mark.parametrize("special", [False, True])
def test_matches_literal_windows(
    dtype: type, shape: tuple[int, ...], per_axis: bool, special: bool
) -> None:
    rng = np.random.default_rng(abs(hash((shape, per_axis, special))) % 2**32)
    array = _random_array(rng, shape, dtype, special)
    ndim = len(shape)
    map_shape = (ndim, *shape) if per_axis else shape
    # Sizes up to twice the longest axis exercise windows wider than the array.
    size = rng.integers(1, 2 * max(shape) + 4, size=map_shape)
    full = size if per_axis else np.broadcast_to(size, (ndim, *shape))
    mixed = tuple(MODES[(axis + 2) % len(MODES)] for axis in range(ndim))
    for modes in [(mode,) * ndim for mode in MODES] + [mixed]:
        for cval in (2.5, np.nan):
            expected = _reference(array, full, modes, cval=cval)
            actual = nf.variable_mean_filter(array, size, mode=modes, cval=cval)
            _assert_close(actual, expected)
            for ignore_inf, min_count in ((False, 1), (True, 3)):
                expected = _reference(
                    array,
                    full,
                    modes,
                    cval=cval,
                    omit_nan=True,
                    ignore_inf=ignore_inf,
                    min_count=min_count,
                )
                actual = nf.nanvariable_mean_filter(
                    array,
                    size,
                    mode=modes,
                    cval=cval,
                    ignore_inf=ignore_inf,
                    min_count=min_count,
                )
                _assert_close(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("window", [(1, 1), (3, 3), (4, 2), (5, 8)])
def test_constant_map_matches_fixed_window_mean(
    dtype: type, mode: str, window: tuple[int, int]
) -> None:
    rng = np.random.default_rng(7)
    array = _random_array(rng, (6, 7), dtype, special=True)
    size = np.stack([np.full(array.shape, length) for length in window])
    expected = nf.nanmean_filter(array, window, mode=mode, cval=1.5, method="boxcar")
    actual = nf.nanvariable_mean_filter(array, size, mode=mode, cval=1.5)
    _assert_close(actual, expected)


def test_radius_equals_odd_size() -> None:
    rng = np.random.default_rng(3)
    array = rng.normal(size=(9, 8))
    radius = rng.integers(0, 6, size=(2, 9, 8))
    np.testing.assert_array_equal(
        nf.variable_mean_filter(array, radius=radius, mode="mirror"),
        nf.variable_mean_filter(array, 2 * radius + 1, mode="mirror"),
    )


def test_window_below_min_count_is_nan() -> None:
    array = np.arange(5.0)
    out = nf.nanvariable_mean_filter(array, np.array([1, 3, 5, 3, 1]), min_count=3)
    assert np.isnan(out[[0, 4]]).all()
    np.testing.assert_allclose(out[1:4], [1.0, 2.0, 3.0])


def test_accuracy_on_a_bright_image_stays_within_the_documented_bound() -> None:
    rng = np.random.default_rng(11)
    array = 1.0e4 + rng.normal(scale=10.0, size=(600, 700))
    size = np.full(array.shape, 3)
    expected = nf.mean_filter(array, 3, method="boxcar")
    actual = nf.variable_mean_filter(array, size)
    # Documented first-order bound on a window sum, divided by the count 9.
    bound = 2.0**-53 * 2**array.ndim * array.ndim * np.abs(array).sum() / 9
    assert np.abs(actual - expected).max() < bound


def test_empty_array_keeps_shape() -> None:
    array = np.empty((0, 3), dtype=np.float32)
    out = nf.variable_mean_filter(array, np.ones((0, 3), dtype=np.int64))
    assert out.shape == (0, 3)
    assert out.dtype == np.float32


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({}, ValueError),
        ({"size": np.ones((3, 4), int), "radius": np.ones((3, 4), int)}, ValueError),
        ({"size": np.ones((3, 4))}, TypeError),
        ({"size": np.ones((3, 4), bool)}, TypeError),
        ({"size": np.ma.masked_array(np.ones((3, 4), int))}, TypeError),
        ({"size": np.ones((4, 3), int)}, ValueError),
        ({"size": np.ones((3, 3, 4), int)}, ValueError),
        ({"size": np.zeros((3, 4), int)}, ValueError),
        ({"radius": -np.ones((3, 4), int)}, ValueError),
        ({"radius": np.full((3, 4), 2**63, np.uint64)}, OverflowError),
        ({"size": np.full((3, 4), 2**40, np.uint64)}, OverflowError),
        ({"size": np.ones((3, 4), int), "mode": "edge"}, ValueError),
        ({"size": np.ones((3, 4), int), "cval": True}, TypeError),
    ],
)
def test_rejects_invalid_arguments(kwargs: dict, error: type) -> None:
    with pytest.raises(error):
        nf.variable_mean_filter(np.zeros((3, 4)), **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"min_count": 0}, ValueError),
        ({"min_count": True}, TypeError),
        ({"ignore_inf": 1}, TypeError),
    ],
)
def test_nan_variant_rejects_invalid_policy(kwargs: dict, error: type) -> None:
    with pytest.raises(error):
        nf.nanvariable_mean_filter(np.zeros(4), np.ones(4, int), **kwargs)
