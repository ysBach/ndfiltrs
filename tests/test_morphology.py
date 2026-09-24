"""Grey-scale morphology with flat structuring elements, checked against SciPy."""

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
CVAL = -0.75

# ndfiltrs function and its SciPy equivalent; erosion is min_filter.
OPERATIONS: list[tuple[Callable[..., NDArray[np.floating]], Callable[..., Any]]] = [
    (nf.min_filter, ndimage.grey_erosion),
    (nf.grey_dilation, ndimage.grey_dilation),
    (nf.grey_opening, ndimage.grey_opening),
    (nf.grey_closing, ndimage.grey_closing),
    (nf.white_tophat, ndimage.white_tophat),
    (nf.black_tophat, ndimage.black_tophat),
    (nf.morphological_gradient, ndimage.morphological_gradient),
    (nf.morphological_laplace, ndimage.morphological_laplace),
]

# Array shape and window: either an integer/tuple size or a footprint array.
WINDOWS: dict[str, tuple[tuple[int, ...], dict[str, Any]]] = {
    "1d-size-odd": ((17,), {"size": 5}),
    "1d-footprint-even": ((17,), {"footprint": np.array([1, 0, 1, 1])}),
    "2d-size-odd": ((9, 10), {"size": 3}),
    "2d-size-even": ((9, 10), {"size": (2, 4)}),
    "2d-footprint-odd": (
        (9, 10),
        {"footprint": np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]])},
    ),
    "2d-footprint-even": (
        (9, 10),
        {"footprint": np.array([[1, 0, 1, 1], [0, 1, 1, 0]])},
    ),
    "3d-footprint-mixed": (
        (5, 6, 7),
        {"footprint": np.array([[[1, 0, 1], [0, 1, 1]], [[1, 1, 0], [0, 0, 1]]])},
    ),
}
ORIGINS = ("zero", "low", "high", "mixed")


def _window_shape(shape: tuple[int, ...], window: dict[str, Any]) -> tuple[int, ...]:
    if "footprint" in window:
        return window["footprint"].shape
    return tuple(np.broadcast_to(window["size"], (len(shape),)).tolist())


def _origin(lengths: tuple[int, ...], kind: str) -> tuple[int, ...]:
    low = tuple(-(length // 2) for length in lengths)
    high = tuple((length - 1) // 2 for length in lengths)
    return {
        "zero": (0,) * len(lengths),
        "low": low,
        "high": high,
        "mixed": tuple(high[i] if i % 2 else low[i] for i in range(len(lengths))),
    }[kind]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("origin_kind", ORIGINS)
@pytest.mark.parametrize("name", WINDOWS)
@pytest.mark.parametrize(("function", "reference"), OPERATIONS)
def test_morphology_matches_scipy(
    function: Callable[..., NDArray[np.floating]],
    reference: Callable[..., Any],
    name: str,
    origin_kind: str,
    mode: str,
    dtype: type[np.floating],
) -> None:
    shape, window = WINDOWS[name]
    values = np.random.default_rng(len(name)).normal(size=shape).astype(dtype)
    origin = _origin(_window_shape(shape, window), origin_kind)
    kwargs = {**window, "origin": origin, "mode": mode, "cval": CVAL}
    actual = function(values, **kwargs)
    expected = reference(values, **kwargs)
    assert actual.dtype == dtype
    if function is nf.morphological_laplace:
        # SciPy forms (d + e) - x - x; ndfiltrs forms d + e - 2x.
        rtol = 1e-6 if dtype == np.float32 else 1e-14
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol)
    else:
        np.testing.assert_array_equal(actual, expected)


def _any_nan(
    indicator: NDArray[np.bool_], dilate: bool, **kwargs: Any
) -> NDArray[np.bool_]:
    """Return where a window (reflected for dilation) touches a flagged pixel."""
    spread = ndimage.grey_dilation if dilate else ndimage.maximum_filter
    return spread(indicator.astype(np.float64), **kwargs, cval=0.0) > 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "name", ["1d-footprint-even", "2d-size-even", "2d-footprint-odd"]
)
def test_nan_propagates_through_each_stage(
    name: str, mode: str, dtype: type[np.floating]
) -> None:
    shape, window = WINDOWS[name]
    values = np.random.default_rng(3).normal(size=shape).astype(dtype)
    values.flat[values.size // 2] = np.nan
    kwargs = {
        **window,
        "origin": _origin(_window_shape(shape, window), "mixed"),
        "mode": mode,
    }
    nan = np.isnan(values)
    eroded = _any_nan(nan, False, **kwargs)
    dilated = _any_nan(nan, True, **kwargs)
    opened = _any_nan(eroded, True, **kwargs)
    closed = _any_nan(dilated, False, **kwargs)
    expected = {
        nf.min_filter: eroded,
        nf.grey_dilation: dilated,
        nf.grey_opening: opened,
        nf.grey_closing: closed,
        nf.white_tophat: opened | nan,
        nf.black_tophat: closed | nan,
        nf.morphological_gradient: dilated | eroded,
        nf.morphological_laplace: dilated | eroded | nan,
    }
    for function, mask in expected.items():
        np.testing.assert_array_equal(
            np.isnan(function(values, cval=CVAL, **kwargs)), mask
        )


@pytest.mark.parametrize("function", [function for function, _ in OPERATIONS])
@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"size": 3, "footprint": np.ones(3)}, ValueError),
        ({"footprint": np.zeros(3)}, ValueError),
        ({"footprint": np.ones((3, 3))}, ValueError),
        ({"size": 3, "origin": 2}, ValueError),
        ({"size": 4, "origin": -3}, ValueError),
        ({"size": 3, "origin": True}, TypeError),
        ({"size": 3, "mode": "edge"}, ValueError),
    ],
)
def test_invalid_windows_are_rejected(
    function: Callable[..., Any], kwargs: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        function(np.arange(8.0), **kwargs)


@pytest.mark.parametrize("function", [function for function, _ in OPERATIONS])
def test_integer_input_is_rejected(function: Callable[..., Any]) -> None:
    with pytest.raises(TypeError, match="dtype"):
        function(np.arange(8))
