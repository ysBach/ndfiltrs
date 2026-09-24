"""Parity and validation tests for N-D discrete derivative operators."""

from __future__ import annotations

import numpy as np
import pytest
from ndfiltrs._derivatives import laplace, prewitt, sobel
from scipy import ndimage as ndi

MODES = ("reflect", "constant", "nearest", "mirror", "wrap")
DTYPES = (np.float32, np.float64)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_sobel_parity_with_scipy(
    dtype: type[np.floating], mode: str, ndim: int
) -> None:
    shape = (5, 6, 7)[:ndim]
    rng = np.random.default_rng(42)
    data = rng.standard_normal(shape).astype(dtype)
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    atol = 1e-5 if dtype == np.float32 else 1e-12

    for axis in range(-ndim, ndim):
        actual = sobel(data, axis=axis, mode=mode, cval=0.0)
        expected = ndi.sobel(data, axis=axis, mode=mode, cval=0.0)
        assert actual.dtype == dtype
        assert actual.shape == shape
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_prewitt_parity_with_scipy(
    dtype: type[np.floating], mode: str, ndim: int
) -> None:
    shape = (5, 6, 7)[:ndim]
    rng = np.random.default_rng(43)
    data = rng.standard_normal(shape).astype(dtype)
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    atol = 1e-5 if dtype == np.float32 else 1e-12

    for axis in range(-ndim, ndim):
        actual = prewitt(data, axis=axis, mode=mode, cval=0.0)
        expected = ndi.prewitt(data, axis=axis, mode=mode, cval=0.0)
        assert actual.dtype == dtype
        assert actual.shape == shape
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("cval", [0.0, 2.5])
def test_laplace_parity_with_scipy(
    dtype: type[np.floating], mode: str, ndim: int, cval: float
) -> None:
    shape = (5, 6, 7)[:ndim]
    rng = np.random.default_rng(44)
    data = rng.standard_normal(shape).astype(dtype)
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    atol = 1e-5 if dtype == np.float32 else 1e-12

    actual = laplace(data, mode=mode, cval=cval)
    expected = ndi.laplace(data, mode=mode, cval=cval)
    assert actual.dtype == dtype
    assert actual.shape == shape
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_1d_discrete_derivative_values() -> None:
    data = np.array([10.0, 20.0, 40.0, 70.0, 110.0], dtype=np.float64)
    # 1-D Sobel and Prewitt reduce to [-1, 0, 1] central difference
    # at index 1: data[2] - data[0] = 40 - 10 = 30
    # at index 2: data[3] - data[1] = 70 - 20 = 50
    # at index 3: data[4] - data[2] = 110 - 40 = 70
    s = sobel(data, mode="reflect")
    p = prewitt(data, mode="reflect")
    np.testing.assert_allclose(s[1:4], [30.0, 50.0, 70.0])
    np.testing.assert_allclose(p[1:4], [30.0, 50.0, 70.0])
    np.testing.assert_array_equal(s, p)

    # 1-D Laplace reduces to [1, -2, 1] second difference
    # at index 2: data[1] - 2*data[2] + data[3] = 20 - 80 + 70 = 10
    lap = laplace(data, mode="reflect")
    assert lap[2] == pytest.approx(10.0)


def test_axis_validation() -> None:
    arr = np.zeros((3, 4), dtype=np.float64)

    with pytest.raises(TypeError, match="axis must be an integer, not a boolean"):
        sobel(arr, axis=True)
    with pytest.raises(TypeError, match="axis must be an integer, not a boolean"):
        prewitt(arr, axis=False)
    with pytest.raises(TypeError, match="axis must be an integer"):
        sobel(arr, axis="0")
    with pytest.raises(ValueError, match="out of bounds"):
        sobel(arr, axis=2)
    with pytest.raises(ValueError, match="out of bounds"):
        sobel(arr, axis=-3)
    with pytest.raises(ValueError, match="out of bounds"):
        prewitt(arr, axis=2)
    with pytest.raises(ValueError, match="out of bounds"):
        prewitt(arr, axis=-3)


def test_array_validation() -> None:
    scalar = np.array(5.0)
    with pytest.raises(ValueError, match="at least one dimension"):
        sobel(scalar)
    with pytest.raises(ValueError, match="at least one dimension"):
        prewitt(scalar)
    with pytest.raises(ValueError, match="at least one dimension"):
        laplace(scalar)

    int_arr = np.arange(9).reshape(3, 3)
    with pytest.raises(TypeError, match="float32 or float64"):
        sobel(int_arr)
    with pytest.raises(TypeError, match="float32 or float64"):
        prewitt(int_arr)
    with pytest.raises(TypeError, match="float32 or float64"):
        laplace(int_arr)


def test_nan_propagation() -> None:
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float64)
    # 1D kernel: [-1, 0, 1]
    # index 0: reflect boundary [-1, 0, 1] on [2, 1, 2] -> 0 (finite)
    # index 1: [-1, 0, 1] on [1, 2, nan] -> nan
    # index 2: center 0 omits NaN in correlate -> -2 + 0*nan + 4 = 2.0
    # index 3: [-1, 0, 1] on [nan, 4, 5] -> nan
    # index 4: reflect boundary [4, 5, 4] -> 0 (finite)
    s = sobel(arr, mode="reflect")
    assert not np.isnan(s[0])
    assert np.isnan(s[1])
    assert s[2] == pytest.approx(2.0)
    assert np.isnan(s[3])
    assert not np.isnan(s[4])

    p = prewitt(arr, mode="reflect")
    np.testing.assert_array_equal(np.isnan(p), np.isnan(s))
    assert p[2] == pytest.approx(2.0)

    # 1D Laplace kernel: [1, -2, 1]
    # At index 2: [1, -2, 1] on [2, nan, 4] has nonzero weight -2 on center
    lap = laplace(arr, mode="reflect")
    assert np.isnan(lap[1])
    assert np.isnan(lap[2])
    assert np.isnan(lap[3])
