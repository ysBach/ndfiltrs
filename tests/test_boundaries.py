"""Differential boundary tests against SciPy for overlapping semantics."""

import ndfiltrs as nf
import numpy as np
import pytest
from scipy import ndimage


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
def test_finite_odd_windows_match_scipy(dtype: np.dtype, mode: str) -> None:
    values = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    expected = ndimage.median_filter(values, size=(3, 1, 5), mode=mode, cval=-7.0)
    actual = nf.median_filter(values, size=(3, 1, 5), mode=mode, cval=-7.0)
    np.testing.assert_array_equal(actual, expected)
