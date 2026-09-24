"""Optional direct comparisons with the legacy astroimred implementation."""

from __future__ import annotations

import ndfiltrs as nf
import numpy as np
import pytest

air = pytest.importorskip("astroimred")


@pytest.mark.parametrize(
    ("values", "mask", "priority"),
    [
        (
            np.arange(7.0),
            np.asarray([True, True, False, False, False, False, False]),
            None,
        ),
        (
            np.arange(7.0),
            np.asarray([False, False, False, False, True, True, True]),
            (0,),
        ),
        (
            np.asarray(
                [
                    [3.0, 31.0, 5.0, 37.0, 9.0],
                    [41.0, 7.0, 43.0, 11.0, 47.0],
                    [13.0, 53.0, 17.0, 59.0, 19.0],
                    [61.0, 23.0, 67.0, 29.0, 71.0],
                    [73.0, 33.0, 79.0, 39.0, 83.0],
                ]
            ),
            np.asarray(
                [
                    [False, False, False, False, False],
                    [False, False, True, False, False],
                    [False, True, True, True, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                ]
            ),
            (1, 0),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fixpix_matches_air_fixpix(
    values: np.ndarray,
    mask: np.ndarray,
    priority: tuple[int, ...] | None,
    dtype: type[np.float32] | type[np.float64],
) -> None:
    array = np.asarray(values, dtype=dtype)
    bad = np.asarray(mask, dtype=np.int8)
    kwargs = {} if priority is None else {"priority": priority}
    legacy = air.fixpix(
        array.copy(),
        bad.copy(),
        update_header=False,
        verbose=False,
        **kwargs,
    )
    actual = nf.fixpix(array, bad, priority=priority)
    np.testing.assert_allclose(actual, legacy.data, rtol=0, atol=0, equal_nan=True)
