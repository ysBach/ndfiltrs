"""Self-tests for the independent reference implementation."""

import numpy as np

from tests._reference import median_filter_reference


def test_reference_nearest_one_dimensional_fixture() -> None:
    values = np.array([1.0, 9.0, 2.0], dtype=np.float64)
    result = median_filter_reference(
        values,
        (3,),
        ("nearest",),
        cval=0.0,
        omit_nan=False,
        ignore_inf=False,
        min_count=1,
    )
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 2.0]))


def test_reference_constant_two_dimensional_fixture() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = median_filter_reference(
        values,
        (3, 3),
        ("constant", "constant"),
        cval=10.0,
        omit_nan=False,
        ignore_inf=False,
        min_count=1,
    )
    np.testing.assert_array_equal(result, np.full((2, 2), 10.0, dtype=np.float32))


def test_reference_preserves_unfiltered_axis_in_three_dimensions() -> None:
    values = np.array([[[1.0]], [[9.0]], [[2.0]]], dtype=np.float64)
    result = median_filter_reference(
        values,
        (3, 1, 1),
        ("nearest", "reflect", "wrap"),
        cval=0.0,
        omit_nan=False,
        ignore_inf=False,
        min_count=1,
    )
    np.testing.assert_array_equal(result[:, 0, 0], np.array([1.0, 2.0, 2.0]))
