"""Public API tests for Rust-backed pixel repair."""

from __future__ import annotations

import inspect
import itertools
import operator

import ndfiltrs as nf
import numpy as np
import pytest

from tests._fixpix_reference import fixpix_reference


class IndexValue:
    """Object accepted through Python's integer-index protocol."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


def _all_masks(shape: tuple[int, ...]) -> list[np.ndarray]:
    """Return every boolean mask for a small shape in C-order bit order."""
    size = int(np.prod(shape))
    return [
        np.asarray([(bits >> index) & 1 for index in range(size)], dtype=bool).reshape(
            shape
        )
        for bits in range(1 << size)
    ]


def _assert_matches_reference(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    exact: bool = False,
) -> None:
    """Compare output while treating NaNs as equal."""
    if exact:
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
    else:
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0,
            atol=0,
            equal_nan=True,
        )


def test_public_signature_is_explicit() -> None:
    signature = inspect.signature(nf.fixpix)
    assert list(signature.parameters) == [
        "array",
        "mask",
        "method",
        "priority",
        "size",
        "footprint",
        "kernel",
        "origin",
        "mode",
        "cval",
        "ignore_inf",
        "min_count",
        "copy",
    ]
    for name in list(signature.parameters)[2:]:
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["mask"].default is None
    assert signature.parameters["priority"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["priority"].default is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mask", _all_masks((2, 3)), ids=lambda value: "mask")
def test_exhaustive_small_2d_masks(
    dtype: type[np.float32] | type[np.float64], mask: np.ndarray
) -> None:
    """Every mask on a 2-by-3 array follows the original-run reference."""
    values = np.asarray([[0.25, 1.5, 9.75], [2.0, 6.25, 11.5]], dtype=dtype)
    for priority in (None, (0, 1), (1, 0)):
        expected = fixpix_reference(values, mask, priority=priority)
        actual = nf.fixpix(values, mask, priority=priority)
        _assert_matches_reference(actual, expected, exact=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numeric_masks_are_cast_to_bool(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    values = np.asarray([1.0, 0.0, 5.0, 9.0], dtype=dtype)
    numeric_mask = np.asarray([0, 2, -1, 0], dtype=np.int16)
    expected = fixpix_reference(values, numeric_mask)
    actual = nf.fixpix(values, numeric_mask)
    _assert_matches_reference(actual, expected, exact=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_seeded_random_nd_inputs_match_all_priority_permutations(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    """Seeded rank-one through rank-four cases exercise every axis ordering."""
    rng = np.random.default_rng(20260906)
    cases = [
        (1, (7,)),
        (2, (4, 5)),
        (3, (3, 4, 2)),
        (4, (2, 3, 2, 3)),
    ]
    for _, shape in cases:
        values = rng.normal(size=shape).astype(dtype)
        mask = rng.random(shape) < 0.38
        for priority in itertools.permutations(range(len(shape))):
            expected = fixpix_reference(values, mask, priority=priority)
            actual = nf.fixpix(values, mask, priority=priority)
            _assert_matches_reference(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_crossing_runs_replay_the_whole_span_in_original_c_order(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    """Crossing spans retain repeated writes from each original bad pixel."""
    values = np.asarray(
        [
            [3.0, 31.0, 5.0, 37.0, 9.0],
            [41.0, 7.0, 43.0, 11.0, 47.0],
            [13.0, 53.0, 17.0, 59.0, 19.0],
            [61.0, 23.0, 67.0, 29.0, 71.0],
            [73.0, 33.0, 79.0, 39.0, 83.0],
        ],
        dtype=dtype,
    )
    mask = np.zeros(values.shape, dtype=bool)
    mask[2, 1:4] = True
    mask[1:4, 2] = True
    expected = fixpix_reference(values, mask)
    actual = nf.fixpix(values, mask)
    _assert_matches_reference(actual, expected, exact=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_last_occurrence_of_repeated_span_wins_over_crossing_run(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    """The final horizontal rewrite must replace the vertical center result."""
    values = np.array(
        [[10, 20, 30, 40], [100, 999, 300, 400], [50, 60, 70, 80], [90, 100, 110, 120]],
        dtype=dtype,
    )
    mask = np.array(
        [[1, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0]],
        dtype=bool,
    )
    expected = np.array(
        [[20, 20, 30, 40], [400, 400, 400, 400], [60, 60, 70, 80], [90, 100, 110, 120]],
        dtype=dtype,
    )
    np.testing.assert_array_equal(nf.fixpix(values, mask), expected)
    np.testing.assert_array_equal(fixpix_reference(values, mask), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_edges_and_full_axes_follow_legacy_endpoint_rules(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    cases = [
        ([True, True, False, False, False], [2.0, 2.0, 2.0, 3.0, 4.0]),
        ([False, False, False, True, True], [0.0, 1.0, 2.0, 2.0, 2.0]),
        ([True, True, True, True, True], [0.0, 1.0, 2.0, 3.0, 4.0]),
        ([False, False, False, False, False], [0.0, 1.0, 2.0, 3.0, 4.0]),
    ]
    values = np.arange(5, dtype=dtype)
    for mask_values, expected_values in cases:
        mask = np.asarray(mask_values, dtype=bool)
        expected = fixpix_reference(values, mask)
        actual = nf.fixpix(values, mask)
        np.testing.assert_array_equal(actual, np.asarray(expected_values, dtype=dtype))
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_strided_array_and_mask_are_accepted_without_mutation(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    data_storage = np.arange(4 * 8 * 10, dtype=dtype).reshape(4, 8, 10)
    mask_storage = np.zeros((4, 8, 10), dtype=np.int8)
    mask_storage[1, 2:6, 3:8] = 1
    values = data_storage[::-1, ::2, 1::2]
    mask = mask_storage[::-1, ::2, 1::2]
    values_before = values.copy()
    mask_before = mask.copy()
    expected = fixpix_reference(values, mask, priority=(2, 0, 1))

    actual = nf.fixpix(values, mask, priority=[2, 0, 1])

    _assert_matches_reference(actual, expected, exact=True)
    assert actual.flags.c_contiguous
    assert actual.flags.aligned
    assert actual.flags.owndata
    assert actual.flags.writeable
    assert actual.base is None
    np.testing.assert_array_equal(values, values_before)
    np.testing.assert_array_equal(mask, mask_before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_no_mask_and_empty_inputs_return_owned_copies(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    values = np.arange(6, dtype=dtype).reshape(2, 3)
    for actual in (
        nf.fixpix(values),
        nf.fixpix(values, np.zeros_like(values, dtype=bool)),
    ):
        np.testing.assert_array_equal(actual, values)
        assert actual is not values
        assert actual.flags.c_contiguous
        assert actual.flags.owndata
        assert actual.flags.writeable
        assert actual.base is None

    empty = np.empty((0, 3, 2), dtype=dtype)
    empty_mask = np.empty(empty.shape, dtype=bool)
    actual = nf.fixpix(empty, empty_mask)
    assert actual.shape == empty.shape
    assert actual.dtype == dtype
    assert actual.size == 0
    assert actual.flags.c_contiguous
    assert actual.flags.owndata
    assert actual.base is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unmasked_nan_is_not_implicitly_repaired(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    values = np.asarray([1.0, np.nan, 3.0, 4.0], dtype=dtype)
    mask = np.zeros(values.shape, dtype=bool)
    actual = nf.fixpix(values, mask)
    np.testing.assert_array_equal(actual, values)
    assert np.isnan(actual[1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nonfinite_and_extreme_endpoints_use_f64_legacy_arithmetic(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    maximum = np.finfo(dtype).max
    cases = [
        ([maximum, 0.0, -maximum], [False, True, False]),
        ([np.inf, 0.0, 3.0], [False, True, False]),
        ([np.nan, 0.0, 3.0], [False, True, False]),
        ([np.nan, 0.0, 3.0], [True, False, False]),
        ([1.0, 0.0, np.nan], [False, False, True]),
    ]
    for values_values, mask_values in cases:
        values = np.asarray(values_values, dtype=dtype)
        mask = np.asarray(mask_values, dtype=bool)
        expected = fixpix_reference(values, mask)
        actual = nf.fixpix(values, mask)
        np.testing.assert_array_equal(actual, expected)


def test_noncanonical_bool_bytes_are_read_as_normalized_mask_values() -> None:
    values = np.asarray([2.0, 99.0, 6.0], dtype=np.float64)
    raw = np.asarray([0, 255, 0], dtype=np.uint8)
    mask = raw.view(np.bool_)
    expected = nf.fixpix(values, raw != 0)

    actual = nf.fixpix(values, mask)

    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


@pytest.mark.parametrize("value", [np.array(1.0), 1.0])
def test_rank_zero_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        nf.fixpix(value)


@pytest.mark.parametrize("dtype", [np.int64, np.bool_, np.complex128, np.float16])
def test_unsupported_array_dtypes_are_rejected(dtype: np.dtype) -> None:
    with pytest.raises(TypeError, match="float32 or float64"):
        nf.fixpix(np.ones(3, dtype=dtype))


def test_non_native_array_byte_order_is_rejected() -> None:
    dtype = np.dtype(">f8" if np.little_endian else "<f8")
    with pytest.raises(TypeError, match="native-endian"):
        nf.fixpix(np.ones(3, dtype=dtype))


def test_unaligned_array_is_rejected() -> None:
    buffer = bytearray(1 + 3 * np.dtype(np.float64).itemsize)
    values = np.ndarray(3, dtype=np.float64, buffer=buffer, offset=1)
    assert not values.flags.aligned
    with pytest.raises(ValueError, match="aligned"):
        nf.fixpix(values)


def test_masked_arrays_are_rejected_without_discarding_their_masks() -> None:
    values = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
    mask = np.ma.array([False, True, False], mask=[False, False, True])
    with pytest.raises(TypeError, match="masked arrays"):
        nf.fixpix(values)
    with pytest.raises(TypeError, match="masked arrays"):
        nf.fixpix(np.asarray(values.data), mask)


def test_mask_must_be_numeric_or_boolean_and_match_shape() -> None:
    values = np.arange(3.0)
    with pytest.raises(TypeError, match="mask"):
        nf.fixpix(values, np.asarray(["", "x", ""]))
    with pytest.raises(ValueError, match="shape"):
        nf.fixpix(values, np.zeros((3, 1), dtype=bool))


def test_priority_accepts_integer_index_values() -> None:
    values = np.arange(27.0).reshape(3, 3, 3)
    mask = np.zeros(values.shape, dtype=bool)
    mask[1, 1, 1] = True
    priority = [IndexValue(2), np.int64(0), 1]
    assert tuple(operator.index(value) for value in priority) == (2, 0, 1)
    expected = fixpix_reference(values, mask, priority=(2, 0, 1))
    actual = nf.fixpix(values, mask, priority=priority)
    _assert_matches_reference(actual, expected, exact=True)


@pytest.mark.parametrize(
    "priority",
    [
        True,
        (0, 1),
        (0, 0, 1),
        (0, 1, 3),
        (0, 1, 1),
        (0, 1, 1.0),
        (0, 1, "2"),
        (0, 1, True),
        0,
    ],
)
def test_invalid_priorities_are_rejected(priority: object) -> None:
    values = np.zeros((2, 2, 2), dtype=np.float64)
    with pytest.raises((TypeError, ValueError)):
        nf.fixpix(values, priority=priority)
