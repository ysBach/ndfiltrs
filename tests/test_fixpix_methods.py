"""Method, median-policy, and ownership tests for ``ndfiltrs.fixpix``."""

from __future__ import annotations

import ndfiltrs as nf
import numpy as np
import pytest

MODES = ("reflect", "constant", "nearest", "mirror", "wrap")
DTYPES = (np.float32, np.float64)


def _median_mask_exclusion_reference(
    array: np.ndarray,
    mask: np.ndarray,
    *,
    size: int | tuple[int, ...],
    mode: str | tuple[str, ...],
    cval: float,
    ignore_inf: bool,
    min_count: int,
) -> np.ndarray:
    """Use the existing nanmedian kernel as a mask-exclusion oracle."""
    bad = np.asarray(mask != 0, dtype=bool)
    masked_input = np.array(array, copy=True, order="C")
    masked_input[bad] = np.nan
    filtered = nf.nanmedian_filter(
        masked_input,
        size=size,
        mode=mode,
        cval=cval,
        ignore_inf=ignore_inf,
        min_count=min_count,
    )
    expected = np.array(array, copy=True, order="C")
    expected[bad] = filtered[bad]
    return expected


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_defaults_match_mask_exclusion_oracle(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    values = np.asarray(
        [[1.0, np.nan, 8.0], [np.inf, 5.0, 2.0], [4.0, 7.0, -np.inf]],
        dtype=dtype,
    )
    mask = np.array([[False, True, False], [True, False, False], [False, False, False]])
    expected = _median_mask_exclusion_reference(
        values,
        mask,
        size=3,
        mode="reflect",
        cval=np.nan,
        ignore_inf=False,
        min_count=1,
    )

    original = values.copy()
    actual = nf.fixpix(values, mask, method="median")

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_median_matches_nanmedian_mask_exclusion_for_all_boundary_modes(
    dtype: type[np.float32] | type[np.float64], mode: str
) -> None:
    rng = np.random.default_rng(20260906)
    values = rng.normal(size=(2, 3, 2)).astype(dtype)
    values[0, 0, 0] = np.nan
    values[1, 1, 0] = np.inf
    values[0, 2, 1] = -np.inf
    mask = np.zeros(values.shape, dtype=bool)
    mask[0, 1, 0] = True
    mask[1, 1, 1] = True
    mask[1, 2, 0] = True
    size = (3, 3, 3)
    cval = -7.25
    expected = _median_mask_exclusion_reference(
        values,
        mask,
        size=size,
        mode=mode,
        cval=cval,
        ignore_inf=True,
        min_count=4,
    )

    actual = nf.fixpix(
        values,
        mask,
        method="median",
        size=size,
        mode=mode,
        cval=cval,
        ignore_inf=True,
        min_count=4,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[~mask], values[~mask])


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_accepts_mixed_nd_configuration_and_omits_nan(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    values = np.arange(2 * 3 * 4, dtype=dtype).reshape(2, 3, 4)
    values[0, 0, 0] = np.nan
    values[1, 2, 3] = np.inf
    mask = np.zeros(values.shape, dtype=np.int8)
    mask[0, 1, 1] = 1
    mask[1, 2, 2] = 1
    size = (3, 1, 5)
    mode = ("reflect", "constant", "wrap")
    cval = 2.5
    expected = _median_mask_exclusion_reference(
        values,
        mask,
        size=size,
        mode=mode,
        cval=cval,
        ignore_inf=False,
        min_count=2,
    )

    actual = nf.fixpix(
        values,
        mask,
        method="median",
        size=size,
        mode=mode,
        cval=cval,
        ignore_inf=False,
        min_count=2,
    )

    np.testing.assert_array_equal(actual, expected)
    assert np.isnan(actual[0, 0, 0])


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_never_uses_repaired_values_as_inputs(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    values = np.asarray([0.0, 100.0, 200.0, 10.0], dtype=dtype)
    mask = np.array([False, True, True, False])
    expected = np.asarray([0.0, 0.0, 10.0, 10.0], dtype=dtype)

    actual = nf.fixpix(
        values,
        mask,
        method="median",
        size=3,
        mode="reflect",
        cval=np.nan,
        ignore_inf=False,
        min_count=1,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        values, np.asarray([0.0, 100.0, 200.0, 10.0], dtype=dtype)
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_edge_cval_and_infinity_policy_are_explicit(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    edge_values = np.asarray([99.0, 5.0], dtype=dtype)
    edge_mask = np.array([True, False])
    edge_expected = _median_mask_exclusion_reference(
        edge_values,
        edge_mask,
        size=3,
        mode="constant",
        cval=11.0,
        ignore_inf=False,
        min_count=1,
    )
    edge_actual = nf.fixpix(
        edge_values,
        edge_mask,
        method="median",
        size=3,
        mode="constant",
        cval=11.0,
        min_count=1,
    )
    np.testing.assert_array_equal(edge_actual, edge_expected)
    assert edge_actual[0] == dtype(8.0)

    inf_values = np.asarray([np.inf, 99.0, 2.0], dtype=dtype)
    inf_mask = np.array([False, True, False])
    retained = nf.fixpix(
        inf_values,
        inf_mask,
        method="median",
        size=3,
        mode="reflect",
        cval=np.nan,
        ignore_inf=False,
        min_count=1,
    )
    omitted = nf.fixpix(
        inf_values,
        inf_mask,
        method="median",
        size=3,
        mode="reflect",
        cval=np.nan,
        ignore_inf=True,
        min_count=1,
    )
    assert np.isposinf(retained[1])
    assert omitted[1] == dtype(2.0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_min_count_counts_duplicate_mapped_samples(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    values = np.asarray([99.0, 7.0], dtype=dtype)
    mask = np.array([True, False])
    enough = nf.fixpix(
        values,
        mask,
        method="median",
        size=5,
        mode="nearest",
        cval=np.nan,
        min_count=2,
    )
    insufficient = nf.fixpix(
        values,
        mask,
        method="median",
        size=5,
        mode="nearest",
        cval=np.nan,
        min_count=3,
    )

    assert enough[0] == dtype(7.0)
    assert np.isnan(insufficient[0])
    assert enough[1] == values[1]
    assert insufficient[1] == values[1]


@pytest.mark.parametrize("dtype", DTYPES)
def test_median_accepts_strided_inputs_and_preserves_unmasked_values(
    dtype: type[np.float32] | type[np.float64],
) -> None:
    data_storage = np.arange(4 * 6 * 8, dtype=dtype).reshape(4, 6, 8)
    mask_storage = np.zeros(data_storage.shape, dtype=np.int8)
    values = data_storage[::-1, ::2, 1::2]
    mask = mask_storage[::-1, ::2, 1::2]
    mask[0, 1, 1] = 1
    mask[2, 0, 2] = 1
    values_before = values.copy()
    mask_before = mask.copy()
    expected = _median_mask_exclusion_reference(
        values,
        mask,
        size=(3, 1, 3),
        mode=("mirror", "nearest", "wrap"),
        cval=-4.0,
        ignore_inf=False,
        min_count=1,
    )

    actual = nf.fixpix(
        values,
        mask,
        method="median",
        size=(3, 1, 3),
        mode=("mirror", "nearest", "wrap"),
        cval=-4.0,
        min_count=1,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, values_before)
    np.testing.assert_array_equal(mask, mask_before)
    assert actual.flags.c_contiguous
    assert actual.flags.owndata
    assert actual.flags.writeable
    assert actual.base is None


def test_noncanonical_bool_mask_bytes_are_treated_as_nonzero() -> None:
    values = np.asarray([1.0, 99.0, 3.0, 7.0], dtype=np.float64)
    raw = np.asarray([0, 255, 0, 2], dtype=np.uint8)
    mask = raw.view(np.bool_)
    canonical = raw != 0
    expected = _median_mask_exclusion_reference(
        values,
        canonical,
        size=3,
        mode="reflect",
        cval=np.nan,
        ignore_inf=False,
        min_count=1,
    )

    actual = nf.fixpix(values, mask, method="median")

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("method", ["linear", "median"])
def test_copy_false_aliases_only_without_repairs(method: str) -> None:
    values = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    empty_mask = np.zeros(values.shape, dtype=bool)
    original = values.copy()

    without_mask = nf.fixpix(values, method=method, copy=False)
    with_empty_mask = nf.fixpix(values, empty_mask, method=method, copy=False)

    assert without_mask is values
    assert with_empty_mask is values
    np.testing.assert_array_equal(values, original)


def test_copy_true_returns_an_independent_noop_copy() -> None:
    values = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    actual = nf.fixpix(values, copy=True)

    np.testing.assert_array_equal(actual, values)
    assert actual is not values
    assert not np.shares_memory(actual, values)
    assert actual.flags.owndata


@pytest.mark.parametrize("method", ["linear", "median"])
def test_copy_false_repairs_allocate_without_mutating_input(method: str) -> None:
    values = np.asarray([1.0, 99.0, 3.0], dtype=np.float64)
    mask = np.array([False, True, False])
    original = values.copy()

    actual = nf.fixpix(values, mask, method=method, copy=False)

    assert actual is not values
    assert not np.shares_memory(actual, values)
    np.testing.assert_array_equal(values, original)
    assert actual[1] == 2.0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"method": "unknown"}, ValueError),
        ({"method": 1}, (TypeError, ValueError)),
        ({"method": "linear", "size": 3}, ValueError),
        ({"method": "linear", "mode": "reflect"}, ValueError),
        ({"method": "linear", "cval": 0.0}, ValueError),
        ({"method": "linear", "ignore_inf": False}, ValueError),
        ({"method": "linear", "min_count": 1}, ValueError),
        ({"method": "median", "size": 0}, ValueError),
        ({"method": "median", "size": True}, TypeError),
        ({"method": "median", "size": (3, 3, 3)}, ValueError),
        ({"method": "median", "mode": "bad"}, ValueError),
        ({"method": "median", "mode": ("reflect",)}, ValueError),
        ({"method": "median", "mode": 1}, TypeError),
        ({"method": "median", "cval": True}, TypeError),
        ({"method": "median", "cval": 1 + 2j}, TypeError),
        ({"method": "median", "ignore_inf": 1}, TypeError),
        ({"method": "median", "min_count": 0}, ValueError),
        ({"method": "median", "min_count": 10}, ValueError),
        ({"method": "median", "min_count": True}, TypeError),
        ({"method": "median", "min_count": 1.5}, TypeError),
        ({"method": "median", "copy": 1}, TypeError),
    ],
)
def test_configuration_is_validated_before_empty_noop(
    kwargs: dict[str, object], error: type[Exception] | tuple[type[Exception], ...]
) -> None:
    values = np.empty((0, 2), dtype=np.float64)
    empty_mask = np.empty(values.shape, dtype=bool)
    for mask in (None, empty_mask):
        with pytest.raises(error):
            nf.fixpix(values, mask, **kwargs)


def test_none_sentinels_preserve_linear_and_median_defaults() -> None:
    values = np.asarray([1.0, 99.0, 3.0], dtype=np.float64)
    mask = np.array([False, True, False])
    linear_default = nf.fixpix(values, mask)
    linear_explicit_none = nf.fixpix(
        values,
        mask,
        method="linear",
        priority=None,
        size=None,
        mode=None,
        cval=None,
        ignore_inf=None,
        min_count=None,
    )
    median_default = nf.fixpix(values, mask, method="median")
    median_explicit_none = nf.fixpix(
        values,
        mask,
        method="median",
        size=None,
        mode=None,
        cval=None,
        ignore_inf=None,
        min_count=None,
    )

    np.testing.assert_array_equal(linear_default, linear_explicit_none)
    np.testing.assert_array_equal(median_default, median_explicit_none)


@pytest.mark.parametrize("method", ["linear", "median"])
def test_noop_alias_preserves_strides_and_readonly_state(method: str) -> None:
    values = np.arange(24.0).reshape(4, 6)[::-1, ::2]
    values.setflags(write=False)
    assert (
        nf.fixpix(values, np.zeros(values.shape), method=method, copy=False) is values
    )
    copied = nf.fixpix(values, np.zeros(values.shape), method=method)
    assert copied.flags.owndata and copied.flags.writeable and copied.flags.c_contiguous
    np.testing.assert_array_equal(copied, values)


@pytest.mark.parametrize("size", [1, 3, 5])
def test_fully_masked_median_has_no_support_without_constant_samples(size: int) -> None:
    values = np.arange(6.0).reshape(2, 3)
    actual = nf.fixpix(
        values, np.ones(values.shape, dtype=bool), method="median", size=size
    )
    assert np.isnan(actual).all()
    assert not np.shares_memory(actual, values)


def test_median_rejects_priority_and_oversized_windows_even_without_mask() -> None:
    values = np.ones((2, 2))
    with pytest.raises(ValueError, match="priority"):
        nf.fixpix(values, method="median", priority=(1, 0))
    with pytest.raises(OverflowError, match="window volume"):
        nf.fixpix(values, method="median", size=(2**63 + 1, 3), copy=False)
