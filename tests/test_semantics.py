"""Public filtering semantics against the independent N-D reference."""

import ndfiltrs as nf
import numpy as np
import pytest

from tests._reference import median_filter_reference


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
def test_plain_filter_matches_reference_for_every_mode(
    dtype: np.dtype, mode: str
) -> None:
    values = np.array([[1.0, 9.0, 2.0], [4.0, 3.0, 8.0]], dtype=dtype)
    expected = median_filter_reference(
        values,
        (3, 3),
        (mode, mode),
        cval=-5.0,
        omit_nan=False,
        ignore_inf=False,
        min_count=1,
    )
    actual = nf.median_filter(values, (3, 3), mode=mode, cval=-5.0)
    np.testing.assert_array_equal(actual, expected)


def test_nanmedian_matches_reference_in_three_dimensions() -> None:
    values = np.array(
        [[[1.0, np.nan], [np.inf, 4.0]], [[5.0, 6.0], [7.0, -np.inf]]],
        dtype=np.float64,
    )
    expected = median_filter_reference(
        values,
        (3, 1, 3),
        ("reflect", "constant", "wrap"),
        cval=np.nan,
        omit_nan=True,
        ignore_inf=True,
        min_count=2,
    )
    actual = nf.nanmedian_filter(
        values,
        (3, 1, 3),
        mode=("reflect", "constant", "wrap"),
        cval=np.nan,
        ignore_inf=True,
        min_count=2,
    )
    np.testing.assert_array_equal(actual, expected)


def test_generated_nan_and_signed_zero_are_bitwise_deterministic() -> None:
    invalid = nf.nanmedian_filter(
        np.array([np.nan], dtype=np.float64), size=1, min_count=1
    )
    assert invalid.view(np.uint64)[0] == np.float64(np.nan).view(np.uint64)

    zero = nf.median_filter(np.array([-0.0, -0.0, 0.0]), size=3, mode="nearest")
    assert zero.view(np.uint64)[1] == np.float64(-0.0).view(np.uint64)


def test_constant_values_participate_in_policy_and_min_count() -> None:
    values = np.array([np.nan], dtype=np.float64)
    retained = nf.nanmedian_filter(values, 3, mode="constant", cval=2.0, min_count=2)
    assert retained[0] == 2.0
    omitted = nf.nanmedian_filter(
        values,
        3,
        mode="constant",
        cval=np.inf,
        ignore_inf=True,
        min_count=1,
    )
    assert np.isnan(omitted[0])


def test_rank_five_mixed_modes_matches_reference() -> None:
    values = np.arange(48.0, dtype=np.float64).reshape(2, 2, 2, 2, 3)
    values[0, 1, 0, 1, 2] = np.nan
    size = (3, 3, 3, 3, 5)
    mode = ("reflect", "constant", "nearest", "mirror", "wrap")
    expected = median_filter_reference(
        values,
        size,
        mode,
        cval=-4.0,
        omit_nan=True,
        ignore_inf=False,
        min_count=7,
    )
    actual = nf.nanmedian_filter(
        values,
        size,
        mode=mode,
        cval=-4.0,
        min_count=7,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "unsigned_dtype", "canonical_nan"),
    [
        (np.float32, np.uint32, 0x7FC00000),
        (np.float64, np.uint64, 0x7FF8000000000000),
    ],
)
def test_both_dtypes_generate_canonical_nan(
    dtype: type[np.float32] | type[np.float64],
    unsigned_dtype: type[np.uint32] | type[np.uint64],
    canonical_nan: int,
) -> None:
    propagated = nf.median_filter(np.array([np.nan], dtype=dtype), size=1)
    below_threshold = nf.nanmedian_filter(
        np.array([1.0, np.nan, np.nan], dtype=dtype),
        size=3,
        mode="nearest",
        min_count=2,
    )
    assert propagated.view(unsigned_dtype)[0] == canonical_nan
    assert below_threshold.view(unsigned_dtype)[1] == canonical_nan
