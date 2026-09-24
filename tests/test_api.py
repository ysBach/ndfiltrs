"""Public API validation tests."""

import inspect
import operator

import ndfiltrs as nf
import numpy as np
import pytest


class IndexValue:
    """Non-integral object implementing Python's integer-index protocol."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


def test_public_signatures_are_explicit() -> None:
    median = inspect.signature(nf.median_filter)
    nanmedian = inspect.signature(nf.nanmedian_filter)
    assert list(median.parameters) == [
        "array",
        "size",
        "footprint",
        "origin",
        "mode",
        "cval",
        "method",
    ]
    assert median.parameters["size"].default is None
    assert median.parameters["footprint"].default is None
    assert median.parameters["origin"].default == 0
    assert median.parameters["mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert median.parameters["mode"].default == "reflect"
    assert median.parameters["cval"].default == 0.0
    assert median.parameters["method"].kind is inspect.Parameter.KEYWORD_ONLY
    assert median.parameters["method"].default == "direct"
    assert list(nanmedian.parameters) == [
        "array",
        "size",
        "footprint",
        "origin",
        "mode",
        "cval",
        "ignore_inf",
        "min_count",
        "method",
    ]
    assert nanmedian.parameters["ignore_inf"].default is False
    assert nanmedian.parameters["min_count"].default == 1
    assert nanmedian.parameters["method"].kind is inspect.Parameter.KEYWORD_ONLY
    assert nanmedian.parameters["method"].default == "direct"


@pytest.mark.parametrize("value", [np.array(1.0), 1.0])
def test_rank_zero_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        nf.median_filter(value)


@pytest.mark.parametrize(
    "dtype",
    [np.int64, np.bool_, np.complex128, np.float16],
)
def test_unsupported_dtypes_are_rejected(dtype: np.dtype) -> None:
    with pytest.raises(TypeError, match="float32 or float64"):
        nf.median_filter(np.ones(3, dtype=dtype))


def test_non_native_byte_order_is_rejected() -> None:
    dtype = np.dtype(">f8" if np.little_endian else "<f8")
    with pytest.raises(TypeError, match="native-endian"):
        nf.median_filter(np.ones(3, dtype=dtype))


@pytest.mark.parametrize("size", [True, 0, -3, 1.5, (3,), (3, 0)])
def test_invalid_sizes_are_rejected(size: object) -> None:
    error = TypeError if size is True or isinstance(size, float) else ValueError
    with pytest.raises(error):
        nf.median_filter(np.ones((3, 3), dtype=np.float64), size=size)


@pytest.mark.parametrize("mode", ["Reflect", "grid-wrap", 1, ("reflect",)])
def test_invalid_modes_are_rejected(mode: object) -> None:
    error = TypeError if mode == 1 else ValueError
    with pytest.raises(error):
        nf.median_filter(np.ones((3, 3), dtype=np.float64), mode=mode)


@pytest.mark.parametrize("cval", [True, 1 + 2j, "zero", [0.0]])
def test_invalid_cval_is_rejected(cval: object) -> None:
    with pytest.raises(TypeError, match="real scalar"):
        nf.median_filter(np.ones(3, dtype=np.float64), cval=cval)


@pytest.mark.parametrize("ignore_inf", [0, 1, "yes", None])
def test_ignore_inf_requires_a_boolean(ignore_inf: object) -> None:
    with pytest.raises(TypeError, match="boolean"):
        nf.nanmedian_filter(np.ones(3, dtype=np.float64), ignore_inf=ignore_inf)


@pytest.mark.parametrize("min_count", [True, 0, 4, 1.5])
def test_min_count_is_validated_against_window_volume(min_count: object) -> None:
    error = (
        TypeError if min_count is True or isinstance(min_count, float) else ValueError
    )
    with pytest.raises(error):
        nf.nanmedian_filter(np.ones(3, dtype=np.float64), size=3, min_count=min_count)


def test_size_and_min_count_accept_integer_index_values() -> None:
    size = IndexValue(3)
    min_count = IndexValue(2)
    assert operator.index(size) == 3
    result = nf.nanmedian_filter(
        np.array([1.0, np.nan, 3.0]), size=size, min_count=min_count
    )
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_masked_arrays_are_rejected_without_discarding_the_mask() -> None:
    value = np.ma.array([1.0, 2.0], mask=[False, True])
    with pytest.raises(TypeError, match="masked arrays"):
        nf.median_filter(value)
