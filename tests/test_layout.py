"""Input normalization and output ownership tests."""

from collections.abc import Callable

import ndfiltrs as nf
import numpy as np
import pytest
from ndfiltrs import _core


@pytest.mark.parametrize(
    "view_factory",
    [
        lambda value: value[::-1, ::2],
        lambda value: np.asfortranarray(value),
        lambda value: value.T,
        lambda value: value.copy(),
    ],
    ids=["strided", "fortran", "transpose", "read-only-c"],
)
def test_public_layout_paths_return_owned_c_arrays(
    view_factory: Callable[[np.ndarray], np.ndarray],
) -> None:
    source = np.arange(24.0, dtype=np.float64).reshape(4, 6)
    view = view_factory(source)
    view.flags.writeable = False
    result = nf.median_filter(view, size=1)
    np.testing.assert_array_equal(result, view)
    assert result.flags.c_contiguous
    assert result.flags.aligned
    assert result.flags.owndata
    assert result.flags.writeable
    assert result.base is None


def test_public_unaligned_input_is_rejected() -> None:
    buffer = bytearray(1 + 3 * np.dtype(np.float64).itemsize)
    value = np.ndarray(3, dtype=np.float64, buffer=buffer, offset=1)
    assert not value.flags.aligned
    with pytest.raises(ValueError, match="aligned"):
        nf.median_filter(value, size=1)


def test_ndarray_subclass_returns_plain_ndarray() -> None:
    class Subclass(np.ndarray):
        pass

    value = np.arange(3.0).view(Subclass)
    result = nf.median_filter(value, size=1)
    assert type(result) is np.ndarray


def test_empty_shape_returns_new_validated_array() -> None:
    value = np.empty((0, 3), dtype=np.float32)
    result = nf.median_filter(value, size=(3, 3), mode=("wrap", "mirror"))
    assert result.shape == (0, 3)
    assert result.dtype == np.float32
    assert result is not value
    assert result.flags.owndata


@pytest.mark.parametrize("shape", [(0, 3), (2, 0, 3), (2, 3, 0)])
def test_zero_length_axis_in_any_position_returns_owned_output(
    shape: tuple[int, ...],
) -> None:
    value = np.empty(shape, dtype=np.float64)
    result = nf.median_filter(value, size=3, mode="wrap")
    assert result.shape == shape
    assert result.size == 0
    assert result.flags.owndata


def test_private_extension_rejects_fortran_order() -> None:
    value = np.asfortranarray(np.arange(6.0).reshape(2, 3))
    with pytest.raises(ValueError, match="C-contiguous"):
        _core.window_filter(
            value,
            "median",
            [1, 1],
            None,
            [0, 0],
            ["nearest", "nearest"],
            0.0,
            False,
            False,
            1,
            0,
        )


def test_output_allocation_never_returns_or_overwrites_external_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = np.full(3, 99.0, dtype=np.float64)

    def return_external_alias(*args: object, **kwargs: object) -> np.ndarray:
        return external

    monkeypatch.setattr(np, "empty", return_external_alias)
    result = nf.median_filter(np.array([1.0, 2.0, 3.0]), 1)

    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(external, [99.0, 99.0, 99.0])
    assert result is not external
    assert result.flags.owndata
    assert result.base is None
