"""Separable Gaussian smoothing against independent full-window averages."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable

import ndfiltrs as nf
import numpy as np
import pytest

from tests._convolution_reference import convolution_reference

FUNCTIONS = (nf.gaussian_filter, nf.nangaussian_filter)
DTYPES = (np.float32, np.float64)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("mode", "cval"),
    [
        (("constant", "mirror"), np.nan),
        (("wrap", "constant"), 2.5),
        (("nearest", "reflect"), 0.0),
    ],
)
def test_separable_selected_axes_match_literal_nd_windows(
    function: Callable,
    dtype: type[np.floating],
    mode: tuple[str, str],
    cval: float,
) -> None:
    array = (np.arange(24).reshape(2, 3, 4) % 13 - 4).astype(dtype)
    array[0, 1, 1] = np.nan
    array[1, 0, 2] = np.inf
    array[0, 2, 3] = -np.inf
    offsets = np.arange(-1.0, 2.0)
    axis0 = np.exp(-0.5 * (offsets / 1.1) ** 2)
    axis2 = np.exp(-0.5 * (offsets / 0.7) ** 2)
    axis0 /= axis0.sum()
    axis2 /= axis2.sum()
    weights = axis0[:, None, None] * axis2[None, None, :]
    expected = convolution_reference(
        array,
        weights,
        (mode[1], "reflect", mode[0]),
        cval=cval,
        omit_nan=function is nf.nangaussian_filter,
    )
    actual = function(
        array,
        (0.7, 1.1),
        axes=(2, 0),
        radius=1,
        mode=mode,
        cval=cval,
        method="separable",
    )
    tolerance = 64 * np.finfo(dtype).eps
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ignore_inf", [False, True])
@pytest.mark.parametrize("min_count", [1, 9])
def test_separable_missing_mass_counts_and_infinity_flags(
    dtype: type[np.floating], ignore_inf: bool, min_count: int
) -> None:
    # Unequal retained support across rows detects normalization after each
    # axis without carrying each intermediate window's retained mass.
    array = np.array(
        [[1, np.nan, np.nan], [3, 7, np.inf], [np.nan, 5, -np.inf]], dtype=dtype
    )
    offsets = np.arange(-1.0, 2.0)
    axis = np.exp(-0.5 * offsets**2)
    axis /= axis.sum()
    expected = convolution_reference(
        array,
        np.outer(axis, axis),
        ("constant", "constant"),
        cval=4,
        omit_nan=True,
        ignore_inf=ignore_inf,
        min_count=min_count,
    )
    actual = nf.nangaussian_filter(
        array,
        1.0,
        radius=1,
        mode="constant",
        cval=4,
        ignore_inf=ignore_inf,
        min_count=min_count,
        method="separable",
    )
    tolerance = 64 * np.finfo(dtype).eps
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True
    )


def test_separable_boundary_repetitions_and_empty_retained_support() -> None:
    np.testing.assert_array_equal(
        nf.nangaussian_filter(
            np.array([[5.0]]), 1.0, radius=1, min_count=9, method="separable"
        ),
        [[5.0]],
    )
    missing = np.array([[np.nan]])
    np.testing.assert_allclose(
        nf.nangaussian_filter(
            missing,
            1.0,
            radius=1,
            mode="constant",
            cval=7,
            min_count=8,
            method="separable",
        ),
        [[7.0]],
        rtol=16 * np.finfo(float).eps,
    )
    for options in (
        {},
        {"mode": "constant", "cval": 7, "min_count": 9},
        {"mode": "constant", "cval": np.inf, "ignore_inf": True},
    ):
        assert np.isnan(
            nf.nangaussian_filter(missing, 1.0, radius=1, method="separable", **options)
        ).all()
    assert np.isposinf(
        nf.nangaussian_filter(
            missing,
            1.0,
            radius=1,
            mode="constant",
            cval=np.inf,
            method="separable",
        )
    ).all()


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_separable_scaling_preserves_tiny_and_maximum_local_constants(
    function: Callable, dtype: type[np.floating]
) -> None:
    tiny = np.nextafter(dtype(0), dtype(1))
    maximum = np.finfo(dtype).max
    array = np.zeros((7, 9), dtype=dtype)
    array[:, :3] = tiny
    array[:, -3:] = maximum
    actual = function(array, 0.8, radius=1, method="separable")
    assert np.isfinite(actual).all()
    # The middle zero columns separate the two support regions. Scaling the
    # entire input by its maximum would destroy the left neighborhood.
    assert actual[3, 1] == tiny
    assert actual[3, 7] == maximum
    assert actual.dtype == array.dtype


@pytest.mark.parametrize("function", FUNCTIONS)
def test_separable_empty_and_strided_inputs_return_owned_arrays(
    function: Callable,
) -> None:
    source = np.arange(24.0).reshape(4, 6)
    for array in (source[::-1, ::2], source.T, source[:0]):
        original = array.copy()
        array.flags.writeable = False
        actual = function(array, 1.0, radius=1, method="separable")
        expected = function(original, 1.0, radius=1, method="direct")
        np.testing.assert_allclose(actual, expected, rtol=64 * np.finfo(float).eps)
        np.testing.assert_array_equal(array, original)
        assert actual.shape == array.shape and actual.dtype == array.dtype
        assert actual.flags.owndata and actual.flags.c_contiguous
        assert actual.flags.writeable and actual.flags.aligned
        assert not np.shares_memory(actual, array)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_separable_zero_sigma_keeps_identity_and_invalid_value_policy(
    function: Callable,
) -> None:
    array = np.array([[np.nan, 1.0, np.inf], [-np.inf, 2.0, 3.0]])
    actual = function(array, 0.0, radius=1, method="separable")
    np.testing.assert_array_equal(actual, array)
    if function is nf.nangaussian_filter:
        actual = function(array, 0.0, radius=1, ignore_inf=True, method="separable")
        np.testing.assert_array_equal(actual, [[np.nan, 1, np.nan], [np.nan, 2, 3]])
        with pytest.raises(ValueError):
            function(array, 0.0, radius=1, min_count=2, method="separable")


@pytest.mark.parametrize("function", FUNCTIONS)
def test_separable_rejects_product_support_that_underflows(function: Callable) -> None:
    # Each radius-one axis has positive side taps exp(-1/(2*.03**2)); their
    # corner product rounds to zero in the dense float64 kernel.
    with pytest.raises(ValueError, match="underflow|direct"):
        function(np.ones((3, 3)), 0.03, radius=1, method="separable")


def test_separable_rejects_nonzero_derivative_orders() -> None:
    with pytest.raises(ValueError, match="order|derivative|smoothing"):
        nf.gaussian_filter(
            np.ones((2, 3)), 1.0, order=(0, 1), radius=1, method="separable"
        )


@pytest.mark.parametrize("function", FUNCTIONS)
def test_separable_validates_method_and_inputs_before_empty_return(
    function: Callable,
) -> None:
    with pytest.raises((TypeError, ValueError), match="method"):
        function(np.empty(0), 1.0, radius=1, method="unknown")
    with pytest.raises(TypeError):
        function(np.empty(0, dtype=np.int64), 1.0, radius=1, method="separable")
    with pytest.raises(ValueError):
        function(np.empty(0), -1.0, radius=1, method="separable")
    if function is nf.nangaussian_filter:
        with pytest.raises(ValueError):
            function(np.empty(0), 1.0, radius=1, min_count=0, method="separable")


@pytest.mark.parametrize("function", FUNCTIONS)
def test_separable_identity_does_not_require_a_dense_kernel_volume(
    function: Callable,
) -> None:
    # Rank 32 is supported by every declared NumPy version. A dense radius-two
    # kernel needs 5**32 coefficients, but these 32 identity passes need only 160.
    array = np.full((1,) * 32, 5.0)
    actual = function(array, 0.0, radius=2, method="separable")
    np.testing.assert_array_equal(actual, array)
    assert actual.flags.owndata


def test_separable_results_are_bitwise_identical_across_thread_counts() -> None:
    script = textwrap.dedent(
        """
        import hashlib
        import json

        import ndfiltrs as nf
        import numpy as np

        results = {}
        for dtype in (np.float32, np.float64):
            array = np.random.default_rng(20260922).normal(size=(13, 17)).astype(dtype)
            array[::7, ::9] = np.nan
            array[2::8, 3::10] = np.inf
            array[4::9, 5::11] = -np.inf
            for function in (nf.gaussian_filter, nf.nangaussian_filter):
                options = {}
                if function is nf.nangaussian_filter:
                    options = {"ignore_inf": True, "min_count": 3}
                actual = function(
                    array, (0.8, 1.1), radius=(1, 2), mode=("mirror", "wrap"),
                    method="separable", **options,
                )
                assert np.isfinite(actual).any()
                assert np.ptp(actual[np.isfinite(actual)]) > 0
                key = function.__name__ + ":" + dtype.__name__
                results[key] = hashlib.sha256(actual.tobytes()).hexdigest()
        print(json.dumps({"threads": nf.get_num_threads(), "digests": results}))
        """
    )
    digests: list[dict[str, str]] = []
    for threads in (1, 2, 4):
        environment = os.environ.copy()
        environment["RAYON_NUM_THREADS"] = str(threads)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert result["threads"] == threads
        digests.append(result["digests"])
    assert digests[0] == digests[1] == digests[2]
