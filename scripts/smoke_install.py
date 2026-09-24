"""Minimal release-artifact installation smoke test."""

import importlib.metadata
import os
import sys
from pathlib import Path

import ndfiltrs as nf
import numpy as np
from ndfiltrs import _core

installed_version = importlib.metadata.version("ndfiltrs")
assert nf.__version__ == installed_version
if expected_version := os.environ.get("EXPECTED_VERSION"):
    assert installed_version == expected_version
for module in (nf, _core):
    assert module.__file__ is not None
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), (
        f"smoke test imported {module.__name__} outside the isolated environment: "
        f"{module.__file__}"
    )
for name in (
    "_DetachProbe",
    "_test_median_filter_after_snapshot",
    "_test_median_filter_during_detach",
    "_test_convolve_after_snapshot",
    "_test_convolve_during_detach",
    "_test_fixpix_after_snapshot",
    "_test_fixpix_during_detach",
):
    assert not hasattr(_core, name), f"release extension exposes test-only {name}"

for dtype in (np.float32, np.float64):
    values = np.array([1.0, 9.0, 2.0], dtype=dtype)
    filtered = nf.median_filter(values, 3, mode="mirror")
    np.testing.assert_array_equal(filtered, [9.0, 2.0, 9.0])
    assert filtered.dtype == dtype
    assert filtered.shape == values.shape
    assert filtered.flags.owndata
    assert filtered.flags.c_contiguous
    assert filtered.flags.aligned
    assert filtered.flags.writeable
    assert filtered.base is None
    np.testing.assert_array_equal(
        nf.nanmedian_filter(np.array([1.0, np.nan, 3.0], dtype=dtype), 3),
        [1.0, 2.0, 3.0],
    )
    thresholded = nf.nanmedian_filter(
        np.array([np.nan], dtype=dtype),
        3,
        mode="constant",
        cval=4.0,
        min_count=2,
    )
    np.testing.assert_array_equal(thresholded, np.array([4.0], dtype=dtype))

    # An asymmetric kernel detects swapped convolution/correlation orientation.
    samples = np.array([1.0, 2.0, 3.0], dtype=dtype)
    for function, expected in (
        (nf.convolve, [2.0, 3.0, 1.0]),
        (nf.correlate, [3.0, 1.0, 2.0]),
    ):
        output = function(samples, [1.0, 0.0, 0.0], mode="wrap")
        np.testing.assert_array_equal(output, expected)
        assert output.dtype == dtype and output.flags.owndata
    for function in (nf.nanconvolve, nf.nancorrelate):
        output = function(
            np.array([1.0, np.nan, 3.0], dtype=dtype),
            [1.0, 2.0, 1.0],
            mode="constant",
            cval=np.nan,
        )
        np.testing.assert_array_equal(output, [1.0, 2.0, 3.0])
        assert output.dtype == dtype and output.flags.owndata

    # Radius one gives Gaussian samples [exp(-1/2), 1, exp(-1/2)].
    impulse = np.array([0.0, 1.0, 0.0], dtype=dtype)
    edge = np.exp(-0.5)
    mass = 1.0 + 2.0 * edge
    for function, kwargs, expected in (
        (nf.gaussian_filter, {}, np.array([edge, 1.0, edge]) / mass),
        (nf.gaussian_filter, {"order": 1}, np.array([edge, 0.0, -edge]) / mass),
        (nf.gaussian_laplace, {}, np.array([0.0, -1.0, 0.0]) / mass),
        (nf.gaussian_gradient_magnitude, {}, np.array([edge, 0.0, edge]) / mass),
    ):
        output = function(impulse, sigma=1.0, radius=1, mode="constant", **kwargs)
        np.testing.assert_allclose(output, expected, rtol=8 * np.finfo(dtype).eps)
        assert output.dtype == dtype and output.flags.owndata
    output = nf.nangaussian_filter(
        np.array([1.0, np.nan, 3.0], dtype=dtype),
        sigma=1.0,
        radius=1,
        mode="constant",
        cval=np.nan,
    )
    np.testing.assert_allclose(output, [1.0, 2.0, 3.0], rtol=8 * np.finfo(dtype).eps)
    assert output.dtype == dtype and output.flags.owndata
    averaged = nf.mean_filter(values, 3, mode="wrap")
    np.testing.assert_array_equal(averaged, np.array([4.0, 4.0, 4.0], dtype=dtype))
    assert averaged.dtype == dtype and averaged.flags.owndata
    np.testing.assert_array_equal(
        nf.nanmean_filter(np.array([1.0, np.nan, 3.0], dtype=dtype)),
        [1.0, 2.0, 3.0],
    )
    repaired = nf.fixpix(values, [False, True, False])
    np.testing.assert_array_equal(repaired, np.array([1.0, 1.5, 2.0], dtype=dtype))
    assert repaired.dtype == dtype
    assert repaired.flags.owndata and repaired.flags.c_contiguous
    assert not np.shares_memory(repaired, values)

    median_repaired = nf.fixpix(values, [False, True, False], method="median")
    np.testing.assert_array_equal(median_repaired, [1.0, 1.5, 2.0])
    assert median_repaired.dtype == dtype
    assert not np.shares_memory(median_repaired, values)
    assert nf.fixpix(values, np.zeros(3, dtype=bool), copy=False) is values

    for function, expected in (
        (nf.min_filter, 1.0),
        (nf.max_filter, 9.0),
        (nf.nanmin_filter, 1.0),
        (nf.nanmax_filter, 9.0),
    ):
        output = function(values, mode="wrap")
        np.testing.assert_array_equal(output, expected)
        assert output.dtype == dtype and output.flags.owndata
    for function in (nf.var_filter, nf.nanvar_filter, nf.std_filter, nf.nanstd_filter):
        # The two finite endpoints have sample variance 2 and std sqrt(2).
        samples = np.array([1.0, 2.0, 3.0], dtype=dtype)
        output = function(samples, mode="wrap", ddof=1)
        np.testing.assert_array_equal(output, np.ones(3, dtype=dtype))
        assert output.dtype == dtype and output.flags.owndata
    for function, expected in (
        (nf.nanvar_filter, 2.0),
        (nf.nanstd_filter, np.sqrt(2.0)),
    ):
        samples = np.array([1.0, np.nan, 3.0], dtype=dtype)
        np.testing.assert_array_equal(
            function(samples, mode="wrap", ddof=1), dtype(expected)
        )

    separable = nf.nangaussian_filter(
        np.array([[1.0, np.nan, 3.0]], dtype=dtype),
        1.0,
        radius=1,
        method="separable",
        min_count=3,
    )
    direct = nf.nangaussian_filter(
        np.array([[1.0, np.nan, 3.0]], dtype=dtype),
        1.0,
        radius=1,
        min_count=3,
    )
    np.testing.assert_allclose(separable, direct, rtol=8 * np.finfo(dtype).eps)
    assert separable.dtype == dtype and separable.flags.owndata

    finite = np.array([1.0, 2.0, 100.0], dtype=dtype)
    missing = np.array([1.0, np.nan, 3.0], dtype=dtype)
    for result, expected in (
        (nf.percentile_filter(finite, 25, mode="wrap"), 1.5),
        (nf.nanpercentile_filter(missing, 25, mode="wrap"), 1.5),
        (nf.mad_filter(finite, mode="wrap"), 1.0),
        (nf.nanmad_filter(missing, mode="wrap"), 1.0),
        (nf.sigma_clipped_filter(finite, spread="mad", mode="wrap"), 1.5),
    ):
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == dtype and result.flags.owndata
