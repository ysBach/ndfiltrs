"""Deterministic execution and snapshot ownership for weighted filters."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading

import numpy as np
import pytest
from ndfiltrs import _core

requires_hooks = pytest.mark.skipif(
    not all(
        hasattr(_core, name)
        for name in (
            "_DetachProbe",
            "_test_convolve_after_snapshot",
            "_test_convolve_during_detach",
        )
    ),
    reason="deterministic convolution ownership and GIL tests require test-hooks",
)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_weighted_and_gaussian_outputs_are_bitwise_identical_across_threads(
    dtype: str,
) -> None:
    script = textwrap.dedent(
        """
        import hashlib
        import json
        import sys

        import ndfiltrs as nf
        import numpy as np

        rng = np.random.default_rng(20260907)
        values = rng.normal(size=(17, 19)).astype(getattr(np, sys.argv[1]))
        values[::7, ::9] = np.nan
        values[3::9, 2::11] = np.inf
        values[5::8, 5::10] = -np.inf
        signed = np.array([[0, 1, -2, 4], [3, 0, -1, 2], [1, -4, 2, 0.]])
        positive = np.abs(signed)
        results = {
            "convolve": nf.convolve(values, signed, mode=("mirror", "wrap")),
            "correlate": nf.correlate(values, signed, mode=("mirror", "wrap")),
            "nanconvolve": nf.nanconvolve(
                values, positive, mode=("mirror", "wrap"), ignore_inf=True,
                min_count=3,
            ),
            "nancorrelate": nf.nancorrelate(
                values, positive, mode=("mirror", "wrap"), ignore_inf=True,
                min_count=3,
            ),
            "gaussian_filter": nf.gaussian_filter(
                values, (0.8, 1.1), radius=(1, 2), order=(1, 0),
                mode=("mirror", "wrap"),
            ),
            "nangaussian_filter": nf.nangaussian_filter(
                values, (0.8, 1.1), radius=(1, 2), ignore_inf=True,
                min_count=3, mode=("mirror", "wrap"),
            ),
            "gaussian_laplace": nf.gaussian_laplace(
                values, (0.8, 1.1), radius=(1, 2), mode=("mirror", "wrap"),
            ),
            "gaussian_gradient_magnitude": nf.gaussian_gradient_magnitude(
                values, (0.8, 1.1), radius=(1, 2), mode=("mirror", "wrap"),
            ),
        }
        for name, result in results.items():
            assert np.isfinite(result).any(), name
            assert np.ptp(result[np.isfinite(result)]) > 0, name
            assert result.dtype == values.dtype, name
        print(json.dumps({
            "threads": nf.get_num_threads(),
            "digests": {
                name: hashlib.sha256(result.tobytes()).hexdigest()
                for name, result in results.items()
            },
        }))
        """
    )
    digests: list[dict[str, str]] = []
    for threads in (1, 2, 4):
        environment = os.environ.copy()
        environment["RAYON_NUM_THREADS"] = str(threads)
        completed = subprocess.run(
            [sys.executable, "-c", script, dtype],
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


@requires_hooks
@pytest.mark.parametrize("correlate", [False, True])
@pytest.mark.parametrize("omit_nan", [False, True])
def test_weighted_filter_snapshots_both_values_and_kernel(
    correlate: bool, omit_nan: bool
) -> None:
    values = np.array([2.0, 4.0, 8.0])
    weights = np.array([1.0, 2.0, 4.0])
    expected = np.array([22.0, 42.0, 52.0] if correlate else [16.0, 24.0, 40.0])
    if omit_nan:
        expected /= 7

    def mutate_sources() -> None:
        values.fill(100)
        weights.fill(0)

    actual = _core._test_convolve_after_snapshot(
        values,
        weights,
        ["reflect"],
        0.0,
        [0],
        mutate_sources,
        correlate=correlate,
        omit_nan=omit_nan,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, 100)
    np.testing.assert_array_equal(weights, 0)


@requires_hooks
@pytest.mark.parametrize("correlate", [False, True])
@pytest.mark.parametrize("omit_nan", [False, True])
def test_weighted_filter_allows_both_sources_to_reallocate_after_snapshot(
    correlate: bool, omit_nan: bool
) -> None:
    script = textwrap.dedent(
        """
        import sys

        import numpy as np
        from ndfiltrs import _core

        correlate = sys.argv[1] == "True"
        omit_nan = sys.argv[2] == "True"
        values = np.array([2.0, 4.0, 8.0])
        weights = np.array([1.0, 2.0, 4.0])
        expected = np.array([22.0, 42.0, 52.0] if correlate else [16.0, 24.0, 40.0])
        if omit_nan:
            expected /= 7

        def reallocate():
            values.resize(100000, refcheck=False)
            weights.resize(100000, refcheck=False)
            values.fill(100)
            weights.fill(0)

        actual = _core._test_convolve_after_snapshot(
            values, weights, ["reflect"], 0.0, [0], reallocate,
            correlate=correlate, omit_nan=omit_nan,
        )
        np.testing.assert_array_equal(actual, expected)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(correlate), str(omit_nan)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


@requires_hooks
@pytest.mark.parametrize("correlate", [False, True])
@pytest.mark.parametrize("omit_nan", [False, True])
def test_python_thread_can_mutate_sources_while_native_weighted_filter_is_detached(
    correlate: bool, omit_nan: bool
) -> None:
    values = np.array([2.0, 4.0, 8.0])
    weights = np.array([1.0, 2.0, 4.0])
    expected = np.array([22.0, 42.0, 52.0] if correlate else [16.0, 24.0, 40.0])
    if omit_nan:
        expected /= 7
    probe = _core._DetachProbe()
    progressed = threading.Event()
    errors: list[Exception] = []

    def worker() -> None:
        try:
            if not probe.wait_until_started(5.0):
                raise AssertionError(
                    "weighted filter did not enter its detached section"
                )
            values.fill(100)
            weights.fill(0)
            progressed.set()
        except Exception as error:
            errors.append(error)
        finally:
            probe.signal_progress()

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        actual = _core._test_convolve_during_detach(
            values,
            weights,
            ["reflect"],
            0.0,
            [0],
            probe,
            correlate=correlate,
            omit_nan=omit_nan,
        )
    finally:
        probe.signal_progress()
        thread.join(timeout=10.0)
    assert not thread.is_alive(), "Python worker did not finish"
    assert not errors, repr(errors)
    assert progressed.is_set(), "Python made no progress in the detached section"
    np.testing.assert_array_equal(actual, expected)
