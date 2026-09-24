"""Verify that native filtering releases the Python GIL."""

import subprocess
import sys
import textwrap
import threading

import numpy as np
import pytest
from ndfiltrs import _core

pytestmark = pytest.mark.skipif(
    not all(
        hasattr(_core, name)
        for name in (
            "_DetachProbe",
            "_test_median_filter_after_snapshot",
            "_test_median_filter_during_detach",
        )
    ),
    reason=(
        "ndfiltrs._core was built without test-hooks; deterministic GIL tests "
        "require the private test hooks"
    ),
)


@pytest.mark.parametrize("mean", [False, True], ids=["median", "mean"])
def test_reallocation_after_snapshot_preserves_original_input(mean: bool) -> None:
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        from ndfiltrs import _core

        values = np.arange(1024, dtype=np.float64)
        result = _core._test_median_filter_after_snapshot(
            values,
            [1],
            ["reflect"],
            0.0,
            lambda: values.resize(2048, refcheck=False),
            mean=sys.argv[1] == "True",
        )
        np.testing.assert_array_equal(result, np.arange(1024, dtype=np.float64))
        print("snapshot-ok")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(mean)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "snapshot-ok" in completed.stdout


@pytest.mark.parametrize("mean", [False, True], ids=["median", "mean"])
def test_python_thread_runs_during_detached_filter(mean: bool) -> None:
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    probe = _core._DetachProbe()
    progress_markers: list[str] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            if not probe.wait_until_started(5.0):
                raise AssertionError("detached filter probe did not start")
            progress_markers.append("Python ran while native code was detached")
        except BaseException as error:
            errors.append(error)
        finally:
            probe.signal_progress()

    worker_thread = threading.Thread(target=worker)
    worker_thread.start()
    result = _core._test_median_filter_during_detach(
        values,
        [1, 1],
        ["reflect", "reflect"],
        0.0,
        probe,
        mean=mean,
    )
    worker_thread.join(timeout=5.0)

    assert not worker_thread.is_alive(), "Python worker did not finish"
    assert not errors, f"detached filter hook failed: {errors[0]!r}"
    assert progress_markers == ["Python ran while native code was detached"]
    np.testing.assert_array_equal(result, values)


@pytest.mark.parametrize("mean", [False, True], ids=["median", "mean"])
def test_python_alias_mutation_does_not_change_running_filter(mean: bool) -> None:
    values = np.zeros((1024, 1024), dtype=np.float64)
    mutation_finished = threading.Event()

    def mutate_after_snapshot() -> None:
        values.fill(1.0)
        mutation_finished.set()

    result = _core._test_median_filter_after_snapshot(
        values,
        [9, 9],
        ["reflect", "reflect"],
        0.0,
        mutate_after_snapshot,
        mean=mean,
    )

    assert mutation_finished.is_set(), "mutation must finish before filter return"
    np.testing.assert_array_equal(result, 0.0)
