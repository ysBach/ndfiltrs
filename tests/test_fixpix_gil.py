"""Deterministic ownership and GIL checks for masked-pixel repair."""

import subprocess
import sys
import textwrap
import threading

import numpy as np
import pytest
from ndfiltrs import _core

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_test_fixpix_after_snapshot"),
    reason="deterministic repair concurrency tests require test-hooks",
)


@pytest.mark.parametrize("median", [False, True])
def test_repair_uses_array_and_mask_snapshots(median: bool) -> None:
    values = np.array([2.0, 99.0, 6.0])
    mask = np.array([0, 1, 0], dtype=np.uint8)

    def mutate_sources() -> None:
        values[:] = 100.0
        mask[:] = 0

    result = _core._test_fixpix_after_snapshot(
        values, mask, [0], mutate_sources, median=median
    )
    np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(values, 100.0)
    np.testing.assert_array_equal(mask, 0)


@pytest.mark.parametrize("median", [False, True])
def test_repair_allows_source_reallocation_after_snapshot(median: bool) -> None:
    script = textwrap.dedent(
        """
        import numpy as np
        import sys
        from ndfiltrs import _core

        values = np.array([2., 99., 6.])
        mask = np.array([0, 1, 0], dtype=np.uint8)
        def reallocate():
            values.resize(100000, refcheck=False)
            mask.resize(100000, refcheck=False)
        result = _core._test_fixpix_after_snapshot(
            values, mask, [0], reallocate, median=sys.argv[1] == "True"
        )
        np.testing.assert_array_equal(result, [2., 4., 6.])
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(median)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("median", [False, True])
def test_python_thread_progresses_during_repair(median: bool) -> None:
    probe = _core._DetachProbe()
    errors: list[BaseException] = []
    progressed = threading.Event()

    def worker() -> None:
        try:
            if not probe.wait_until_started(5.0):
                raise AssertionError("repair did not enter its detached section")
            progressed.set()
        except BaseException as error:
            errors.append(error)
        finally:
            probe.signal_progress()

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        result = _core._test_fixpix_during_detach(
            np.array([2.0, 99.0, 6.0]),
            np.array([0, 1, 0], dtype=np.uint8),
            [0],
            probe,
            median=median,
        )
    finally:
        probe.signal_progress()
        thread.join(timeout=10.0)
    assert not thread.is_alive()
    assert not errors
    assert progressed.is_set()
    np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])
