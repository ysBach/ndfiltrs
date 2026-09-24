"""Fresh-process tests for Rayon configuration and determinism."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_fresh_process_observes_rayon_thread_count(threads: int) -> None:
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(threads)
    result = subprocess.run(
        [sys.executable, "-c", "import ndfiltrs; print(ndfiltrs.get_num_threads())"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout.strip() == str(threads)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    ("function_name", "ignore_inf"),
    [
        ("median_filter", False),
        ("nanmedian_filter", False),
        ("nanmedian_filter", True),
        ("mean_filter", False),
        ("nanmean_filter", False),
        ("nanmean_filter", True),
        ("min_filter", False),
        ("max_filter", False),
        ("nanmin_filter", False),
        ("nanmin_filter", True),
        ("nanmax_filter", False),
        ("nanmax_filter", True),
        ("var_filter", False),
        ("std_filter", False),
        ("nanvar_filter", False),
        ("nanvar_filter", True),
        ("nanstd_filter", False),
        ("nanstd_filter", True),
    ],
)
def test_output_is_bitwise_identical_across_thread_counts(
    dtype: str,
    function_name: str,
    ignore_inf: bool,
) -> None:
    program = """
import hashlib
import sys
import numpy as np
import ndfiltrs

rng = np.random.default_rng(319)
values = rng.normal(size=(17, 19)).astype(getattr(np, sys.argv[1]))
values[::5, ::4] = np.nan
values[1::6, 2::7] = np.inf
function = getattr(ndfiltrs, sys.argv[2])
kwargs = {}
if sys.argv[2].startswith("nan"):
    kwargs["ignore_inf"] = sys.argv[3] == "true"
output = function(values, (5, 3), mode=("mirror", "wrap"), **kwargs)
print(hashlib.sha256(output.tobytes()).hexdigest())
"""
    digests: list[str] = []
    for threads in (1, 2, 4):
        environment = os.environ.copy()
        environment["RAYON_NUM_THREADS"] = str(threads)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                program,
                dtype,
                function_name,
                str(ignore_inf).lower(),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        digests.append(result.stdout.strip())
    assert len(set(digests)) == 1
