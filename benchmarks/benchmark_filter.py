"""Measure the installed public filtering API."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import ndfiltrs as nf
import numpy as np

from benchmarks._common import make_input


def main() -> None:
    """Run deterministic public-call timings and emit JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    arguments = parser.parse_args()
    records: list[dict[str, object]] = []
    for dtype in (np.float32, np.float64):
        for policy in ("plain", "nan"):
            values = make_input((256, 256), dtype)
            if policy == "nan":
                values[::11, ::13] = np.nan
            function = nf.median_filter if policy == "plain" else nf.nanmedian_filter
            function(values, 5)
            samples = []
            for _ in range(arguments.repeats):
                start = time.perf_counter()
                function(values, 5)
                samples.append(time.perf_counter() - start)
            records.append(
                {
                    "dtype": np.dtype(dtype).name,
                    "policy": policy,
                    "shape": [256, 256],
                    "size": 5,
                    "threads": nf.get_num_threads(),
                    "median_seconds": statistics.median(samples),
                    "min_seconds": min(samples),
                    "max_seconds": max(samples),
                }
            )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
