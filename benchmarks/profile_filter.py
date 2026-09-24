"""Keep a validated native filter busy for an external CPU sampler.

Example: run with RAYON_NUM_THREADS=1, then use the logged process ID with
``sample PID 8 1 -file /tmp/filter.sample.txt`` on macOS. Sampling is separate
from comparative timing; it must not run during benchmark measurements.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from functools import partial

import ndfiltrs as nf
import numpy as np

from benchmarks._common import make_input


def main() -> None:
    """Run one selected filter repeatedly for the requested sampling interval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("sortnet", "boxcar"), required=True)
    parser.add_argument("--shape", type=int, nargs="+", default=(256, 256))
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    if any(length <= 0 for length in args.shape) or args.size <= 0:
        parser.error("shape and size must be positive")
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("seconds must be finite and positive")
    array = make_input(tuple(args.shape), getattr(np, args.dtype))
    function = nf.median_filter if args.method == "sortnet" else nf.mean_filter
    run = partial(function, array, args.size, method=args.method)
    run()
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "pid=%d method=%s threads=%d", os.getpid(), args.method, nf.get_num_threads()
    )
    deadline = time.perf_counter() + args.seconds
    calls = 0
    while time.perf_counter() < deadline:
        run()
        calls += 1
    logging.info("Completed %d calls; sampling run, not a speed comparison", calls)


if __name__ == "__main__":
    main()
