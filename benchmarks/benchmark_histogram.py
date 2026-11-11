"""Time exact median methods on large square images.

Run with a release extension and no concurrent CPU-heavy work. Set
``RAYON_NUM_THREADS`` before the run; SciPy runs on one thread. Each timed
call is a complete public call: validation, input copy, allocation, and
filtering. A method is skipped for the remaining larger windows of a case when
its predicted call time exceeds ``--budget``. The prediction scales the last
measured time by window area, because `direct`, `sortnet`, and SciPy visit
every window sample.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import ndfiltrs as nf
import numpy as np
import scipy
from numpy.typing import NDArray
from scipy import ndimage

from benchmarks._common import make_input

FloatArray = NDArray[np.float32] | NDArray[np.float64]
METHODS = ("histogram", "direct", "sortnet", "scipy")
LOGGER = logging.getLogger(__name__)


def runner(method: str, array: FloatArray, size: int) -> Callable[[], FloatArray]:
    """Return one complete median call with reflect boundaries."""
    if method == "scipy":
        return partial(ndimage.median_filter, array, size=size, mode="reflect")
    return partial(nf.median_filter, array, size, mode="reflect", method=method)


def bits(array: FloatArray) -> NDArray[np.uint32] | NDArray[np.uint64]:
    """View floats as unsigned integers for bitwise comparison."""
    return array.view(np.uint32 if array.dtype == np.float32 else np.uint64)


def timed(call: Callable[[], FloatArray]) -> tuple[float, FloatArray]:
    """Return the wall time and result of one call."""
    start = time.perf_counter()
    result = call()
    return time.perf_counter() - start, result


def crop_check(dtype: type[np.floating], size: int) -> bool:
    """Compare histogram and direct results on a 256 x 256 input."""
    array = make_input((256, 256), dtype)
    return bool(
        np.array_equal(
            bits(runner("histogram", array, size)()),
            bits(runner("direct", array, size)()),
        )
    )


def run_case(
    array: FloatArray,
    size: int,
    repeats: int,
    budget: float,
    previous: dict[str, tuple[int, float]],
) -> dict[str, Any]:
    """Validate, then time each method that fits the budget."""
    result: dict[str, Any] = {"validation": {}, "seconds": {}, "skipped": {}}
    outputs = {}
    first = {}
    for method in METHODS:
        if method in previous:
            last_size, last_seconds = previous[method]
            predicted = last_seconds * (size / last_size) ** 2
            if predicted > budget:
                result["skipped"][method] = f"predicted {predicted:.0f} s > budget"
                continue
        seconds, outputs[method] = timed(runner(method, array, size))
        first[method] = seconds
        previous[method] = (size, seconds)
    reference = outputs["histogram"]
    for method in ("direct", "sortnet"):
        if method in outputs:
            equal = np.array_equal(bits(outputs[method]), bits(reference))
            result["validation"][f"{method}_bitwise"] = bool(equal)
            if not equal:
                raise AssertionError(f"{method} differs from histogram at size {size}")
    if "direct" not in outputs:
        equal = crop_check(array.dtype.type, size)
        result["validation"]["direct_bitwise_on_256x256"] = equal
        if not equal:
            raise AssertionError(f"histogram differs on the 256 x 256 check, {size}")
    if "scipy" in outputs and size % 2 == 1:
        # Finite inputs and odd windows: SciPy's rank median is the exact median.
        equal = np.array_equal(outputs["scipy"], reference)
        result["validation"]["scipy_values"] = bool(equal)
        if not equal:
            raise AssertionError(f"SciPy differs from histogram at size {size}")
    del outputs
    for method, seconds in first.items():
        samples = [seconds]
        # Calls longer than the budget keep their single validation sample.
        if seconds * repeats <= budget:
            samples = [timed(runner(method, array, size))[0] for _ in range(repeats)]
        result["seconds"][method] = {
            "median": statistics.median(samples),
            "samples": samples,
            "includes_first_call": len(samples) == 1,
        }
    return result


def main() -> None:
    """Write a JSON report of the size matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="1024,4096")
    parser.add_argument("--sizes", default="3,5,7,11,15,21,31,51,71,101,151,201")
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    records = []
    for side in (int(value) for value in args.shapes.split(",")):
        for dtype in (np.dtype(name).type for name in args.dtypes.split(",")):
            array = make_input((side, side), dtype)
            previous: dict[str, tuple[int, float]] = {}
            for size in (int(value) for value in args.sizes.split(",")):
                load = os.getloadavg()
                case = run_case(array, size, args.repeats, args.budget, previous)
                record = {
                    "shape": [side, side],
                    "dtype": np.dtype(dtype).name,
                    "size": size,
                    "load_average_before": load,
                    "load_average_after": os.getloadavg(),
                    **case,
                }
                records.append(record)
                LOGGER.info(
                    "%d %s %d %s",
                    side,
                    np.dtype(dtype).name,
                    size,
                    {k: round(v["median"], 4) for k, v in case["seconds"].items()},
                )
                args.output.write_text(
                    json.dumps(
                        {
                            "scope": __doc__,
                            "platform": platform.platform(),
                            "logical_cpus": os.cpu_count(),
                            "python": platform.python_version(),
                            "python_executable": sys.executable,
                            "versions": {
                                "ndfiltrs": nf.__version__,
                                "numpy": np.__version__,
                                "scipy": scipy.__version__,
                            },
                            "rayon_threads": nf.get_num_threads(),
                            "repeats": args.repeats,
                            "budget_seconds": args.budget,
                            "records": records,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )


if __name__ == "__main__":
    main()
