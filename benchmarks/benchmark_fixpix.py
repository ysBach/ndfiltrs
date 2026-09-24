"""Compare public fixpix calls after exact numerical parity checks.

Run from the source checkout in an environment containing astroimred and SciPy.
The SciPy reference reproduces the repair contract with labeling and linear
interpolation; it is not a claim about the fastest possible SciPy algorithm.
All timed calls include input copies and public API validation. Astroimred
also constructs CCDData, but header updates and logging are disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path

import ndfiltrs as nf
import numpy as np
import scipy
from ndfiltrs import _core

from tests._fixpix_reference import fixpix_reference


def cases(
    dtype: type[np.float32] | type[np.float64],
) -> Iterator[tuple[str, np.ndarray, np.ndarray, bool]]:
    """Yield fixed mask geometries, including sparse overhead cases."""
    rng = np.random.default_rng(20260906)
    for name in (
        "single_pixel",
        "empty_mask",
        "random_1pct",
        "block",
        "columns",
        "crossings",
        "dense",
        "three_dimensional",
        "strided",
        "scipy_2d",
        "scipy_3d",
    ):
        shape = (512, 512)
        if name == "dense":
            shape = (128, 160)
        elif name == "three_dimensional":
            shape = (24, 28, 36)
        elif name == "scipy_2d":
            shape = (64, 64)
        elif name == "scipy_3d":
            shape = (8, 9, 10)
        array = rng.normal(size=shape).astype(dtype)
        mask = np.zeros(shape, dtype=bool)
        if name == "single_pixel":
            mask[10, 10] = True
        elif name == "block":
            mask[200:225, 200:225] = True
        elif name == "columns":
            mask[:, 100:110] = True
        elif name == "crossings":
            mask[100:400, 200:203] = True
            mask[200:203, 100:400] = True
        elif name == "dense":
            mask = rng.random(shape) < 0.7
        elif name in ("three_dimensional", "scipy_2d", "scipy_3d"):
            mask = rng.random(shape) < 0.05
        elif name in ("random_1pct", "strided"):
            mask = rng.random(shape) < 0.01
        if name == "strided":
            array, mask = array[::-1, ::2], mask[::-1, ::2]
        yield name, array, mask, name.startswith("scipy_")


def main() -> None:
    """Run equivalent public calls and emit timings with source identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--astroimred-src",
        type=Path,
        help="directory containing the astroimred package (e.g. ../astroimred/src)",
    )
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    if arguments.astroimred_src is not None:
        source = arguments.astroimred_src.resolve()
        if not (source / "astroimred/__init__.py").is_file():
            parser.error("--astroimred-src must contain astroimred/__init__.py")
        sys.path.insert(0, str(source))
    air = importlib.import_module("astroimred")
    records: list[dict[str, object]] = []
    for dtype in (np.float32, np.float64):
        for name, array, mask, compare_scipy in cases(dtype):
            priority = tuple(range(array.ndim - 1, -1, -1))
            original = array.copy()
            original_mask = mask.copy()
            runners: dict[str, Callable[[], object]] = {
                "ndfiltrs": partial(nf.fixpix, array, mask, priority=priority),
                "astroimred": partial(
                    air.fixpix,
                    array,
                    mask,
                    priority=priority,
                    update_header=False,
                    verbose=False,
                ),
            }
            if compare_scipy:
                runners["scipy_reference"] = partial(
                    fixpix_reference, array, mask, priority=priority
                )
            expected = runners["astroimred"]().data
            for key, runner in runners.items():
                result = runner()  # parity check also warms each implementation
                if key == "astroimred":
                    result = result.data
                np.testing.assert_allclose(
                    result, expected, rtol=0, atol=0, equal_nan=True
                )
                assert result.dtype == dtype and result.shape == array.shape
                assert not np.shares_memory(result, array)
                np.testing.assert_array_equal(array, original)
                np.testing.assert_array_equal(mask, original_mask)
            samples: dict[str, list[float]] = {key: [] for key in runners}
            for repeat in range(arguments.repeats):
                order = list(runners)
                if repeat % 2:
                    order.reverse()
                for key in order:
                    start = time.perf_counter()
                    result = runners[key]()
                    samples[key].append(time.perf_counter() - start)
                    del result
            np.testing.assert_array_equal(array, original)
            np.testing.assert_array_equal(mask, original_mask)
            timings = {
                key: {
                    "median_seconds": statistics.median(values),
                    "min_seconds": min(values),
                    "max_seconds": max(values),
                    "samples_seconds": values,
                }
                for key, values in samples.items()
            }
            records.append(
                {
                    "case": name,
                    "dtype": np.dtype(dtype).name,
                    "shape": list(array.shape),
                    "priority": list(priority),
                    "bad_pixels": int(np.count_nonzero(mask)),
                    "c_contiguous": bool(array.flags.c_contiguous),
                    "timings": timings,
                    "speedup_air_over_nf": timings["astroimred"]["median_seconds"]
                    / timings["ndfiltrs"]["median_seconds"],
                }
            )
    identities = {}
    for name, file in (
        ("astroimred", inspect.getsourcefile(air.fixpix)),
        ("ndfiltrs_extension", _core.__file__),
        ("scipy_reference", inspect.getsourcefile(fixpix_reference)),
        ("benchmark", __file__),
    ):
        assert file is not None
        path = Path(file).resolve()
        identities[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    json.dump(
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "astroimred": importlib.metadata.version("astroimred"),
            "ndfiltrs": nf.__version__,
            "repeats": arguments.repeats,
            "warmup_calls": 1,
            "method": "public calls, alternating timing order; no header updates",
            "sources": identities,
            "records": records,
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
