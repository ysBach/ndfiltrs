"""Measure sparse repairs, output-copy cost, and an optional prior release.

All fixpix timings include public validation. Copy-free no-ops are labeled
separately because their ownership contract differs. The median reference
filters the whole masked image; it is a composition baseline, not SciPy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import ndfiltrs as nf
import numpy as np
from ndfiltrs import _core

from benchmarks._common import load_baseline


def median_reference(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Build replacement estimates by filtering the entire masked image."""
    work = array.copy()
    work[mask] = np.nan
    filtered = nf.nanmedian_filter(work, size=3)
    output = array.copy()
    output[mask] = filtered[mask]
    return output


def main() -> None:
    """Check results before timing sparse public calls in shuffled order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, help="prior installed ndfiltrs package directory"
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--loops", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or args.loops < 1:
        parser.error("repeats and loops must be positive")
    baseline = None
    if args.baseline is not None:
        baseline = load_baseline(args.baseline)
    records = []
    rng = np.random.default_rng(20260906)
    for dtype in (np.float32, np.float64):
        for side in (512, 2048):
            array = rng.normal(size=(side, side)).astype(dtype)
            offsets = rng.permutation(array.size)
            for count in (0, 1, int(array.size * 0.0001), int(array.size * 0.001)):
                mask = np.zeros(array.shape, dtype=bool)
                mask.flat[offsets[:count]] = True
                original, original_mask = array.copy(), mask.copy()
                runners: dict[str, Callable[[], object]] = {
                    "linear_copy": partial(nf.fixpix, array, mask),
                    "linear_copy_false": partial(nf.fixpix, array, mask, copy=False),
                    "median_copy": partial(nf.fixpix, array, mask, method="median"),
                    "median_copy_false": partial(
                        nf.fixpix, array, mask, method="median", copy=False
                    ),
                    "array_copy": array.copy,
                    "mask_any": mask.any,
                }
                expected_linear = nf.fixpix(array, mask)
                if baseline is not None:
                    runners["previous_linear"] = partial(baseline.fixpix, array, mask)
                    np.testing.assert_array_equal(
                        runners["previous_linear"](), expected_linear
                    )
                expected_median = median_reference(array, mask)
                # Keep full filtering only on the smaller benchmark image.
                if side == 512:
                    runners["median_full_filter"] = partial(
                        median_reference, array, mask
                    )
                for key, runner in runners.items():
                    result = runner()
                    if key == "mask_any":
                        assert bool(result) == (count != 0)
                        continue
                    expected = (
                        expected_median if key.startswith("median") else expected_linear
                    )
                    if key == "array_copy":
                        expected = array
                    np.testing.assert_array_equal(result, expected)
                    assert result.dtype == array.dtype and result.shape == array.shape
                    if key.endswith("copy_false") and count == 0:
                        assert result is array
                    else:
                        assert not np.shares_memory(result, array)
                samples: dict[str, list[float]] = {key: [] for key in runners}
                for _ in range(args.repeats):
                    for key in rng.permutation(list(runners)):
                        start = time.perf_counter_ns()
                        for _ in range(args.loops):
                            result = runners[key]()
                            del result
                        samples[key].append(
                            (time.perf_counter_ns() - start) / args.loops / 1e9
                        )
                np.testing.assert_array_equal(array, original)
                np.testing.assert_array_equal(mask, original_mask)
                records.append(
                    {
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "bad_pixels": count,
                        "timings": {
                            key: {
                                "median_seconds": statistics.median(values),
                                "samples_seconds": values,
                            }
                            for key, values in samples.items()
                        },
                    }
                )
    sources = {}
    for name, file in (
        ("current_extension", _core.__file__),
        ("benchmark", __file__),
        ("wrapper", nf.fixpix.__code__.co_filename),
    ):
        path = Path(file).resolve()
        sources[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    if baseline is not None:
        path = Path(baseline._core.__file__).resolve()
        sources["previous_extension"] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    json.dump(
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "sources": sources,
            "repeats": args.repeats,
            "loops": args.loops,
            "warmup_calls": 1,
            "method": (
                "public calls; shuffled order; parity and ownership preflight; "
                "copy_false aliases only no-ops"
            ),
            "records": records,
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
