"""Compare direct and separable Gaussian smoothing on fixed reproducible inputs."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import ndfiltrs as nf
import numpy as np


def benchmark(repeats: int) -> dict:
    """Measure median call times after checking both methods agree.

    Parameters
    ----------
    repeats : int
        Positive number of timed calls for each method.

    Returns
    -------
    dict
        Environment, workload parameters, and timings in seconds.
    """
    rng = np.random.default_rng(20260922)
    results = []
    for shape, radius in (((128, 128), 2), ((128, 128), 6), ((32, 32, 32), 3)):
        for missing in (False, True):
            data = rng.normal(size=shape)
            if missing:
                data[rng.random(shape) < 0.2] = np.nan
            function = nf.nangaussian_filter if missing else nf.gaussian_filter
            sigma = radius / 3
            reference = function(data, sigma, radius=radius, method="direct")
            actual = function(data, sigma, radius=radius, method="separable")
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
            timings = {}
            for method in ("direct", "separable"):
                elapsed = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    function(data, sigma, radius=radius, method=method)
                    elapsed.append(time.perf_counter() - start)
                timings[method] = float(np.median(elapsed))
            results.append(
                {
                    "shape": shape,
                    "radius": radius,
                    "sigma": sigma,
                    "missing": missing,
                    "seconds": timings,
                    "speedup": timings["direct"] / timings["separable"],
                }
            )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rayon_threads": nf.get_num_threads(),
        "repeats": repeats,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    arguments.output.write_text(
        json.dumps(benchmark(arguments.repeats), indent=2) + "\n"
    )
