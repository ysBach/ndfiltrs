"""Compare ``mesh_background`` with sep and photutils after checking agreement.

Each timed call produces both full-resolution maps (background and RMS):
``sep.Background(...)`` plus ``back()`` and ``rms()``, and photutils
``Background2D(...)`` plus ``background`` and ``background_rms``. Agreement is
checked on every input before any timer is read.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ndfiltrs as nf
import numpy as np
import photutils
import sep
from numpy.typing import NDArray
from photutils.background import Background2D

LOGGER = logging.getLogger(__name__)
SIGMA = 5.0
PHOTUTILS = {
    "estimator": "sigma_clip",
    "maxiters": 10,
    "statistic": "mode",
    "mode_estimator": "sex",
    "rms": "std",
    "min_valid_fraction": 0.9,
    "valid_count": "retained",
    "valid_base": "box",
    "fill": "idw",
    "filter_edge": "truncate",
}


def make_image(size: int, dtype: type[np.float32] | type[np.float64]) -> NDArray:
    """Sloped sky with Gaussian noise (sigma 5) and 2000 bright point sources."""
    rng = np.random.default_rng(2026)
    y, x = np.mgrid[:size, :size] / size
    image = 100.0 + 20.0 * x - 10.0 * y + rng.normal(0.0, SIGMA, (size, size))
    rows, cols = rng.integers(2, size - 2, (2, 2000))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            image[rows + dy, cols + dx] += 300.0 / (1 + dy * dy + dx * dx)
    return image.astype(dtype)


def check(image: NDArray, box: int) -> dict[str, float]:
    """Agreement of ndfiltrs with sep and photutils on one image.

    Returns
    -------
    dict
        Maximum differences in units of the noise sigma (sep, photutils maps)
        or relative (photutils meshes).

    Raises
    ------
    AssertionError
        A difference exceeds its tolerance (see ``docs/background.md``).
    """
    reference = sep.Background(image, bw=box, bh=box, fw=3, fh=3)
    ours = nf.mesh_background(image, box, interpolation="sextractor")
    sep_back, sep_rms = reference.back(), reference.rms()
    # Histogram bin of sep: 10 sigma / nlevels with nlevels ~ 0.95 N.
    tolerance = 4 * 10 * SIGMA / (0.95 * box * box)
    result = {
        "sep_background_max": float(np.abs(ours.background - sep_back).max()),
        "sep_rms_max": float(np.abs(ours.rms - sep_rms).max()),
        "sep_tolerance": tolerance,
    }
    assert result["sep_background_max"] <= tolerance, result
    assert result["sep_rms_max"] <= tolerance, result
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phot = Background2D(image, box, filter_size=3)
        phot_back = phot.background
    clipped = nf.mesh_background(image, box, **PHOTUTILS)
    mesh_error = np.abs(clipped.mesh_background / phot.background_mesh - 1).max()
    rms_error = np.abs(clipped.mesh_rms / phot.background_rms_mesh - 1).max()
    result |= {
        "photutils_mesh_relative": float(mesh_error),
        "photutils_rms_mesh_relative": float(rms_error),
        # Different interpolators: natural spline vs scipy.ndimage.zoom.
        "photutils_background_max": float(np.abs(clipped.background - phot_back).max()),
    }
    # photutils computes in float64, then rounds meshes to the input dtype.
    limit = 2 * np.finfo(np.float32).eps if image.dtype == np.float32 else 1e-12
    assert mesh_error < limit and rms_error < limit, result
    return result


def timed(call: Callable[[], object], repeats: int) -> list[float]:
    """Wall times in milliseconds after one warmup call."""
    call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def sep_call(image: NDArray, box: int) -> tuple[NDArray, NDArray]:
    background = sep.Background(image, bw=box, bh=box, fw=3, fh=3)
    return background.back(), background.rms()


def photutils_call(image: NDArray, box: int, threads: int) -> tuple[NDArray, NDArray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        background = Background2D(image, box, filter_size=3, n_threads=threads)
        return background.background, background.background_rms


def main() -> None:
    """Check agreement, then time each implementation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--box", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    threads = nf.get_num_threads()
    report: dict[str, Any] = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sep": sep.__version__,
            "photutils": photutils.__version__,
            "ndfiltrs_threads": threads,
            "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS"),
        },
        "cases": [],
    }
    box = arguments.box
    for size in arguments.sizes:
        for dtype in (np.float32, np.float64):
            image = make_image(size, dtype)
            agreement = check(image, box)
            calls = {
                "ndfiltrs spline": lambda image=image: nf.mesh_background(image, box),
                "ndfiltrs sextractor": lambda image=image: nf.mesh_background(
                    image, box, interpolation="sextractor"
                ),
                "ndfiltrs photutils rules": lambda image=image: nf.mesh_background(
                    image, box, **PHOTUTILS
                ),
                "sep": lambda image=image: sep_call(image, box),
                "photutils n_threads=1": lambda image=image: photutils_call(
                    image, box, 1
                ),
                f"photutils n_threads={threads}": lambda image=image: photutils_call(
                    image, box, threads
                ),
            }
            medians = {}
            for name, call in calls.items():
                samples = timed(call, arguments.repeats)
                medians[name] = statistics.median(samples)
                LOGGER.info(
                    "%d %s %s: %.1f ms", size, dtype.__name__, name, medians[name]
                )
            report["cases"].append(
                {
                    "size": size,
                    "dtype": dtype.__name__,
                    "box": box,
                    "agreement": agreement,
                    "median_ms": medians,
                }
            )
            LOGGER.info("agreement: %s", agreement)
    if arguments.output:
        arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
