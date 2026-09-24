"""Generate the versioned synthetic reference scene without using ndfiltrs.

The committed NPZ is the canonical input. Regeneration is explicit and writes
elsewhere by default; NumPy RNG/transcendental changes can alter regenerated
bytes even with the same seed. A new scientific scene needs a new version.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import platform
import zipfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)
SCENE_ID = "ndfiltrs-gold-v1"
SHAPE = (256, 320)
SEED = 20260906
# Coordinates are (y, x), amplitudes are pixel-center model units, not fluxes.
STARS = (
    (48.25, 56.75, 220.0, 1.4),
    (81.5, 139.25, 120.0, 2.2),
    (153.25, 75.5, 300.0, 1.7),
    (194.5, 243.25, 160.0, 3.0),
    (4.25, 208.75, 180.0, 1.5),
    (229.0, 314.5, 90.0, 2.0),
)


def build_scene() -> tuple[dict[str, NDArray], dict[str, object]]:
    """Construct analytic layers, a fixed noisy realization, and defects.

    Returns
    -------
    arrays : dict of numpy.ndarray
        Numeric layers and masks; no Python objects are stored in the NPZ.
    metadata : dict
        Versioned scientific parameters and coordinate conventions.
    """
    y, x = np.indices(SHAPE, dtype=np.float64)
    background = 100.0 + 0.03 * x + 0.02 * y
    sources = np.zeros(SHAPE, dtype=np.float64)
    for cy, cx, amplitude, sigma in STARS:
        sources += amplitude * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2))
    extended = 35.0 * np.exp(
        -0.5 * (((x - 230.0) / 22.0) ** 2 + ((y - 108.0) / 12.0) ** 2)
    )
    step = 18.0 * (x >= 160)
    texture = 2.5 * np.sin(2 * np.pi * x / 19.0) * np.cos(2 * np.pi * y / 31.0)
    model = background + sources + extended + step + texture
    # Synthetic Gaussian approximation to signal-dependent shot + read noise.
    noise_sigma = np.sqrt(model / 4.0 + 2.0**2)
    noise_rng = np.random.Generator(np.random.PCG64(SEED))
    noise = noise_rng.standard_normal(SHAPE) * noise_sigma
    clean = model + noise

    # Separate seed makes the mask independent of the noise-generation calls.
    mask_rng = np.random.Generator(np.random.PCG64(SEED + 1))
    sparse = np.zeros(SHAPE, dtype=bool)
    anchors = [(0, 0), (255, 319), (0, 100), (130, 0), (120, 159), (120, 160)]
    anchors.extend((int(round(cy)), int(round(cx))) for cy, cx, _, _ in STARS)
    for cy, cx in anchors:
        sparse[cy, cx] = True
    sparse_count = int(np.prod(SHAPE) // 1000)
    candidates = np.flatnonzero(~sparse)
    selected = mask_rng.choice(
        candidates, size=sparse_count - int(sparse.sum()), replace=False
    )
    sparse.flat[selected] = True
    stress = sparse.copy()
    stress[:, 34] = True
    stress[176, 180:255] = True
    stress[175:184, 40:49] = True
    stress[90:127, 93] = True
    stress[109, 76:112] = True
    stress[:4, 260:267] = True
    stress[199:204, 315:] = True

    arrays = {
        "background": background,
        "model": model,
        "noise_sigma": noise_sigma,
        "noise": noise,
        "clean": clean,
    }
    for name, mask in (("sparse", sparse), ("stress", stress)):
        observed = clean.copy()
        kinds = np.zeros(SHAPE, dtype=np.uint8)
        offsets = np.flatnonzero(mask)
        # Interleaved defect kinds are fixture parameters, never dispatch rules.
        for kind in range(1, 6):
            locations = offsets[kind - 1 :: 5]
            kinds.flat[locations] = kind
            if kind == 1:
                observed.flat[locations] = clean.flat[locations] + 500.0
            elif kind == 2:
                observed.flat[locations] = -100.0
            elif kind == 3:
                observed.flat[locations] = np.nan
            elif kind == 4:
                observed.flat[locations] = np.inf
            else:
                observed.flat[locations] = -np.inf
        arrays[f"observed_{name}"] = observed
        arrays[f"mask_{name}"] = mask
        arrays[f"defect_kind_{name}"] = kinds

    cy, cx = np.indices((9, 11), dtype=np.float64)
    arrays["control_flat"] = np.full((9, 11), 16.0)
    arrays["control_ramp"] = 16.0 + 2.0 * cx + 4.0 * cy
    control_mask = np.zeros((9, 11), dtype=bool)
    control_mask[4, 5] = True
    arrays["control_mask"] = control_mask
    # Stable little-endian storage makes the digest independent of host byte order.
    arrays = {
        key: np.asarray(value, dtype="<f8" if value.dtype.kind == "f" else value.dtype)
        for key, value in arrays.items()
    }
    metadata = {
        "schema_version": 1,
        "scene_id": SCENE_ID,
        "shape": list(SHAPE),
        "coordinates": (
            "zero-based array[y, x], y increases downward; "
            "model sampled at integer pixel centers"
        ),
        "units": (
            "arbitrary signal units; synthetic Gaussian noise, "
            "not a calibrated detector simulation"
        ),
        "model": {
            "background": "100 + 0.03*x + 0.02*y",
            "stars": [
                dict(y=cy, x=cx, amplitude=amp, sigma_pixels=sigma)
                for cy, cx, amp, sigma in STARS
            ],
            "extended": "35*exp(-0.5*(((x-230)/22)**2 + ((y-108)/12)**2))",
            "step": "18*(x >= 160)",
            "texture": "2.5*sin(2*pi*x/19)*cos(2*pi*y/31)",
            "noise_sigma": "sqrt(model/4 + 2**2)",
        },
        "rng": {"algorithm": "PCG64", "noise_seed": SEED, "mask_seed": SEED + 1},
        "defect_kinds": {
            "0": "valid",
            "1": "clean + 500",
            "2": "-100",
            "3": "NaN",
            "4": "+inf",
            "5": "-inf",
        },
        "cases": {
            "sparse": {
                "count": int(sparse.sum()),
                "fraction": float(sparse.mean()),
                "fixed_yx": anchors,
            },
            "stress": {
                "count": int(stress.sum()),
                "fraction": float(stress.mean()),
                "regions": [
                    "full column x=34",
                    "row y=176,x=180:255",
                    "block y=175:184,x=40:49",
                    "cross y=90:127,x=93 and y=109,x=76:112",
                    "edge y=0:4,x=260:267",
                    "edge y=199:204,x=315:320",
                ],
            },
        },
        "controls": {
            "flat": "16 everywhere",
            "ramp": "16 + 2*x + 4*y",
            "mask": (
                "only [y=4,x=5]; exact interior reconstruction "
                "for linear and median size=3"
            ),
        },
        "truth": {
            "model": "noiseless signal",
            "clean": (
                "uncorrupted noisy realization; not an exact expected repair output"
            ),
        },
    }
    return arrays, metadata


def save_scene(output: Path) -> None:
    """Write the scene, fixed-timestamp NPZ, and array integrity manifest.

    Parameters
    ----------
    output : pathlib.Path
        New or empty directory. Existing scene files are never overwritten.
    """
    if (output / "scene.npz").exists() or (output / "manifest.json").exists():
        raise FileExistsError("scene already exists; regenerate into a new directory")
    arrays, metadata = build_scene()
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output / "scene.npz", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name, value in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, buffer.getvalue())
    metadata["arrays"] = {
        name: {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        }
        for name, value in sorted(arrays.items())
    }
    metadata["archive_sha256"] = hashlib.sha256(
        (output / "scene.npz").read_bytes()
    ).hexdigest()
    metadata["generation"] = {
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Wrote %s with %d arrays to %s", SCENE_ID, len(arrays), output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ndfiltrs-gold-v1"))
    save_scene(parser.parse_args().output)
