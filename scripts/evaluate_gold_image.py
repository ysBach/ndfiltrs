"""Measure repair quality on frozen inputs; no empirical pass/fail thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import ndfiltrs as nf
import numpy as np
from ndfiltrs import _core


def evaluate(data: Path, repeats: int) -> dict[str, object]:
    """Report resolved fraction, conditional errors, and public-call timings.

    Parameters
    ----------
    data : pathlib.Path
        Frozen NPZ input bundle.
    repeats : int
        Timed calls after one untimed warmup and validity check.

    Returns
    -------
    dict
        Diagnostic metrics and source identities. RMSE/bias include only
        finite repaired masked pixels; unresolved counts are always explicit.
    """
    with np.load(data, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    records = []
    for dtype in (np.float32, np.float64):
        for case in ("sparse", "stress"):
            image = arrays[f"observed_{case}"].astype(dtype)
            mask = arrays[f"mask_{case}"]
            clean = arrays["clean"].astype(dtype)
            model = arrays["model"].astype(dtype)
            for method in ("linear", "median"):
                options = {"method": method}
                if method == "median":
                    options.update(
                        size=3, mode="constant", min_count=3, ignore_inf=True
                    )
                output = nf.fixpix(image, mask, **options)
                np.testing.assert_array_equal(
                    output[~mask].view(np.uint8), image[~mask].view(np.uint8)
                )
                timings = []
                for _ in range(repeats):
                    start = time.perf_counter_ns()
                    output = nf.fixpix(image, mask, **options)
                    timings.append((time.perf_counter_ns() - start) / 1e9)
                resolved = mask & np.isfinite(output)
                residual = output[resolved].astype(np.float64) - clean[resolved]
                model_residual = output[resolved].astype(np.float64) - model[resolved]
                count, total = int(resolved.sum()), int(mask.sum())
                records.append(
                    {
                        "case": case,
                        "dtype": np.dtype(dtype).name,
                        "options": options,
                        "masked_pixels": total,
                        "resolved_pixels": count,
                        "unresolved_pixels": total - count,
                        "resolved_fraction": count / total,
                        "rmse_vs_clean_resolved_only": float(
                            np.sqrt(np.mean(residual**2))
                        )
                        if count
                        else None,
                        "bias_vs_clean_resolved_only": float(np.mean(residual))
                        if count
                        else None,
                        "rmse_vs_model_resolved_only": float(
                            np.sqrt(np.mean(model_residual**2))
                        )
                        if count
                        else None,
                        "median_seconds": statistics.median(timings),
                        "samples_seconds": timings,
                    }
                )
    sources = {}
    for name, path in (
        ("data", data),
        ("extension", Path(_core.__file__)),
        ("wrapper", Path(nf.fixpix.__code__.co_filename)),
        ("evaluator", Path(__file__)),
    ):
        sources[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "scene_id": "ndfiltrs-gold-v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sources": sources,
        "interpretation": (
            "Synthetic diagnostic, not scientific acceptance thresholds. "
            "Errors exclude unresolved masked pixels; compare coverage too."
        ),
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("tests/data/gold_v1/scene.npz")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluate(args.data, args.repeats), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
