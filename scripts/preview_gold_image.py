"""Render a contact sheet for a gold image; requires NumPy and Pillow only."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = np.array(
    [
        [0, 0, 0],
        [255, 75, 75],
        [60, 155, 255],
        [246, 206, 67],
        [217, 115, 255],
        [60, 230, 177],
    ],
    dtype=np.uint8,
)


def render_preview(data_path: Path, output: Path) -> None:
    """Show truth, corruptions, and their masks using a common display scale.

    Parameters
    ----------
    data_path : pathlib.Path
        Canonical numeric NPZ fixture.
    output : pathlib.Path
        Destination PNG; no input or expected test output is changed.
    """
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    # Display-only asinh stretch; clipping here never changes numerical inputs.
    lower, upper = np.percentile(arrays["clean"], (0.5, 99.9))
    panels = []
    for name, label in (
        ("model", "Noiseless model"),
        ("clean", "Clean noisy realization"),
        ("observed_sparse", "Sparse defects"),
        ("observed_stress", "Stress defects"),
    ):
        data = arrays[name]
        scaled = np.arcsinh(
            8 * np.clip((data - lower) / (upper - lower), 0, 1)
        ) / np.arcsinh(8)
        gray = (np.nan_to_num(scaled) * 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        rgb[np.isnan(data)] = COLORS[3]
        rgb[np.isposinf(data)] = COLORS[4]
        rgb[np.isneginf(data)] = COLORS[5]
        panels.append((label, Image.fromarray(rgb)))
    for case in ("sparse", "stress"):
        count = int(arrays[f"mask_{case}"].sum())
        label = f"{case.capitalize()} mask: {count} pixels"
        panels.append((label, Image.fromarray(COLORS[arrays[f"defect_kind_{case}"]])))
    font = ImageFont.load_default(size=19)
    small = ImageFont.load_default(size=15)
    panel_width, panel_height = 384, 307
    margin, gap, top, label_height = 22, 18, 86, 30
    canvas = Image.new(
        "RGB",
        (
            3 * panel_width + 2 * gap + 2 * margin,
            top + 2 * (panel_height + label_height + gap) + 84,
        ),
        "#10141d",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 16), "ndfiltrs / GOLD v1", fill="white", font=font)
    draw.text(
        (margin, 44),
        ("256 x 320 pixels | fixed signal + noise | original arrays preserved"),
        fill="#b9c5d8",
        font=small,
    )
    for index, (label, image) in enumerate(panels):
        x = margin + (index % 3) * (panel_width + gap)
        y = top + (index // 3) * (panel_height + label_height + gap)
        draw.text((x, y), label, fill="white", font=font)
        canvas.paste(
            image.resize((panel_width, panel_height), Image.Resampling.NEAREST),
            (x, y + label_height),
        )
    y = canvas.height - 65
    for index, (label, color) in enumerate(
        zip(("hot", "dead", "NaN", "+inf", "-inf"), COLORS[1:], strict=True)
    ):
        x = margin + index * 150
        draw.rectangle((x, y, x + 13, y + 13), fill=tuple(color))
        draw.text((x + 21, y - 2), label, fill="#c9d3e2", font=small)
    draw.text(
        (margin, canvas.height - 32),
        (
            "Display: shared asinh stretch; color-coded nonfinite pixels. "
            "Synthetic truth is not exact repair output."
        ),
        fill="#b9c5d8",
        font=small,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    logging.info("Wrote %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("tests/data/gold_v1/scene.npz")
    )
    parser.add_argument("--output", type=Path, default=Path("docs/images/gold-v1.png"))
    arguments = parser.parse_args()
    render_preview(arguments.data, arguments.output)
