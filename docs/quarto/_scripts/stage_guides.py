"""Stage the canonical Markdown guides as QMD without duplicating their sources."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlsplit

REPOSITORY = Path(__file__).resolve().parents[3]
SITE = REPOSITORY / "docs" / "quarto"
GUIDES = (
    "semantics",
    "convolution",
    "kernels",
    "robust",
    "morphology",
    "fixpix",
    "benchmarks",
)
GITHUB = "https://github.com/ysBach/ndfiltrs/blob/main/"
INTRODUCTIONS = {
    "semantics": (
        "**When to use this guide:** choose a neighborhood statistic when you "
        "need a value or scatter estimate at every output position."
    ),
    "convolution": (
        "**When to use this guide:** choose convolution for weighted sums, "
        "weighted smoothing, or Gaussian derivatives."
    ),
    "kernels": (
        "**When to use this guide:** build Gaussian, Moffat, Student-t, box, "
        "top-hat, or ring weights, or a disk or annulus footprint."
    ),
    "morphology": (
        "**When to use this guide:** find edges or remove compact features "
        "with Sobel, Laplace, or grey morphology."
    ),
    "robust": (
        "**When to use this guide:** choose local percentiles, robust scatter, "
        "or iterative outlier rejection before estimating a mean or median."
    ),
    "fixpix": (
        "**When to use this guide:** choose masked-pixel repair when known bad "
        "positions may change while unmasked samples must stay unchanged."
    ),
    "benchmarks": (
        "**When to use this guide:** check these measurements before making a "
        "performance claim or sizing a production workload."
    ),
}


def main() -> None:
    """Build guide pages and copy the reference image.

    Raises
    ------
    ValueError
        A source guide lacks its title or links outside the repository.
    FileNotFoundError
        A local link refers to a missing file.
    """
    destinations = {
        REPOSITORY / "docs" / f"{name}.md": f"{name}.qmd" for name in GUIDES
    }
    output = SITE / "guides"
    output.mkdir(parents=True, exist_ok=True)
    for source, destination in destinations.items():
        name = source.stem
        heading, body = source.read_text(encoding="utf-8").split("\n", 1)
        if not heading.startswith("# "):
            raise ValueError(f"Expected one leading H1 title in {source}")

        def rewrite_link(match: re.Match[str], *, source: Path = source) -> str:
            """Resolve each local Markdown link against its canonical source."""
            target = match.group(1)
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                return match.group(0)
            linked = (source.parent / parsed.path).resolve()
            relative = linked.relative_to(REPOSITORY)
            if not linked.is_file():
                raise FileNotFoundError(f"Missing link target in {source}: {target}")
            rewritten = destinations.get(linked, GITHUB + relative.as_posix())
            if parsed.query:
                rewritten += "?" + parsed.query
            if parsed.fragment:
                rewritten += "#" + parsed.fragment
            return "](" + rewritten + ")"

        body = re.sub(r"\]\(([^)]+)\)", rewrite_link, body)
        title = json.dumps(heading[2:])
        introduction = INTRODUCTIONS[name]
        (output / destination).write_text(
            f"---\ntitle: {title}\n---\n{introduction}\n\n{body}",
            encoding="utf-8",
        )
    assets = SITE / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        REPOSITORY / "docs" / "images" / "gold-v1.png", assets / "gold-v1.png"
    )


if __name__ == "__main__":
    main()
