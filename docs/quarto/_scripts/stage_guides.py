"""Stage the canonical Markdown guides as QMD without duplicating their sources."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlsplit

import yaml

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


def main() -> None:
    """Stage guides and copy the reference image.

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
        (output / destination).write_text(
            f"---\ntitle: {title}\n---\n{body}",
            encoding="utf-8",
        )

    assets = SITE / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        REPOSITORY / "docs" / "images" / "gold-v1.png", assets / "gold-v1.png"
    )


def stage_navigation() -> None:
    """Expose the guides and generated API pages in the navbar.

    Run after quartodoc has generated the API pages and sidebar.

    Raises
    ------
    ValueError
        An API section uses a non-string page entry.
    FileNotFoundError
        A navigation target has not been generated.
    """
    config = yaml.safe_load((SITE / "_quarto.yml").read_text(encoding="utf-8"))
    api = config["quartodoc"]
    api_dir = api["dir"]
    api_menu = [{"href": f"{api_dir}/index.qmd"}]
    for section in api["sections"]:
        api_menu.append({"text": section["title"]})
        for name in section["contents"]:
            if not isinstance(name, str):
                raise ValueError("API navigation requires named quartodoc pages")
            api_menu.append({"href": f"{api_dir}/{name}.qmd"})

    # Quarto resolves href-only labels from the destination title. In particular,
    # API filenames retain module prefixes, while short page titles need not.
    navigation = {
        "website": {
            "navbar": {
                "left": [
                    {"href": "index.qmd"},
                    {"href": "quickstart.qmd"},
                    {
                        "text": "Guides",
                        "menu": [{"href": "guides/index.qmd"}]
                        + [{"href": f"guides/{name}.qmd"} for name in GUIDES],
                    },
                    {"text": "API", "menu": api_menu},
                ]
            }
        }
    }
    for entry in navigation["website"]["navbar"]["left"]:
        for link in entry.get("menu", [entry]):
            if "href" in link and not (SITE / link["href"]).is_file():
                raise FileNotFoundError(f"Missing navigation target: {link['href']}")
    api_output = SITE / api_dir
    (api_output / "_navbar.yml").write_text(
        yaml.safe_dump(navigation, sort_keys=False), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--navigation-only",
        action="store_true",
        help="Generate navbar metadata after quartodoc has built the API pages.",
    )
    if parser.parse_args().navigation_only:
        stage_navigation()
    else:
        main()
