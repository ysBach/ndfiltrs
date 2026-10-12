"""Check the rendered website after running ``make docs-build``."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

SITE = Path(__file__).resolve().parents[1] / "docs" / "quarto"


class NavigationParser(HTMLParser):
    """Collect page headings and links inside the top navbar only."""

    def __init__(self) -> None:
        super().__init__()
        self.in_navbar = False
        self.in_heading = False
        self.href: str | None = None
        self.label: list[str] = []
        self.heading: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "nav" and "navbar" in (attributes.get("class") or "").split():
            self.in_navbar = True
        if tag == "h1":
            self.in_heading = True
        if tag == "a" and self.in_navbar:
            self.href = attributes.get("href")
            self.label = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.href is not None:
            self.links.append((self.href, " ".join("".join(self.label).split())))
            self.href = None
        if tag == "nav":
            self.in_navbar = False
        if tag == "h1":
            self.in_heading = False

    def handle_data(self, data: str) -> None:
        if self.href is not None:
            self.label.append(data)
        if self.in_heading:
            self.heading.append(data)


def test_rendered_pages_are_in_every_navbar_with_matching_titles() -> None:
    """Every published page is reachable by its title from every navbar."""
    output = SITE / "_site"
    if not (output / "index.html").is_file():
        pytest.skip("Run make docs-build to check rendered documentation")

    # Derive coverage from page sources, not from the generated menu itself.
    sources = [*SITE.glob("*.qmd"), *SITE.glob("guides/*.qmd"), *SITE.glob("api/*.qmd")]
    pages: dict[Path, NavigationParser] = {}
    for source in sources:
        page = output / source.relative_to(SITE).with_suffix(".html")
        assert page.is_file(), f"Page was not rendered: {source}"
        parser = NavigationParser()
        parser.feed(page.read_text(encoding="utf-8"))
        assert parser.heading, f"Page has no H1: {page}"
        pages[page.resolve()] = parser

    for page, parser in pages.items():
        reachable = set()
        for href, label in parser.links:
            target = urlsplit(href)
            if target.scheme or target.netloc or not target.path:
                continue
            destination = (page.parent / unquote(target.path)).resolve()
            assert destination in pages, f"{page}: unknown navbar target {href}"
            title = " ".join("".join(pages[destination].heading).split())
            assert label == title, f"{page}: {href}: {label!r} != {title!r}"
            reachable.add(destination)
        assert reachable == pages.keys(), f"{page}: missing {pages.keys() - reachable}"
