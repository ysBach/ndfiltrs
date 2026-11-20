"""Check the rendered website after running ``make docs-build``."""

import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

SITE = Path(__file__).resolve().parents[1] / "docs" / "quarto"


class NavigationParser(HTMLParser):
    """Collect headings, fragment IDs, and links in the navbar and article."""

    def __init__(self) -> None:
        super().__init__()
        self.in_navbar = False
        self.in_heading = False
        self.in_main = False
        self.in_pre = False
        self.in_code = False
        self.in_term = False
        self.unlinked_inline: set[str] = set()
        self.unlinked_code: list[str] = []
        self.href: str | None = None
        self.label: list[str] = []
        self.heading: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.article: list[tuple[str, str, bool]] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if element_id := attributes.get("id"):
            self.ids.add(element_id)
        if tag == "main":
            self.in_main = True
        if tag == "pre":
            self.in_pre = True
        if tag == "code":
            self.in_code = True
        if tag == "dt":
            self.in_term = True
        if tag == "nav" and "navbar" in (attributes.get("class") or "").split():
            self.in_navbar = True
        if tag == "h1":
            self.in_heading = True
        if tag == "a" and (self.in_navbar or self.in_main):
            self.href = attributes.get("href")
            self.label = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.href is not None:
            label = " ".join("".join(self.label).split())
            if self.in_navbar:
                self.links.append((self.href, label))
            elif self.in_main:
                self.article.append((self.href, label, self.in_pre))
            self.href = None
        if tag == "main":
            self.in_main = False
        if tag == "pre":
            self.in_pre = False
        if tag == "code":
            self.in_code = False
        if tag == "dt":
            self.in_term = False
        if tag == "nav":
            self.in_navbar = False
        if tag == "h1":
            self.in_heading = False

    def handle_data(self, data: str) -> None:
        if self.in_main and self.in_code and self.href is None:
            self.unlinked_code.append(data)
            if not self.in_pre and not self.in_term:
                self.unlinked_inline.add(data.strip())
        if self.href is not None:
            self.label.append(data)
        if self.in_heading:
            self.heading.append(data)


@pytest.fixture(scope="module")
def pages() -> dict[Path, NavigationParser]:
    """Read each published page once."""
    output = SITE / "_site"
    if not (output / "index.html").is_file():
        pytest.skip("Run make docs-build to check rendered documentation")

    # Derive coverage from page sources, not from the generated menu itself.
    sources = [*SITE.glob("*.qmd"), *SITE.glob("guides/*.qmd"), *SITE.glob("api/*.qmd")]
    parsed = {}
    for source in sources:
        page = output / source.relative_to(SITE).with_suffix(".html")
        assert page.is_file(), f"Page was not rendered: {source}"
        parser = NavigationParser()
        parser.feed(page.read_text(encoding="utf-8"))
        assert parser.heading, f"Page has no H1: {page}"
        parsed[page.resolve()] = parser
    return parsed


def test_rendered_pages_are_in_every_navbar_with_matching_titles(pages) -> None:
    """Every published page is reachable by its title from every navbar."""

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


def test_numbered_guides_match_index_and_menu(pages) -> None:
    """The guide index and menu show the same numbered page titles in order."""
    index = SITE / "_site/guides/index.html"
    parser = pages[index]
    guide_pages = {page for page in pages if page.parent == index.parent} - {index}

    def guide_links(links):
        result = []
        for href, label in links:
            target = urlsplit(href)
            page = (index.parent / target.path).resolve()
            if page in guide_pages and not target.fragment:
                result.append((page, label))
        return result

    menu = guide_links(parser.links)
    article = guide_links((url, label) for url, label, _ in parser.article)
    assert article == menu
    assert {page for page, _ in menu} == guide_pages
    for number, (page, label) in enumerate(menu, start=1):
        assert label == "".join(pages[page].heading)
        assert label.startswith(f"{number:02d}. ")


def test_article_api_links_and_guide_shortcuts(pages) -> None:
    """Check article targets, including calls that used to have no link."""
    site = SITE / "_site"
    public_names = {p.stem for p in pages if p.parent == site / "api"} - {"index"}
    tutorial = pages[site / "guides/background-tutorial.html"]
    mesh = "../api/mesh_background.html#ndfiltrs.mesh_background"
    assert (mesh, "mesh_background", False) in tutorial.article
    assert tutorial.article.count((mesh, "nf.mesh_background", True)) == 2
    for name, domain in (("sep.Background", "sep"), ("Background2D", "photutils")):
        assert any(
            block
            and label == name
            and urlsplit(href).hostname == f"{domain}.readthedocs.io"
            for href, label, block in tutorial.article
        )

    repair = pages[site / "guides/fixpix.html"]
    assert any(
        label == "air.fixpix"
        and href
        == "https://ysbach.github.io/astroimred/api/imutil.pixels.fixpix.html#astroimred.imutil.pixels.fixpix"
        for href, label, _ in repair.article
    )

    for page, parser in pages.items():
        missing = public_names & parser.unlinked_inline
        assert not missing, (page, missing)
        for text in parser.unlinked_code:
            calls = re.findall(r"\b(?:ndfiltrs|nf)\.([\w.]+)\(", text)
            assert public_names.isdisjoint(calls), (page, text)
        for href, label, _ in parser.article:
            target = urlsplit(href)
            if target.netloc or target.scheme or not target.path:
                continue
            destination = (page.parent / unquote(target.path)).resolve()
            if target.path.endswith(".html"):
                assert destination in pages, (page, label, href)
                if target.fragment:
                    assert unquote(target.fragment) in pages[destination].ids, (
                        page,
                        href,
                    )
