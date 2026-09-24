"""Release identity, changelog, and clean-tree checks without a native build."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 11):
    pytest.skip(
        "release checker requires Python 3.11+ tomllib", allow_module_level=True
    )

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_release.py"
_SPEC = importlib.util.spec_from_file_location("check_release", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create bounded release inputs; the package must never be imported."""
    files = {
        "Cargo.toml": '[package]\nname = "ndfiltrs"\nversion = "0.1.0"\n',
        "pyproject.toml": '[project]\nname = "ndfiltrs"\nversion = "0.1.0"\n',
        "Cargo.lock": '[[package]]\nname = "ndfiltrs"\nversion = "0.1.0"\n',
        "uv.lock": (
            '[[package]]\nname = "ndfiltrs"\nversion = "0.1.0"\n'
            'source = { editable = "." }\n'
        ),
        "python/ndfiltrs/__init__.py": (
            'raise RuntimeError("must not import")\n__version__ = "0.1.0"\n'
        ),
        "CHANGELOG.md": (
            "# Changelog\n\n## 0.1.0 - Unreleased\n\n### Added\n\n- Filters.\n\n"
            "## 0.0.9 - 2026-01-01\n\n- Older notes.\n"
        ),
    }
    for name, contents in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    return tmp_path


def test_rehearsal_extracts_only_current_section_without_import(
    repository: Path,
) -> None:
    assert release.check_release(repository) == ("0.1.0", "### Added\n\n- Filters.\n")
    assert release.check_release(repository, tag="") == release.check_release(
        repository
    )


@pytest.mark.parametrize(
    "filename",
    [
        "Cargo.toml",
        "pyproject.toml",
        "Cargo.lock",
        "uv.lock",
        "python/ndfiltrs/__init__.py",
    ],
)
def test_version_drift_is_rejected(repository: Path, filename: str) -> None:
    path = repository / filename
    path.write_text(path.read_text().replace("0.1.0", "0.2.0"))
    with pytest.raises(ValueError, match="version"):
        release.check_release(repository)


@pytest.mark.parametrize(
    "filename", ["Cargo.toml", "pyproject.toml", "Cargo.lock", "uv.lock"]
)
def test_package_name_drift_is_rejected(repository: Path, filename: str) -> None:
    path = repository / filename
    path.write_text(path.read_text().replace("ndfiltrs", "different"))
    with pytest.raises(ValueError, match=filename.replace(".", r"\.")):
        release.check_release(repository)


@pytest.mark.parametrize(
    "version", ["1.2", "1.2.3rc1", "1.2.3-dev", "01.2.3", "v1.2.3"]
)
def test_only_stable_versions_are_accepted(repository: Path, version: str) -> None:
    path = repository / "Cargo.toml"
    path.write_text(path.read_text().replace("0.1.0", version))
    with pytest.raises(ValueError, match="stable X.Y.Z"):
        release.check_release(repository)


@pytest.mark.parametrize(
    "table",
    [
        "dependencies",
        "dev-dependencies",
        "build-dependencies",
        "target.'cfg(unix)'.dependencies",
        "workspace.dependencies",
        "patch.crates-io",
    ],
)
@pytest.mark.parametrize(
    "source", ['path = "../reducers"', 'git = "https://example.invalid/reducers.git"']
)
def test_cargo_dependency_overrides_are_rejected(
    repository: Path, table: str, source: str
) -> None:
    path = repository / "Cargo.toml"
    path.write_text(
        path.read_text()
        + f'\n[{table}]\nreducers = {{ version = "0.3.2", {source} }}\n'
    )
    with pytest.raises(ValueError, match="reducers uses a path/git"):
        release.check_release(repository)


def test_registry_workspace_dependency_is_resolved(repository: Path) -> None:
    path = repository / "Cargo.toml"
    path.write_text(
        path.read_text() + "\n[dependencies]\nreducers = { workspace = true }\n"
        "[workspace.dependencies]\n"
        'reducers = { version = "0.3.2", default-features = false }\n'
    )
    assert release.check_release(repository)[0] == "0.1.0"
    path.write_text(path.read_text().replace('version = "0.3.2", ', ""))
    with pytest.raises(ValueError, match="requires a registry version"):
        release.check_release(repository)


def test_python_uv_source_overrides_are_rejected(repository: Path) -> None:
    path = repository / "pyproject.toml"
    path.write_text(
        path.read_text()
        + '\n[tool.uv.sources]\nreducers = { path = "../reducers", editable = true }\n'
    )
    with pytest.raises(ValueError, match="tool.uv.sources"):
        release.check_release(repository)


@pytest.mark.parametrize(
    "requirement",
    [
        "reducers @ file:///tmp/reducers",
        "reducers @ https://example.invalid/source.tar.gz",
        "git+https://example.invalid/reducers.git",
        "../reducers",
        "/tmp/reducers",
    ],
)
@pytest.mark.parametrize(
    "table,key",
    [
        ("project", "dependencies"),
        ("project.optional-dependencies", "dev"),
        ("dependency-groups", "dev"),
    ],
)
def test_python_direct_sources_are_rejected(
    repository: Path, requirement: str, table: str, key: str
) -> None:
    path = repository / "pyproject.toml"
    header = "" if table == "project" else f"\n[{table}]\n"
    path.write_text(path.read_text() + header + f'{key} = ["{requirement}"]\n')
    with pytest.raises(ValueError, match="direct local/URL"):
        release.check_release(repository)


def test_python_registry_requirements_remain_supported(repository: Path) -> None:
    path = repository / "pyproject.toml"
    path.write_text(
        path.read_text() + '\ndependencies = ["numpy>=1.24"]\n[dependency-groups]\n'
        'dev = ["pytest", { include-group = "lint" }]\nlint = ["ruff"]\n'
    )
    assert release.check_release(repository)[0] == "0.1.0"


@pytest.mark.parametrize("literal", ['"0." + "1.0"', 'str("0.1.0")', "(0, 1, 0)"])
def test_version_must_be_a_literal_string(repository: Path, literal: str) -> None:
    (repository / "python/ndfiltrs/__init__.py").write_text(
        f"__version__ = {literal}\n"
    )
    with pytest.raises(ValueError, match="literal __version__"):
        release.check_release(repository)


def test_duplicate_version_assignments_are_rejected(repository: Path) -> None:
    path = repository / "python/ndfiltrs/__init__.py"
    path.write_text('__version__ = "0.1.0"\n__version__ = "0.1.0"\n')
    with pytest.raises(ValueError, match="one literal"):
        release.check_release(repository)


@pytest.mark.parametrize(
    "filename,source",
    [
        ("Cargo.lock", 'source = "registry+https://example.invalid/index"'),
        ("uv.lock", 'source = { registry = "https://example.invalid/simple" }'),
    ],
)
def test_lockfile_selects_local_entry_and_rejects_duplicates(
    repository: Path,
    filename: str,
    source: str,
) -> None:
    path = repository / filename
    local = path.read_text()
    external = f'[[package]]\nname = "ndfiltrs"\nversion = "9.9.9"\n{source}\n\n'
    path.write_text(external + local)
    assert release.check_release(repository)[0] == "0.1.0"
    path.write_text(external)
    with pytest.raises(ValueError, match="exactly one local"):
        release.check_release(repository)
    path.write_text(local + "\n" + local)
    with pytest.raises(ValueError, match="exactly one local"):
        release.check_release(repository)


@pytest.mark.parametrize("tag", ["v0.2.0", "0.1.0", "v0.1.0rc1", " "])
def test_wrong_tag_is_rejected(repository: Path, tag: str) -> None:
    with pytest.raises(ValueError, match="tag .* must match v0.1.0"):
        release.check_release(repository, tag=tag)


def test_tag_requires_released_changelog(repository: Path) -> None:
    with pytest.raises(ValueError, match="cannot be Unreleased"):
        release.check_release(repository, tag="v0.1.0")
    path = repository / "CHANGELOG.md"
    path.write_text(path.read_text().replace("Unreleased", "2026-09-07"))
    assert release.check_release(repository, tag="v0.1.0") == (
        "0.1.0",
        "### Added\n\n- Filters.\n",
    )


@pytest.mark.parametrize("status", ["2026-02-30", "2026-13-01", "2026-9-7", "soon"])
def test_release_date_must_be_valid_iso_date(repository: Path, status: str) -> None:
    path = repository / "CHANGELOG.md"
    path.write_text(path.read_text().replace("Unreleased", status))
    with pytest.raises(ValueError, match="date"):
        release.check_release(repository, tag="v0.1.0")


@pytest.mark.parametrize(
    "notes,error",
    [
        ("## 0.1.0 - 2026-09-07\n\n## 0.0.9 - 2026-01-01\n- Old.\n", "nonempty"),
        ("## 0.2.0 - Unreleased\n- Next.\n", "exactly one"),
        ("## 0.1.0 - 2026-09-07\n- A.\n## 0.1.0 - 2026-09-07\n- B.\n", "exactly one"),
    ],
)
def test_tag_requires_unique_nonempty_current_section(
    repository: Path,
    notes: str,
    error: str,
) -> None:
    (repository / "CHANGELOG.md").write_text(notes)
    with pytest.raises(ValueError, match=error):
        release.check_release(repository, tag="v0.1.0")


def test_cli_writes_exact_notes_and_appends_github_output(
    repository: Path, tmp_path: Path
) -> None:
    notes = tmp_path / "notes.md"
    github = tmp_path / "github-output"
    github.write_bytes(b"earlier=value\n")
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--root",
            str(repository),
            "--tag",
            "",
            "--notes-output",
            str(notes),
            "--github-output",
            str(github),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert notes.read_bytes() == b"### Added\n\n- Filters.\n"
    assert github.read_bytes() == b"earlier=value\nversion=0.1.0\n"


@pytest.fixture
def clean_repository(repository: Path) -> Path:
    """Commit only the synthetic fixture in a temporary Git repository."""
    subprocess.run(
        ["git", "init", "--quiet"], cwd=repository, check=True, capture_output=True
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True, capture_output=True)
    hooks = repository / ".git" / "empty-hooks"
    hooks.mkdir()
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=release@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            f"core.hooksPath={hooks}",
            "commit",
            "--quiet",
            "-m",
            "Fixture",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return repository


def test_clean_tree_is_checked_before_creating_outputs(clean_repository: Path) -> None:
    assert release.check_release(clean_repository, require_clean=True)[0] == "0.1.0"
    notes = clean_repository / "release-notes.md"
    assert (
        release.main(
            [
                "--root",
                str(clean_repository),
                "--require-clean",
                "--notes-output",
                str(notes),
            ]
        )
        == 0
    )
    assert notes.read_bytes() == b"### Added\n\n- Filters.\n"
    with pytest.raises(ValueError, match="clean tree"):
        release.check_release(clean_repository, require_clean=True)


@pytest.mark.parametrize("change", ["tracked", "staged", "untracked"])
def test_dirty_tree_rejects_without_touching_outputs(
    clean_repository: Path, change: str
) -> None:
    root = clean_repository
    if change == "untracked":
        (root / "new-source.py").write_text("# Untracked\n")
        subprocess.run(
            ["git", "config", "status.showUntrackedFiles", "no"], cwd=root, check=True
        )
    else:
        (root / "CHANGELOG.md").write_text((root / "CHANGELOG.md").read_text() + "\n")
        if change == "staged":
            subprocess.run(["git", "add", "CHANGELOG.md"], cwd=root, check=True)
    notes = root / "notes-output.md"
    github = root / "github-output"
    assert (
        release.main(
            [
                "--root",
                str(root),
                "--require-clean",
                "--notes-output",
                str(notes),
                "--github-output",
                str(github),
            ]
        )
        == 1
    )
    assert not notes.exists() and not github.exists()


def test_invalid_metadata_preserves_existing_outputs(repository: Path) -> None:
    notes = repository / "notes.md"
    github = repository / "github-output"
    notes.write_bytes(b"keep notes\n")
    github.write_bytes(b"keep=output\n")
    assert (
        release.main(
            [
                "--root",
                str(repository),
                "--tag",
                "v9.9.9",
                "--notes-output",
                str(notes),
                "--github-output",
                str(github),
            ]
        )
        == 1
    )
    assert notes.read_bytes() == b"keep notes\n"
    assert github.read_bytes() == b"keep=output\n"


def test_require_clean_reports_git_failure(repository: Path) -> None:
    with pytest.raises(ValueError, match="git status failed"):
        release.check_release(repository, require_clean=True)
