"""Dependency closure, notice provenance, and deterministic drift checks."""

from __future__ import annotations

import copy
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts import update_rust_notices as notices

_ROOT = Path(__file__).resolve().parents[1]
_MIT = (_ROOT / "LICENSE").read_bytes()
_REDUCERS = _ROOT / "python/ndfiltrs/licenses/reducers-MIT.txt"
_REDUCERS_BSD = _ROOT / "python/ndfiltrs/licenses/reducers-BSD-3-Clause.txt"
_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"


def _metadata(tmp_path: Path) -> dict[str, Any]:
    """Build a small graph with target, build, mixed-kind, and dev-only edges."""
    packages = []
    for name in ("root", "runtime", "build", "target", "shared", "dev", "nested-dev"):
        directory = tmp_path / name
        directory.mkdir(parents=True)
        manifest = directory / "Cargo.toml"
        manifest.write_text(f'[package]\nname = "{name}"\nversion = "1.0.0"\n')
        (directory / "LICENSE-MIT").write_bytes(_MIT)
        packages.append(
            {
                "id": name,
                "name": name,
                "version": "1.0.0",
                "source": None if name == "root" else _SOURCE,
                "manifest_path": str(manifest),
                "license": "MIT",
                "license_file": None,
                "targets": [{"kind": ["proc-macro" if name == "build" else "lib"]}],
            }
        )
    edges = {
        "root": [
            ("runtime", None, None),
            ("build", "build", None),
            ("dev", "dev", None),
        ],
        "runtime": [
            ("target", None, 'cfg(target_os = "windows")'),
            ("nested-dev", "dev", None),
        ],
        "build": [("shared", None, None), ("nested-dev", "dev", None)],
        "target": [("shared", "dev", None), ("shared", "build", "cfg(unix)")],
        "shared": [("runtime", None, None)],
        "dev": [("nested-dev", None, None)],
        "nested-dev": [],
    }
    nodes = []
    for name, entries in edges.items():
        dependencies: dict[str, list[dict[str, str | None]]] = {}
        for dependency, kind, target in entries:
            dependencies.setdefault(dependency, []).append(
                {"kind": kind, "target": target}
            )
        nodes.append(
            {
                "id": name,
                "deps": [
                    {"pkg": dependency, "dep_kinds": kinds}
                    for dependency, kinds in dependencies.items()
                ],
            }
        )
    return {"packages": packages, "resolve": {"root": "root", "nodes": nodes}}


def test_closure_keeps_build_proc_macros_and_targets_but_excludes_dev(
    tmp_path: Path,
) -> None:
    metadata = _metadata(tmp_path)
    packages = notices.dependency_closure(metadata)
    assert [package["name"] for package in packages] == [
        "build",
        "runtime",
        "shared",
        "target",
    ]
    # A development-only local dependency must not invalidate a release graph.
    metadata["packages"][-1]["source"] = None
    assert notices.dependency_closure(metadata) == packages


@pytest.mark.parametrize(
    "source",
    [None, "git+https://example.com/repo#abc", "registry+https://example.com/index"],
)
def test_nonregistry_release_dependency_is_rejected(
    tmp_path: Path, source: str | None
) -> None:
    metadata = _metadata(tmp_path)
    metadata["packages"][1]["source"] = source
    with pytest.raises(
        notices.NoticeError, match="non-registry or unsupported registry"
    ):
        notices.render_notices(metadata, _REDUCERS)


def test_unknown_edge_kind_is_not_silently_excluded(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    metadata["resolve"]["nodes"][0]["deps"][0]["dep_kinds"][0]["kind"] = "future-kind"
    with pytest.raises(notices.NoticeError, match="unknown dependency edge kind"):
        notices.dependency_closure(metadata)


def test_render_is_portable_sorted_and_preserves_every_notice_byte(
    tmp_path: Path,
) -> None:
    metadata = _metadata(tmp_path)
    directory = Path(metadata["packages"][1]["manifest_path"]).parent
    nested = directory / "embedded"
    nested.mkdir()
    extra = b"Copyright 2026 Example\r\nAttribution must stay verbatim."
    (nested / "COPYRIGHT").write_bytes(extra)
    (nested / "NOTICE.txt").write_bytes(b"A distinct attribution notice.\n")
    custom = directory / "terms.txt"
    custom.write_bytes(b"Additional declared terms.\n")
    metadata["packages"][1]["license_file"] = "terms.txt"
    expected = notices.render_notices(metadata, _REDUCERS)
    assert extra in expected
    assert b"Source file: embedded/COPYRIGHT" in expected
    assert b"Source file: embedded/NOTICE.txt" in expected
    assert b"Source file: terms.txt" in expected
    assert hashlib.sha256(extra).hexdigest().encode() in expected
    assert str(tmp_path).encode() not in expected
    assert b"Declared license: MIT" in expected
    assert b"Registry: https://crates.io/crates/runtime/1.0.0" in expected
    metadata["packages"].reverse()
    metadata["resolve"]["nodes"].reverse()
    for node in metadata["resolve"]["nodes"]:
        node["deps"].reverse()
    assert notices.render_notices(metadata, _REDUCERS) == expected


@pytest.mark.parametrize("expression", [None, "GPL-3.0-only", "MIT AND Unreviewed-1.0"])
def test_unknown_declarations_fail_before_generation(
    tmp_path: Path, expression: str | None
) -> None:
    metadata = _metadata(tmp_path)
    metadata["packages"][1]["license"] = expression
    with pytest.raises(notices.NoticeError, match="unreviewed license expression"):
        notices.render_notices(metadata, _REDUCERS)


def test_missing_required_family_is_rejected_despite_other_notices(
    tmp_path: Path,
) -> None:
    metadata = _metadata(tmp_path)
    package = metadata["packages"][1]
    directory = Path(package["manifest_path"]).parent
    package["license"] = "MIT OR Apache-2.0"
    with pytest.raises(notices.NoticeError, match="missing required Apache-2.0"):
        notices.collect_texts(package, _REDUCERS)
    package["license"] = "MIT"
    (directory / "LICENSE-MIT").unlink()
    (directory / "NOTICE").write_bytes(b"An attribution alone is not the license.\n")
    with pytest.raises(notices.NoticeError, match="missing required MIT"):
        notices.collect_texts(package, _REDUCERS)


def test_dual_license_requires_and_collects_both_license_families(
    tmp_path: Path,
) -> None:
    metadata = _metadata(tmp_path)
    package = metadata["packages"][1]
    directory = Path(package["manifest_path"]).parent
    package["license"] = "MIT AND BSD-3-Clause"
    (directory / "LICENSE-BSD-3-Clause").write_bytes(_REDUCERS_BSD.read_bytes())

    texts = notices.collect_texts(package, _REDUCERS)

    assert texts["LICENSE-MIT"] == _MIT
    assert texts["LICENSE-BSD-3-Clause"] == _REDUCERS_BSD.read_bytes()


@pytest.mark.parametrize("data", [b"", b"not a text\x00file", b"\xff"])
def test_unreadable_notice_is_not_silently_skipped(tmp_path: Path, data: bytes) -> None:
    metadata = _metadata(tmp_path)
    package = metadata["packages"][1]
    directory = Path(package["manifest_path"]).parent
    (directory / "NOTICE").write_bytes(data)
    with pytest.raises(notices.NoticeError, match="not UTF-8|empty or not text"):
        notices.collect_texts(package, _REDUCERS)


def test_explicit_missing_license_file_fails_even_with_mit_present(
    tmp_path: Path,
) -> None:
    metadata = _metadata(tmp_path)
    package = metadata["packages"][1]
    package["license_file"] = "unavailable.txt"
    with pytest.raises(notices.NoticeError, match="missing or symlinked notice"):
        notices.collect_texts(package, _REDUCERS)


def test_reducers_fallback_is_version_and_content_specific(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    package = metadata["packages"][1]
    package.update(name="reducers", version="0.3.2")
    (Path(package["manifest_path"]).parent / "LICENSE-MIT").unlink()
    text = notices.render_notices(metadata, _REDUCERS)
    assert _REDUCERS.read_bytes() in text
    assert b"7625507ac1ff4101c3f97348c0a79a7d07e6ab5e/LICENSE" in text
    replacement = tmp_path / "replacement.txt"
    replacement.write_bytes(_MIT + b"\nChanged attribution.\n")
    with pytest.raises(notices.NoticeError, match="checksum differs"):
        notices.collect_texts(package, replacement)
    package["version"] = "0.3.3"
    with pytest.raises(notices.NoticeError, match="missing required MIT"):
        notices.collect_texts(package, _REDUCERS)


def test_check_recomputes_and_never_overwrites_stale_or_missing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata(tmp_path)
    current = copy.deepcopy(metadata)
    monkeypatch.setattr(notices, "load_metadata", lambda manifest: current)
    output = tmp_path / "aggregate.txt"
    arguments = ["--output", str(output)]
    assert notices.main([*arguments, "--check"]) == 1
    assert not output.exists()
    assert notices.main(arguments) == 0
    initial = output.read_bytes()
    assert notices.main([*arguments, "--check"]) == 0
    current["packages"][1]["version"] = "1.0.1"
    assert notices.main([*arguments, "--check"]) == 1
    assert output.read_bytes() == initial
    current["packages"][1]["version"] = "1.0.0"
    directory = Path(current["packages"][1]["manifest_path"]).parent
    (directory / "NOTICE").write_bytes(b"A newly added upstream notice.\n")
    assert notices.main([*arguments, "--check"]) == 1
    assert output.read_bytes() == initial
    current["packages"][1]["source"] = None
    assert notices.main(arguments) == 1
    assert output.read_bytes() == initial


def test_metadata_command_preserves_locked_features_and_all_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(notices.subprocess, "run", run)
    manifest = tmp_path / "Cargo.toml"
    assert notices.load_metadata(manifest) == {}
    assert calls == [
        [
            "cargo",
            "metadata",
            "--locked",
            "--offline",
            "--no-default-features",
            "--features",
            "extension-module",
            "--format-version",
            "1",
            "--manifest-path",
            str(manifest.resolve()),
        ]
    ]
