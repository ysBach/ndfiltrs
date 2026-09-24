"""Collect reviewed Cargo dependency notices; use --check to detect drift.

The scope is the normal/build dependency closure of the extension-module
feature, including proc macros and every target condition. Development edges
are excluded at every node. This is a conservative source-notice inventory,
not an assertion about linked code or a legal-compliance determination.

Only inspected license expressions and crates.io sources are supported.
All discovered LICENSE*, LICENCE*, COPYING*, COPYRIGHT*, NOTICE*, files in
license/notice directories, and explicit package license_file texts are kept.
Text signatures catch missing license families, not every possible legal or
source-attribution requirement. New expressions require explicit review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote

_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT = _ROOT / "python/ndfiltrs/licenses/RUST-DEPENDENCIES.txt"
_REDUCERS_NOTICE = _ROOT / "python/ndfiltrs/licenses/reducers-MIT.txt"
_REDUCERS_SHA256 = "a2672db0608261148bbb525fd3869ab0a42b16d24343b1079c646bc67dd5d4da"
_REDUCERS_ORIGIN = (
    "https://raw.githubusercontent.com/ysbach/reducers/"
    "7625507ac1ff4101c3f97348c0a79a7d07e6ab5e/LICENSE"
)
_CRATES_IO_SOURCES = {
    "registry+https://github.com/rust-lang/crates.io-index",
    "sparse+https://index.crates.io/",
}
_NOTICE_PREFIXES = ("license", "licence", "copying", "copyright", "notice")

# Reviewed declarations, including Cargo's legacy slash syntax. For OR grants
# we retain both families, rather than select one on behalf of the distributor.
_REQUIRED_FAMILIES = {
    "MIT": ("MIT",),
    "MIT AND BSD-3-Clause": ("MIT", "BSD-3-Clause"),
    "MIT OR Apache-2.0": ("MIT", "Apache-2.0"),
    "Apache-2.0 OR MIT": ("MIT", "Apache-2.0"),
    "MIT/Apache-2.0": ("MIT", "Apache-2.0"),
    "BSD-2-Clause": ("BSD-2-Clause",),
    "Apache-2.0 WITH LLVM-exception": ("Apache-2.0", "LLVM-exception"),
    "(MIT OR Apache-2.0) AND Unicode-3.0": ("MIT", "Apache-2.0", "Unicode-3.0"),
}
_TEXT_SIGNATURES = {
    "MIT": (
        "permission is hereby granted, free of charge",
        "the above copyright notice and this permission notice",
        'the software is provided "as is"',
    ),
    "Apache-2.0": (
        "apache license",
        "version 2.0, january 2004",
        "terms and conditions for use, reproduction, and distribution",
        "end of terms and conditions",
    ),
    "BSD-2-Clause": (
        "redistribution and use in source and binary forms",
        "redistributions of source code must retain",
        "redistributions in binary form must reproduce",
        "this software is provided by the copyright holders and contributors",
    ),
    "BSD-3-Clause": (
        "redistribution and use in source and binary forms",
        "redistributions of source code must retain",
        "redistributions in binary form must reproduce",
        "neither the name of the copyright holder",
        "this software is provided by the copyright holders and contributors",
    ),
    "LLVM-exception": (
        "llvm exceptions to the apache 2.0 license",
        "portions of this software are embedded into an object form",
        "without complying with the conditions of sections 4(a), 4(b) and 4(d)",
    ),
    "Unicode-3.0": (
        "unicode license v3",
        "copyright and permission notice",
        "this copyright and permission notice appear with all copies",
    ),
}


class NoticeError(ValueError):
    """Raised when dependency identity or required notice coverage is unresolved."""


def load_metadata(manifest_path: Path) -> dict[str, Any]:
    """Read Cargo's locked, offline graph without filtering target conditions.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Root package manifest. Its lockfile must already be consistent.

    Returns
    -------
    dict
        Cargo metadata in format version `1`.
    """
    result = subprocess.run(
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
            str(manifest_path.resolve()),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode:
        raise NoticeError(
            "Cargo metadata failed with --locked --offline. Restore a consistent "
            "release manifest/lockfile and cached registry dependencies; path/git "
            "overrides cannot be represented as registry notices.\n"
            + result.stderr.strip()
        )
    return json.loads(result.stdout)


def dependency_closure(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Select registry packages reachable through normal or build edges.

    Parameters
    ----------
    metadata : dict
        Unfiltered output of ``cargo metadata --format-version 1``.

    Returns
    -------
    list of dict
        Non-root packages sorted by name and version, retaining all target
        conditions and proc-macro dependencies. Unsupported sources raise
        `NoticeError` instead of receiving an invented registry identity.
    """
    resolve = metadata.get("resolve")
    if not resolve or not resolve.get("root"):
        raise NoticeError("metadata must contain a resolved root package")
    packages = {package["id"]: package for package in metadata["packages"]}
    nodes = {node["id"]: node for node in resolve["nodes"]}
    if len(packages) != len(metadata["packages"]) or len(nodes) != len(
        resolve["nodes"]
    ):
        raise NoticeError("metadata contains duplicate package or node identities")
    root = resolve["root"]
    visited: set[str] = set()
    pending = [root]
    while pending:
        package_id = pending.pop()
        if package_id in visited:
            continue
        if package_id not in packages or package_id not in nodes:
            raise NoticeError(f"unresolved dependency identity: {package_id}")
        visited.add(package_id)
        if (
            package_id != root
            and packages[package_id].get("source") not in _CRATES_IO_SOURCES
        ):
            package = packages[package_id]
            raise NoticeError(
                f"{package['name']} {package['version']}: non-registry or unsupported "
                "registry source; remove path/git overrides for the release graph"
            )
        for dependency in nodes[package_id]["deps"]:
            kinds = dependency.get("dep_kinds")
            if not kinds or any(
                kind["kind"] not in (None, "build", "dev") for kind in kinds
            ):
                raise NoticeError(f"unknown dependency edge kind in {package_id}")
            # Do not inspect target expressions: this is a conservative union.
            if any(kind["kind"] != "dev" for kind in kinds):
                pending.append(dependency["pkg"])
    return sorted(
        (packages[key] for key in visited if key != root),
        key=lambda package: (package["name"], package["version"], package["source"]),
    )


def _walk_error(error: OSError) -> None:
    raise error


def collect_texts(package: dict[str, Any], reducers_notice: Path) -> dict[str, bytes]:
    """Read all named notices and verify the inspected declaration's coverage.

    Parameters
    ----------
    package : dict
        Cargo package metadata including `manifest_path` and `license`.
    reducers_notice : pathlib.Path
        Authoritative bundled MIT notice for registry reducers 0.3.2 only.

    Returns
    -------
    dict of str to bytes
        Sorted relative source filenames mapped to unchanged UTF-8 bytes.
        Missing, empty, non-text, or unsupported coverage raises `NoticeError`.
    """
    label = f"{package['name']} {package['version']}"
    expression = package.get("license")
    if expression not in _REQUIRED_FAMILIES:
        raise NoticeError(f"{label}: unreviewed license expression {expression!r}")
    manifest = Path(package["manifest_path"])
    if not manifest.is_file():
        raise NoticeError(f"{label}: cached manifest is missing")
    directory = manifest.resolve().parent
    selected: set[Path] = set()
    for parent, directories, files in os.walk(directory, onerror=_walk_error):
        parent_path = Path(parent)
        for name in directories:
            if (parent_path / name).is_symlink():
                raise NoticeError(
                    f"{label}: symlinked directory requires manual review"
                )
        for name in files:
            path = parent_path / name
            relative = path.relative_to(directory)
            if any(
                part.lower().startswith(_NOTICE_PREFIXES) for part in relative.parts
            ):
                selected.add(path)
    if package.get("license_file"):
        explicit = directory / package["license_file"]
        if not explicit.resolve().is_relative_to(directory):
            raise NoticeError(f"{label}: license_file escapes the package")
        selected.add(explicit)

    texts: dict[str, bytes] = {}
    for path in sorted(selected):
        if path.is_symlink() or not path.is_file():
            raise NoticeError(f"{label}: missing or symlinked notice {path.name}")
        texts[path.relative_to(directory).as_posix()] = path.read_bytes()

    # This is the explicitly reviewed provenance repair for the published crate
    # that omitted LICENSE. Never apply it to another version or local checkout.
    if not texts and (package["name"], package["version"], expression) == (
        "reducers",
        "0.3.2",
        "MIT",
    ):
        if package.get("source") not in _CRATES_IO_SOURCES:
            raise NoticeError("reducers fallback requires its registry identity")
        data = reducers_notice.read_bytes()
        if hashlib.sha256(data).hexdigest() != _REDUCERS_SHA256:
            raise NoticeError("authoritative reducers 0.3.2 notice checksum differs")
        texts["ndfiltrs:python/ndfiltrs/licenses/reducers-MIT.txt"] = data

    normalized: list[str] = []
    for name, data in texts.items():
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise NoticeError(f"{label}: notice {name} is not UTF-8 text") from error
        if not text.strip() or "\x00" in text:
            raise NoticeError(f"{label}: notice {name} is empty or not text")
        normalized.append(" ".join(text.lower().split()))
    for family in _REQUIRED_FAMILIES[expression]:
        signatures = _TEXT_SIGNATURES[family]
        if not any(
            all(fragment in text for fragment in signatures) for text in normalized
        ):
            raise NoticeError(f"{label}: missing required {family} license text")
    return dict(sorted(texts.items()))


def render_notices(metadata: dict[str, Any], reducers_notice: Path) -> bytes:
    """Render a deterministic inventory with full source texts and fingerprints.

    Parameters
    ----------
    metadata : dict
        Current Cargo metadata for the release feature graph.
    reducers_notice : pathlib.Path
        Existing authoritative notice for the reducers 0.3.2 packaging defect.

    Returns
    -------
    bytes
        UTF-8 aggregate without generated timestamps or local absolute paths.
    """
    packages = dependency_closure(metadata)
    parts = [
        (
            "Rust dependency notices for ndfiltrs\n"
            "Generated by scripts/update_rust_notices.py\n"
            "Scope: --no-default-features --features extension-module; normal and "
            "build edges, including proc macros; all target conditions; no dev edges.\n"
            "All discovered license/copyright/notice texts are reproduced. No choice "
            "among alternative grants is made.\n"
            "This source inventory does not establish linked-code coverage or legal "
            "compliance. Python dependencies, system libraries, and toolchain notices "
            "are outside this Cargo graph.\n"
            f"Dependency count: {len(packages)}\n"
        ).encode()
    ]
    for package in packages:
        texts = collect_texts(package, reducers_notice)
        name, version = package["name"], package["version"]
        parts.append(
            (
                f"\n=== {name} {version} ===\n"
                f"Declared license: {package['license']}\n"
                f"Cargo source: {package['source']}\n"
                f"Registry: https://crates.io/crates/{quote(name, safe='')}/"
                f"{quote(version, safe='')}\n"
            ).encode()
        )
        for filename, contents in texts.items():
            parts.append(f"\nSource file: {filename}\n".encode())
            if filename.startswith("ndfiltrs:"):
                parts.append(f"Upstream source: {_REDUCERS_ORIGIN}\n".encode())
            parts.append(
                (
                    f"SHA-256: {hashlib.sha256(contents).hexdigest()}\n"
                    f"Bytes: {len(contents)}\n"
                    "--- BEGIN SOURCE TEXT ---\n"
                ).encode()
            )
            parts.append(contents)
            parts.append(b"\n--- END SOURCE TEXT ---\n")
    return b"".join(parts)


def main(argv: list[str] | None = None) -> int:
    """Generate the aggregate or compare it without modifying files.

    Parameters
    ----------
    argv : list of str or None, optional
        CLI arguments, or `None` to read the process arguments.

    Returns
    -------
    int
        `0` for success; `1` for stale, missing, or unreviewed notice coverage.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="compare only; never write"
    )
    parser.add_argument("--manifest-path", type=Path, default=_ROOT / "Cargo.toml")
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    arguments = parser.parse_args(argv)
    try:
        expected = render_notices(
            load_metadata(arguments.manifest_path), _REDUCERS_NOTICE
        )
        if arguments.check:
            if not arguments.output.is_file():
                raise NoticeError(
                    "Rust dependency notice aggregate is missing; regenerate it"
                )
            if arguments.output.read_bytes() != expected:
                raise NoticeError(
                    "Rust dependency notice aggregate is stale; inspect "
                    "dependency/license changes and regenerate it"
                )
        else:
            arguments.output.write_bytes(expected)
    except (NoticeError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        logging.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
