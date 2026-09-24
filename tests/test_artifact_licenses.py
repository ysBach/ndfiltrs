"""Tests for release-artifact license verification."""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.metadata
import importlib.util
import re
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_LICENSE = b"MIT License\n\nCopyright 2026 Example\n"


def _toml_literal_fields(
    path: Path,
    section: str,
    field_names: set[str],
) -> dict[str, object]:
    current_section: str | None = None
    values: dict[str, object] = {}
    for line in path.read_text().splitlines():
        section_match = re.fullmatch(r"\s*\[([^]]+)]\s*", line)
        if section_match:
            current_section = section_match.group(1)
            continue
        if current_section != section:
            continue
        field_match = re.fullmatch(r"\s*([\w-]+)\s*=\s*(.+?)\s*", line)
        if field_match and field_match.group(1) in field_names:
            values[field_match.group(1)] = ast.literal_eval(field_match.group(2))
    assert values.keys() == field_names
    return values


_VERSION = _toml_literal_fields(_ROOT / "pyproject.toml", "project", {"version"})[
    "version"
]
assert isinstance(_VERSION, str)
_ARCHIVE_ROOT = f"ndfiltrs-{_VERSION}"
_DIST_INFO_ROOT = f"{_ARCHIVE_ROOT}.dist-info"
_METADATA = (
    "Metadata-Version: 2.4\n"
    "Name: ndfiltrs\n"
    f"Version: {_VERSION}\n"
    "License-Expression: MIT\n"
    "License-File: LICENSE\n"
).encode()
_CRATE_MANIFEST = (
    f'[package]\nname = "ndfiltrs"\nversion = "{_VERSION}"\nlicense = "MIT"\n'
).encode()
_WHEEL_LICENSE_NAME = f"{_DIST_INFO_ROOT}/licenses/LICENSE"


def test_release_identity_is_consistent_across_sources() -> None:
    cargo = _toml_literal_fields(
        Path("Cargo.toml"),
        "package",
        {"name", "version", "license"},
    )
    project = _toml_literal_fields(
        Path("pyproject.toml"),
        "project",
        {"name", "version", "license", "license-files"},
    )
    installed = importlib.metadata.metadata("ndfiltrs")
    verifier_spec = importlib.util.spec_from_file_location(
        "artifact_verifier",
        Path("scripts/verify_artifact_licenses.py"),
    )
    assert verifier_spec is not None and verifier_spec.loader is not None
    verifier = importlib.util.module_from_spec(verifier_spec)
    verifier_spec.loader.exec_module(verifier)
    source_versions = [
        ast.literal_eval(node.value)
        for node in ast.parse(
            (_ROOT / "python/ndfiltrs/__init__.py").read_text(encoding="utf-8")
        ).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    ]
    assert len(source_versions) == 1

    assert cargo["name"] == project["name"] == installed["Name"] == "ndfiltrs"
    assert (
        cargo["version"]
        == project["version"]
        == source_versions[0]
        == importlib.metadata.version("ndfiltrs")
    )
    assert cargo["license"] == project["license"] == installed["License-Expression"]
    assert project["license-files"] == installed.get_all("License-File") == ["LICENSE"]
    assert verifier._PACKAGE_NAME == project["name"]
    assert verifier._PACKAGE_VERSION == project["version"]
    assert verifier._LICENSE_EXPRESSION == project["license"]


def _write_tar(
    path: Path,
    files: Mapping[str, bytes] | Iterable[tuple[str, bytes]],
) -> None:
    staging = path.parent / f"{path.stem}-contents"
    entries = files.items() if isinstance(files, Mapping) else files
    with tarfile.open(path, "w:gz") as archive:
        for name, contents in entries:
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
            archive.add(staging / name, arcname=name)


def _write_wheel(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)


def _record_row(name: str, contents: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=")
    return f"{name},sha256={digest.decode('ascii')},{len(contents)}"


def _artifacts(
    tmp_path: Path,
    wheel_license: bytes,
    *,
    metadata: bytes | None = None,
    crate_manifest: bytes = _CRATE_MANIFEST,
    roots: tuple[str, str] = (_ARCHIVE_ROOT, _ARCHIVE_ROOT),
    extra_crate_root: str | None = None,
    record: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    license_path = tmp_path / "LICENSE"
    license_path.write_bytes(_LICENSE)
    if metadata is None:
        metadata = _METADATA
    crate_root, sdist_root = roots
    crate = tmp_path / f"{_ARCHIVE_ROOT}.crate"
    crate_files = {
        f"{crate_root}/LICENSE": _LICENSE,
        f"{crate_root}/Cargo.toml": crate_manifest,
    }
    if extra_crate_root is not None:
        crate_files[f"{extra_crate_root}/decoy.txt"] = b"decoy"
    _write_tar(crate, crate_files)
    sdist = tmp_path / f"{_ARCHIVE_ROOT}.tar.gz"
    _write_tar(
        sdist,
        {
            f"{sdist_root}/LICENSE": _LICENSE,
            f"{sdist_root}/PKG-INFO": metadata,
        },
    )
    wheel = tmp_path / f"{_ARCHIVE_ROOT}-cp310-abi3-any.whl"
    if record is None:
        record = _record_row(_WHEEL_LICENSE_NAME, wheel_license)
    _write_wheel(
        wheel,
        {
            _WHEEL_LICENSE_NAME: wheel_license,
            f"{_DIST_INFO_ROOT}/METADATA": metadata,
            f"{_DIST_INFO_ROOT}/RECORD": f"{record}\n".encode(),
        },
    )
    return license_path, crate, sdist, wheel


def _run_verifier(
    paths: tuple[Path, Path, Path, Path],
) -> subprocess.CompletedProcess[str]:
    license_path, crate, sdist, wheel = paths
    return subprocess.run(
        [
            sys.executable,
            "scripts/verify_artifact_licenses.py",
            "--license",
            str(license_path),
            "--crate",
            str(crate),
            "--sdist",
            str(sdist),
            "--wheel",
            str(wheel),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_artifact_license_verifier_accepts_consistent_artifacts(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path, _LICENSE)
    result = _run_verifier(paths)
    assert result.returncode == 0, result.stderr


def test_artifact_license_verifier_rejects_changed_license(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path, b"different license\n")
    result = _run_verifier(paths)
    assert result.returncode != 0
    assert "wheel license does not match" in result.stderr


def test_artifact_license_verifier_rejects_missing_spdx_expression(
    tmp_path: Path,
) -> None:
    paths = _artifacts(
        tmp_path,
        _LICENSE,
        metadata=(
            "Metadata-Version: 2.4\n"
            "Name: ndfiltrs\n"
            f"Version: {_VERSION}\n"
            "Classifier: License :: OSI Approved :: MIT License\n"
            "License-File: LICENSE\n"
        ).encode(),
    )
    result = _run_verifier(paths)
    assert result.returncode != 0
    assert "License-Expression: MIT" in result.stderr


def test_artifact_license_verifier_accepts_wheel_only(tmp_path: Path) -> None:
    license_path, _, _, wheel = _artifacts(tmp_path, _LICENSE)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_artifact_licenses.py",
            "--license",
            str(license_path),
            "--wheel",
            str(wheel),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_artifact_license_verifier_requires_an_artifact(tmp_path: Path) -> None:
    license_path = tmp_path / "LICENSE"
    license_path.write_text("MIT License\n")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_artifact_licenses.py",
            "--license",
            str(license_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "at least one artifact" in result.stderr


def test_artifact_license_verifier_accepts_one_artifact_directory(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path, _LICENSE)
    crate_dir = tmp_path / "crate"
    crate_dir.mkdir()
    paths[1].rename(crate_dir / paths[1].name)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_artifact_licenses.py",
            "--license",
            str(paths[0]),
            "--crate",
            str(crate_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_artifact_license_verifier_rejects_mixed_archive_roots(
    tmp_path: Path,
) -> None:
    paths = _artifacts(
        tmp_path,
        _LICENSE,
        extra_crate_root=f"other-{_VERSION}",
    )
    result = _run_verifier(paths)
    assert result.returncode != 0
    assert "archive must have one root" in result.stderr


def test_artifact_license_verifier_rejects_duplicate_tar_members(
    tmp_path: Path,
) -> None:
    license_path, crate, sdist, wheel = _artifacts(tmp_path, _LICENSE)
    _write_tar(
        crate,
        [
            (f"{_ARCHIVE_ROOT}/LICENSE", _LICENSE),
            (f"{_ARCHIVE_ROOT}/LICENSE", _LICENSE),
            (f"{_ARCHIVE_ROOT}/Cargo.toml", _CRATE_MANIFEST),
        ],
    )

    result = _run_verifier((license_path, crate, sdist, wheel))

    assert result.returncode != 0
    assert "duplicate member names" in result.stderr


def test_artifact_license_verifier_scopes_cargo_identity_to_package_table(
    tmp_path: Path,
) -> None:
    paths = _artifacts(
        tmp_path,
        _LICENSE,
        crate_manifest=(
            "\n[package]\n"
            'name = "ndfiltrs"\n'
            f'version = "{_VERSION}"\n'
            'license = "MIT"\n'
            "\n[bad]\n"
            'license = "GPL-3.0"\n'
        ).encode(),
    )
    result = _run_verifier(paths)
    assert result.returncode == 0, result.stderr


def test_artifact_license_verifier_rejects_decoy_license_outside_package_table(
    tmp_path: Path,
) -> None:
    paths = _artifacts(
        tmp_path,
        _LICENSE,
        crate_manifest=(
            "\n[package]\n"
            'name = "ndfiltrs"\n'
            f'version = "{_VERSION}"\n'
            'license = "GPL-3.0"\n'
            "\n[bad]\n"
            'license = "MIT"\n'
        ).encode(),
    )

    result = _run_verifier(paths)

    assert result.returncode != 0
    assert "Cargo.toml package identity" in result.stderr


def test_artifact_license_verifier_rejects_wrong_cargo_identity(
    tmp_path: Path,
) -> None:
    paths = _artifacts(
        tmp_path,
        _LICENSE,
        crate_manifest=(
            b'[package]\nname = "decoy"\nversion = "9.9.9"\nlicense = "MIT"\n'
        ),
    )
    result = _run_verifier(paths)
    assert result.returncode != 0
    assert "Cargo.toml package identity" in result.stderr


def test_artifact_license_verifier_rejects_bad_record_integrity(
    tmp_path: Path,
) -> None:
    paths = _artifacts(
        tmp_path,
        _LICENSE,
        record=f"{_WHEEL_LICENSE_NAME},,",
    )
    result = _run_verifier(paths)
    assert result.returncode != 0
    assert "RECORD integrity" in result.stderr


def test_artifact_license_verifier_rejects_wrong_record_digest(
    tmp_path: Path,
) -> None:
    paths = _artifacts(
        tmp_path,
        _LICENSE,
        record=f"{_WHEEL_LICENSE_NAME},sha256=AAAA,{len(_LICENSE)}",
    )

    result = _run_verifier(paths)

    assert result.returncode != 0
    assert "RECORD integrity" in result.stderr


def test_artifact_license_verifier_rejects_wrong_record_size(
    tmp_path: Path,
) -> None:
    digest = _record_row(_WHEEL_LICENSE_NAME, _LICENSE).rsplit(",", maxsplit=1)[0]
    paths = _artifacts(
        tmp_path,
        _LICENSE,
        record=f"{digest},{len(_LICENSE) + 1}",
    )

    result = _run_verifier(paths)

    assert result.returncode != 0
    assert "RECORD integrity" in result.stderr


def test_artifact_license_verifier_rejects_wrong_python_identity(
    tmp_path: Path,
) -> None:
    metadata = _METADATA.replace(b"Name: ndfiltrs", b"Name: other")
    paths = _artifacts(tmp_path, _LICENSE, metadata=metadata)
    result = _run_verifier(paths)
    assert result.returncode != 0
    assert "metadata package identity" in result.stderr


def test_artifact_license_verifier_rejects_wrong_archive_root(
    tmp_path: Path,
) -> None:
    paths = _artifacts(
        tmp_path,
        _LICENSE,
        roots=(f"other-{_VERSION}", _ARCHIVE_ROOT),
    )
    result = _run_verifier(paths)
    assert result.returncode != 0
    assert "archive root" in result.stderr


def test_artifact_license_verifier_rejects_wheel_version_mismatch(
    tmp_path: Path,
) -> None:
    metadata = _METADATA.replace(
        f"Version: {_VERSION}\n".encode(),
        f"Version: {_VERSION}.post1\n".encode(),
    )
    license_path, _, _, wheel = _artifacts(tmp_path, _LICENSE, metadata=metadata)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_artifact_licenses.py",
            "--license",
            str(license_path),
            "--wheel",
            str(wheel),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "wheel metadata package identity" in result.stderr
