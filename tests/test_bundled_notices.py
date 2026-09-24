"""Tests for bundled third-party notices verification in release artifacts."""

from __future__ import annotations

import base64
import hashlib
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

import pytest

from scripts.verify_artifact_licenses import (
    ArtifactLicenseError,
    verify_artifact_licenses,
)
from tests.test_artifact_licenses import _VERSION

_LICENSE = b"MIT License\n\nCopyright 2026 Example\n"
_ARCHIVE_ROOT = f"ndfiltrs-{_VERSION}"
_DIST_INFO_ROOT = f"{_ARCHIVE_ROOT}.dist-info"
_METADATA = (
    "Metadata-Version: 2.4\n"
    "Name: ndfiltrs\n"
    f"Version: {_VERSION}\n"
    "License-Expression: MIT\n"
    "License-File: LICENSE\n"
).encode()
_WHEEL_LICENSE_NAME = f"{_DIST_INFO_ROOT}/licenses/LICENSE"
_NOTICE_A = b"Notice A text\n"
_NOTICE_B = b"Notice B text\n"


def _write_tar(
    path: Path,
    files: Mapping[str, bytes],
) -> None:
    staging = path.parent / f"{path.stem}-contents"
    with tarfile.open(path, "w:gz") as archive:
        for name, contents in files.items():
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
            archive.add(destination, arcname=name)


def _write_wheel(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)


def _record_row(name: str, contents: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=")
    return f"{name},sha256={digest.decode('ascii')},{len(contents)}"


def _build_test_artifacts(
    tmp_path: Path,
    notices: dict[str, bytes],
    *,
    crate_notices: dict[str, bytes] | None = None,
    sdist_notices: dict[str, bytes] | None = None,
    wheel_notices: dict[str, bytes] | None = None,
    record_rows: list[str] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    license_path = tmp_path / "LICENSE"
    license_path.write_bytes(_LICENSE)

    notices_dir = tmp_path / "notices"
    notices_dir.mkdir(parents=True, exist_ok=True)
    for name, content in notices.items():
        (notices_dir / name).write_bytes(content)

    c_notices = notices if crate_notices is None else crate_notices
    s_notices = notices if sdist_notices is None else sdist_notices
    w_notices = notices if wheel_notices is None else wheel_notices

    crate_files: dict[str, bytes] = {
        f"{_ARCHIVE_ROOT}/LICENSE": _LICENSE,
        f"{_ARCHIVE_ROOT}/Cargo.toml": (
            f'[package]\nname = "ndfiltrs"\nversion = "{_VERSION}"\nlicense = "MIT"\n'
        ).encode(),
    }
    for name, content in c_notices.items():
        crate_files[f"{_ARCHIVE_ROOT}/python/ndfiltrs/licenses/{name}"] = content
    crate = tmp_path / f"{_ARCHIVE_ROOT}.crate"
    _write_tar(crate, crate_files)

    sdist_files: dict[str, bytes] = {
        f"{_ARCHIVE_ROOT}/LICENSE": _LICENSE,
        f"{_ARCHIVE_ROOT}/PKG-INFO": _METADATA,
    }
    for name, content in s_notices.items():
        sdist_files[f"{_ARCHIVE_ROOT}/python/ndfiltrs/licenses/{name}"] = content
    sdist = tmp_path / f"{_ARCHIVE_ROOT}.tar.gz"
    _write_tar(sdist, sdist_files)

    wheel_files: dict[str, bytes] = {
        _WHEEL_LICENSE_NAME: _LICENSE,
        f"{_DIST_INFO_ROOT}/METADATA": _METADATA,
    }
    rows = [_record_row(_WHEEL_LICENSE_NAME, _LICENSE)]
    for name, content in w_notices.items():
        member_name = f"ndfiltrs/licenses/{name}"
        wheel_files[member_name] = content
        rows.append(_record_row(member_name, content))

    if record_rows is not None:
        rows = record_rows

    wheel_files[f"{_DIST_INFO_ROOT}/RECORD"] = ("\n".join(rows) + "\n").encode()
    wheel = tmp_path / f"{_ARCHIVE_ROOT}-cp310-abi3-any.whl"
    _write_wheel(wheel, wheel_files)

    return license_path, crate, sdist, wheel, notices_dir


def test_verify_artifact_licenses_accepts_consistent_notices(tmp_path: Path) -> None:
    notices = {"a.txt": _NOTICE_A, "b.txt": _NOTICE_B}
    license_path, crate, sdist, wheel, notices_dir = _build_test_artifacts(
        tmp_path, notices
    )

    verify_artifact_licenses(
        license_path=license_path,
        crate_path=crate,
        sdist_path=sdist,
        wheel_path=wheel,
        notices_path=notices_dir,
    )


def test_verify_artifact_licenses_cli_accepts_notices_dir(tmp_path: Path) -> None:
    notices = {"dep.txt": _NOTICE_A}
    license_path, crate, sdist, wheel, notices_dir = _build_test_artifacts(
        tmp_path, notices
    )

    result = subprocess.run(
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
            "--notices",
            str(notices_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_verify_artifact_licenses_rejects_missing_or_empty_notices_dir(
    tmp_path: Path,
) -> None:
    notices = {"a.txt": _NOTICE_A}
    license_path, crate, sdist, wheel, _ = _build_test_artifacts(tmp_path, notices)

    nonexistent_dir = tmp_path / "nonexistent"
    with pytest.raises(ArtifactLicenseError, match="must be a directory"):
        verify_artifact_licenses(
            license_path=license_path,
            crate_path=crate,
            sdist_path=sdist,
            wheel_path=wheel,
            notices_path=nonexistent_dir,
        )

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    (empty_dir / "not_a_txt.md").write_text("license info")
    with pytest.raises(ArtifactLicenseError, match="no .txt notices"):
        verify_artifact_licenses(
            license_path=license_path,
            crate_path=crate,
            sdist_path=sdist,
            wheel_path=wheel,
            notices_path=empty_dir,
        )


def test_verify_artifact_licenses_rejects_crate_missing_or_mismatched_notice(
    tmp_path: Path,
) -> None:
    notices = {"a.txt": _NOTICE_A}

    # Missing notice in crate
    license_path, crate, sdist, wheel, notices_dir = _build_test_artifacts(
        tmp_path / "missing", notices, crate_notices={}
    )
    with pytest.raises(ArtifactLicenseError, match="is missing"):
        verify_artifact_licenses(
            license_path=license_path,
            crate_path=crate,
            sdist_path=sdist,
            wheel_path=wheel,
            notices_path=notices_dir,
        )

    # Mismatched notice bytes in crate
    license_path, crate, sdist, wheel, notices_dir = _build_test_artifacts(
        tmp_path / "mismatched", notices, crate_notices={"a.txt": b"wrong content"}
    )
    with pytest.raises(ArtifactLicenseError, match="crate notice does not match"):
        verify_artifact_licenses(
            license_path=license_path,
            crate_path=crate,
            sdist_path=sdist,
            wheel_path=wheel,
            notices_path=notices_dir,
        )


def test_verify_artifact_licenses_rejects_sdist_missing_or_mismatched_notice(
    tmp_path: Path,
) -> None:
    notices = {"a.txt": _NOTICE_A}

    # Missing notice in sdist
    license_path, crate, sdist, wheel, notices_dir = _build_test_artifacts(
        tmp_path / "missing", notices, sdist_notices={}
    )
    with pytest.raises(ArtifactLicenseError, match="is missing"):
        verify_artifact_licenses(
            license_path=license_path,
            crate_path=crate,
            sdist_path=sdist,
            wheel_path=wheel,
            notices_path=notices_dir,
        )

    # Mismatched notice bytes in sdist
    license_path, crate, sdist, wheel, notices_dir = _build_test_artifacts(
        tmp_path / "mismatched", notices, sdist_notices={"a.txt": b"wrong content"}
    )
    with pytest.raises(ArtifactLicenseError, match="sdist notice does not match"):
        verify_artifact_licenses(
            license_path=license_path,
            crate_path=crate,
            sdist_path=sdist,
            wheel_path=wheel,
            notices_path=notices_dir,
        )


def test_verify_artifact_licenses_rejects_wheel_missing_or_mismatched_notice(
    tmp_path: Path,
) -> None:
    notices = {"a.txt": _NOTICE_A}

    # Missing notice in wheel
    license_path, crate, sdist, wheel, notices_dir = _build_test_artifacts(
        tmp_path / "missing",
        notices,
        wheel_notices={},
        record_rows=[_record_row(_WHEEL_LICENSE_NAME, _LICENSE)],
    )
    with pytest.raises(ArtifactLicenseError, match="is missing"):
        verify_artifact_licenses(
            license_path=license_path,
            crate_path=crate,
            sdist_path=sdist,
            wheel_path=wheel,
            notices_path=notices_dir,
        )

    # Mismatched notice bytes in wheel
    license_path, crate, sdist, wheel, notices_dir = _build_test_artifacts(
        tmp_path / "mismatched",
        notices,
        wheel_notices={"a.txt": b"corrupted bytes"},
        record_rows=[
            _record_row(_WHEEL_LICENSE_NAME, _LICENSE),
            _record_row("ndfiltrs/licenses/a.txt", b"corrupted bytes"),
        ],
    )
    with pytest.raises(ArtifactLicenseError, match="wheel notice does not match"):
        verify_artifact_licenses(
            license_path=license_path,
            crate_path=crate,
            sdist_path=sdist,
            wheel_path=wheel,
            notices_path=notices_dir,
        )


def test_verify_artifact_licenses_rejects_bad_record_for_notices(
    tmp_path: Path,
) -> None:
    notices = {"a.txt": _NOTICE_A}
    member = "ndfiltrs/licenses/a.txt"

    # Missing RECORD entry for notice
    license_path, crate, sdist, wheel, notices_dir = _build_test_artifacts(
        tmp_path / "missing_row",
        notices,
        record_rows=[_record_row(_WHEEL_LICENSE_NAME, _LICENSE)],
    )
    with pytest.raises(ArtifactLicenseError, match="RECORD integrity"):
        verify_artifact_licenses(
            license_path=license_path,
            crate_path=crate,
            sdist_path=sdist,
            wheel_path=wheel,
            notices_path=notices_dir,
        )

    # Wrong digest in RECORD
    license_path, crate, sdist, wheel, notices_dir = _build_test_artifacts(
        tmp_path / "bad_digest",
        notices,
        record_rows=[
            _record_row(_WHEEL_LICENSE_NAME, _LICENSE),
            f"{member},sha256=invalidhash,{len(_NOTICE_A)}",
        ],
    )
    with pytest.raises(ArtifactLicenseError, match="RECORD integrity"):
        verify_artifact_licenses(
            license_path=license_path,
            crate_path=crate,
            sdist_path=sdist,
            wheel_path=wheel,
            notices_path=notices_dir,
        )

    # Wrong size in RECORD
    license_path, crate, sdist, wheel, notices_dir = _build_test_artifacts(
        tmp_path / "bad_size",
        notices,
        record_rows=[
            _record_row(_WHEEL_LICENSE_NAME, _LICENSE),
            _record_row(member, _NOTICE_A).rsplit(",", maxsplit=1)[0]
            + f",{len(_NOTICE_A) + 10}",
        ],
    )
    with pytest.raises(ArtifactLicenseError, match="RECORD integrity"):
        verify_artifact_licenses(
            license_path=license_path,
            crate_path=crate,
            sdist_path=sdist,
            wheel_path=wheel,
            notices_path=notices_dir,
        )
