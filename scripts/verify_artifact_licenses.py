"""Verify MIT license identity and integrity in release artifacts."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import logging
import re
import tarfile
import zipfile
from email.message import Message
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath

_PACKAGE_NAME = "ndfiltrs"
_PACKAGE_VERSION = next(
    ast.literal_eval(node.value)
    for node in ast.parse(
        (Path(__file__).resolve().parents[1] / "python/ndfiltrs/__init__.py").read_text(
            encoding="utf-8"
        )
    ).body
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name) and target.id == "__version__"
        for target in node.targets
    )
)
_LICENSE_EXPRESSION = "MIT"
_ARCHIVE_ROOT = f"{_PACKAGE_NAME}-{_PACKAGE_VERSION}"
_DIST_INFO_ROOT = f"{_PACKAGE_NAME}-{_PACKAGE_VERSION}.dist-info"


class ArtifactLicenseError(ValueError):
    """Raised when release-artifact license data is inconsistent."""


def _resolve_artifact(path: Path, suffix: str, label: str) -> Path:
    if path.is_file():
        if not path.name.endswith(suffix):
            raise ArtifactLicenseError(f"{label} artifact must end with {suffix}")
        return path
    if not path.is_dir():
        raise ArtifactLicenseError(f"{label} artifact path does not exist: {path}")

    matches = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.name.endswith(suffix)
    )
    if len(matches) != 1:
        raise ArtifactLicenseError(
            f"{label} directory must contain exactly one {suffix} artifact; "
            f"found {len(matches)}"
        )
    return matches[0]


def _checked_tar_members(
    archive: tarfile.TarFile,
    label: str,
) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    roots: set[str] = set()
    names: set[str] = set()
    for member in archive.getmembers():
        pure_path = PurePosixPath(member.name)
        if pure_path.is_absolute() or ".." in pure_path.parts or not pure_path.parts:
            raise ArtifactLicenseError(f"{label} archive contains an unsafe path")
        if member.name in names:
            raise ArtifactLicenseError(
                f"{label} archive contains duplicate member names"
            )
        names.add(member.name)
        roots.add(pure_path.parts[0])
        if member.isfile():
            members[member.name] = member

    if len(roots) != 1:
        raise ArtifactLicenseError(f"{label} archive must have one root")
    root = next(iter(roots))
    if root != _ARCHIVE_ROOT:
        raise ArtifactLicenseError(
            f"{label} archive root must be {_ARCHIVE_ROOT}; found {root}"
        )
    return members


def _tar_bytes(path: Path, member_name: str, label: str) -> bytes:
    with tarfile.open(path, "r:*") as archive:
        members = _checked_tar_members(archive, label)
        member = members.get(member_name)
        if member is None:
            raise ArtifactLicenseError(f"{label} is missing {member_name}")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ArtifactLicenseError(f"{label} member is not readable: {member_name}")
        return extracted.read()


def _verify_cargo_metadata(contents: bytes) -> None:
    section: str | None = None
    fields: dict[str, str] = {}
    section_pattern = re.compile(r"^\s*\[([^]]+)]\s*(?:#.*)?$")
    field_pattern = re.compile(
        r'^\s*(name|version|license)\s*=\s*"([^"]*)"\s*(?:#.*)?$'
    )

    for line in contents.decode("utf-8").splitlines():
        section_match = section_pattern.match(line)
        if section_match:
            section = section_match.group(1).strip()
            continue
        if section != "package":
            continue
        field_match = field_pattern.match(line)
        if field_match:
            key, value = field_match.groups()
            if key in fields:
                raise ArtifactLicenseError(f"crate Cargo.toml duplicates package {key}")
            fields[key] = value

    expected = {
        "name": _PACKAGE_NAME,
        "version": _PACKAGE_VERSION,
        "license": _LICENSE_EXPRESSION,
    }
    if fields != expected:
        raise ArtifactLicenseError(
            "crate Cargo.toml package identity or license is inconsistent"
        )


def _one_metadata_value(message: Message, field: str) -> str:
    values = message.get_all(field, [])
    if len(values) != 1:
        raise ArtifactLicenseError(f"metadata must contain exactly one {field} field")
    return str(values[0])


def _verify_python_metadata(contents: bytes, label: str) -> None:
    message = BytesParser(policy=default).parsebytes(contents, headersonly=True)
    name = _one_metadata_value(message, "Name")
    version = _one_metadata_value(message, "Version")
    if (name, version) != (_PACKAGE_NAME, _PACKAGE_VERSION):
        raise ArtifactLicenseError(f"{label} metadata package identity is inconsistent")
    expressions = message.get_all("License-Expression", [])
    if expressions != [_LICENSE_EXPRESSION]:
        raise ArtifactLicenseError(
            f"{label} metadata is missing License-Expression: MIT"
        )
    license_files = message.get_all("License-File", [])
    if license_files != ["LICENSE"]:
        raise ArtifactLicenseError(f"{label} metadata is missing License-File: LICENSE")


def _zip_member_bytes(archive: zipfile.ZipFile, member_name: str, label: str) -> bytes:
    try:
        return archive.read(member_name)
    except KeyError as error:
        raise ArtifactLicenseError(f"{label} is missing {member_name}") from error


def _verify_wheel(
    path: Path, expected_license: bytes, notices: dict[str, bytes]
) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ArtifactLicenseError("wheel contains duplicate member names")
        if any(
            (pure_path := PurePosixPath(name)).is_absolute()
            or ".." in pure_path.parts
            or not pure_path.parts
            for name in names
        ):
            raise ArtifactLicenseError("wheel contains an unsafe path")
        dist_info_roots = {
            pure_path.parts[0]
            for name in names
            if (pure_path := PurePosixPath(name)).parts
            and pure_path.parts[0].endswith(".dist-info")
        }
        if dist_info_roots != {_DIST_INFO_ROOT}:
            raise ArtifactLicenseError(
                f"wheel must contain exactly one {_DIST_INFO_ROOT} directory"
            )

        license_name = f"{_DIST_INFO_ROOT}/licenses/LICENSE"
        metadata_name = f"{_DIST_INFO_ROOT}/METADATA"
        record_name = f"{_DIST_INFO_ROOT}/RECORD"
        wheel_license = _zip_member_bytes(archive, license_name, "wheel")
        if wheel_license != expected_license:
            raise ArtifactLicenseError(
                "wheel license does not match repository LICENSE"
            )
        _verify_python_metadata(
            _zip_member_bytes(archive, metadata_name, "wheel"),
            "wheel",
        )
        record = _zip_member_bytes(archive, record_name, "wheel").decode("utf-8")
        expected_files = {license_name: wheel_license}
        for name, contents in notices.items():
            member = f"ndfiltrs/licenses/{name}"
            if _zip_member_bytes(archive, member, "wheel") != contents:
                raise ArtifactLicenseError(
                    f"wheel notice does not match source: {name}"
                )
            expected_files[member] = contents

    all_rows = list(csv.reader(io.StringIO(record)))
    for name, contents in expected_files.items():
        rows = [row for row in all_rows if row and row[0] == name]
        if len(rows) != 1 or len(rows[0]) != 3:
            raise ArtifactLicenseError(
                f"wheel RECORD integrity check failed for {name}"
            )
        digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest())
        expected_digest = f"sha256={digest.rstrip(b'=').decode('ascii')}"
        if rows[0][1:] != [expected_digest, str(len(contents))]:
            raise ArtifactLicenseError(
                f"wheel RECORD integrity check failed for {name}"
            )


def verify_artifact_licenses(
    license_path: Path,
    crate_path: Path | None,
    sdist_path: Path | None,
    wheel_path: Path | None,
    notices_path: Path | None = None,
) -> None:
    """Verify license metadata and optional bundled notices in artifacts.

    Parameters
    ----------
    license_path : pathlib.Path
        Authoritative first-party license file.
    crate_path, sdist_path, wheel_path : pathlib.Path or None
        Artifact file or directory with a single matching artifact.
    notices_path : pathlib.Path or None, optional
        Directory of authoritative bundled `*.txt` third-party notices.
    """
    if crate_path is None and sdist_path is None and wheel_path is None:
        raise ArtifactLicenseError("at least one artifact is required")

    expected_license = license_path.read_bytes()
    notices = {}
    if notices_path is not None:
        if not notices_path.is_dir():
            raise ArtifactLicenseError("notices path must be a directory")
        notices = {
            file.name: file.read_bytes() for file in sorted(notices_path.glob("*.txt"))
        }
        if not notices:
            raise ArtifactLicenseError("notices directory contains no .txt notices")
    if crate_path is not None:
        crate = _resolve_artifact(crate_path, ".crate", "crate")
        crate_license = _tar_bytes(crate, f"{_ARCHIVE_ROOT}/LICENSE", "crate")
        if crate_license != expected_license:
            raise ArtifactLicenseError(
                "crate license does not match repository LICENSE"
            )
        _verify_cargo_metadata(
            _tar_bytes(crate, f"{_ARCHIVE_ROOT}/Cargo.toml", "crate")
        )
        for name, contents in notices.items():
            if (
                _tar_bytes(
                    crate, f"{_ARCHIVE_ROOT}/python/ndfiltrs/licenses/{name}", "crate"
                )
                != contents
            ):
                raise ArtifactLicenseError(
                    f"crate notice does not match source: {name}"
                )

    if sdist_path is not None:
        sdist = _resolve_artifact(sdist_path, ".tar.gz", "sdist")
        sdist_license = _tar_bytes(sdist, f"{_ARCHIVE_ROOT}/LICENSE", "sdist")
        if sdist_license != expected_license:
            raise ArtifactLicenseError(
                "sdist license does not match repository LICENSE"
            )
        _verify_python_metadata(
            _tar_bytes(sdist, f"{_ARCHIVE_ROOT}/PKG-INFO", "sdist"),
            "sdist",
        )
        for name, contents in notices.items():
            if (
                _tar_bytes(
                    sdist, f"{_ARCHIVE_ROOT}/python/ndfiltrs/licenses/{name}", "sdist"
                )
                != contents
            ):
                raise ArtifactLicenseError(
                    f"sdist notice does not match source: {name}"
                )

    if wheel_path is not None:
        wheel = _resolve_artifact(wheel_path, ".whl", "wheel")
        _verify_wheel(wheel, expected_license, notices)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--license", required=True, type=Path)
    parser.add_argument("--crate", type=Path)
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--notices", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run artifact license verification from the command line."""
    arguments = _parse_args()
    try:
        verify_artifact_licenses(
            arguments.license,
            arguments.crate,
            arguments.sdist,
            arguments.wheel,
            arguments.notices,
        )
    except (
        ArtifactLicenseError,
        OSError,
        tarfile.TarError,
        UnicodeDecodeError,
        zipfile.BadZipFile,
    ) as error:
        logging.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
