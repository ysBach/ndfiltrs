"""Check release metadata without importing the extension (Python 3.11+)."""

from __future__ import annotations

import argparse
import ast
import logging
import re
import subprocess
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import tomllib

_NAME = "ndfiltrs"
_VERSION = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"


def _check_dependency_sources(cargo: dict[str, Any], python: dict[str, Any]) -> None:
    """Reject dependency sources that bypass published registry packages.

    Parameters
    ----------
    cargo, python : dict
        Parsed Cargo and Python manifests. Inspect direct, target, workspace,
        and patched Cargo dependencies and Python requirement/source tables.

    Raises
    ------
    ValueError
        Dependencies use local/Git overrides or direct Python URL references.
    """
    scopes = [("Cargo.toml", cargo)]
    workspace = cargo.get("workspace", {})
    scopes.append(("Cargo.toml workspace", workspace))
    scopes.extend(
        (f"Cargo.toml target {name}", table)
        for name, table in cargo.get("target", {}).items()
    )
    for label, scope in scopes:
        tables = [
            (f"{label} {key}", scope.get(key, {}))
            for key in ("dependencies", "dev-dependencies", "build-dependencies")
        ]
        tables.extend(
            (f"{label} patch {registry}", table)
            for registry, table in scope.get("patch", {}).items()
        )
        if scope.get("replace"):
            raise ValueError(
                f"{label}: dependency replace overrides are not release sources"
            )
        for location, dependencies in tables:
            for name, specification in dependencies.items():
                if (
                    isinstance(specification, dict)
                    and specification.get("workspace") is True
                ):
                    specification = workspace.get("dependencies", {}).get(name)
                if isinstance(specification, dict):
                    if "path" in specification or "git" in specification:
                        raise ValueError(
                            f"{location}: {name} uses a path/git dependency override"
                        )
                    version = specification.get("version")
                else:
                    version = specification
                if not isinstance(version, str) or not version.strip():
                    raise ValueError(f"{location}: {name} requires a registry version")

    if python.get("tool", {}).get("uv", {}).get("sources"):
        raise ValueError(
            "pyproject.toml: tool.uv.sources overrides are not release sources"
        )
    project = python.get("project", {})
    groups = [
        ("project.dependencies", project.get("dependencies", [])),
        ("build-system.requires", python.get("build-system", {}).get("requires", [])),
    ]
    groups.extend(
        (f"project.optional-dependencies.{name}", values)
        for name, values in project.get("optional-dependencies", {}).items()
    )
    groups.extend(
        (f"dependency-groups.{name}", values)
        for name, values in python.get("dependency-groups", {}).items()
    )
    for label, requirements in groups:
        for requirement in requirements:
            # Dependency-group include directives do not change package sources.
            if isinstance(requirement, dict) and set(requirement) == {"include-group"}:
                continue
            if not isinstance(requirement, str):
                raise ValueError(
                    f"pyproject.toml {label}: expected a registry requirement"
                )
            reference = requirement.split(";", 1)[0].strip()
            if "@" in reference or re.match(
                r"(?:[A-Za-z][A-Za-z0-9+.-]*:|[./\\])", reference
            ):
                raise ValueError(
                    f"pyproject.toml {label}: direct local/URL dependency "
                    f"{requirement!r}"
                )


def check_release(
    root: Path, *, tag: str | None = None, require_clean: bool = False
) -> tuple[str, str]:
    """Validate package identity, changelog, optional tag, and Git cleanliness.

    Parameters
    ----------
    root : pathlib.Path
        Repository containing the manifests, lockfiles, and Python sources.
    tag : str or None, optional
        Exact ``vX.Y.Z`` release tag. `None` and the empty string mean rehearsal.
    require_clean : bool, optional
        Reject tracked or untracked changes, including staged changes.

    Returns
    -------
    version : str
        Consistent stable release version.
    notes : str
        Current changelog section body, without its level-two heading. Nonempty
        notes have one final newline; surrounding blank lines are removed.

    Raises
    ------
    ValueError
        Metadata, tag, changelog, or working-tree checks fail.
    OSError
        A required file or Git executable cannot be accessed.
    """
    metadata: dict[str, dict] = {}
    for filename in ("Cargo.toml", "pyproject.toml", "Cargo.lock", "uv.lock"):
        with (root / filename).open("rb") as source:
            metadata[filename] = tomllib.load(source)
    cargo = metadata["Cargo.toml"].get("package", {})
    project = metadata["pyproject.toml"].get("project", {})
    for filename, package in (("Cargo.toml", cargo), ("pyproject.toml", project)):
        if not isinstance(package, dict) or package.get("name") != _NAME:
            raise ValueError(f"{filename}: package name must be {_NAME!r}")
    version = cargo.get("version")
    if not isinstance(version, str) or re.fullmatch(_VERSION, version) is None:
        raise ValueError("Cargo.toml: version must be a stable X.Y.Z release")
    if project.get("version") != version:
        raise ValueError(f"pyproject.toml: version must match {version}")
    _check_dependency_sources(metadata["Cargo.toml"], metadata["pyproject.toml"])

    for filename in ("Cargo.lock", "uv.lock"):
        entries = metadata[filename].get("package", [])
        if not isinstance(entries, list):
            raise ValueError(f"{filename}: expected a package list")
        local = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("name") == _NAME
            and (
                "source" not in entry
                if filename == "Cargo.lock"
                else entry.get("source") in ({"editable": "."}, {"virtual": "."})
            )
        ]
        if len(local) != 1:
            raise ValueError(f"{filename}: expected exactly one local {_NAME} entry")
        if local[0].get("version") != version:
            raise ValueError(f"{filename}: local version must match {version}")

    init_path = root / "python" / _NAME / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    assignments: list[ast.expr | None] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in targets
        ):
            assignments.append(node.value)
    if (
        len(assignments) != 1
        or not isinstance(assignments[0], ast.Constant)
        or not isinstance(assignments[0].value, str)
        or assignments[0].value != version
    ):
        raise ValueError(f"{init_path}: require one literal __version__ = {version!r}")
    if tag and tag != f"v{version}":
        raise ValueError(f"tag {tag!r} must match v{version}")

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = list(re.finditer(r"^##[ \t]+(.+?)[ \t]*$", changelog, re.MULTILINE))
    current = [
        (index, heading)
        for index, heading in enumerate(headings)
        if re.fullmatch(rf"{re.escape(version)} - (.+)", heading.group(1))
    ]
    if len(current) != 1:
        raise ValueError(
            f"CHANGELOG.md: expected exactly one '## {version} - ...' section"
        )
    index, heading = current[0]
    status = heading.group(1).removeprefix(f"{version} - ")
    if status == "Unreleased":
        if tag:
            raise ValueError("CHANGELOG.md: tagged release cannot be Unreleased")
    else:
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", status) is None:
            raise ValueError("CHANGELOG.md: date must be YYYY-MM-DD or Unreleased")
        try:
            date.fromisoformat(status)
        except ValueError as error:
            raise ValueError(
                f"CHANGELOG.md: invalid release date {status!r}"
            ) from error
    end = headings[index + 1].start() if index + 1 < len(headings) else len(changelog)
    notes = changelog[heading.end() : end].strip()
    if tag and not notes:
        raise ValueError("CHANGELOG.md: tagged release requires nonempty notes")

    if require_clean:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise ValueError(f"git status failed: {result.stderr.strip()}")
        if result.stdout:
            raise ValueError(
                "release requires a clean tree (tracked and untracked files)"
            )
    return version, notes + "\n" if notes else ""


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a release, then write requested notes and GitHub output.

    Parameters
    ----------
    argv : sequence of str or None, optional
        CLI arguments; `None` reads the process arguments. Relative output
        paths use the current working directory, independently of `--root`.

    Returns
    -------
    int
        Zero on success, one on validation or I/O failure. Notes overwrite the
        requested file; GitHub output appends ``version=X.Y.Z`` and a newline.
        All validation, including cleanliness, precedes either output write.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--tag", help="matching vX.Y.Z tag; omit for rehearsal")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--notes-output", type=Path)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)
    try:
        version, notes = check_release(
            args.root, tag=args.tag, require_clean=args.require_clean
        )
        if args.notes_output is not None:
            with args.notes_output.open("w", encoding="utf-8", newline="\n") as output:
                output.write(notes)
        if args.github_output is not None:
            with args.github_output.open("a", encoding="utf-8", newline="\n") as output:
                output.write(f"version={version}\n")
    except (ValueError, OSError, SyntaxError) as error:
        logging.error("Release check failed: %s", error)
        return 1
    logging.info("Release metadata verified: %s", version)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
