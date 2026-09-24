# Releasing ndfiltrs

Use the ImCombiners release pattern across these packages: **manual rehearsal,
then a matching version tag to publish**. One `release.yml` owns the release;
it may call the existing `ci.yml` for tests. Build/test commands vary by package:

| Package | Build | Registries |
| --- | --- | --- |
| ndfiltrs | Maturin; platform wheels + sdist | PyPI |
| imcombiners / reducers | Maturin; platform wheels + sdist | PyPI + crates.io |
| astroimred | uv_build; pure-Python wheel + sdist | PyPI |

Shared rules: check versions, use published dependencies, test built artifacts,
use trusted publishing, and make `workflow_dispatch` build-only even on tags.
Only a `vX.Y.Z` tag push publishes; GitHub release creation follows successful
registry uploads. Keep publisher credentials confined to publishing jobs.
Crates.io applies only where a supported Rust API is deliberately published.
This is the chosen convention; sibling repositories still need to adopt it.

## One-time setup

1. Review and commit the source, including new files and intended deletions.
   Create the GitHub repository with no initial files. Assumed owner/name:
   `ysBach/ndfiltrs`; adjust manifest URLs and publisher settings if different.

   ```sh
   gh repo create ysBach/ndfiltrs --public --source=. --remote=origin
   ```

   If the repository already exists, inspect `git remote -v`, configure its
   actual URL as `origin`, and complete the setup below before pushing `main`.

2. Create a GitHub environment named `pypi`, restricted to `v*` tags. Register a
   [PyPI pending publisher](https://pypi.org/manage/account/publishing/) for the
   new project (or use Publishing settings for an existing project you own):

   | Setting | Value |
   | --- | --- |
   | Project / owner / repository | `ndfiltrs` / `ysBach` / `ndfiltrs` |
   | Workflow filename / environment | `release.yml` / `pypi` |

   Pending publishers do not reserve names. Use the
   [official first-project instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
   No permanent PyPI token is needed in GitHub secrets.

3. Set **Settings → Pages → Source** to **GitHub Actions** before the first
   push to `main`; otherwise the Docs deploy job fails. Details:
   [docs/quarto/contributing.qmd](docs/quarto/contributing.qmd).

   Push only the reviewed branch. Do not use `--all` or `--mirror`: local backup
   branches may contain files that are intentionally absent from the public history.

   ```sh
   git push -u origin main
   ```

## Each release

1. Keep dependencies on published registry versions. The build pins
   `reducers 0.4.0` without a local path override. After dependency changes,
   update `Cargo.lock`, then regenerate notices:

   ```sh
   cargo fetch --locked
   uv run --no-sync python scripts/update_rust_notices.py
   ```

2. Keep the version identical in `Cargo.toml`, `pyproject.toml`, both lockfiles,
   and `python/ndfiltrs/__init__.py`. Date its changelog section as
   `## X.Y.Z - YYYY-MM-DD`. Commit and push. Use the project `.venv`, with
   Python 3.11+ for release tooling; never synchronize `~/.venvs/ysbpy`.

3. Check the committed tree and rehearse:

   ```sh
   uv run --no-sync python scripts/check_release.py --require-clean
   gh workflow run release.yml --ref main
   gh run list --workflow release.yml
   ```

   Inspect the run. CI and all artifact jobs must pass; publishing must be
   skipped. An `Unreleased` changelog section is allowed for early rehearsals.

4. Tag exactly the successfully rehearsed commit. For the first release:

   ```sh
   uv run --no-sync python scripts/check_release.py --tag v0.1.0 --require-clean
   git tag -a v0.1.0 -m "Release 0.1.0"
   git push origin v0.1.0
   ```

   **Pushing the tag publishes.** Check the Actions run, PyPI version, and GitHub
   release. From the repository root, verify installation separately:

   ```sh
   uv venv --no-project /tmp/ndfiltrs-pypi-check --python 3.13
   uv pip install --python /tmp/ndfiltrs-pypi-check/bin/python --index-url https://pypi.org/simple ndfiltrs==0.1.0
   EXPECTED_VERSION=0.1.0 /tmp/ndfiltrs-pypi-check/bin/python -I scripts/smoke_install.py
   ```

   These paths are for macOS/Linux; Windows uses the environment's
   `Scripts/python.exe`. Choose an unused environment directory.

## Coverage and recovery

NF builds `cp310-abi3` wheels for Linux/macOS x86-64 and ARM64, and Windows
x86-64. Linux requires glibc 2.28+. Wheels receive Python 3.10 tests and a 3.13
smoke test; CI separately verifies test-hook builds and sdist installation.
These are configured checks; platform support requires successful hosted runs.
The Rust API is private, so NF retains `publish = false`.

On failure, rerun **failed jobs** in the original run, preserving its artifacts.
`uv publish --check-url` skips identical uploads and rejects differing content;
see [uv's retry behavior](https://docs.astral.sh/uv/guides/package/#publishing-your-package).
For source/artifact changes, use a new version; never move a published tag.
Registries and GitHub release creation are not one atomic transaction.
