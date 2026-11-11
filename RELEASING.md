# Releasing ndfiltrs

First, run a manual release rehearsal. Publish only after that rehearsal
succeeds, by pushing a matching version tag.

This follows the ImCombiners release pattern. One `release.yml` controls the
release. It may call the existing `ci.yml` for tests.
Build and test commands vary by package:

| Package | Build | Registries |
| --- | --- | --- |
| ndfiltrs | Maturin: platform wheels + sdist | PyPI + crates.io |
| [imcombiners](https://github.com/ysBach/imcombiners) / [reducers](https://github.com/ysBach/reducers) | Maturin: platform wheels + sdist | PyPI + crates.io |
| [astroimred](https://github.com/ysBach/astroimred) | uv_build: pure-Python wheel + sdist | PyPI |

Apply these shared rules:

- Check package versions.
- Use published dependencies.
- Test built artifacts.
- Use trusted publishing. Keep publisher credentials in publishing jobs only.
- Keep `workflow_dispatch` build-only, even on tags.

Only a `vX.Y.Z` tag push publishes. The workflow creates a GitHub release after
registry uploads succeed. Use crates.io only for packages with a deliberately
published, supported Rust API. Sibling repositories still need to adopt this
release convention.

## One-time setup

1. Review the source, including new files and intended deletions.
   Commit the reviewed source. Create the GitHub repository with no initial
   files if it does not exist. These commands assume `ysBach/ndfiltrs`.
   Adjust manifest URLs and publisher settings if the owner or repository name differs.

   ```sh
   gh repo create ysBach/ndfiltrs --public --source=. --remote=origin
   ```

   If the repository already exists, inspect `git remote -v`.
   Configure its actual URL as `origin`.
   Complete the setup below before pushing `main`.

2. Create GitHub environments named `pypi` and `crates-io`, each restricted to
   `v*` tags. Register a
   [PyPI pending publisher](https://pypi.org/manage/account/publishing/) for the
   new project (or use Publishing settings for an existing project you own):

   | Setting | Value |
   | --- | --- |
   | Project / owner / repository | `ndfiltrs` / `ysBach` / `ndfiltrs` |
   | Workflow filename / environment | `release.yml` / `pypi` |

   Pending publishers do not reserve names. Use the
   [official first-project instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
   This setup does not need a permanent PyPI token in GitHub secrets.

   crates.io accepts a trusted publisher only for an existing crate
   ([RFC 3691](https://rust-lang.github.io/rfcs/3691-trusted-publishing-cratesio.html)).
   Publish the first version by hand from the rehearsed commit, before you
   push its tag:

   ```sh
   cargo login          # paste a crates.io API token with publish-new scope
   cargo publish --locked
   ```

   Then add the trusted publisher in the crate's crates.io settings:
   owner `ysBach`, repository `ndfiltrs`, workflow `release.yml`, environment
   `crates-io`. Revoke the API token after use. The `crates-io` job skips a
   version that already exists on crates.io, so the tag of the hand-published
   version still creates the GitHub release.

3. Set **Settings → Pages → Source** to **GitHub Actions** before the first
   push to `main`. Otherwise, the Docs deploy job fails. Details:
   [Build the documentation](README.md#build-the-documentation).

   Push only the reviewed branch. Do not use `--all` or `--mirror`: local backup
   branches may contain files that are intentionally absent from the public history.

   ```sh
   git push -u origin main
   ```

## Each release

1. Keep dependencies on published registry versions. The build pins
   [reducers 0.4.0](https://crates.io/crates/reducers/0.4.0) without a local path
   override. After dependency changes, update `Cargo.lock`.
   Regenerate notices with:

   ```sh
   cargo fetch --locked
   uv run --no-sync python scripts/update_rust_notices.py
   ```

2. Keep the version identical in `Cargo.toml`, `pyproject.toml`, both lockfiles,
   and `python/ndfiltrs/__init__.py`. Date its changelog section as
   `## X.Y.Z - YYYY-MM-DD`. Commit the changes. Push the commit.
   Use the project `.venv`, with Python 3.11+ for release tooling.
   Never synchronize `~/.venvs/ysbpy`.

3. Check the committed tree and rehearse:

   ```sh
   uv run --no-sync python scripts/check_release.py --require-clean
   gh workflow run release.yml --ref main
   gh run list --workflow release.yml
   ```

   Inspect the run. CI and all artifact jobs must pass.
   Publishing must remain skipped.
   Early rehearsals may use an `Unreleased` changelog section.

4. Tag exactly the successfully rehearsed commit. For the first release:

   ```sh
   uv run --no-sync python scripts/check_release.py --tag v0.1.0 --require-clean
   git tag -a v0.1.0 -m "Release 0.1.0"
   git push origin v0.1.0
   ```

   **Pushing the tag publishes** to PyPI and crates.io. A plain `git push`
   does not push tags. Check the Actions run, the PyPI and crates.io versions,
   and the GitHub release. From the repository root, check installation separately:

   ```sh
   uv venv --no-project /tmp/ndfiltrs-pypi-check --python 3.13
   uv pip install --python /tmp/ndfiltrs-pypi-check/bin/python --index-url https://pypi.org/simple ndfiltrs==0.1.0
   EXPECTED_VERSION=0.1.0 /tmp/ndfiltrs-pypi-check/bin/python -I scripts/smoke_install.py
   ```

   These paths are for macOS/Linux. Windows uses the environment's
   `Scripts/python.exe`. Choose an unused environment directory.

## Coverage and recovery

`ndfiltrs` builds `cp310-abi3` wheels for Linux/macOS x86-64 and ARM64, and Windows
x86-64. Linux requires glibc 2.28+. Wheels receive Python 3.10 tests and a 3.13
smoke test. CI separately checks test-hook builds and sdist installation.
The workflows configure these checks. Platform support requires successful
hosted runs. The crate's public Rust API is the set of plans and functions
re-exported from `src/lib.rs`. Kernel modules stay private.

On failure, rerun **failed jobs** in the original run. Preserve its artifacts.
`uv publish --check-url` skips identical uploads and rejects differing content.
See [uv's retry behavior](https://docs.astral.sh/uv/guides/package/#publishing-your-package).
For changes to source or artifacts, use a new version.
Never move a published tag.
Registries and GitHub release creation are not one atomic transaction.
