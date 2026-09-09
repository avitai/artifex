# Releasing Artifex

Artifex publishes through `.github/workflows/publish.yml` with PyPI trusted
publishing. No commit or tag push creates a release by itself. Release timing and
versioning stay under operator control. The manual `target=github-release` workflow
path creates a GitHub Release for an explicit existing tag with
`softprops/action-gh-release@v3` and `generate_release_notes: true`, then
publishes to PyPI. Publishing an existing GitHub Release also runs the PyPI upload
path.

## Release Checklist

1. Activate the local environment.

   ```bash
   source activate.sh
   ```

2. Bump `version` in `pyproject.toml`. The version is static, so refresh the lock in
   the same commit:

   ```bash
   uv lock
   ```

3. Update `CHANGELOG.md` by moving unreleased entries under the new version and
   date.
4. Run the release checks.

   ```bash
   uv lock --check
   uv run pre-commit run --all-files
   uv run mkdocs build --strict --clean
   rm -rf dist/
   uv build
   uv run twine check --strict dist/*
   ```

5. Commit the version, lock and changelog updates, push, and read CI at the job level
   for that commit. Every workflow must be green before the tag exists.
6. Create and push an annotated tag from the exact release commit.

   ```bash
   target_sha=$(git rev-parse HEAD)
   git tag -a vX.Y.Z -m "artifex X.Y.Z"
   git push origin main vX.Y.Z
   ```

7. In GitHub Actions, manually run `Publish to PyPI` with:

   - `target=github-release`
   - `version_tag=vX.Y.Z`

   The workflow verifies that the tag exists, creates the GitHub Release with
   generated release notes, then publishes to PyPI.

8. Confirm the upload from a throwaway environment.

   ```bash
   uv venv /tmp/artifex-smoke && uv pip install --python /tmp/artifex-smoke avitai-artifex==X.Y.Z
   ```

## Manual Release Recovery

If the manual generated-release workflow is interrupted before creating the
GitHub Release, create the release from the exact tagged commit:

```bash
gh release create vX.Y.Z --target "$target_sha" --generate-notes
```

Publishing that release triggers the same PyPI upload workflow.

## TestPyPI

Use the manual `workflow_dispatch` path in `publish.yml` with
`target=testpypi` when validating the trusted publishing setup before a real
release.

## PyPI Trusted Publishing

PyPI must trust, for the project `avitai-artifex`:

- Owner: `avitai`
- Repository: `artifex`
- Workflow: `publish.yml`
- Environment: `pypi`

If PyPI rejects the publish with `invalid-publisher`, verify the trusted
publisher registration before looking for repository secrets. The expected
publisher identity is:

```text
repo:avitai/artifex:environment:pypi
```

For TestPyPI, the expected environment is `testpypi`.
