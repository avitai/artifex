# Artifex Scripts Directory

This directory contains repo-maintenance helpers. Supported contributor
workflows should use `uv run python ...` for Python entrypoints or checked-in
shell wrappers such as `./scripts/serve_docs_fast.sh`.

Some local one-off maintenance scripts remain intentionally undocumented here
until their safety contracts are repaired. This catalog only covers retained
surfaces with reviewed behavior.

## 📋 Table of Contents

- [Environment Setup](#-environment-setup)
- [Testing](#-testing)
- [GPU/Hardware Verification](#-gpuhardware-verification)
- [Code Analysis](#-code-analysis)
- [Documentation](#-documentation)
- [Script Guidelines](#-script-guidelines)
- [Maintenance](#-maintenance)

## 🚀 Environment Setup

### setup_env.py

**Purpose:** Write and inspect the generated `.artifex.env` file loaded by
`activate.sh`.

**Usage:**

```bash
uv run python scripts/setup_env.py show
uv run python scripts/setup_env.py write --backend cpu
uv run python scripts/setup_env.py write --backend cuda12
```

For full environment setup, use `./setup.sh` in the project root. The generated
file is `.artifex.env`; `.env` remains user-owned.

## 🧪 Testing

Ad hoc test wrappers have been removed. Use direct `uv run pytest` commands so
the test surface stays aligned with the central pytest configuration.

```bash
uv run pytest tests/ -v
uv run pytest -m gpu -v
uv run pytest -m blackjax -v
uv run pytest tests/artifex/generative_models/core/sampling/test_blackjax_samplers.py -v
```

## 🎮 GPU/Hardware Verification

### verify_gpu_setup.py

**Purpose:** Verify the active JAX backend and device visibility.

```bash
uv run python scripts/verify_gpu_setup.py
uv run python scripts/verify_gpu_setup.py --require-gpu
uv run python scripts/verify_gpu_setup.py --json
```

### gpu_utils.py

**Purpose:** Run repo-local GPU diagnostics against the current runtime device
surface.

```bash
uv run python scripts/gpu_utils.py
uv run python scripts/gpu_utils.py --detailed
uv run python scripts/gpu_utils.py --test
uv run python scripts/gpu_utils.py --test-critical
```

### check_jax_nnx_compatibility.py

**Purpose:** Compare the installed JAX ecosystem packages against the live repo
dependency contract declared in `pyproject.toml`.

```bash
uv run python scripts/check_jax_nnx_compatibility.py
uv run python scripts/check_jax_nnx_compatibility.py --json
```

## 📝 Code Analysis

### analyze_dependencies.py

**Purpose:** Analyze repo-local import relationships and render dependency
artifacts into ignored output paths.

```bash
uv run python scripts/analyze_dependencies.py
uv run python scripts/analyze_dependencies.py --source-dir src/artifex
uv run python scripts/analyze_dependencies.py --output-dir test_artifacts/code_analysis/custom
```

### analyze_test_structure.py

**Purpose:** Analyze test-to-source coverage and write reports into ignored
repo-local artifacts.

Default outputs land under
`test_artifacts/code_analysis/test_structure/`, not under curated docs.

```bash
uv run python scripts/analyze_test_structure.py
uv run python scripts/analyze_test_structure.py --test-dir tests --src-dir src/artifex
uv run python scripts/analyze_test_structure.py --json-output test_artifacts/code_analysis/test_structure/custom.json
```

### find_circular_imports.py

**Purpose:** Report full repo-local circular dependency groups from the import
graph.

```bash
uv run python scripts/find_circular_imports.py
uv run python scripts/find_circular_imports.py --source-dir src/artifex --fail-on-cycles
```

## 📚 Documentation

### build_docs.py

**Purpose:** Run the docs validator, then perform a strict MkDocs build. The
optional serve mode uses the same validated config path.

```bash
uv run python scripts/build_docs.py
uv run python scripts/build_docs.py --config-path mkdocs.yml --docs-path docs --src-path src
uv run python scripts/build_docs.py --serve --port 8000
```

### validate_docs.py

**Purpose:** Validate the curated docs contract: nav coverage, internal links,
and mkdocstrings module references.

`--fix` is intentionally narrow. It only repairs safe scaffolding gaps such as
a missing `theme.custom_dir`; unresolved issues still exit non-zero.

```bash
uv run python scripts/validate_docs.py --check-only
uv run python scripts/validate_docs.py --fix
uv run python scripts/validate_docs.py --fix --verbose --config-path mkdocs.yml --docs-path docs --src-path src
```

### serve_docs_fast.sh

**Purpose:** Serve the local docs quickly with `mkdocs-dev.yml` and dirty
reload.

```bash
./scripts/serve_docs_fast.sh
```

## 🧹 Maintenance

### clean_cache.sh

**Purpose:** Remove common cache files and temporary artifacts while preserving
the virtual environment, lock files, and Git metadata.

```bash
./scripts/clean_cache.sh
```

## 🔧 Script Guidelines

When adding or changing scripts:

1. Keep the CLI surface narrow and truthful.
2. Prefer `uv run python ...` for contributor-facing Python workflows.
3. Default write-capable helpers to ignored output paths or explicit reviewed targets.
4. Add or update repo-contract tests before expanding the documented surface.
5. Document only the retained, supported behavior in this file.

## 📊 Examples

Example workflows live under `examples/` and are documented separately through
the example-specific docs contracts.

## 🚦 Quick Reference

| Task | Command |
|------|---------|
| Check environment | `uv run python scripts/setup_env.py show` |
| Verify GPU setup | `uv run python scripts/verify_gpu_setup.py` |
| Run critical GPU diagnostics | `uv run python scripts/gpu_utils.py --test-critical` |
| Run all tests | `uv run pytest tests/ -v` |
| Clean caches | `./scripts/clean_cache.sh` |
| Check docs | `uv run python scripts/validate_docs.py --check-only` |
| Build docs | `uv run python scripts/build_docs.py` |
| Serve docs quickly | `./scripts/serve_docs_fast.sh` |
| Check dependencies | `uv run python scripts/analyze_dependencies.py` |
| Analyze test structure | `uv run python scripts/analyze_test_structure.py` |

## 🤝 Contributing

When modifying this directory:

1. Update the relevant repo-contract tests first.
2. Keep public documentation aligned with the retained CLI surface.
3. Avoid introducing blind in-place rewrites of tracked files.
4. Prefer shared setup flows over script-local environment assumptions.

## 📝 License

All scripts in this directory are part of the Artifex project and are licensed
under the MIT License.
