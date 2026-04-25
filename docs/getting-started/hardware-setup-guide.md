# Hardware Setup Guide

This guide describes the supported JAX backend model for Artifex.

## Supported backends

### CPU

- Default package install: `pip install avitai-artifex`
- Default repo sync: `uv sync --extra dev --extra test`
- JAX uses CPU when no accelerator runtime is available

### NVIDIA CUDA 12 on Linux

- Package install: `pip install "avitai-artifex[cuda12]"`
- Repo sync: `uv sync --extra cuda-dev`
- Recommended repo bootstrap: `./setup.sh --backend cuda12`

Artifex relies on JAX's pip-managed CUDA runtime for this path. The required host dependency is the NVIDIA driver. Artifex does not require `/usr/local/cuda`, manual `LD_LIBRARY_PATH` edits, or a system CUDA toolkit for the normal pip-managed JAX flow.

### Apple Silicon Metal

- Package install: `pip install "avitai-artifex[metal]"`
- Repo sync: `uv sync --extra dev --extra test --extra metal`
- Recommended repo bootstrap: `./setup.sh --backend metal`

## Backend-selection policy

- GPU-capable environments leave `JAX_PLATFORMS` unset by default.
- JAX then selects GPU when the installed runtime supports it and falls back to CPU otherwise.
- Use `JAX_PLATFORMS=cpu` only when you explicitly want a CPU-only run.
- Tests use the same rule: GPU-marked tests are skipped only when JAX cannot actually see a GPU backend.

## Developer workflow

```bash
git clone https://github.com/avitai/artifex.git
cd artifex

# Auto-detect CPU, CUDA 12, or Metal
./setup.sh
source ./activate.sh

# Verify what JAX can actually use
uv run python scripts/verify_gpu_setup.py
```

The generated backend settings live in `.artifex.env`. Optional `.env` and `.env.local` files are user-owned override layers and are loaded after `.artifex.env`.
Re-sourcing `activate.sh` refreshes the managed backend state from `.artifex.env` before applying those user-owned overrides.
On the `cuda12` path, `activate.sh` also strips inherited CUDA toolkit library directories from `LD_LIBRARY_PATH` so JAX keeps using the pip-managed runtime bundled by the environment.

## Explicit backend workflows

```bash
# CPU
./setup.sh --backend cpu

# Linux CUDA 12
./setup.sh --backend cuda12

# Apple Silicon Metal
./setup.sh --backend metal
```

## Verification

```bash
# Human-readable report
uv run python scripts/verify_gpu_setup.py

# Fail unless a GPU backend is available
uv run python scripts/verify_gpu_setup.py --require-gpu

# Machine-readable output
uv run python scripts/verify_gpu_setup.py --json
```

## Troubleshooting

### JAX stays on CPU on a Linux GPU machine

This usually means the NVIDIA driver is missing or incompatible with the JAX CUDA runtime. Start with:

```bash
uv run python scripts/verify_gpu_setup.py --json
```

Do not try to fix this by pointing Artifex at `/usr/local/cuda` or by hand-editing `LD_LIBRARY_PATH`.

### You want to force CPU for a single command

```bash
JAX_PLATFORMS=cpu uv run pytest -m "not gpu"
```

### You want to rebuild the repo environment

```bash
./setup.sh --recreate
source ./activate.sh
```

If you also want to clear repo-local test and coverage artifacts without touching user-owned `.env` files:

```bash
./setup.sh --force-clean
source ./activate.sh
```
