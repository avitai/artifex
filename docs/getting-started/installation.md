# Installation Guide

Artifex supports CPU-first installation by default, with explicit extras for accelerator backends.

## Package users

### CPU

```bash
pip install artifex
```

### NVIDIA CUDA 12 on Linux

```bash
pip install "artifex[cuda12]"
```

This uses JAX's official pip-managed CUDA runtime. Artifex does not require `/usr/local/cuda` or manual `LD_LIBRARY_PATH` edits for this install path.

### Apple Silicon Metal

```bash
pip install "artifex[metal]"
```

## Repository developers

Artifex uses `uv` for all repo-maintained setup, sync, and test workflows.

### Recommended setup

```bash
git clone https://github.com/avitai/artifex.git
cd artifex

# Auto-detect CPU, CUDA 12, or Metal policy
./setup.sh
source ./activate.sh
```

`setup.sh` writes a generated `.artifex.env` file. Optional `.env` and `.env.local` files are treated as user-owned override layers and are loaded after `.artifex.env`.
Re-sourcing `activate.sh` refreshes the managed backend variables from `.artifex.env` before applying `.env` and `.env.local`.
When the resolved backend is `cuda12`, `activate.sh` also removes inherited CUDA toolkit library directories from `LD_LIBRARY_PATH` so JAX uses its pip-managed runtime instead of stale system CUDA libraries.

### Explicit backend selection

```bash
./setup.sh --backend cpu
./setup.sh --backend cuda12
./setup.sh --backend metal
source ./activate.sh
```

### Recreate vs force-clean

```bash
# Rebuild the virtual environment and regenerate the managed backend file
./setup.sh --recreate

# Also remove repo-local test/coverage artifacts, but keep user-owned .env files
./setup.sh --force-clean
```

### Direct uv sync

```bash
# CPU development
uv sync --extra dev --extra test

# Linux CUDA development
uv sync --extra cuda-dev

# Apple Silicon development
uv sync --extra dev --extra test --extra metal
```

## Backend verification

```bash
uv run python scripts/verify_gpu_setup.py

# Require a working GPU backend
uv run python scripts/verify_gpu_setup.py --require-gpu
```

The verification script checks what JAX can actually use, not just what hardware exists.

## Backend behavior

- CPU installs work everywhere JAX CPU wheels work.
- Linux CUDA installs rely on the NVIDIA driver plus the JAX-managed CUDA runtime from `jax[cuda12]`.
- Metal installs are for Apple Silicon development.
- Artifex leaves `JAX_PLATFORMS` unset by default on GPU-capable environments so JAX can select GPU when available and CPU otherwise.
- If you want a CPU-only run on a machine with accelerators, set `JAX_PLATFORMS=cpu` for that command.

## Troubleshooting

### JAX falls back to CPU on Linux

Run:

```bash
uv run python scripts/verify_gpu_setup.py --json
```

If the default backend is still CPU, the usual problem is the NVIDIA driver or a driver/runtime mismatch. It is not fixed by editing `LD_LIBRARY_PATH` or installing a system CUDA toolkit.

### You want to rebuild the repo environment

```bash
./setup.sh --recreate
source ./activate.sh
```
