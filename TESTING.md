# Artifex Testing Guide

Artifex uses the standard `uv run pytest` surface. Ad hoc GPU wrappers and backend-patching test runners are intentionally not part of the supported workflow.

## Setup

### Development environment

```bash
# Auto-detect an appropriate backend policy
./setup.sh
source ./activate.sh

# Or choose one explicitly
./setup.sh --backend cpu
./setup.sh --backend cuda12
./setup.sh --backend metal
```

`setup.sh` now writes a generated `.artifex.env` file. If you also keep a `.env` or `.env.local`, those files are treated as user-owned override layers and are loaded after `.artifex.env`.
Re-sourcing `activate.sh` refreshes the managed backend variables before those user-owned overrides are applied.

### Direct uv sync

```bash
# CPU-focused development
uv sync --extra dev --extra test

# Linux CUDA development
uv sync --extra cuda-dev

# Apple Silicon Metal development
uv sync --extra dev --extra test --extra metal
```

## Backend policy

- Artifex leaves `JAX_PLATFORMS` unset by default for GPU-capable environments.
- JAX then chooses GPU when the installed runtime supports it, otherwise CPU.
- Use `JAX_PLATFORMS=cpu` only when you explicitly want to force CPU execution.
- Artifex does not rely on `/usr/local/cuda` or custom `LD_LIBRARY_PATH` injection for the JAX pip-managed CUDA runtime.
- `./setup.sh --force-clean` removes `.venv`, `.artifex.env`, and repo-local test artifacts without touching `.env` or `.env.local`.

Verify the active runtime with:

```bash
uv run python scripts/verify_gpu_setup.py
uv run python scripts/verify_gpu_setup.py --require-gpu
uv run python scripts/verify_gpu_setup.py --json
```

## Test backend and devices

The test suite chooses its own JAX backend before JAX is imported, so a local run matches CI:

- Tests run on the CPU with eight emulated devices, so multi-device code paths run on any machine. An exported
  `JAX_PLATFORMS` does not change this.
- `ARTIFEX_TEST_JAX_PLATFORMS=cuda uv run pytest ...` runs the tests on the GPU, without CPU emulation. Asking for
  CUDA when JAX's CUDA plugin is not installed fails instead of falling back to the CPU.
- `ARTIFEX_TEST_DEVICE_COUNT=N` sets the number of emulated CPU devices; `0` turns emulation off.

## Running tests

```bash
# Run the standard suite
uv run pytest

# Run a focused file
uv run pytest tests/path/to/test_file.py -xvs --no-cov

# Run a single test
uv run pytest tests/path/to/test_file.py::TestClass::test_method -xvs --no-cov

# Run the GPU-only tests on a GPU
ARTIFEX_TEST_JAX_PLATFORMS=cuda uv run pytest -m accelerator --no-cov

# Run BlackJAX tests only
uv run pytest -m blackjax --no-cov
```

## Coverage

```bash
# HTML coverage report
uv run pytest --cov=src/artifex --cov-report=html

# JSON coverage report
uv run pytest \
  --cov=src/ \
  --cov-report=json:temp/coverage.json \
  --cov-report=term-missing
```

## Markers

- `accelerator(kind=None)`, from the substrax pytest plugin: tests that need an accelerator backend, of `kind` when given; they skip otherwise
- `devices(count, kind=None)`, from the same plugin: tests that need at least `count` visible devices
- `x64`, from the same plugin: runs one test with 64-bit types enabled
- `blackjax`: tests that exercise BlackJAX integration
- `slow`, `integration`, `e2e`, `benchmark`, `contract`: standard suite categorization markers

## Notes

- BlackJAX is a first-class dependency. Its tests are part of the normal pytest contract.
- The supported suite lives under `tests/` and should import live Artifex owners; shadow local-replica suites are not part of the supported workflow.
- `uv run pytest` runs on the CPU on a GPU machine too; set `ARTIFEX_TEST_JAX_PLATFORMS=cuda` to test on the GPU.
- If `uv run python scripts/verify_gpu_setup.py --require-gpu` fails on Linux, the usual problem is a missing or incompatible NVIDIA driver, not a missing system CUDA toolkit.
