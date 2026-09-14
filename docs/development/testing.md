# Testing Guide

Artifex uses the standard `uv run pytest` surface. Treat the root `TESTING.md` file as the canonical command and policy reference, and keep contributor-facing docs aligned with that contract.

## Running Tests

### Standard Commands

```bash
# Run the standard suite
uv run pytest

# Run a focused file
uv run pytest tests/path/to/test_file.py -xvs --no-cov

# Run a single test
uv run pytest tests/path/to/test_file.py::TestClass::test_method -xvs --no-cov

# Run GPU-only tests on a GPU
ARTIFEX_TEST_JAX_PLATFORMS=cuda uv run pytest -m accelerator --no-cov

# Run BlackJAX tests
uv run pytest -m blackjax --no-cov
```

### Coverage Commands

```bash
# HTML coverage report
uv run pytest --cov=src/artifex --cov-report=html

# JSON coverage report
uv run pytest \
  --cov=src/ \
  --cov-report=json:temp/coverage.json \
  --cov-report=term-missing
```

The repo-wide pytest fail-under is `80%`, matching the target recorded in `tool.artifex.repo_standards.coverage`.

## Current Test Layout

Tests should import live Artifex owners instead of recreating local shadow configs or models. The maintained suite currently lives under `tests/` as:

- `tests/artifex/` for package, integration, visualization, benchmark, and repo-contract coverage
- `tests/artifex/repo_contracts/` for docs and public-surface contract checks
- `tests/unit/` for narrower low-level unit coverage where that layout already exists

Do not add local replica suites; shadow tests disconnected from live imports are not part of the supported workflow.

## Markers And Fixtures

### Markers

- `@pytest.mark.accelerator(kind="gpu")`: requires a GPU backend and skips otherwise. Tests run on the CPU unless `ARTIFEX_TEST_JAX_PLATFORMS` names an accelerator
- `@pytest.mark.devices(count)`: requires at least `count` visible devices
- `@pytest.mark.blackjax`: exercises BlackJAX integration
- `@pytest.mark.integration`, `@pytest.mark.e2e`, `@pytest.mark.contract`, `@pytest.mark.benchmark`, `@pytest.mark.slow`: standard suite categorization markers

### Shared Fixtures

- `test_device`: provides a preferred live device for device-aware tests
- `base_rngs`, `standard_shapes`, and related shared fixtures live under `tests/artifex/fixtures/base.py`

Example GPU-aware test:

```python
import pytest


@pytest.mark.accelerator(kind="gpu")
def test_gpu_training_path(test_device):
    """Exercise a GPU-only path on the active runtime device."""
    assert test_device.platform in {"gpu", "cuda"}
```

## Writing Tests

- Import live Artifex owners or helpers instead of recreating local config or model replicas.
- Use explicit pytest markers for expensive or backend-specific coverage.
- Keep focused file or test-node commands in docs and review notes so failures are easy to reproduce.
- Prefer shared fixtures over ad hoc setup duplication when the fixture reflects a real supported runtime contract.

## Pre-Commit Check

```bash
uv run pre-commit run --all-files
uv run pytest -x
```

## See Also

- the root `TESTING.md` file
- [Design Philosophy](philosophy.md)
