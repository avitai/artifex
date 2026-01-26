# Artifex Testing Guide

This document outlines the comprehensive testing infrastructure for the Artifex library, including both CPU and GPU testing capabilities.

## 🚀 Quick Start: CUDA Development Testing

### Streamlined CUDA Testing Setup

For the fastest path to CUDA development and testing:

```bash
# 1. Install complete CUDA development environment
uv sync --extra cuda-dev

# 2. Setup fresh CUDA environment
./scripts/fresh_cuda_setup.sh
source activate.sh
```

That's it! The `cuda-dev` environment includes:

- ✅ All development tools (pytest, ruff, coverage)
- ✅ GPU dependencies (JAX with CUDA support)
- ✅ Testing frameworks (pytest with parallel execution)
- ✅ Automatic CUDA configuration

### Quick Testing Commands

```bash
# Run tests with intelligent GPU/CPU routing
./scripts/smart_test_runner.sh tests/ -v

# Run all tests (includes GPU tests if available)
uv run pytest

# Verify CUDA setup
uv run python scripts/verify_gpu_setup.py
```

### Environment Options for Testing

| Environment | Command | Best For |
|-------------|---------|----------|
| **cuda-dev** | `uv sync --extra cuda-dev` | **Recommended**: Complete CUDA development |
| **test** | `uv sync --extra test` | Testing dependencies only |
| **dev** | `uv sync --extra dev` | CPU development and testing |

---

## Comprehensive Testing Setup

The testing infrastructure leverages uv and pytest to provide a comprehensive and easy-to-use test runner system. Tests run in parallel by default for better performance, and the framework automatically detects if a GPU with CUDA is available to provide seamless testing across both CPU and GPU backends.

## Test Structure

The tests are organized into two main categories:

1. **Main Tests**: Tests that directly import from the module and test the actual code
   - Path: `tests/artifex/`

2. **Standalone Tests**: Independent tests that recreate the module structure to test core functionality
   - Path: `tests/standalone/`

## Environment Setup

### Standard Setup

To set up the testing environment:

```bash
# Clean up any existing environment
if [ -d ".venv" ]; then rm -rf .venv; fi
if [ -f "uv.lock" ]; then rm -f uv.lock; fi
uv cache clean

# Create and activate fresh virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[test]"  # Standard install with test dependencies
```

### GPU Setup

Artifex supports seamless testing on GPU backends when available. The framework automatically detects if a GPU with CUDA is available and configures the environment appropriately.

#### Quick GPU Setup

To set up the environment with GPU support:

```bash
# One-command setup
deactivate && uv cache clean && rm -rf .venv && rm -f uv.lock && uv venv && source .venv/bin/activate && uv sync

# OR use the more user-friendly wrapper script:
scripts/gpu_setup.sh
```

This will:

1. Clean the existing environment
2. Create a new virtual environment with `uv`
3. Detect if a GPU with CUDA is available
4. Install the appropriate JAX versions for your hardware
5. Configure environment variables for optimal performance

#### Installing GPU Dependencies

You can install the project with GPU support using:

```bash
# Install with GPU support
uv pip install -e ".[gpu]"
```

This installation uses the `gpu` extra in `pyproject.toml` which includes:

- `jax[cuda12]==0.6.1` with proper version pinning
- Uses the JAX CUDA releases find-links to get the appropriate CUDA builds

For a clean GPU install:

```bash
# Clean up existing environment
if [ -d ".venv" ]; then rm -rf .venv; fi
if [ -f "uv.lock" ]; then rm -f uv.lock; fi
uv cache clean 2>/dev/null || true

# Create and activate fresh virtual environment
uv venv
source .venv/bin/activate

# Install with GPU support
uv pip install -e ".[gpu]"
```

#### Activating GPU Environment

After setup, you can activate the environment with GPU-specific configurations using:

```bash
source scripts/activate_gpu_env.sh
```

This script activates the virtual environment and loads the appropriate environment variables for your hardware.

#### Testing GPU Availability

After installation, you can verify GPU availability with:

```bash
python scripts/check_gpu.py
```

For a more comprehensive GPU diagnostic test that checks for common issues:

```bash
python scripts/verify_gpu_setup.py
```

## Running Tests

### Using the GPU-Aware Test Runner

The easiest way to run tests is using the GPU-aware test runner:

```bash
# Run all tests (parallel execution by default)
scripts/gpu_test_runner.sh

# Run specific test categories
scripts/gpu_test_runner.sh --standalone  # Only standalone tests
scripts/gpu_test_runner.sh --main        # Only main module tests

# Run without parallel execution
scripts/gpu_test_runner.sh --no-parallel

# Run a specific test file
scripts/gpu_test_runner.sh -t tests/standalone/test_distributed_config.py

# Run with verbose output
scripts/gpu_test_runner.sh -v

# Run with coverage report
scripts/gpu_test_runner.sh -c

# Run tests in parallel with specific number of processes
scripts/gpu_test_runner.sh -j 4

# Run with GPU-specific tests
scripts/gpu_test_runner.sh --gpu-tests

# Force CPU-only mode
scripts/gpu_test_runner.sh --cpu-only

# Run with BlackJAX tests enabled
scripts/gpu_test_runner.sh --all

# Generate JUnit XML report
scripts/gpu_test_runner.sh --junit-xml
```

### Convenience Testing Commands

The `cuda-dev` environment provides streamlined testing commands that automatically handle CUDA setup:

```bash
# Run GPU tests with automatic CUDA configuration and benchmarks
./artifex-test-gpu

# Verify CUDA setup and diagnostics
./artifex-setup-cuda

# Standard pytest commands work with auto-configured CUDA
uv run pytest                        # All tests
uv run pytest tests/gpu_testing_demo.py  # Specific GPU tests
uv run pytest -v --tb=short         # Verbose with short traceback
```

### Automatic CUDA Setup in Tests

For automatic CUDA configuration in your test scripts:

```python
# Import at the top of test files for auto-CUDA setup
import scripts.cuda.cuda_autosetup

import jax
# JAX will now use GPU if available
```

### Parallel Testing

Tests run in parallel by default using pytest-xdist to optimize execution speed. To control parallel execution:

```bash
# Run with default parallel settings (auto-detection)
scripts/gpu_test_runner.sh       # Uses all available CPU cores

# Disable parallel execution
scripts/gpu_test_runner.sh --no-parallel

# Specify number of processes
scripts/gpu_test_runner.sh -j 4    # Use 4 processes
```

### Using Hatch Directly

You can also run tests using hatch directly:

```bash
# Run all tests
uv run hatch run test:all

# Run standalone tests
uv run hatch run test:standalone

# Run main tests
uv run hatch run test:main

# Run with coverage report
uv run hatch run test:cov

# Run quick tests (subset for pre-commit)
uv run hatch run test:quick
```

## Writing GPU-Aware Tests

The Artifex testing framework provides several utilities for writing GPU-aware tests.

### GPU Detection

```python
from tests.utils.gpu_test_utils import is_gpu_available

def test_something():
    if is_gpu_available():
        # GPU-specific code
    else:
        # CPU-specific code
```

### GPU-Specific Tests

Use any of these methods to mark tests that require a GPU:

```python
# Method 1: Using the built-in pytest.mark.gpu marker
@pytest.mark.gpu
def test_gpu_feature():
    # This test runs only when GPU is available
    ...

# Method 2: Using the requires_gpu decorator
from tests.utils.gpu_test_utils import requires_gpu

@requires_gpu
def test_another_gpu_feature():
    # This test runs only when GPU is available
    ...

# Method 3: Using the skipif marker directly
@pytest.mark.skipif(not is_gpu_available(), reason="Test requires GPU")
def test_yet_another_gpu_feature():
    # This test runs only when GPU is available
    ...
```

### GPU-Specific Test Classes

For entire test classes that require GPU:

```python
# Method 1: Using the gpu marker
@pytest.mark.gpu
class TestGPUFeatures:
    def test_one(self):
        ...

    def test_two(self):
        ...

# Method 2: Using the fixture
class TestMoreGPUFeatures:
    pytestmark = pytest.mark.usefixtures("gpu_test_fixture")

    def test_something(self):
        ...
```

### CPU-Only Tests

For tests that should only run on CPU:

```python
from tests.utils.gpu_test_utils import skip_on_gpu

@skip_on_gpu
def test_cpu_only_feature():
    # This test will be skipped when a GPU is being used
    ...
```

## BlackJAX Test Integration

### BlackJAX Test Helper

A helper script is available to make it easy to enable or disable BlackJAX tests:

```bash
# Show current status of BlackJAX tests
scripts/blackjax_test_helper.py --status

# Enable BlackJAX tests
scripts/blackjax_test_helper.py --enable

# Disable BlackJAX tests
scripts/blackjax_test_helper.py --disable

# Enable tests and run them immediately
scripts/blackjax_test_helper.py --enable --run

# Run only BlackJAX tests in parallel
scripts/blackjax_test_helper.py --enable --run --only-blackjax -p
```

The helper script provides a convenient interface to manage environment variables that control whether BlackJAX tests are run.

### Running BlackJAX Tests

BlackJAX integration tests are disabled by default as they are computationally expensive and can be slow to run. These tests can be easily enabled when needed.

To run with BlackJAX tests disabled (default):

```bash
# Default behavior - no need to set anything
pytest tests/

# Explicitly disable
SKIP_BLACKJAX_TESTS=1 pytest tests/
```

To enable BlackJAX tests (preferred method):

```bash
# Enable BlackJAX tests (preferred method)
ENABLE_BLACKJAX_TESTS=1 pytest tests/

# Alternative method
SKIP_BLACKJAX_TESTS="" pytest tests/

# Run only BlackJAX tests
pytest tests/ -m blackjax
```

To use the test runner:

```bash
# Run without BlackJAX tests (default)
scripts/gpu_test_runner.sh

# Run with BlackJAX tests
scripts/gpu_test_runner.sh --all

# Run only BlackJAX tests
scripts/gpu_test_runner.sh -t tests/artifex/generative_models/core/sampling/
```

For detailed instructions including how to make these tests enabled by default, see `tests/artifex/generative_models/core/sampling/README_BLACKJAX_TESTS.md`.

## Implementation Details

### Environment Variables

The following environment variables control the test behavior and backends:

- `JAX_PLATFORMS=cpu` - Forces JAX to use CPU backend only
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` - Controls memory allocation for GPU (when GPU is enabled)
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` - Prevents JAX from pre-allocating GPU memory
- `JAX_ENABLE_X64=0` - Disables 64-bit computations for better performance
- `TF_CPP_MIN_LOG_LEVEL=1` - Reduces TensorFlow log verbosity
- `ENABLE_BLACKJAX_TESTS=1` - Enables BlackJAX tests
- `SKIP_BLACKJAX_TESTS=1` - Disables BlackJAX tests (default)

### Backend Detection

The framework uses the following methods to detect and configure the backend:

1. Checks for NVIDIA GPU with `nvidia-smi`
2. Determines CUDA version if available
3. Configures appropriate JAX installation based on CUDA version
4. Sets optimal environment variables for JAX
5. Allows tests to detect which backend is being used at runtime

### Pre-commit Integration

Tests are integrated with pre-commit to ensure basic functionality is maintained with each commit:

1. **Configuration Test Suite**: Runs standalone tests to verify the configuration system
2. **General pytest checks**: Runs a minimal set of quick tests

To run the pre-commit test hooks manually:

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run only the test hooks
pre-commit run config-tests --all-files
pre-commit run pytest-check --all-files
```

The pre-commit tests are configured to be fast, running only a subset of the full test suite to avoid slowing down the commit process.

## Test Coverage

The tests aim to provide comprehensive coverage of the library, including:

- Configuration system (schemas, serialization, environment handling)
- Model implementations (VAE, GAN, Diffusion, Flow, etc.)
- Core utilities and abstractions
- Extension mechanism
- Sampling utilities
- Protein-specific components

Coverage reports are generated in HTML format when using the `-c` flag.

## Troubleshooting

### Common GPU Issues

1. **Segmentation Faults During Matrix Multiplication**
   - See `memory-bank/matrix_multiplication_fix.md` for detailed fixes
   - Use `scripts/verify_gpu_setup.py` to diagnose issues

2. **Out of Memory Errors**
   - Adjust `XLA_PYTHON_CLIENT_MEM_FRACTION` to a lower value
   - Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`
   - Run with `--cpu-only` for large tests

3. **JAX/CUDA Version Mismatch**
   - The setup scripts automatically detect and install the appropriate JAX version
   - If you need a specific version, see `memory-bank/matrix_multiplication_fix.md`

### Testing on Systems Without GPU

When no GPU is available, the framework automatically falls back to CPU-only mode. GPU-specific tests will be skipped with a clear message indicating why.

### Code Quality and Test Warnings

The following test-related issues have been addressed:

- **PytestCollectionWarning**: Fixed warnings about test classes with constructors
- **Visualization Warnings**: Fixed matplotlib warnings in protein visualization
- **Unused Variables**: Removed unused variables in example files
- **ResNetBlock Issues**: Fixed the ResNetBlock and BottleneckBlock implementations

All tests now run without warnings when using the standard test runner. The pre-commit hooks are configured to catch code quality issues before they are committed.

## Example Tests

- See `tests/gpu_testing_demo.py` for examples of GPU-aware tests
- See `tests/standalone/` for examples of standalone tests

## Development Workflow

When developing new features:

1. Add standalone tests first to verify core functionality
2. Add main tests once the module structure is stable
3. Run tests with coverage to ensure adequate test coverage
4. Ensure all tests pass in pre-commit before committing changes
5. For GPU-specific features, use the GPU-aware testing utilities
