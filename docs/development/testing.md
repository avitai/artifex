# Testing Guide

Artifex follows test-driven development practices with comprehensive test coverage.

## Running Tests

### Basic Test Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/path/to/test_file.py -xvs

# Run single test
uv run pytest tests/path/to/test_file.py::TestClass::test_method -xvs

# Run with coverage
uv run pytest --cov=src/artifex --cov-report=html

# Run all tests with JSON report
uv run pytest -vv --json-report --json-report-file=temp/test-results.json \
    --json-report-indent=2 --json-report-verbosity=2 \
    --cov=src/ --cov-report=json:temp/coverage.json --cov-report=term-missing
```

### GPU-Aware Testing

```bash
# Use smart test runner for automatic GPU/CPU routing
./scripts/smart_test_runner.sh tests/ -v

# Run GPU-specific tests only (requires CUDA)
uv run pytest -m gpu

# Run expensive BlackJAX tests (optional)
uv run pytest -m blackjax
```

## Test Organization

### Directory Structure

Tests mirror the source structure:

```
tests/
├── standalone/          # Isolated component tests
│   └── artifex/
│       └── generative_models/
│           ├── models/
│           │   ├── vae/
│           │   ├── gan/
│           │   └── ...
│           └── core/
└── artifex/            # Integrated system tests
    └── generative_models/
        └── ...
```

### Test Categories

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **GPU tests**: Marked with `@pytest.mark.gpu`, automatically skipped on CPU-only systems
- **BlackJAX tests**: Marked with `@pytest.mark.blackjax`, expensive MCMC tests

## Writing Tests

### Basic Test Structure

```python
import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import VAEConfig, EncoderConfig
from artifex.generative_models.factory import create_model


class TestVAE:
    """Test suite for VAE model."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        rngs = nnx.Rngs(params=42, dropout=42, sample=42)

        encoder_config = EncoderConfig(
            name="encoder",
            input_shape=(28, 28, 1),
            latent_dim=32,
            hidden_dims=(64, 32),
            activation="relu",
        )

        config = VAEConfig(
            name="test_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            kl_weight=1.0,
        )

        model = create_model(config, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (4, 28, 28, 1))

        outputs = model(x, rngs=rngs)

        assert outputs.reconstruction.shape == x.shape
        assert outputs.z.shape == (4, 32)
```

### GPU Test Marking

```python
import pytest

@pytest.mark.gpu
def test_gpu_operation():
    """Test that requires GPU."""
    # This test is skipped on CPU-only systems
    ...
```

### Using Device Fixtures

```python
def test_with_device(device):
    """Test with automatic GPU/CPU selection."""
    # 'device' fixture automatically selects GPU if available
    ...
```

## Test Best Practices

### Do's

1. **Test behavior, not implementation**: Focus on what the code does, not how
2. **Use specific assertions**: Compare specific properties, not entire objects
3. **Provide meaningful test names**: Describe what's being tested
4. **Clean up GPU memory**: Use `device_manager.cleanup()` when needed
5. **Use fixtures for common setup**: Avoid duplicating initialization code

### Don'ts

1. **Don't compare complex objects with `==`**: Use specific property comparisons
2. **Don't modify tests to accommodate bugs**: Fix the implementation instead
3. **Don't skip tests without documentation**: Mark with reason
4. **Don't use random seeds without purpose**: Be explicit about reproducibility

### Assertion Patterns

```python
# CORRECT: Compare specific properties
assert model.config.name == "expected_name"
assert outputs.reconstruction.shape == expected_shape
assert jnp.allclose(outputs.loss, expected_loss, rtol=1e-5)

# WRONG: Compare entire objects
assert model.config == expected_config  # May fail unexpectedly
assert outputs == expected_outputs      # Too broad
```

## Code Coverage

### Running Coverage Reports

```bash
# Generate HTML coverage report
uv run pytest --cov=src/artifex --cov-report=html

# View report
open htmlcov/index.html
```

### Coverage Requirements

- Minimum 80% coverage for new code
- Critical paths should have 100% coverage
- Document any intentionally uncovered code

## Pre-Commit Testing

Before committing, ensure tests pass:

```bash
# Run all quality checks
uv run pre-commit run --all-files

# Quick test run
uv run pytest tests/ -x  # Stop on first failure
```

## Debugging Tests

### Verbose Output

```bash
# Maximum verbosity
uv run pytest tests/path/to/test.py -xvs --tb=long

# Show local variables on failure
uv run pytest --tb=long --showlocals
```

### Running Single Tests

```bash
# Run specific test method
uv run pytest tests/path/to/test.py::TestClass::test_method -xvs

# Run tests matching pattern
uv run pytest -k "vae and forward" -xvs
```

## See Also

- [Core Concepts](../getting-started/core-concepts.md) - Architecture overview
- [Design Philosophy](philosophy.md) - Testing philosophy
