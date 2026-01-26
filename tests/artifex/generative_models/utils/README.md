# Test Utilities for Generative Models

This directory contains utility functions and helpers for testing the generative models package.

## Test Fixtures

To improve test isolation and eliminate redefined-outer-name warnings, we provide utility functions for creating standard test fixtures.

### Available Fixtures

The `test_fixtures.py` module provides the following utilities:

- `get_rng_key(seed=42)`: Returns a JAX random key with a consistent seed
- `get_rng_keys(seed=42)`: Returns a dictionary of JAX random keys for different purposes
- `get_standard_dims()`: Returns a dictionary of standard dimensions for tests
- `get_image_sample()`: Returns a sample image tensor with configurable dimensions
- `get_sequence_sample()`: Returns a sample sequence tensor with configurable dimensions

### Usage Examples

#### Creating Random Keys

Instead of redefining RNG keys in each test, use the utility function:

```python
from tests.artifex.generative_models.utils import get_rng_key

def test_my_function():
    # Use a consistent seed but different for each test
    key = get_rng_key(0)  # For the first test
    key2 = get_rng_key(1)  # For the second test

    # Test with the keys...
```

#### Creating Sample Data

For tests that need sample data, use the sample generation functions:

```python
from tests.artifex.generative_models.utils import get_sequence_sample

def test_transformer():
    # Get a sample sequence with standard dimensions
    sequence = get_sequence_sample()

    # Or with custom dimensions
    batch_size = 8
    seq_length = 32
    embed_dim = 64
    custom_sequence = get_sequence_sample(
        batch_size=batch_size,
        seq_length=seq_length,
        embed_dim=embed_dim
    )
```

## Migrating Existing Tests

When updating existing tests to use these utilities:

1. Remove local fixture definitions for RNG keys, dimensions, etc.
2. Use the utility functions to create the needed values directly in the test
3. If a different seed is needed for different tests, use a unique number for each test
4. For derived fixtures (that depend on the common ones), update the parameters to use the utility values

## Benefits

- Eliminates redefined-outer-name warnings
- Improves test isolation by avoiding shared state between tests
- Makes tests more maintainable with consistent dimensions and patterns
- Simplifies test authoring with reusable utility functions
