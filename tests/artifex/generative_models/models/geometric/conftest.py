"""Shared fixtures for geometric model tests."""

import jax
import pytest
from flax import nnx


@pytest.fixture
def default_rng():
    """Create a default JAX random key for testing."""
    return jax.random.key(42)


@pytest.fixture
def model_rngs():
    """Create a default set of RNGs for model initialization."""
    return nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))


@pytest.fixture
def batch_size():
    """Return a default batch size for testing."""
    return 2
