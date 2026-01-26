"""Simple test to verify Flax NNX functionality."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


class SimpleModule(nnx.Module):
    """A simple module for testing."""

    def __init__(self, features: int, *, rngs: nnx.Rngs):
        self.dense = nnx.Linear(in_features=features, out_features=features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dense(x)


@pytest.fixture
def rng_keys():
    """Fixture providing deterministic RNG keys."""
    return {
        "params": jax.random.key(0),
        "dropout": jax.random.key(1),
    }


def test_simple_module(rng_keys):
    """Test a simple NNX module."""
    rngs = nnx.Rngs(params=rng_keys["params"])

    # Initialize the module
    features = 4
    module = SimpleModule(features=features, rngs=rngs)

    # Test forward pass
    batch_size = 2
    x = jnp.ones((batch_size, features))
    y = module(x)

    # Check shape
    assert y.shape == (batch_size, features)
