"""Simple test for distribution classes."""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.distributions import (
    Bernoulli,
    Categorical,
)


def test_bernoulli():
    """Test Bernoulli distribution."""
    print("Testing Bernoulli...")

    # Create RNGs
    key = jax.random.PRNGKey(0)
    key, sample_key = jax.random.split(key)
    rngs = nnx.Rngs(sample=sample_key)

    # Initialize with probability
    dist = Bernoulli(probs=jnp.array(0.7), rngs=rngs)

    # Test sampling
    try:
        samples = dist.sample(sample_shape=(10,), rngs=rngs)
        print(f"Sampling success! Shape: {samples.shape}")
        print(f"Sample values: {samples}")
    except Exception as e:
        print(f"Sampling failed: {e}")

    # Test log probability
    try:
        log_prob = dist.log_prob(jnp.array([0, 1]))
        print(f"Log prob success! Shape: {log_prob.shape}")
        print(f"Log prob values: {log_prob}")
    except Exception as e:
        print(f"Log prob failed: {e}")


def test_categorical():
    """Test Categorical distribution."""
    print("\nTesting Categorical...")

    # Create RNGs
    key = jax.random.PRNGKey(1)
    key, sample_key = jax.random.split(key)
    rngs = nnx.Rngs(sample=sample_key)

    # Initialize with probability
    dist = Categorical(probs=jnp.array([0.2, 0.3, 0.5]), rngs=rngs)

    # Test sampling
    try:
        samples = dist.sample(sample_shape=(10,), rngs=rngs)
        print(f"Sampling success! Shape: {samples.shape}")
        print(f"Sample values: {samples}")
    except Exception as e:
        print(f"Sampling failed: {e}")

    # Test log probability
    try:
        log_prob = dist.log_prob(jnp.array([0, 1, 2]))
        print(f"Log prob success! Shape: {log_prob.shape}")
        print(f"Log prob values: {log_prob}")
    except Exception as e:
        print(f"Log prob failed: {e}")


if __name__ == "__main__":
    test_bernoulli()
    test_categorical()
