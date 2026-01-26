"""Simple test for mixture distributions."""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.distributions import (
    Bernoulli,
    Mixture,
    MixtureOfGaussians,
)


def test_mixture_of_gaussians():
    """Test MixtureOfGaussians."""
    print("Testing MixtureOfGaussians...")

    # Create RNGs
    key = jax.random.PRNGKey(0)
    key, sample_key = jax.random.split(key)
    rngs = nnx.Rngs(sample=sample_key)

    # Initialize mixture with 2 components
    dist = MixtureOfGaussians(
        locs=jnp.array([[0.0, 0.0], [5.0, 5.0]]),  # [components, event_size]
        scales=jnp.array([[1.0, 1.0], [0.5, 0.5]]),
        weights=jnp.array([0.3, 0.7]),
        rngs=rngs,
    )

    # Test sampling
    try:
        samples = dist.sample(sample_shape=(10,), rngs=rngs)
        print(f"Sampling success! Shape: {samples.shape}")
        print(f"Sample values (first 3):\n{samples[:3]}")
    except Exception as e:
        print(f"Sampling failed: {e}")

    # Test log probability
    try:
        test_points = jnp.array([[0.0, 0.0], [5.0, 5.0]])
        log_prob = dist.log_prob(test_points)
        print(f"Log prob success! Shape: {log_prob.shape}")
        print(f"Log prob values: {log_prob}")
    except Exception as e:
        print(f"Log prob failed: {e}")


def test_general_mixture():
    """Test general Mixture with Bernoulli components."""
    print("\nTesting general Mixture...")

    # Create RNGs
    key = jax.random.PRNGKey(1)
    key, sample_key = jax.random.split(key)
    rngs = nnx.Rngs(sample=sample_key)

    # Create component distributions
    components = [
        Bernoulli(probs=jnp.array(0.2), rngs=rngs),
        Bernoulli(probs=jnp.array(0.8), rngs=rngs),
    ]

    # Initialize mixture
    dist = Mixture(
        components=components,
        weights=jnp.array([0.4, 0.6]),
        rngs=rngs,
    )

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


if __name__ == "__main__":
    test_mixture_of_gaussians()
    test_general_mixture()
