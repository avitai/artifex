"""Test file for distribution transformations with NNX compatibility."""

import distrax
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.distributions import (
    AffineTransform,
    Normal,
    TransformedDistribution,
)


@pytest.mark.parametrize("seed", [0])
def test_affine_transform(seed):
    """Test AffineTransform with NNX Rngs."""
    print("Testing AffineTransform...")

    # Create RNGs
    key = jax.random.PRNGKey(seed)
    key, sample_key = jax.random.split(key)
    rngs = nnx.Rngs(sample=sample_key)

    # Create base distribution
    base_dist = Normal(
        loc=jnp.array(0.0),
        scale=jnp.array(1.0),
        rngs=rngs,
    )

    # Create affine transform
    shift = jnp.array(5.0)
    scale = jnp.array(2.0)
    transformed_dist = AffineTransform(
        base_distribution=base_dist,
        shift=shift,
        scale=scale,
        rngs=rngs,
    )

    # Test sampling
    samples = transformed_dist.sample(sample_shape=(10,), rngs=rngs)
    assert samples.shape == (10,)

    # Test log probability
    x = jnp.array([5.0, 7.0, 9.0])
    log_prob = transformed_dist.log_prob(x)
    assert log_prob.shape == (3,)

    # Test that transformation shifted and scaled correctly
    # For a standard normal transformed by shift=5, scale=2:
    # - Mean should be around 5
    # - Std should be around 2
    mean = jnp.mean(samples)
    std = jnp.std(samples)
    assert 3.0 < mean < 7.0  # Approximate range for mean (5 ± 2)
    assert 1.0 < std < 3.0  # Approximate range for std (2 ± 1)


@pytest.mark.parametrize("seed", [1])
def test_transformed_distribution_with_bijector(seed):
    """Test TransformedDistribution with a custom bijector."""
    print("Testing TransformedDistribution...")

    # Create RNGs
    key = jax.random.PRNGKey(seed)
    key, sample_key = jax.random.split(key)
    rngs = nnx.Rngs(sample=sample_key)

    # Create base distribution
    base_dist = Normal(
        loc=jnp.array(0.0),
        scale=jnp.array(1.0),
        rngs=rngs,
    )

    # Create a Sigmoid bijector (transforms R -> (0,1))
    bijector = distrax.Sigmoid()

    # Create transformed distribution
    transformed_dist = TransformedDistribution(
        base_distribution=base_dist,
        bijector=bijector,
        rngs=rngs,
    )

    # Test sampling
    samples = transformed_dist.sample(sample_shape=(100,), rngs=rngs)
    assert samples.shape == (100,)

    # Test log probability
    x = jnp.array([0.1, 0.5, 0.9])
    log_prob = transformed_dist.log_prob(x)
    assert log_prob.shape == (3,)

    # Test that transformation worked correctly
    # For a sigmoid transformation of a standard normal:
    # - All values should be between 0 and 1
    # - Mean should be around 0.5
    assert jnp.all(samples > 0) and jnp.all(samples < 1)
    mean = jnp.mean(samples)
    assert 0.3 < mean < 0.7  # Approximate range for mean (0.5 ± 0.2)


def test_affine_transform_edge_cases():
    """Test AffineTransform edge cases and error conditions."""
    print("Testing AffineTransform edge cases...")

    # Create RNGs
    key = jax.random.PRNGKey(42)
    key, sample_key = jax.random.split(key)
    rngs = nnx.Rngs(sample=sample_key)

    # Create base distribution
    base_dist = Normal(
        loc=jnp.array(0.0),
        scale=jnp.array(1.0),
        rngs=rngs,
    )

    # Test with None shift and scale (should use defaults)
    transformed_dist = AffineTransform(
        base_distribution=base_dist,
        shift=None,
        scale=None,
        rngs=rngs,
    )

    # Test __call__ method with x=None (sampling)
    sample = transformed_dist(x=None, rngs=rngs)
    assert sample.shape == ()

    # Test __call__ method with x provided (log_prob)
    x = jnp.array(1.0)
    log_prob = transformed_dist(x=x)
    assert log_prob.shape == ()

    # Test entropy method
    entropy = transformed_dist.entropy()
    assert entropy.shape == ()

    # Test sampling with different RNG configurations
    # Test with None rngs (should use internal)
    sample_no_rngs = transformed_dist.sample(rngs=None)
    assert sample_no_rngs.shape == ()

    # Test with rngs that has "default" key instead of "sample"
    rngs_default = nnx.Rngs(default=jax.random.key(123))
    sample_default = transformed_dist.sample(rngs=rngs_default)
    assert sample_default.shape == ()


def test_transformed_distribution_edge_cases():
    """Test TransformedDistribution edge cases and error conditions."""
    print("Testing TransformedDistribution edge cases...")

    # Create RNGs
    key = jax.random.PRNGKey(99)
    key, sample_key = jax.random.split(key)
    rngs = nnx.Rngs(sample=sample_key)

    # Create base distribution
    base_dist = Normal(
        loc=jnp.array(0.0),
        scale=jnp.array(1.0),
        rngs=rngs,
    )

    # Create a simple bijector
    bijector = distrax.ScalarAffine(shift=2.0, scale=3.0)

    # Create transformed distribution
    transformed_dist = TransformedDistribution(
        base_distribution=base_dist,
        bijector=bijector,
        rngs=rngs,
    )

    # Test __call__ method with x=None (sampling)
    sample = transformed_dist(x=None, rngs=rngs)
    assert sample.shape == ()

    # Test __call__ method with x provided (log_prob)
    x = jnp.array(5.0)
    log_prob = transformed_dist(x=x)
    assert log_prob.shape == ()

    # Test entropy method
    entropy = transformed_dist.entropy()
    assert entropy.shape == ()

    # Test sampling with different RNG configurations
    # Test with None rngs (should use internal)
    sample_no_rngs = transformed_dist.sample(rngs=None)
    assert sample_no_rngs.shape == ()

    # Test with rngs that has "default" key instead of "sample"
    rngs_default = nnx.Rngs(default=jax.random.key(456))
    sample_default = transformed_dist.sample(rngs=rngs_default)
    assert sample_default.shape == ()


def test_affine_transform_with_batch_dimensions():
    """Test AffineTransform with batch dimensions."""
    print("Testing AffineTransform with batch dimensions...")

    # Create RNGs
    key = jax.random.PRNGKey(777)
    key, sample_key = jax.random.split(key)
    rngs = nnx.Rngs(sample=sample_key)

    # Create base distribution with batch dimensions
    base_dist = Normal(
        loc=jnp.array([0.0, 1.0, 2.0]),
        scale=jnp.array([1.0, 0.5, 2.0]),
        rngs=rngs,
    )

    # Create affine transform with batch dimensions
    shift = jnp.array([10.0, 20.0, 30.0])
    scale = jnp.array([2.0, 3.0, 0.5])
    transformed_dist = AffineTransform(
        base_distribution=base_dist,
        shift=shift,
        scale=scale,
        rngs=rngs,
    )

    # Test sampling
    samples = transformed_dist.sample(sample_shape=(5,), rngs=rngs)
    assert samples.shape == (5, 3)

    # Test log probability
    x = jnp.array([[10.0, 20.0, 30.0], [12.0, 23.0, 29.0]])
    log_prob = transformed_dist.log_prob(x)
    assert log_prob.shape == (2, 3)


def test_error_conditions():
    """Test error conditions for transformed distributions."""
    print("Testing error conditions...")

    # Create RNGs
    key = jax.random.PRNGKey(888)
    key, sample_key = jax.random.split(key)
    rngs = nnx.Rngs(sample=sample_key)

    # Create base distribution
    base_dist = Normal(
        loc=jnp.array(0.0),
        scale=jnp.array(1.0),
        rngs=rngs,
    )

    # Create transformed distribution
    bijector = distrax.Sigmoid()
    transformed_dist = TransformedDistribution(
        base_distribution=base_dist,
        bijector=bijector,
        rngs=rngs,
    )

    # Manually set _dist to None to test error condition
    original_dist = transformed_dist._dist
    transformed_dist._dist = None

    # Test that sampling raises ValueError when _dist is None
    try:
        transformed_dist.sample(rngs=rngs)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Distribution not initialized" in str(e)

    # Restore the distribution
    transformed_dist._dist = original_dist

    # Test that it works again
    sample = transformed_dist.sample(rngs=rngs)
    assert sample.shape == ()


def test_fallback_rng_handling():
    """Test fallback RNG handling when no proper RNG is provided."""
    print("Testing fallback RNG handling...")

    # Create base distribution
    base_dist = Normal(
        loc=jnp.array(0.0),
        scale=jnp.array(1.0),
        rngs=None,  # No RNGs provided
    )

    # Create transformed distribution
    transformed_dist = AffineTransform(
        base_distribution=base_dist,
        shift=jnp.array(5.0),
        scale=jnp.array(2.0),
        rngs=None,  # No RNGs provided
    )

    # Test sampling with no RNGs (should use fallback)
    sample = transformed_dist.sample(rngs=None)
    assert sample.shape == ()

    # Test with empty RNGs object
    empty_rngs = nnx.Rngs()
    sample_empty = transformed_dist.sample(rngs=empty_rngs)
    assert sample_empty.shape == ()


if __name__ == "__main__":
    test_affine_transform(0)
    test_transformed_distribution_with_bijector(1)
    test_affine_transform_edge_cases()
    test_transformed_distribution_edge_cases()
    test_affine_transform_with_batch_dimensions()
    test_error_conditions()
    test_fallback_rng_handling()
