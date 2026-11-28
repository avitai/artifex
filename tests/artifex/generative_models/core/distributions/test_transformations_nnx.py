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
    )

    # Create affine transform
    shift = jnp.array(5.0)
    scale = jnp.array(2.0)
    transformed_dist = AffineTransform(
        base_distribution=base_dist,
        shift=shift,
        scale=scale,
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
    )

    # Create a Sigmoid bijector (transforms R -> (0,1))
    bijector = distrax.Sigmoid()

    # Create transformed distribution
    transformed_dist = TransformedDistribution(
        base_distribution=base_dist,
        bijector=bijector,
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
    )

    # Test with None shift and scale (should use defaults)
    transformed_dist = AffineTransform(
        base_distribution=base_dist,
        shift=None,
        scale=None,
    )

    sample = transformed_dist.sample(rngs=rngs)
    assert sample.shape == ()

    x = jnp.array(1.0)
    log_prob = transformed_dist.log_prob(x)
    assert log_prob.shape == ()

    # Test entropy method
    entropy = transformed_dist.entropy()
    assert entropy.shape == ()

    # A distribution holds no key: a missing nnx.Rngs is refused.
    with pytest.raises(TypeError, match="rngs must be an nnx.Rngs"):
        transformed_dist.sample(rngs=None)

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
    )

    # Create a simple bijector
    bijector = distrax.ScalarAffine(shift=2.0, scale=3.0)

    # Create transformed distribution
    transformed_dist = TransformedDistribution(
        base_distribution=base_dist,
        bijector=bijector,
    )

    sample = transformed_dist.sample(rngs=rngs)
    assert sample.shape == ()

    x = jnp.array(5.0)
    log_prob = transformed_dist.log_prob(x)
    assert log_prob.shape == ()

    # Test entropy method
    entropy = transformed_dist.entropy()
    assert entropy.shape == ()

    # A distribution holds no key: a missing nnx.Rngs is refused.
    with pytest.raises(TypeError, match="rngs must be an nnx.Rngs"):
        transformed_dist.sample(rngs=None)

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
    )

    # Create affine transform with batch dimensions
    shift = jnp.array([10.0, 20.0, 30.0])
    scale = jnp.array([2.0, 3.0, 0.5])
    transformed_dist = AffineTransform(
        base_distribution=base_dist,
        shift=shift,
        scale=scale,
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
    )

    # Create transformed distribution
    bijector = distrax.Sigmoid()
    transformed_dist = TransformedDistribution(
        base_distribution=base_dist,
        bijector=bijector,
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
    """A missing nnx.Rngs, and one with neither the stream nor a default, are refused."""
    print("Testing fallback RNG handling...")

    # Create base distribution
    base_dist = Normal(
        loc=jnp.array(0.0),
        scale=jnp.array(1.0),  # No RNGs provided
    )

    # Create transformed distribution
    transformed_dist = AffineTransform(
        base_distribution=base_dist,
        shift=jnp.array(5.0),
        scale=jnp.array(2.0),  # No RNGs provided
    )

    with pytest.raises(TypeError, match="rngs must be an nnx.Rngs"):
        transformed_dist.sample(rngs=None)

    with pytest.raises(KeyError, match="No RngStream named 'sample'"):
        transformed_dist.sample(rngs=nnx.Rngs(params=jax.random.key(0)))
