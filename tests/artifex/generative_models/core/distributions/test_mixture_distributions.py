"""Tests for mixture distributions."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.distributions import (
    Categorical,
    Mixture,
    MixtureOfGaussians,
    Normal,
)


class TestMixtureOfGaussians:
    """Test suite for MixtureOfGaussians distribution."""

    @pytest.fixture
    def mog_params(self):
        """Fixture for mixture of Gaussians parameters."""
        # 2 components, 3-dimensional Gaussians
        locs = jnp.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        scales = jnp.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
        weights = jnp.array([0.3, 0.7])
        return locs, scales, weights

    @pytest.fixture
    def mog_dist(self, mog_params):
        """Fixture for mixture of Gaussians distribution."""
        locs, scales, weights = mog_params
        return MixtureOfGaussians(locs=locs, scales=scales, weights=weights)

    def test_initialization(self, mog_params):
        """Test initialization of MixtureOfGaussians."""
        locs, scales, weights = mog_params
        mog = MixtureOfGaussians(locs=locs, scales=scales, weights=weights)

        assert mog.num_components == 2
        assert jnp.allclose(mog.locs, locs)
        assert jnp.allclose(mog.scales, scales)
        # Check that weights are normalized
        assert jnp.allclose(jnp.sum(mog.weights), 1.0)

    def test_uniform_weights(self, mog_params):
        """Test initialization with uniform weights."""
        locs, scales, _ = mog_params
        mog = MixtureOfGaussians(locs=locs, scales=scales)

        # Check that weights are uniform and normalized
        assert jnp.allclose(mog.weights, jnp.array([0.5, 0.5]))
        assert jnp.allclose(jnp.sum(mog.weights), 1.0)

    def test_sample(self, mog_dist):
        """Test sampling from MixtureOfGaussians."""
        key = jax.random.PRNGKey(0)
        n_samples = 100

        # Create nnx Rngs object
        rngs = nnx.Rngs(sample=key)

        # Get samples
        samples = mog_dist.sample(sample_shape=(n_samples,), rngs=rngs)

        # Check shape
        assert samples.shape == (n_samples, 3)

    def test_log_prob(self, mog_dist, mog_params):
        """Test log probability calculation for MixtureOfGaussians."""
        locs, _, _ = mog_params

        # Test points - first component mean and second component mean
        x = jnp.array([0.0, 0.0, 0.0])
        y = jnp.array([5.0, 5.0, 5.0])

        # Calculate log probabilities
        log_prob_x = mog_dist.log_prob(x)
        log_prob_y = mog_dist.log_prob(y)

        # Convert to probabilities
        prob_x = jnp.exp(log_prob_x)
        prob_y = jnp.exp(log_prob_y)

        # Both should be positive
        assert prob_x > 0
        assert prob_y > 0

        # Check weights influence (second component has higher weight)
        assert prob_y > prob_x

    def test_call(self, mog_dist):
        """Test __call__ method for MixtureOfGaussians."""
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(sample=key)

        # Test with x provided (should return log_prob)
        x = jnp.array([0.0, 0.0, 0.0])
        result_x = mog_dist(x)
        assert jnp.shape(result_x) == ()  # scalar log probability

        # Test with x=None (should return sample)
        result_sample = mog_dist(x=None, rngs=rngs)
        assert result_sample.shape == (3,)  # Single sample

    def test_with_nnx_rng(self, mog_dist):
        """Test using nnx.Rngs for sampling."""
        rngs = nnx.Rngs(params=0)

        # Sample with nnx.Rngs
        samples = mog_dist.sample(sample_shape=(10,), rngs=rngs)

        # Check shape
        assert samples.shape == (10, 3)


class TestMixture:
    """Test suite for general Mixture distribution."""

    @pytest.fixture
    def components(self):
        """Fixture for mixture components."""
        # Create two Normal distributions
        normal1 = Normal(loc=jnp.array(0.0), scale=jnp.array(1.0))
        normal2 = Normal(loc=jnp.array(5.0), scale=jnp.array(0.5))
        return [normal1, normal2]

    @pytest.fixture
    def weights(self):
        """Fixture for mixture weights."""
        return jnp.array([0.3, 0.7])

    @pytest.fixture
    def mixture_dist(self, components, weights):
        """Fixture for Mixture distribution."""
        return Mixture(components=components, weights=weights)

    def test_initialization(self, components, weights):
        """Test initialization of Mixture."""
        mixture = Mixture(components=components, weights=weights)

        assert mixture.num_components == 2
        assert len(mixture.components) == 2
        # Check that weights are normalized
        assert jnp.allclose(jnp.sum(mixture.weights), 1.0)

    def test_uniform_weights(self, components):
        """Test initialization with uniform weights."""
        mixture = Mixture(components=components)

        # Check that weights are uniform and normalized
        assert jnp.allclose(mixture.weights, jnp.array([0.5, 0.5]))
        assert jnp.allclose(jnp.sum(mixture.weights), 1.0)

    def test_sample(self, mixture_dist):
        """Test sampling from Mixture."""
        key = jax.random.PRNGKey(0)
        n_samples = 100

        # Create nnx Rngs object
        rngs = nnx.Rngs(sample=key)

        # Get samples
        samples = mixture_dist.sample(sample_shape=(n_samples,), rngs=rngs)

        # Check shape
        assert samples.shape == (n_samples,)

    def test_log_prob(self, mixture_dist):
        """Test log probability calculation for Mixture."""
        # Test points - means of the two normal components
        x = jnp.array(0.0)
        y = jnp.array(5.0)

        # Calculate log probabilities
        log_prob_x = mixture_dist.log_prob(x)
        log_prob_y = mixture_dist.log_prob(y)

        # Convert to probabilities
        prob_x = jnp.exp(log_prob_x)
        prob_y = jnp.exp(log_prob_y)

        # Both should be positive
        assert prob_x > 0
        assert prob_y > 0

        # Check that weights influence probabilities correctly
        assert prob_y > prob_x

    def test_call(self, mixture_dist):
        """Test __call__ method for Mixture."""
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(sample=key)

        # Test with x provided (should return log_prob)
        x = jnp.array(0.0)
        result_x = mixture_dist(x)
        assert jnp.shape(result_x) == ()  # scalar log probability

        # Test with x=None (should return sample)
        result_sample = mixture_dist(x=None, rngs=rngs)
        assert result_sample.shape == ()  # Single sample (scalar)

    def test_with_categorical(self):
        """Test Mixture with a categorical component."""
        # Create a Normal and a Categorical distribution
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(sample=key)

        normal = Normal(loc=jnp.array(0.0), scale=jnp.array(1.0))
        categorical = Categorical(probs=jnp.array([0.2, 0.3, 0.5]))

        # Create mixture with equal weights
        mixture = Mixture(components=[normal, categorical])

        # Try sampling
        samples = mixture.sample(sample_shape=(100,), rngs=rngs)

        # Check that we got samples
        assert samples.shape == (100,)
