"""Tests for discrete distributions: Bernoulli and Categorical."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.distributions import (
    Bernoulli,
    Categorical,
    OneHotCategorical,
)


@pytest.fixture
def rngs():
    """Fixture for random number generators."""
    return nnx.Rngs(0)


class TestBernoulli:
    """Test cases for the Bernoulli distribution."""

    @pytest.fixture
    def bernoulli_probs(self):
        """Fixture for Bernoulli with probability parameter."""
        return Bernoulli(probs=jnp.array(0.7))

    @pytest.fixture
    def bernoulli_logits(self):
        """Fixture for Bernoulli with logits parameter."""
        return Bernoulli(logits=jnp.array(1.0))

    def test_initialization(self, bernoulli_probs, bernoulli_logits):
        """Test initialization of Bernoulli distribution."""
        # Test initialization with probs
        assert jnp.allclose(bernoulli_probs.probs, 0.7)

        # Test initialization with logits
        assert jnp.allclose(bernoulli_logits.logits, 1.0)

        # Test initialization with default parameters
        default_bernoulli = Bernoulli()
        assert isinstance(default_bernoulli.logits, nnx.Param)
        assert jnp.allclose(default_bernoulli.logits.value, 0.0)

    def test_invalid_initialization(self):
        """Test invalid initialization."""
        with pytest.raises(ValueError):
            Bernoulli(probs=jnp.array(0.7), logits=jnp.array(1.0))

    def test_sample(self, rngs, bernoulli_probs):
        """Test sampling from Bernoulli distribution."""
        samples = bernoulli_probs.sample(sample_shape=(1000,), rngs=rngs)
        assert samples.shape == (1000,)
        assert jnp.all((samples == 0) | (samples == 1))
        # Check mean is approximately p
        assert jnp.allclose(jnp.mean(samples), 0.7, atol=0.1)

    def test_call(self, rngs, bernoulli_probs):
        """Test __call__ method."""
        # Test sampling
        samples = bernoulli_probs(rngs=rngs)
        assert isinstance(samples, jax.Array)

        # Test log probability computation
        x = jnp.array([0.0, 1.0])
        log_probs = bernoulli_probs(x)
        assert log_probs.shape == (2,)
        assert log_probs[1] > log_probs[0]

    def test_log_prob(self, bernoulli_probs):
        """Test log probability computation."""
        x = jnp.array([0.0, 1.0])
        log_probs = bernoulli_probs.log_prob(x)
        assert log_probs.shape == (2,)
        # Check that log probability of 1 is higher than 0 for p > 0.5
        assert log_probs[1] > log_probs[0]

    def test_entropy(self, bernoulli_probs):
        """Test entropy computation."""
        entropy = bernoulli_probs.entropy()
        assert isinstance(entropy, jax.Array)
        # Entropy should be positive for non-deterministic distribution
        assert entropy > 0

    def test_kl_divergence(self, bernoulli_probs):
        """Test KL divergence computation."""
        other = Bernoulli(probs=jnp.array(0.5))
        kl = bernoulli_probs.kl_divergence(other)
        assert isinstance(kl, jax.Array)
        # KL divergence should be positive for different distributions
        assert kl > 0


class TestCategorical:
    """Test cases for the Categorical distribution."""

    @pytest.fixture
    def categorical_probs(self):
        """Fixture for Categorical with probability parameter."""
        return Categorical(probs=jnp.array([0.2, 0.3, 0.5]))

    @pytest.fixture
    def categorical_logits(self):
        """Fixture for Categorical with logits parameter."""
        return Categorical(logits=jnp.array([1.0, 2.0, 3.0]))

    def test_initialization(self, categorical_probs, categorical_logits):
        """Test initialization of Categorical distribution."""
        # Test initialization with probs
        assert jnp.allclose(categorical_probs.probs, jnp.array([0.2, 0.3, 0.5]))

        # Test initialization with logits
        assert jnp.allclose(categorical_logits.logits, jnp.array([1.0, 2.0, 3.0]))

        # Test initialization with num_classes
        default_categorical = Categorical(num_classes=3)
        assert isinstance(default_categorical.logits, nnx.Param)
        assert default_categorical.logits.value.shape == (3,)

    def test_invalid_initialization(self):
        """Test invalid initialization."""
        with pytest.raises(ValueError):
            Categorical(
                probs=jnp.array([0.2, 0.3, 0.5]),
                logits=jnp.array([1.0, 2.0, 3.0]),
            )

        with pytest.raises(ValueError):
            Categorical()  # No num_classes specified

    def test_sample(self, rngs, categorical_probs):
        """Test sampling from Categorical distribution."""
        samples = categorical_probs.sample(sample_shape=(1000,), rngs=rngs)
        assert samples.shape == (1000,)
        assert jnp.all((samples >= 0) & (samples <= 2))
        # Check empirical probabilities
        counts = jnp.bincount(samples.astype(jnp.int32), length=3)
        probs = counts / 1000
        probs_expected = jnp.array([0.2, 0.3, 0.5])
        assert jnp.allclose(probs, probs_expected, atol=0.1)

    def test_call(self, rngs, categorical_probs):
        """Test __call__ method."""
        # Test sampling
        samples = categorical_probs(rngs=rngs)
        assert isinstance(samples, jax.Array)

        # Test log probability computation
        x = jnp.array([0, 1, 2])
        log_probs = categorical_probs(x)
        assert log_probs.shape == (3,)
        assert jnp.allclose(jnp.exp(log_probs), jnp.array([0.2, 0.3, 0.5]))

    def test_log_prob(self, categorical_probs):
        """Test log probability computation."""
        x = jnp.array([0, 1, 2])
        log_probs = categorical_probs.log_prob(x)
        assert log_probs.shape == (3,)
        # Check that log probabilities match the given probabilities
        assert jnp.allclose(jnp.exp(log_probs), jnp.array([0.2, 0.3, 0.5]))

    def test_entropy(self, categorical_probs):
        """Test entropy computation."""
        entropy = categorical_probs.entropy()
        assert isinstance(entropy, jax.Array)
        # Entropy should be positive for non-deterministic distribution
        assert entropy > 0

    def test_kl_divergence(self, categorical_probs):
        """Test KL divergence computation."""
        other = Categorical(probs=jnp.array([0.3, 0.3, 0.4]))
        kl = categorical_probs.kl_divergence(other)
        assert isinstance(kl, jax.Array)
        # KL divergence should be positive for different distributions
        assert kl > 0

    def test_mode(self, categorical_probs):
        """Test mode computation."""
        mode = categorical_probs.mode()
        assert mode == 2  # Index of highest probability


class TestOneHotCategorical:
    """Test cases for the OneHotCategorical distribution."""

    @pytest.fixture
    def onehot_probs(self):
        """Fixture for OneHotCategorical with probability parameter."""
        return OneHotCategorical(probs=jnp.array([0.2, 0.3, 0.5]))

    @pytest.fixture
    def onehot_logits(self):
        """Fixture for OneHotCategorical with logits parameter."""
        return OneHotCategorical(logits=jnp.array([1.0, 2.0, 3.0]))

    def test_initialization(self, onehot_probs, onehot_logits):
        """Test initialization of OneHotCategorical distribution."""
        # Test that underlying categorical is initialized correctly
        assert hasattr(onehot_probs, "categorical")
        assert hasattr(onehot_logits, "categorical")

        # Test that num_classes is set correctly
        assert onehot_probs.num_classes == 3
        assert onehot_logits.num_classes == 3

        # Test initialization with num_classes
        default_onehot = OneHotCategorical(num_classes=3)
        assert default_onehot.num_classes == 3

    def test_invalid_initialization(self):
        """Test invalid initialization."""
        with pytest.raises(ValueError):
            OneHotCategorical(
                probs=jnp.array([0.2, 0.3, 0.5]),
                logits=jnp.array([1.0, 2.0, 3.0]),
            )

        with pytest.raises(ValueError):
            OneHotCategorical()  # No num_classes specified

    def test_sample(self, rngs, onehot_probs):
        """Test sampling from OneHotCategorical distribution."""
        samples = onehot_probs.sample(sample_shape=(1000,), rngs=rngs)
        assert samples.shape == (1000, 3)

        # Check one-hot property: each row should sum to 1
        assert jnp.all(jnp.sum(samples, axis=-1) == 1)

        # Each row should have exactly one 1 and the rest 0s
        assert jnp.all(jnp.max(samples, axis=-1) == 1)

        # Check empirical probabilities by counting which position has the 1
        positions = jnp.argmax(samples, axis=-1)
        counts = jnp.bincount(positions, length=3)
        probs = counts / 1000
        probs_expected = jnp.array([0.2, 0.3, 0.5])
        assert jnp.allclose(probs, probs_expected, atol=0.1)

    def test_call(self, rngs, onehot_probs):
        """Test __call__ method."""
        # Test sampling
        samples = onehot_probs(rngs=rngs)
        assert isinstance(samples, jax.Array)
        assert samples.shape == (3,)
        assert jnp.sum(samples) == 1

        # Test log probability computation
        x = jnp.array(
            [
                [1, 0, 0],  # one-hot for class 0
                [0, 1, 0],  # one-hot for class 1
                [0, 0, 1],  # one-hot for class 2
            ]
        )
        log_probs = onehot_probs(x)
        assert log_probs.shape == (3,)
        assert jnp.allclose(jnp.exp(log_probs), jnp.array([0.2, 0.3, 0.5]))

    def test_log_prob(self, onehot_probs):
        """Test log probability computation."""
        # Test with one-hot encoded input
        x = jnp.array(
            [
                [1, 0, 0],  # one-hot for class 0
                [0, 1, 0],  # one-hot for class 1
                [0, 0, 1],  # one-hot for class 2
            ]
        )
        log_probs = onehot_probs.log_prob(x)
        assert log_probs.shape == (3,)
        assert jnp.allclose(jnp.exp(log_probs), jnp.array([0.2, 0.3, 0.5]), atol=1e-5)

        # Test with categorical indices
        x_cat = jnp.array([0, 1, 2])
        log_probs_cat = onehot_probs.log_prob(x_cat)
        assert jnp.allclose(log_probs, log_probs_cat)

    def test_entropy(self, onehot_probs):
        """Test entropy computation."""
        entropy = onehot_probs.entropy()
        assert isinstance(entropy, jax.Array)
        # Entropy should be positive for non-deterministic distribution
        assert entropy > 0

        # Should match entropy of underlying categorical
        categorical_entropy = onehot_probs.categorical.entropy()
        assert jnp.allclose(entropy, categorical_entropy)

    def test_kl_divergence(self, onehot_probs):
        """Test KL divergence computation."""
        other = OneHotCategorical(probs=jnp.array([0.3, 0.3, 0.4]))
        kl = onehot_probs.kl_divergence(other)
        assert isinstance(kl, jax.Array)
        # KL divergence should be positive for different distributions
        assert kl > 0

    def test_mode(self, onehot_probs):
        """Test mode computation."""
        mode = onehot_probs.mode()
        assert mode.shape == (3,)
        # Should be one-hot encoding of class with highest probability
        assert jnp.array_equal(mode, jnp.array([0, 0, 1]))
