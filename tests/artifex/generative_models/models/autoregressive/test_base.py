"""Tests for autoregressive base models."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.models.autoregressive.base import (
    AutoregressiveModel,
)


class SimpleAutoregressiveModel(AutoregressiveModel):
    """Simple test implementation of autoregressive model."""

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        hidden_size: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            rngs=rngs,
        )

        self.hidden_size = hidden_size

        # Simple embedding and output layers
        self.embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            rngs=rngs,
        )

        self.output_proj = nnx.Linear(
            in_features=hidden_size,
            out_features=vocab_size,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, *, rngs=None, training=False, **kwargs):
        """Simple forward pass."""
        # Embed tokens
        embedded = self.embedding(x)

        # Simple linear transformation (not truly autoregressive for testing)
        logits = self.output_proj(embedded)

        return {"logits": logits}


class TestAutoregressiveModel:
    """Test cases for AutoregressiveModel base class."""

    @pytest.fixture
    def model_config(self):
        """Basic model configuration."""
        return {
            "vocab_size": 100,
            "sequence_length": 32,
            "hidden_size": 64,
        }

    @pytest.fixture
    def rngs(self):
        """Random number generators."""
        return nnx.Rngs(params=jax.random.key(42), sample=jax.random.key(123))

    @pytest.fixture
    def model(self, model_config, rngs):
        """Create test model."""
        return SimpleAutoregressiveModel(rngs=rngs, **model_config)

    @pytest.fixture
    def batch_data(self, model_config):
        """Sample batch data."""
        batch_size = 4
        sequence_length = model_config["sequence_length"]
        vocab_size = model_config["vocab_size"]

        key = jax.random.key(456)
        x = jax.random.randint(key, (batch_size, sequence_length), 0, vocab_size)
        return {"x": x}

    def test_model_initialization(self, model, model_config):
        """Test model initialization."""
        assert model.vocab_size == model_config["vocab_size"]
        assert model.sequence_length == model_config["sequence_length"]
        assert model.hidden_size == model_config["hidden_size"]
        assert hasattr(model, "_rngs")

    def test_forward_pass(self, model, batch_data):
        """Test forward pass."""
        x = batch_data["x"]
        outputs = model(x, training=True)

        assert "logits" in outputs
        logits = outputs["logits"]

        # Check output shape
        expected_shape = (x.shape[0], x.shape[1], model.vocab_size)
        assert logits.shape == expected_shape

        # Check output is finite
        assert jnp.all(jnp.isfinite(logits))

    def test_forward_pass_evaluation_mode(self, model, batch_data):
        """Test forward pass in evaluation mode."""
        x = batch_data["x"]
        outputs = model(x, training=False)

        assert "logits" in outputs
        logits = outputs["logits"]
        assert logits.shape == (x.shape[0], x.shape[1], model.vocab_size)

    def test_generate_basic(self, model, rngs):
        """Test basic generation."""
        n_samples = 3
        sequences = model.generate(n_samples=n_samples, rngs=rngs)

        # Check output shape
        assert sequences.shape == (n_samples, model.sequence_length)

        # Check output is valid tokens
        assert jnp.all(sequences >= 0)
        assert jnp.all(sequences < model.vocab_size)

    def test_generate_with_max_length(self, model, rngs):
        """Test generation with custom max length."""
        n_samples = 2
        max_length = 16
        sequences = model.generate(n_samples=n_samples, max_length=max_length, rngs=rngs)

        assert sequences.shape == (n_samples, max_length)
        assert jnp.all(sequences >= 0)
        assert jnp.all(sequences < model.vocab_size)

    def test_generate_with_temperature(self, model, rngs):
        """Test generation with different temperatures."""
        n_samples = 2

        # Low temperature (more deterministic)
        seq_low = model.generate(n_samples=n_samples, temperature=0.1, rngs=rngs)

        # High temperature (more random)
        seq_high = model.generate(n_samples=n_samples, temperature=2.0, rngs=rngs)

        assert seq_low.shape == seq_high.shape
        assert jnp.all(seq_low >= 0) and jnp.all(seq_low < model.vocab_size)
        assert jnp.all(seq_high >= 0) and jnp.all(seq_high < model.vocab_size)

    def test_generate_with_top_k(self, model, rngs):
        """Test generation with top-k sampling."""
        n_samples = 2
        top_k = 10

        sequences = model.generate(n_samples=n_samples, top_k=top_k, rngs=rngs)

        assert sequences.shape == (n_samples, model.sequence_length)
        assert jnp.all(sequences >= 0)
        assert jnp.all(sequences < model.vocab_size)

    def test_generate_with_top_p(self, model, rngs):
        """Test generation with top-p (nucleus) sampling."""
        n_samples = 2
        top_p = 0.9

        sequences = model.generate(n_samples=n_samples, top_p=top_p, rngs=rngs)

        assert sequences.shape == (n_samples, model.sequence_length)
        assert jnp.all(sequences >= 0)
        assert jnp.all(sequences < model.vocab_size)

    def test_generate_without_rngs(self, model):
        """Test generation without providing rngs."""
        n_samples = 1
        sequences = model.generate(n_samples=n_samples)

        assert sequences.shape == (n_samples, model.sequence_length)
        assert jnp.all(sequences >= 0)
        assert jnp.all(sequences < model.vocab_size)

    def test_compute_loss_basic(self, model, batch_data):
        """Test basic loss computation."""
        x = batch_data["x"]
        outputs = model(x, training=True)
        logits = outputs["logits"]

        # Use shifted input as targets for testing
        targets = x
        loss = model.compute_loss(logits, targets)

        # Check loss is scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0  # Cross-entropy loss should be non-negative

    def test_compute_loss_with_mask(self, model, batch_data):
        """Test loss computation with mask."""
        x = batch_data["x"]
        outputs = model(x, training=True)
        logits = outputs["logits"]

        # Create mask (some positions are valid)
        mask = jnp.ones_like(x)
        mask = mask.at[:, -5:].set(0.0)  # Mask last 5 positions

        targets = x
        loss = model.compute_loss(logits, targets, mask=mask)

        # Check loss is scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0

    def test_log_prob(self, model, batch_data, rngs):
        """Test log probability computation."""
        x = batch_data["x"]
        log_probs = model.log_prob(x, rngs=rngs)

        # Check output shape
        assert log_probs.shape == (x.shape[0],)

        # Check output is finite and negative (log probabilities)
        assert jnp.all(jnp.isfinite(log_probs))
        assert jnp.all(log_probs <= 0)  # Log probabilities should be ≤ 0

    def test_sample_with_conditioning(self, model, rngs):
        """Test conditional sampling."""
        conditioning = jnp.array([[1, 2, 3]])  # Shape: [1, 3]
        n_samples = 2

        samples = model.sample_with_conditioning(
            conditioning=conditioning, n_samples=n_samples, rngs=rngs
        )

        # Check output shape
        cond_len = conditioning.shape[1]
        expected_length = cond_len + (model.sequence_length - cond_len)
        assert samples.shape == (n_samples, expected_length)

        # Check starts with conditioning
        assert jnp.allclose(samples[:, : conditioning.shape[1]], conditioning)

        # Check rest is valid tokens
        assert jnp.all(samples >= 0)
        assert jnp.all(samples < model.vocab_size)

    def test_get_rng_key(self, model, rngs):
        """Test RNG key extraction."""
        key = model._get_rng_key(rngs, "sample", 0)
        # Check that we get some kind of key/array - shape might vary
        assert hasattr(key, "shape")
        # Allow for scalar or proper JAX keys
        assert jnp.isscalar(key) or len(key.shape) <= 2

        # Test with non-existent key name
        key_default = model._get_rng_key(rngs, "nonexistent", 123)
        assert hasattr(key_default, "shape")
        # Should fallback to jax.random.key which now has scalar shape ()
        assert key_default.shape == ()

    def test_loss_fn_structure(self, model, batch_data):
        """Test loss function structure."""
        x = batch_data["x"]
        outputs = model(x, training=True)

        # Create batch format
        batch = {"inputs": x, "targets": x}

        loss_dict = model.loss_fn(batch, outputs)

        assert isinstance(loss_dict, dict)
        # This may vary based on the actual implementation

    def test_edge_cases(self, model_config, rngs):
        """Test edge cases."""
        # Test with sequence_length=1
        small_model = SimpleAutoregressiveModel(
            vocab_size=10,
            sequence_length=1,
            rngs=rngs,
            **{k: v for k, v in model_config.items() if k not in ["vocab_size", "sequence_length"]},
        )

        x = jnp.array([[5]])  # Single token
        outputs = small_model(x, training=False)
        assert outputs["logits"].shape == (1, 1, 10)

        # Test generation
        sequences = small_model.generate(n_samples=2, rngs=rngs)
        assert sequences.shape == (2, 1)

    def test_model_reproducibility(self, model_config):
        """Test model reproducibility with same random seed."""
        rngs1 = nnx.Rngs(params=jax.random.key(42), sample=jax.random.key(123))
        rngs2 = nnx.Rngs(params=jax.random.key(42), sample=jax.random.key(123))

        model1 = SimpleAutoregressiveModel(rngs=rngs1, **model_config)
        model2 = SimpleAutoregressiveModel(rngs=rngs2, **model_config)

        x = jnp.ones((1, model_config["sequence_length"]), dtype=jnp.int32)

        outputs1 = model1(x, training=False)
        outputs2 = model2(x, training=False)

        # Should be the same with same random seed
        assert jnp.allclose(outputs1["logits"], outputs2["logits"], atol=1e-6)


class TestAutoregressiveModelAbstractMethods:
    """Test abstract method enforcement."""

    def test_abstract_call_method(self):
        """Test that abstract methods must be implemented."""
        rngs = nnx.Rngs(params=jax.random.key(42))

        # Should be able to instantiate the base class, but calling
        # it should raise NotImplementedError
        model = AutoregressiveModel(vocab_size=100, sequence_length=32, rngs=rngs)

        # Calling the abstract method should raise NotImplementedError
        x = jnp.ones((1, 32))
        with pytest.raises(NotImplementedError):
            model(x)


if __name__ == "__main__":
    pytest.main([__file__])
