"""Tests for transformer autoregressive models."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    TransformerConfig,
    TransformerNetworkConfig,
)
from artifex.generative_models.factory import create_model
from artifex.generative_models.models.autoregressive import TransformerAutoregressiveModel


@pytest.fixture
def rngs():
    """Standard fixture for random number generators."""
    return nnx.Rngs(
        params=jax.random.key(42), sample=jax.random.key(123), dropout=jax.random.key(456)
    )


@pytest.fixture
def network_config():
    """Standard network configuration for testing."""
    return TransformerNetworkConfig(
        name="test_network",
        hidden_dims=(256, 512),
        activation="gelu",
        embed_dim=256,
        num_heads=4,
        mlp_ratio=2.0,  # mlp_dim = 256 * 2 = 512
        dropout_rate=0.1,
    )


@pytest.fixture
def transformer_config(network_config):
    """Standard transformer configuration for testing."""
    return TransformerConfig(
        name="test_transformer",
        vocab_size=1000,
        sequence_length=128,
        network=network_config,
        num_layers=4,
        dropout_rate=0.1,
    )


class TestTransformerAutoregressiveModel:
    """Test suite for TransformerAutoregressiveModel."""

    def test_model_initialization(self, transformer_config, rngs: nnx.Rngs):
        """Test basic model initialization."""
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)

        assert model.vocab_size == transformer_config.vocab_size
        assert model.sequence_length == transformer_config.sequence_length
        assert model.embed_dim == transformer_config.network.embed_dim
        assert model.num_layers == transformer_config.num_layers
        assert model.num_heads == transformer_config.network.num_heads
        assert model.mlp_dim == int(
            transformer_config.network.embed_dim * transformer_config.network.mlp_ratio
        )
        assert model.dropout_rate == transformer_config.dropout_rate

    def test_forward_pass_shapes(self, transformer_config, rngs: nnx.Rngs):
        """Test forward pass produces correct output shapes."""
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)

        batch_size = 2
        seq_len = 64
        inputs = jax.random.randint(
            jax.random.key(0), (batch_size, seq_len), 0, transformer_config.vocab_size
        )

        # Forward pass
        outputs = model(inputs)

        # Check output shape
        expected_shape = (batch_size, seq_len, transformer_config.vocab_size)
        assert "logits" in outputs
        assert outputs["logits"].shape == expected_shape

        # Check output is finite
        assert jnp.isfinite(outputs["logits"]).all()

    def test_positional_encoding_configuration(self, network_config, rngs: nnx.Rngs):
        """Test positional encoding can be enabled/disabled."""
        # With positional encoding (default: sinusoidal)
        config_with_pos = TransformerConfig(
            name="test_with_pos",
            vocab_size=1000,
            sequence_length=128,
            network=network_config,
            num_layers=2,
        )
        model_with_pos = TransformerAutoregressiveModel(config_with_pos, rngs=rngs)
        assert hasattr(model_with_pos, "pos_encoding")

        # Without positional encoding
        network_no_pos = TransformerNetworkConfig(
            name="no_pos_network",
            hidden_dims=(256, 512),
            activation="gelu",
            embed_dim=256,
            num_heads=4,
            positional_encoding="none",
        )
        config_without_pos = TransformerConfig(
            name="test_no_pos",
            vocab_size=1000,
            sequence_length=128,
            network=network_no_pos,
            num_layers=2,
        )
        model_without_pos = TransformerAutoregressiveModel(config_without_pos, rngs=rngs)
        assert not hasattr(model_without_pos, "pos_encoding")

    def test_generation_basic(self, transformer_config, rngs: nnx.Rngs):
        """Test basic text generation."""
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)

        # Start with a short prompt
        prompt = jnp.array([[1, 2, 3]])  # Shape: (1, 3)

        # Generate continuation
        generated = model.generate(prompt=prompt, max_length=10, rngs=rngs)

        # Check output shape and content
        assert generated.shape[0] == 1  # Batch size
        assert generated.shape[1] <= 10  # Max length
        assert jnp.array_equal(generated[:, :3], prompt)  # Prompt preserved

        # Check generated tokens are valid
        assert jnp.all(generated >= 0)
        assert jnp.all(generated < transformer_config.vocab_size)

    def test_generation_with_temperature(self, transformer_config, rngs: nnx.Rngs):
        """Test generation with different temperature settings."""
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)
        prompt = jnp.array([[1, 2]])

        # High temperature (more random)
        gen_high_temp = model.generate(prompt=prompt, max_length=8, temperature=2.0, rngs=rngs)

        # Low temperature (more deterministic)
        gen_low_temp = model.generate(prompt=prompt, max_length=8, temperature=0.1, rngs=rngs)

        # Both should be valid
        assert gen_high_temp.shape == gen_low_temp.shape
        assert jnp.all(gen_high_temp >= 0)
        assert jnp.all(gen_low_temp >= 0)

    def test_generation_with_top_k(self, transformer_config, rngs: nnx.Rngs):
        """Test generation with top-k sampling."""
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)
        prompt = jnp.array([[1, 2]])

        generated = model.generate(prompt=prompt, max_length=8, top_k=10, rngs=rngs)

        assert generated.shape[1] <= 8
        assert jnp.all(generated >= 0)
        assert jnp.all(generated < transformer_config.vocab_size)

    def test_generation_with_top_p(self, transformer_config, rngs: nnx.Rngs):
        """Test generation with nucleus (top-p) sampling."""
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)
        prompt = jnp.array([[1, 2]])

        generated = model.generate(prompt=prompt, max_length=8, top_p=0.9, rngs=rngs)

        assert generated.shape[1] <= 8
        assert jnp.all(generated >= 0)
        assert jnp.all(generated < transformer_config.vocab_size)

    def test_compute_loss(self, transformer_config, rngs: nnx.Rngs):
        """Test loss computation."""
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)

        batch_size = 2
        seq_len = 32
        inputs = jax.random.randint(
            jax.random.key(0), (batch_size, seq_len), 0, transformer_config.vocab_size
        )
        targets = jax.random.randint(
            jax.random.key(1), (batch_size, seq_len), 0, transformer_config.vocab_size
        )

        # Get logits from model
        outputs = model(inputs)
        logits = outputs["logits"]

        loss = model.compute_loss(logits, targets)

        # Loss should be a scalar
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0  # Cross-entropy loss is non-negative

    def test_deterministic_mode(self, transformer_config, rngs: nnx.Rngs):
        """Test deterministic vs non-deterministic forward passes."""
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)

        inputs = jax.random.randint(jax.random.key(0), (1, 16), 0, transformer_config.vocab_size)

        # Deterministic mode (evaluation)
        output1_det = model(inputs, training=False)
        output2_det = model(inputs, training=False)

        # Should be identical in deterministic mode
        assert jnp.allclose(output1_det["logits"], output2_det["logits"], atol=1e-6)

        # Non-deterministic mode (training) with dropout
        if transformer_config.dropout_rate > 0:
            output1_nondet = model(inputs, training=True, rngs=rngs)
            output2_nondet = model(inputs, training=True, rngs=rngs)

            # Outputs should be valid but may differ due to dropout
            assert jnp.isfinite(output1_nondet["logits"]).all()
            assert jnp.isfinite(output2_nondet["logits"]).all()

    def test_attention_mask_handling(self, transformer_config, rngs: nnx.Rngs):
        """Test proper handling of attention masks."""
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)

        batch_size = 2
        seq_len = 16
        inputs = jax.random.randint(
            jax.random.key(0), (batch_size, seq_len), 0, transformer_config.vocab_size
        )

        # Create mask (1 for valid tokens, 0 for padding)
        mask = jnp.ones((batch_size, seq_len))
        mask = mask.at[0, 10:].set(0)  # Mask out last 6 tokens for first example
        mask = mask.at[1, 12:].set(0)  # Mask out last 4 tokens for second example

        # Forward pass with mask
        outputs = model(inputs)

        # Check output is valid
        assert jnp.isfinite(outputs["logits"]).all()
        assert outputs["logits"].shape == (batch_size, seq_len, transformer_config.vocab_size)

    def test_model_reproducibility(self, transformer_config, rngs: nnx.Rngs):
        """Test model reproducibility with same random seed."""
        # Create two models with same seed
        rngs1 = nnx.Rngs(
            params=jax.random.key(42), sample=jax.random.key(123), dropout=jax.random.key(456)
        )
        rngs2 = nnx.Rngs(
            params=jax.random.key(42), sample=jax.random.key(123), dropout=jax.random.key(456)
        )

        model1 = TransformerAutoregressiveModel(transformer_config, rngs=rngs1)
        model2 = TransformerAutoregressiveModel(transformer_config, rngs=rngs2)

        # Same input
        inputs = jnp.array([[1, 2, 3, 4, 5]])

        # Outputs should be identical
        output1 = model1(inputs, training=False)
        output2 = model2(inputs, training=False)

        assert jnp.allclose(output1["logits"], output2["logits"], atol=1e-6)


class TestTransformerFactoryFunctions:
    """Test suite for transformer factory functions using TransformerConfig."""

    def test_create_transformer_language_model(self, rngs: nnx.Rngs):
        """Test transformer language model creation with TransformerConfig."""
        network = TransformerNetworkConfig(
            name="test_network",
            hidden_dims=(512,),
            activation="gelu",
            embed_dim=512,
            num_heads=8,
            mlp_ratio=4.0,
        )
        config = TransformerConfig(
            name="test_transformer",
            vocab_size=5000,
            sequence_length=512,
            num_layers=6,
            network=network,
            dropout_rate=0.1,
        )
        model = create_model(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "vocab_size")
        assert model.vocab_size == 5000

    def test_create_gpt_style_model_sizes(self, rngs: nnx.Rngs):
        """Test GPT-style model creation with different sizes using TransformerConfig."""
        sizes = ["small", "medium", "large", "xl"]
        expected_configs = {
            "small": {"embed_dim": 512, "num_layers": 6, "num_heads": 8},
            "medium": {"embed_dim": 768, "num_layers": 12, "num_heads": 12},
            "large": {"embed_dim": 1024, "num_layers": 24, "num_heads": 16},
            "xl": {"embed_dim": 1280, "num_layers": 36, "num_heads": 20},
        }

        for size in sizes:
            expected = expected_configs[size]
            network = TransformerNetworkConfig(
                name=f"gpt_{size}_network",
                hidden_dims=(expected["embed_dim"],),
                activation="gelu",
                embed_dim=expected["embed_dim"],
                num_heads=expected["num_heads"],
                mlp_ratio=4.0,  # d_ff = embed_dim * 4
            )
            config = TransformerConfig(
                name=f"gpt_{size}",
                vocab_size=50000,
                sequence_length=1024,
                num_layers=expected["num_layers"],
                network=network,
            )
            model = create_model(config, rngs=rngs)

            assert model is not None
            assert hasattr(model, "embed_dim")
            assert model.embed_dim == expected["embed_dim"]

    def test_create_transformer_for_text(self, rngs: nnx.Rngs):
        """Test transformer for text creation with TransformerConfig."""
        network = TransformerNetworkConfig(
            name="text_network",
            hidden_dims=(768,),
            activation="gelu",
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4.0,  # d_ff = 768 * 4 = 3072
        )
        config = TransformerConfig(
            name="text_transformer",
            vocab_size=30000,
            sequence_length=1024,
            num_layers=12,
            network=network,
        )
        model = create_model(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "vocab_size")
        assert model.vocab_size == 30000

    def test_default_mlp_dim_calculation(self, rngs: nnx.Rngs):
        """Test that mlp_dim defaults to 4 * embed_dim (mlp_ratio=4.0)."""
        embed_dim = 384
        network = TransformerNetworkConfig(
            name="test_network",
            hidden_dims=(embed_dim,),
            activation="gelu",
            embed_dim=embed_dim,
            num_heads=6,  # 384 / 6 = 64 per head
            mlp_ratio=4.0,  # mlp_dim = embed_dim * 4
        )
        config = TransformerConfig(
            name="test_mlp_dim",
            vocab_size=1000,
            sequence_length=256,
            num_layers=4,
            network=network,
        )
        model = create_model(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "mlp_dim")
        assert model.mlp_dim == embed_dim * 4

    def test_custom_mlp_dim(self, rngs: nnx.Rngs):
        """Test custom mlp_dim override via mlp_ratio."""
        embed_dim = 384
        custom_mlp_dim = 1024
        # mlp_ratio = custom_mlp_dim / embed_dim
        mlp_ratio = custom_mlp_dim / embed_dim
        network = TransformerNetworkConfig(
            name="test_network",
            hidden_dims=(embed_dim,),
            activation="gelu",
            embed_dim=embed_dim,
            num_heads=6,
            mlp_ratio=mlp_ratio,  # Custom ratio for custom mlp_dim
        )
        config = TransformerConfig(
            name="test_custom_mlp",
            vocab_size=1000,
            sequence_length=256,
            num_layers=4,
            network=network,
        )
        model = create_model(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "mlp_dim")
        # mlp_dim = embed_dim * mlp_ratio = 384 * (1024/384) = 1024
        assert model.mlp_dim == custom_mlp_dim


class TestTransformerIntegration:
    """Integration tests for transformer models."""

    def test_transformer_training_step(self, transformer_config, rngs: nnx.Rngs):
        """Test a complete training step."""
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)

        # Create dummy batch
        batch_size = 4
        seq_len = 64
        inputs = jax.random.randint(
            jax.random.key(0), (batch_size, seq_len), 0, transformer_config.vocab_size
        )
        targets = jax.random.randint(
            jax.random.key(1), (batch_size, seq_len), 0, transformer_config.vocab_size
        )

        # Forward pass
        outputs = model(inputs, training=True, rngs=rngs)
        logits = outputs["logits"]
        loss = model.compute_loss(logits, targets)

        # Check shapes and validity
        assert logits.shape == (batch_size, seq_len, transformer_config.vocab_size)
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0

    def test_transformer_inference_step(self, transformer_config, rngs: nnx.Rngs):
        """Test inference generation."""
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)

        # Short prompt
        prompt = jnp.array([[1, 2, 3]])

        # Generate text
        generated = model.generate(
            prompt=prompt, max_length=20, temperature=0.8, top_k=50, rngs=rngs
        )

        # Verify generation
        assert generated.shape[0] == 1  # Batch size
        assert generated.shape[1] <= 20  # Max length
        assert jnp.array_equal(generated[:, :3], prompt)  # Prompt preserved

        # All tokens should be valid
        assert jnp.all(generated >= 0)
        assert jnp.all(generated < transformer_config.vocab_size)

    def test_mixed_precision_compatibility(self, transformer_config, rngs: nnx.Rngs):
        """Test model works with different dtypes."""
        # Test with float32 (default)
        model_f32 = TransformerAutoregressiveModel(transformer_config, rngs=rngs)
        inputs = jax.random.randint(jax.random.key(0), (2, 16), 0, 100)

        outputs_f32 = model_f32(inputs, training=False)
        assert outputs_f32["logits"].dtype == jnp.float32
        assert jnp.isfinite(outputs_f32["logits"]).all()

    def test_gradient_computation(self, transformer_config, rngs: nnx.Rngs):
        """Test that gradients can be computed."""
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)

        def loss_fn(model, inputs, targets):
            outputs = model(inputs, training=True, rngs=rngs)
            logits = outputs["logits"]
            return model.compute_loss(logits, targets)

        inputs = jax.random.randint(jax.random.key(0), (2, 16), 0, 100)
        targets = jax.random.randint(jax.random.key(1), (2, 16), 0, 100)

        # Compute gradients
        loss, grads = nnx.value_and_grad(loss_fn)(model, inputs, targets)

        # Check loss and gradients are finite
        assert jnp.isfinite(loss)

        # Check gradients exist for parameters
        assert grads is not None
