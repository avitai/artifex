"""Tests for autoregressive model integration with dataclass configs.

This module tests that autoregressive models can be created from the new
nested dataclass configuration system, following the TDD approach.

The migration pattern is that `__init__` takes a config object directly,
following the pattern established in flow models (Glow, RealNVP, etc.).
"""

import jax
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    PixelCNNConfig,
    TransformerConfig,
    TransformerNetworkConfig,
)
from artifex.generative_models.models.autoregressive import (
    PixelCNN,
    TransformerAutoregressiveModel,
)


# =============================================================================
# TransformerAutoregressiveModel Tests with Config
# =============================================================================
class TestTransformerWithConfig:
    """Test TransformerAutoregressiveModel with config-based initialization."""

    @pytest.fixture
    def network_config(self):
        """Create a basic network config."""
        return TransformerNetworkConfig(
            name="test_network",
            hidden_dims=(256, 512),
            activation="gelu",
            embed_dim=256,
            num_heads=4,
            mlp_ratio=4.0,
            dropout_rate=0.1,
        )

    @pytest.fixture
    def transformer_config(self, network_config):
        """Create a basic transformer config."""
        return TransformerConfig(
            name="test_transformer",
            vocab_size=1000,
            sequence_length=128,
            network=network_config,
            num_layers=2,
            dropout_rate=0.1,
        )

    def test_init_with_config_creates_model(self, transformer_config):
        """Test that __init__ with config creates a model instance."""
        rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)
        assert isinstance(model, TransformerAutoregressiveModel)

    def test_init_with_config_stores_config(self, transformer_config):
        """Test that __init__ stores the config."""
        rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)
        assert model.config == transformer_config

    def test_init_with_config_sets_vocab_size(self, transformer_config):
        """Test that __init__ correctly sets vocab_size from config."""
        rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)
        assert model.vocab_size == transformer_config.vocab_size

    def test_init_with_config_sets_sequence_length(self, transformer_config):
        """Test that __init__ correctly sets sequence_length from config."""
        rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)
        assert model.sequence_length == transformer_config.sequence_length

    def test_init_with_config_sets_embed_dim(self, transformer_config):
        """Test that __init__ correctly sets embed_dim from network config."""
        rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)
        assert model.embed_dim == transformer_config.network.embed_dim

    def test_init_with_config_sets_num_layers(self, transformer_config):
        """Test that __init__ correctly sets num_layers from config."""
        rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)
        assert model.num_layers == transformer_config.num_layers

    def test_init_with_config_sets_num_heads(self, transformer_config):
        """Test that __init__ correctly sets num_heads from network config."""
        rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)
        assert model.num_heads == transformer_config.network.num_heads

    def test_config_validates_type(self, transformer_config):
        """Test that invalid config type raises TypeError."""
        rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
        with pytest.raises(TypeError, match="config must be TransformerConfig"):
            TransformerAutoregressiveModel({"invalid": "config"}, rngs=rngs)

    def test_model_forward_pass(self, transformer_config):
        """Test that model created from config can perform forward pass."""
        rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
        model = TransformerAutoregressiveModel(transformer_config, rngs=rngs)

        # Create sample input
        batch_size = 2
        seq_len = 64
        x = jax.random.randint(
            jax.random.key(0), (batch_size, seq_len), 0, transformer_config.vocab_size
        )

        # Forward pass
        outputs = model(x, training=False)
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, transformer_config.vocab_size)


# =============================================================================
# PixelCNN Tests with Config
# =============================================================================
class TestPixelCNNWithConfig:
    """Test PixelCNN with config-based initialization."""

    @pytest.fixture
    def pixelcnn_config(self):
        """Create a basic PixelCNN config."""
        return PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(8, 8, 3),
            hidden_channels=32,
            num_layers=3,
            num_residual_blocks=2,
        )

    def test_init_with_config_creates_model(self, pixelcnn_config):
        """Test that __init__ with config creates a model instance."""
        rngs = nnx.Rngs(params=jax.random.key(0))
        model = PixelCNN(pixelcnn_config, rngs=rngs)
        assert isinstance(model, PixelCNN)

    def test_init_with_config_stores_config(self, pixelcnn_config):
        """Test that __init__ stores the config."""
        rngs = nnx.Rngs(params=jax.random.key(0))
        model = PixelCNN(pixelcnn_config, rngs=rngs)
        assert model.config == pixelcnn_config

    def test_init_with_config_sets_image_shape(self, pixelcnn_config):
        """Test that __init__ correctly sets image_shape from config."""
        rngs = nnx.Rngs(params=jax.random.key(0))
        model = PixelCNN(pixelcnn_config, rngs=rngs)
        assert model.image_shape == pixelcnn_config.image_shape

    def test_init_with_config_sets_hidden_channels(self, pixelcnn_config):
        """Test that __init__ correctly sets hidden_channels from config."""
        rngs = nnx.Rngs(params=jax.random.key(0))
        model = PixelCNN(pixelcnn_config, rngs=rngs)
        assert model.hidden_channels == pixelcnn_config.hidden_channels

    def test_init_with_config_sets_num_layers(self, pixelcnn_config):
        """Test that __init__ correctly sets num_layers from config."""
        rngs = nnx.Rngs(params=jax.random.key(0))
        model = PixelCNN(pixelcnn_config, rngs=rngs)
        assert model.num_layers == pixelcnn_config.num_layers

    def test_init_with_config_sets_num_residual_blocks(self, pixelcnn_config):
        """Test that __init__ correctly sets num_residual_blocks from config."""
        rngs = nnx.Rngs(params=jax.random.key(0))
        model = PixelCNN(pixelcnn_config, rngs=rngs)
        assert model.num_residual_blocks == pixelcnn_config.num_residual_blocks

    def test_config_validates_type(self, pixelcnn_config):
        """Test that invalid config type raises TypeError."""
        rngs = nnx.Rngs(params=jax.random.key(0))
        with pytest.raises(TypeError, match="config must be PixelCNNConfig"):
            PixelCNN({"invalid": "config"}, rngs=rngs)

    def test_model_forward_pass(self, pixelcnn_config):
        """Test that model created from config can perform forward pass."""
        rngs = nnx.Rngs(params=jax.random.key(0))
        model = PixelCNN(pixelcnn_config, rngs=rngs)

        # Create sample input
        batch_size = 2
        h, w, c = pixelcnn_config.image_shape
        x = jax.random.randint(jax.random.key(0), (batch_size, h, w, c), 0, 256)

        # Forward pass
        outputs = model(x, training=False)
        assert "logits" in outputs
        assert "logits_spatial" in outputs
