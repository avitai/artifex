"""Tests for Autoregressive builder using dataclass configs.

These tests verify the AutoregressiveBuilder functionality with the new dataclass-based
configuration system (TransformerConfig, PixelCNNConfig, WaveNetConfig) following Principle #4.
"""

import jax
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.autoregressive_config import (
    PixelCNNConfig,
    TransformerConfig,
    TransformerNetworkConfig,
    WaveNetConfig,
)
from artifex.generative_models.factory.builders.autoregressive import AutoregressiveBuilder


class TestAutoregressiveBuilder:
    """Test Autoregressive builder functionality with dataclass configs."""

    @pytest.fixture
    def rngs(self):
        """Create RNGs for testing.

        Note: nnx.Dropout requires a 'dropout' stream, so we provide both
        'params' for weight initialization and 'dropout' for stochastic layers.
        """
        return nnx.Rngs(params=jax.random.PRNGKey(42), dropout=jax.random.PRNGKey(43))

    @pytest.fixture
    def transformer_network_config(self):
        """Create TransformerNetworkConfig for testing."""
        return TransformerNetworkConfig(
            name="test_network",
            hidden_dims=(512,),
            activation="gelu",
            embed_dim=512,
            num_heads=8,
            mlp_ratio=4.0,
            dropout_rate=0.1,
            positional_encoding="sinusoidal",
        )

    def test_build_transformer(self, rngs, transformer_network_config):
        """Test building a Transformer autoregressive model."""
        config = TransformerConfig(
            name="test_transformer",
            vocab_size=50257,
            sequence_length=1024,
            network=transformer_network_config,
            num_layers=4,
            dropout_rate=0.1,
        )

        builder = AutoregressiveBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "vocab_size")
        assert model.vocab_size == 50257
        assert hasattr(model, "sequence_length")
        assert model.sequence_length == 1024

    def test_build_pixelcnn(self, rngs):
        """Test building a PixelCNN model."""
        config = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(32, 32, 3),
            hidden_channels=128,
            num_layers=7,
            num_residual_blocks=5,
            dropout_rate=0.0,
        )

        builder = AutoregressiveBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "image_shape")
        assert model.image_shape == (32, 32, 3)

    def test_build_wavenet(self, rngs):
        """Test building a WaveNet model."""
        config = WaveNetConfig(
            name="test_wavenet",
            vocab_size=256,
            sequence_length=16000,
            residual_channels=32,
            skip_channels=256,
            num_blocks=3,
            layers_per_block=10,
            kernel_size=2,
            dilation_base=2,
            dropout_rate=0.0,
        )

        builder = AutoregressiveBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "vocab_size")
        assert model.vocab_size == 256

    def test_transformer_with_custom_params(self, rngs):
        """Test building Transformer with custom parameters."""
        # Create network config with 768 embed_dim and 12 heads
        network = TransformerNetworkConfig(
            name="custom_network",
            hidden_dims=(768,),
            activation="gelu",
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            positional_encoding="learned",
        )

        config = TransformerConfig(
            name="test_custom_transformer",
            vocab_size=30000,
            sequence_length=512,
            network=network,
            num_layers=6,
            dropout_rate=0.2,
        )

        builder = AutoregressiveBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert model.vocab_size == 30000
        assert model.sequence_length == 512

    def test_pixelcnn_mnist(self, rngs):
        """Test PixelCNN with MNIST-like configuration."""
        config = PixelCNNConfig(
            name="test_pixelcnn_mnist",
            image_shape=(28, 28, 1),
            hidden_channels=64,
            num_layers=8,
            num_residual_blocks=3,
            kernel_size=5,
            dropout_rate=0.05,
        )

        builder = AutoregressiveBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert model.image_shape == (28, 28, 1)

    def test_wavenet_with_custom_params(self, rngs):
        """Test WaveNet with custom parameters."""
        config = WaveNetConfig(
            name="test_wavenet_custom",
            vocab_size=256,
            sequence_length=8000,
            residual_channels=128,
            skip_channels=128,
            num_blocks=2,
            layers_per_block=10,
            kernel_size=3,
            dilation_base=2,
            use_gated_activation=True,
            dropout_rate=0.0,
        )

        builder = AutoregressiveBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        # Check receptive field calculation
        assert config.receptive_field > 0

    def test_build_with_none_config(self, rngs):
        """Test that None config raises TypeError."""
        builder = AutoregressiveBuilder()

        with pytest.raises(TypeError, match="config cannot be None"):
            builder.build(None, rngs=rngs)

    def test_build_with_dict_config(self, rngs):
        """Test that dict config raises TypeError."""
        builder = AutoregressiveBuilder()
        config = {"name": "test", "vocab_size": 1000}

        with pytest.raises(TypeError, match="config must be a dataclass"):
            builder.build(config, rngs=rngs)

    def test_build_with_invalid_config_type(self, rngs):
        """Test that unsupported config type raises TypeError."""
        builder = AutoregressiveBuilder()

        # Create a mock object that's not a supported config type
        class FakeConfig:
            pass

        fake_config = FakeConfig()

        with pytest.raises(TypeError, match="Unsupported config type"):
            builder.build(fake_config, rngs=rngs)

    def test_transformer_network_validation(self):
        """Test TransformerNetworkConfig validation."""
        # Valid config should work
        valid_config = TransformerNetworkConfig(
            name="valid",
            hidden_dims=(512,),
            activation="gelu",
            embed_dim=512,
            num_heads=8,
        )
        assert valid_config.embed_dim == 512

        # embed_dim not divisible by num_heads should fail
        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            TransformerNetworkConfig(
                name="invalid",
                hidden_dims=(512,),
                activation="gelu",
                embed_dim=513,  # Not divisible by 8
                num_heads=8,
            )

    def test_pixelcnn_config_validation(self):
        """Test PixelCNNConfig validation."""
        # Valid config
        valid = PixelCNNConfig(
            name="valid",
            image_shape=(32, 32, 3),
            hidden_channels=64,
            num_layers=5,
        )
        assert valid.image_shape == (32, 32, 3)
        assert valid.derived_vocab_size == 256  # 8-bit pixels
        assert valid.derived_sequence_length == 32 * 32 * 3

        # Invalid image_shape (2D instead of 3D)
        with pytest.raises(ValueError, match="image_shape must have 3 dimensions"):
            PixelCNNConfig(
                name="invalid",
                image_shape=(32, 32),  # Missing channels
                hidden_channels=64,
            )

    def test_wavenet_config_validation(self):
        """Test WaveNetConfig validation."""
        # Valid config
        valid = WaveNetConfig(
            name="valid",
            vocab_size=256,
            sequence_length=16000,
            residual_channels=32,
            skip_channels=256,
        )
        assert valid.receptive_field > 0

        # Invalid dilation_base
        with pytest.raises(ValueError, match="dilation_base must be at least 2"):
            WaveNetConfig(
                name="invalid",
                vocab_size=256,
                sequence_length=16000,
                dilation_base=1,  # Invalid
            )
