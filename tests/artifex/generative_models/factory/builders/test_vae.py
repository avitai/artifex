"""Tests for VAE builder using dataclass configs.

These tests verify the VAEBuilder functionality with the new dataclass-based
configuration system (VAEConfig, BetaVAEConfig, etc.) following Principle #4.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.core.configuration.vae_config import (
    BetaVAEConfig,
    ConditionalVAEConfig,
    VAEConfig,
)
from artifex.generative_models.factory.builders.vae import VAEBuilder
from artifex.generative_models.models.vae.base import VAE


class TestVAEBuilder:
    """Test VAE builder functionality with dataclass configs."""

    @pytest.fixture
    def rngs(self):
        """Create RNGs for testing."""
        return nnx.Rngs(params=jax.random.PRNGKey(42))

    @pytest.fixture
    def encoder_config(self):
        """Create EncoderConfig for testing."""
        return EncoderConfig(
            name="test_encoder",
            input_shape=(28, 28, 1),
            hidden_dims=(512, 256),
            latent_dim=64,
            activation="relu",
        )

    @pytest.fixture
    def decoder_config(self):
        """Create DecoderConfig for testing."""
        return DecoderConfig(
            name="test_decoder",
            latent_dim=64,
            hidden_dims=(256, 512),
            output_shape=(28, 28, 1),
            activation="relu",
        )

    def test_build_standard_vae(self, rngs, encoder_config, decoder_config):
        """Test building a standard VAE."""
        config = VAEConfig(
            name="test_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            kl_weight=1.0,
        )

        builder = VAEBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "encode")
        assert hasattr(model, "decode")
        assert hasattr(model, "latent_dim")
        assert model.latent_dim == 64

    def test_build_beta_vae(self, rngs, encoder_config, decoder_config):
        """Test building a Beta-VAE."""
        config = BetaVAEConfig(
            name="test_beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=4.0,
        )

        builder = VAEBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "beta_default")
        assert model.beta_default == 4.0

    def test_build_conditional_vae(self, rngs):
        """Test building a conditional VAE."""
        encoder = EncoderConfig(
            name="cvae_encoder",
            input_shape=(28, 28, 1),
            hidden_dims=(128, 64),
            latent_dim=16,
            activation="gelu",
        )
        decoder = DecoderConfig(
            name="cvae_decoder",
            latent_dim=16,
            hidden_dims=(64, 128),
            output_shape=(28, 28, 1),
            activation="gelu",
        )
        config = ConditionalVAEConfig(
            name="test_cvae",
            encoder=encoder,
            decoder=decoder,
            num_classes=10,
        )

        builder = VAEBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert model.latent_dim == 16
        assert hasattr(model, "condition_dim")
        assert model.condition_dim == 10

    def test_vae_forward_pass(self, rngs, encoder_config, decoder_config):
        """Test VAE forward pass produces correct shapes."""
        config = VAEConfig(
            name="test_vae_forward",
            encoder=encoder_config,
            decoder=decoder_config,
            kl_weight=1.0,
        )

        builder = VAEBuilder()
        vae = builder.build(config, rngs=rngs)

        # Create test input
        batch_size = 2
        x = jnp.ones((batch_size, 28, 28, 1))

        # Forward pass
        outputs = vae(x)

        # Check outputs
        assert "reconstructed" in outputs
        assert "mean" in outputs
        assert "log_var" in outputs

        assert outputs["reconstructed"].shape == x.shape
        assert outputs["mean"].shape == (batch_size, 64)
        assert outputs["log_var"].shape == (batch_size, 64)

    def test_vae_encode_decode(self, rngs):
        """Test VAE encode and decode methods."""
        encoder = EncoderConfig(
            name="test_encoder",
            input_shape=(784,),
            hidden_dims=(256, 128),
            latent_dim=20,
            activation="relu",
        )
        decoder = DecoderConfig(
            name="test_decoder",
            latent_dim=20,
            hidden_dims=(128, 256),
            output_shape=(784,),
            activation="relu",
        )
        config = VAEConfig(
            name="test_vae",
            encoder=encoder,
            decoder=decoder,
        )

        builder = VAEBuilder()
        vae = builder.build(config, rngs=rngs)

        batch_size = 4
        x = jnp.ones((batch_size, 784))

        # Test encode
        mean, log_var = vae.encode(x)
        assert mean.shape == (batch_size, 20)
        assert log_var.shape == (batch_size, 20)

        # Sample z from the latent distribution
        z = vae.reparameterize(mean, log_var)
        assert z.shape == (batch_size, 20)

        # Test decode
        x_recon = vae.decode(z)
        assert x_recon.shape == x.shape

    def test_vae_sample(self, rngs, encoder_config, decoder_config):
        """Test VAE sampling."""
        config = VAEConfig(
            name="test_vae_sample",
            encoder=encoder_config,
            decoder=decoder_config,
        )

        builder = VAEBuilder()
        vae = builder.build(config, rngs=rngs)

        # Generate samples
        num_samples = 3
        samples = vae.sample(num_samples)

        assert samples.shape == (num_samples, 28, 28, 1)

    def test_build_with_none_config(self, rngs):
        """Test that None config raises TypeError."""
        builder = VAEBuilder()

        with pytest.raises(TypeError, match="config cannot be None"):
            builder.build(None, rngs=rngs)

    def test_build_with_dict_config(self, rngs):
        """Test that dict config raises TypeError."""
        builder = VAEBuilder()
        config = {"name": "test", "latent_dim": 64}

        with pytest.raises(TypeError, match="config must be a dataclass"):
            builder.build(config, rngs=rngs)

    def test_build_with_invalid_config_type(self, rngs):
        """Test that unsupported config type raises TypeError."""
        builder = VAEBuilder()

        class FakeConfig:
            pass

        fake_config = FakeConfig()

        with pytest.raises(TypeError, match="Unsupported config type"):
            builder.build(fake_config, rngs=rngs)

    def test_config_validation(self):
        """Test that VAE configs properly validate."""
        encoder = EncoderConfig(
            name="valid_encoder",
            input_shape=(28, 28, 1),
            hidden_dims=(128, 64),
            latent_dim=32,
            activation="relu",
        )
        decoder = DecoderConfig(
            name="valid_decoder",
            latent_dim=32,
            hidden_dims=(64, 128),
            output_shape=(28, 28, 1),
            activation="relu",
        )

        # Valid VAE config
        valid_config = VAEConfig(
            name="valid_vae",
            encoder=encoder,
            decoder=decoder,
            kl_weight=1.0,
        )
        assert valid_config.kl_weight == 1.0

        # Valid BetaVAE config
        valid_beta = BetaVAEConfig(
            name="valid_beta",
            encoder=encoder,
            decoder=decoder,
            beta_default=4.0,
        )
        assert valid_beta.beta_default == 4.0

    def test_vae_with_different_kl_weight(self, rngs, encoder_config, decoder_config):
        """Test VAE with custom KL weight."""
        config = VAEConfig(
            name="test_vae_kl",
            encoder=encoder_config,
            decoder=decoder_config,
            kl_weight=0.5,
        )

        builder = VAEBuilder()
        vae = builder.build(config, rngs=rngs)

        assert isinstance(vae, VAE)
        assert vae.kl_weight == 0.5
