"""Tests for GAN builder using dataclass configs.

These tests verify the GANBuilder functionality with the new dataclass-based
configuration system (WGANConfig, LSGANConfig, etc.) following Principle #4.

Note: Many GAN configs require ConvGeneratorConfig and ConvDiscriminatorConfig
since they are designed for convolutional architectures.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.gan_config import (
    DCGANConfig,
    LSGANConfig,
    WGANConfig,
)
from artifex.generative_models.core.configuration.network_configs import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
)
from artifex.generative_models.factory.builders.gan import GANBuilder


class TestGANBuilder:
    """Test GAN builder functionality with dataclass configs."""

    @pytest.fixture
    def rngs(self):
        """Create RNGs for testing (includes sample stream for DCGAN)."""
        key = jax.random.PRNGKey(42)
        return nnx.Rngs(
            params=key,
            dropout=jax.random.fold_in(key, 1),
            sample=jax.random.fold_in(key, 2),
        )

    @pytest.fixture
    def conv_generator_config(self):
        """Create a ConvGeneratorConfig for testing."""
        return ConvGeneratorConfig(
            name="test_generator",
            latent_dim=100,
            hidden_dims=(512, 256, 128),
            output_shape=(3, 32, 32),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm=True,
        )

    @pytest.fixture
    def conv_discriminator_config(self):
        """Create a ConvDiscriminatorConfig for testing."""
        return ConvDiscriminatorConfig(
            name="test_discriminator",
            input_shape=(3, 32, 32),
            hidden_dims=(64, 128, 256),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

    def test_build_dcgan(self, rngs, conv_generator_config, conv_discriminator_config):
        """Test building a DCGAN."""
        config = DCGANConfig(
            name="test_dcgan",
            generator=conv_generator_config,
            discriminator=conv_discriminator_config,
        )

        builder = GANBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "generator")
        assert hasattr(model, "discriminator")
        assert model.latent_dim == 100

    def test_build_wgan(self, rngs, conv_generator_config, conv_discriminator_config):
        """Test building a WGAN."""
        config = WGANConfig(
            name="test_wgan",
            generator=conv_generator_config,
            discriminator=conv_discriminator_config,
            critic_iterations=5,
            use_gradient_penalty=True,
        )

        builder = GANBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "generator")
        assert hasattr(model, "discriminator")

    def test_build_lsgan(self, rngs, conv_generator_config, conv_discriminator_config):
        """Test building an LSGAN."""
        config = LSGANConfig(
            name="test_lsgan",
            generator=conv_generator_config,
            discriminator=conv_discriminator_config,
            a=0.0,
            b=1.0,
            c=1.0,
        )

        builder = GANBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "generator")
        assert hasattr(model, "discriminator")

    def test_dcgan_forward_pass(self, rngs, conv_generator_config, conv_discriminator_config):
        """Test DCGAN forward pass produces correct shapes."""
        config = DCGANConfig(
            name="test_dcgan_forward",
            generator=conv_generator_config,
            discriminator=conv_discriminator_config,
        )

        builder = GANBuilder()
        model = builder.build(config, rngs=rngs)

        # Test generation
        noise = jnp.ones((2, 100))
        generated = model.generator(noise)
        # DCGAN returns (batch, channels, height, width) format
        assert generated.shape == (2, 3, 32, 32)

        # Test discrimination
        images = jnp.ones((2, 3, 32, 32))
        discriminated = model.discriminator(images)
        assert discriminated.shape == (2, 1)

    def test_lsgan_forward_pass(self, rngs, conv_generator_config, conv_discriminator_config):
        """Test LSGAN forward pass produces correct shapes."""
        config = LSGANConfig(
            name="test_lsgan_forward",
            generator=conv_generator_config,
            discriminator=conv_discriminator_config,
        )

        builder = GANBuilder()
        model = builder.build(config, rngs=rngs)

        # Test generation
        noise = jnp.ones((2, 100))
        generated = model.generator(noise)
        assert generated.shape == (2, 3, 32, 32)

        # Test discrimination
        images = jnp.ones((2, 3, 32, 32))
        discriminated = model.discriminator(images)
        assert discriminated.shape == (2, 1)

    def test_build_with_none_config(self, rngs):
        """Test that None config raises TypeError."""
        builder = GANBuilder()

        with pytest.raises(TypeError, match="config cannot be None"):
            builder.build(None, rngs=rngs)

    def test_build_with_dict_config(self, rngs):
        """Test that dict config raises TypeError."""
        builder = GANBuilder()
        config = {"name": "test", "latent_dim": 100}

        with pytest.raises(TypeError, match="config must be a dataclass"):
            builder.build(config, rngs=rngs)

    def test_build_with_invalid_config_type(self, rngs):
        """Test that unsupported config type raises TypeError."""
        builder = GANBuilder()

        class FakeConfig:
            pass

        fake_config = FakeConfig()

        with pytest.raises(TypeError, match="Unsupported config type"):
            builder.build(fake_config, rngs=rngs)

    def test_config_validation(self):
        """Test that GAN configs properly validate."""
        gen_config = ConvGeneratorConfig(
            name="valid_gen",
            latent_dim=100,
            hidden_dims=(128, 256),
            output_shape=(3, 32, 32),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        disc_config = ConvDiscriminatorConfig(
            name="valid_disc",
            input_shape=(3, 32, 32),
            hidden_dims=(64, 128),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        # Valid WGAN config
        valid_config = WGANConfig(
            name="valid_wgan",
            generator=gen_config,
            discriminator=disc_config,
            critic_iterations=5,
        )
        assert valid_config.critic_iterations == 5
        assert valid_config.gradient_penalty_weight == 10.0  # WGAN default

        # Valid LSGAN config
        valid_lsgan = LSGANConfig(
            name="valid_lsgan",
            generator=gen_config,
            discriminator=disc_config,
            a=0.0,
            b=1.0,
            c=1.0,
        )
        assert valid_lsgan.a == 0.0
        assert valid_lsgan.b == 1.0
        assert valid_lsgan.c == 1.0
