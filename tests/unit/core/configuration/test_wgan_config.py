"""Tests for WGANConfig frozen dataclass.

Following TDD principles: Tests define the expected behavior BEFORE implementation.
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration.gan_config import GANConfig, WGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
)


class TestWGANConfigBasics:
    """Test basic WGANConfig creation and properties."""

    def test_wgan_config_creation_minimal(self):
        """Test creating WGANConfig with minimal required parameters."""
        generator = ConvGeneratorConfig(
            name="wgan_gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512, 256, 128),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="wgan_disc",
            input_shape=(32, 32, 3),
            hidden_dims=(128, 256, 512),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = WGANConfig(
            name="test_wgan",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.name == "test_wgan"
        # WGAN-specific defaults
        assert config.loss_type == "wasserstein"
        assert config.gradient_penalty_weight == 10.0
        assert config.critic_iterations == 5
        assert config.use_gradient_penalty is True

    def test_wgan_config_inherits_from_gan_config(self):
        """Test that WGANConfig properly inherits from GANConfig."""
        generator = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = WGANConfig(
            name="test",
            generator=generator,
            discriminator=discriminator,
        )

        assert isinstance(config, GANConfig)
        assert isinstance(config, WGANConfig)

    def test_wgan_config_is_frozen(self):
        """Test that WGANConfig is immutable."""
        generator = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = WGANConfig(
            name="frozen_test",
            generator=generator,
            discriminator=discriminator,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.critic_iterations = 10  # type: ignore

    def test_wgan_config_with_all_parameters(self):
        """Test creating WGANConfig with all parameters."""
        generator = ConvGeneratorConfig(
            name="full_gen",
            latent_dim=128,
            output_shape=(64, 64, 3),
            hidden_dims=(512, 256),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="full_disc",
            input_shape=(64, 64, 3),
            hidden_dims=(256, 512),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            dropout_rate=0.0,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = WGANConfig(
            name="full_wgan",
            description="Full WGAN config",
            generator=generator,
            discriminator=discriminator,
            generator_lr=0.0001,
            discriminator_lr=0.0002,
            beta1=0.0,
            beta2=0.9,
            gradient_penalty_weight=10.0,
            critic_iterations=5,
            use_gradient_penalty=True,
            tags=("test", "wgan"),
        )

        assert config.name == "full_wgan"
        assert config.critic_iterations == 5
        assert config.use_gradient_penalty is True


class TestWGANConfigValidation:
    """Test WGANConfig validation logic."""

    def test_critic_iterations_must_be_positive(self):
        """Test that critic_iterations must be positive."""
        generator = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        with pytest.raises(ValueError, match="critic_iterations.*positive"):
            WGANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
                critic_iterations=0,
            )

        with pytest.raises(ValueError, match="critic_iterations.*positive"):
            WGANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
                critic_iterations=-5,
            )

    def test_loss_type_must_be_wasserstein(self):
        """Test that loss_type is fixed to wasserstein for WGAN."""
        generator = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        # Default should be wasserstein
        config = WGANConfig(
            name="test",
            generator=generator,
            discriminator=discriminator,
        )
        assert config.loss_type == "wasserstein"

    def test_wgan_requires_conv_generator_config(self):
        """Test that WGANConfig requires ConvGeneratorConfig."""
        from artifex.generative_models.core.configuration.network_configs import GeneratorConfig

        generator = GeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
        )
        discriminator = ConvDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        with pytest.raises(TypeError, match="WGANConfig requires ConvGeneratorConfig"):
            WGANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
            )

    def test_wgan_requires_conv_discriminator_config(self):
        """Test that WGANConfig requires ConvDiscriminatorConfig."""
        from artifex.generative_models.core.configuration.network_configs import (
            DiscriminatorConfig,
        )

        generator = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = DiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
        )

        with pytest.raises(TypeError, match="WGANConfig requires ConvDiscriminatorConfig"):
            WGANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
            )


class TestWGANConfigDefaults:
    """Test WGANConfig default values."""

    def test_default_values(self):
        """Test that default values are applied correctly."""
        generator = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = WGANConfig(
            name="defaults_test",
            generator=generator,
            discriminator=discriminator,
        )

        # Check WGAN-specific defaults
        assert config.loss_type == "wasserstein"
        assert config.gradient_penalty_weight == 10.0
        assert config.critic_iterations == 5
        assert config.use_gradient_penalty is True
        assert config.generator_lr == 0.0001
        assert config.discriminator_lr == 0.0001
        assert config.beta1 == 0.0
        assert config.beta2 == 0.9


class TestWGANConfigSerialization:
    """Test WGANConfig serialization to/from dict."""

    def test_from_dict_minimal(self):
        """Test creation from dictionary with minimal parameters."""
        config_dict = {
            "name": "test_wgan",
            "generator": {
                "name": "gen",
                "latent_dim": 128,
                "output_shape": [64, 64, 3],
                "hidden_dims": [512, 256],
                "activation": "relu",
                "kernel_size": [4, 4],
                "stride": [2, 2],
                "padding": "SAME",
            },
            "discriminator": {
                "name": "disc",
                "input_shape": [64, 64, 3],
                "hidden_dims": [256, 512],
                "activation": "leaky_relu",
                "kernel_size": [4, 4],
                "stride": [2, 2],
                "padding": "SAME",
            },
        }

        config = WGANConfig.from_dict(config_dict)

        assert config.name == "test_wgan"
        assert config.generator.latent_dim == 128
        assert config.discriminator.input_shape == (64, 64, 3)
        assert config.loss_type == "wasserstein"

    def test_from_dict_with_all_parameters(self):
        """Test creation from dictionary with all parameters."""
        config_dict = {
            "name": "full_wgan",
            "generator": {
                "name": "gen",
                "latent_dim": 256,
                "output_shape": [64, 64, 1],
                "hidden_dims": [512, 256],
                "activation": "gelu",
                "kernel_size": [4, 4],
                "stride": [2, 2],
                "padding": "SAME",
            },
            "discriminator": {
                "name": "disc",
                "input_shape": [64, 64, 1],
                "hidden_dims": [256, 512],
                "activation": "relu",
                "kernel_size": [4, 4],
                "stride": [2, 2],
                "padding": "SAME",
            },
            "critic_iterations": 10,
            "use_gradient_penalty": False,
        }

        config = WGANConfig.from_dict(config_dict)

        assert config.critic_iterations == 10
        assert config.use_gradient_penalty is False
        assert config.generator.activation == "gelu"
        assert config.discriminator.activation == "relu"

    def test_to_dict_preserves_structure(self):
        """Test that to_dict preserves the nested structure."""
        generator = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512, 256),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(256, 512),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = WGANConfig(
            name="test_wgan",
            generator=generator,
            discriminator=discriminator,
            critic_iterations=10,
        )

        config_dict = config.to_dict()

        assert config_dict["name"] == "test_wgan"
        assert config_dict["critic_iterations"] == 10
        assert isinstance(config_dict["generator"], dict)
        assert isinstance(config_dict["discriminator"], dict)
        assert config_dict["generator"]["latent_dim"] == 100
        assert config_dict["discriminator"]["input_shape"] == (32, 32, 3)

    def test_roundtrip_serialization(self):
        """Test that from_dict(to_dict(config)) == config."""
        generator = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512, 256),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(256, 512),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        original = WGANConfig(
            name="test_wgan",
            generator=generator,
            discriminator=discriminator,
            critic_iterations=7,
        )

        # Roundtrip
        config_dict = original.to_dict()
        restored = WGANConfig.from_dict(config_dict)

        # Compare field by field (can't use == because of metadata dict)
        assert restored.name == original.name
        assert restored.critic_iterations == original.critic_iterations
        assert restored.use_gradient_penalty == original.use_gradient_penalty
        assert restored.generator.latent_dim == original.generator.latent_dim
        assert restored.discriminator.input_shape == original.discriminator.input_shape


class TestWGANConfigEdgeCases:
    """Test WGANConfig edge cases."""

    def test_single_channel_image(self):
        """Test config with single channel (grayscale)."""
        generator = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(512,),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="disc",
            input_shape=(28, 28, 1),
            hidden_dims=(512,),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = WGANConfig(
            name="grayscale",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.generator.output_shape == (28, 28, 1)
        assert config.discriminator.input_shape == (28, 28, 1)

    def test_large_image_size(self):
        """Test config with large image size."""
        generator = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(256, 256, 3),
            hidden_dims=(512,),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="disc",
            input_shape=(256, 256, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = WGANConfig(
            name="large_image",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.generator.output_shape == (256, 256, 3)
        assert config.discriminator.input_shape == (256, 256, 3)

    def test_high_critic_iterations(self):
        """Test config with high critic iterations."""
        generator = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = WGANConfig(
            name="high_critic",
            generator=generator,
            discriminator=discriminator,
            critic_iterations=100,
        )

        assert config.critic_iterations == 100

    def test_wgan_without_gradient_penalty(self):
        """Test WGAN without gradient penalty (original WGAN, not WGAN-GP)."""
        generator = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = WGANConfig(
            name="wgan_no_gp",
            generator=generator,
            discriminator=discriminator,
            use_gradient_penalty=False,
            gradient_penalty_weight=0.0,
        )

        assert config.use_gradient_penalty is False
        assert config.gradient_penalty_weight == 0.0

    def test_wgan_specific_learning_rates(self):
        """Test that WGAN uses different learning rates from vanilla GAN."""
        generator = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = WGANConfig(
            name="test",
            generator=generator,
            discriminator=discriminator,
        )

        # WGAN typically uses lower learning rates
        assert config.generator_lr == 0.0001
        assert config.discriminator_lr == 0.0001
        # WGAN typically uses different beta values
        assert config.beta1 == 0.0
        assert config.beta2 == 0.9

    def test_custom_network_configurations(self):
        """Test WGAN with custom generator and discriminator configs."""
        generator = ConvGeneratorConfig(
            name="custom_gen",
            latent_dim=128,
            output_shape=(64, 64, 3),
            hidden_dims=(1024, 512, 256, 128),
            activation="gelu",
            batch_norm=True,
            dropout_rate=0.1,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="custom_disc",
            input_shape=(64, 64, 3),
            hidden_dims=(128, 256, 512, 1024),
            activation="silu",
            leaky_relu_slope=0.3,
            use_spectral_norm=True,
            dropout_rate=0.2,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = WGANConfig(
            name="custom_wgan",
            generator=generator,
            discriminator=discriminator,
        )

        # Verify generator config
        assert config.generator.latent_dim == 128
        assert config.generator.hidden_dims == (1024, 512, 256, 128)
        assert config.generator.activation == "gelu"
        assert config.generator.batch_norm is True
        assert config.generator.dropout_rate == 0.1

        # Verify discriminator config
        assert config.discriminator.input_shape == (64, 64, 3)
        assert config.discriminator.hidden_dims == (128, 256, 512, 1024)
        assert config.discriminator.activation == "silu"
        assert config.discriminator.use_spectral_norm is True
        assert config.discriminator.dropout_rate == 0.2
