"""Tests for LSGANConfig frozen dataclass.

Following TDD principles: Tests define the expected behavior BEFORE implementation.
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration.gan_config import GANConfig, LSGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
)


class TestLSGANConfigBasics:
    """Test basic LSGANConfig creation and properties."""

    def test_lsgan_config_creation_minimal(self):
        """Test creating LSGANConfig with minimal required parameters."""
        generator = ConvGeneratorConfig(
            name="lsgan_gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512, 256, 128),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConvDiscriminatorConfig(
            name="lsgan_disc",
            input_shape=(32, 32, 3),
            hidden_dims=(128, 256, 512),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = LSGANConfig(
            name="test_lsgan",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.name == "test_lsgan"
        # LSGAN-specific defaults
        assert config.loss_type == "least_squares"
        assert config.a == 0.0
        assert config.b == 1.0
        assert config.c == 1.0

    def test_lsgan_config_inherits_from_gan_config(self):
        """Test that LSGANConfig properly inherits from GANConfig."""
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

        config = LSGANConfig(
            name="test",
            generator=generator,
            discriminator=discriminator,
        )

        assert isinstance(config, GANConfig)
        assert isinstance(config, LSGANConfig)

    def test_lsgan_config_is_frozen(self):
        """Test that LSGANConfig is immutable."""
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

        config = LSGANConfig(
            name="frozen_test",
            generator=generator,
            discriminator=discriminator,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.a = 0.5  # type: ignore

    def test_lsgan_config_with_all_parameters(self):
        """Test creating LSGANConfig with all parameters."""
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

        config = LSGANConfig(
            name="full_lsgan",
            description="Full LSGAN config",
            generator=generator,
            discriminator=discriminator,
            generator_lr=0.0002,
            discriminator_lr=0.0002,
            beta1=0.5,
            beta2=0.999,
            a=-1.0,
            b=1.0,
            c=0.0,
            tags=("test", "lsgan"),
        )

        assert config.name == "full_lsgan"
        assert config.a == -1.0
        assert config.b == 1.0
        assert config.c == 0.0


class TestLSGANConfigValidation:
    """Test LSGANConfig validation logic."""

    def test_loss_type_must_be_least_squares(self):
        """Test that loss_type is fixed to least_squares for LSGAN."""
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

        # Default should be least_squares
        config = LSGANConfig(
            name="test",
            generator=generator,
            discriminator=discriminator,
        )
        assert config.loss_type == "least_squares"

    def test_lsgan_requires_conv_generator_config(self):
        """Test that LSGANConfig requires ConvGeneratorConfig."""
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

        with pytest.raises(TypeError, match="LSGANConfig requires ConvGeneratorConfig"):
            LSGANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
            )

    def test_lsgan_requires_conv_discriminator_config(self):
        """Test that LSGANConfig requires ConvDiscriminatorConfig."""
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

        with pytest.raises(TypeError, match="LSGANConfig requires ConvDiscriminatorConfig"):
            LSGANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
            )


class TestLSGANConfigDefaults:
    """Test LSGANConfig default values."""

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

        config = LSGANConfig(
            name="defaults_test",
            generator=generator,
            discriminator=discriminator,
        )

        # Check LSGAN-specific defaults
        assert config.loss_type == "least_squares"
        assert config.a == 0.0
        assert config.b == 1.0
        assert config.c == 1.0
        assert config.generator_lr == 0.0002
        assert config.discriminator_lr == 0.0002
        assert config.beta1 == 0.5
        assert config.beta2 == 0.999


class TestLSGANConfigLossParameters:
    """Test LSGAN-specific loss parameters (a, b, c)."""

    def test_standard_lsgan_parameters(self):
        """Test standard LSGAN loss parameters (a=0, b=1, c=1)."""
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

        config = LSGANConfig(
            name="standard_lsgan",
            generator=generator,
            discriminator=discriminator,
            a=0.0,
            b=1.0,
            c=1.0,
        )

        assert config.a == 0.0
        assert config.b == 1.0
        assert config.c == 1.0

    def test_alternative_lsgan_parameters(self):
        """Test alternative LSGAN loss parameters (a=-1, b=1, c=0)."""
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

        config = LSGANConfig(
            name="alt_lsgan",
            generator=generator,
            discriminator=discriminator,
            a=-1.0,
            b=1.0,
            c=0.0,
        )

        assert config.a == -1.0
        assert config.b == 1.0
        assert config.c == 0.0

    def test_custom_lsgan_parameters(self):
        """Test that LSGAN allows any real number values for a, b, c."""
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

        config = LSGANConfig(
            name="custom_lsgan",
            generator=generator,
            discriminator=discriminator,
            a=-0.5,
            b=0.5,
            c=0.75,
        )

        assert config.a == -0.5
        assert config.b == 0.5
        assert config.c == 0.75

    def test_negative_values_allowed(self):
        """Test that negative values are allowed for a, b, c."""
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

        config = LSGANConfig(
            name="neg_lsgan",
            generator=generator,
            discriminator=discriminator,
            a=-2.0,
            b=-1.0,
            c=-0.5,
        )

        assert config.a == -2.0
        assert config.b == -1.0
        assert config.c == -0.5

    def test_zero_values_allowed(self):
        """Test that zero values are allowed for a, b, c."""
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

        config = LSGANConfig(
            name="zero_lsgan",
            generator=generator,
            discriminator=discriminator,
            a=0.0,
            b=0.0,
            c=0.0,
        )

        assert config.a == 0.0
        assert config.b == 0.0
        assert config.c == 0.0


class TestLSGANConfigSerialization:
    """Test LSGANConfig serialization to/from dict."""

    def test_from_dict_minimal(self):
        """Test creation from dictionary with minimal parameters."""
        config_dict = {
            "name": "test_lsgan",
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

        config = LSGANConfig.from_dict(config_dict)

        assert config.name == "test_lsgan"
        assert config.generator.latent_dim == 128
        assert config.discriminator.input_shape == (64, 64, 3)
        assert config.loss_type == "least_squares"

    def test_from_dict_with_all_parameters(self):
        """Test creation from dictionary with all parameters."""
        config_dict = {
            "name": "full_lsgan",
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
            "a": -1.0,
            "b": 1.0,
            "c": 0.0,
        }

        config = LSGANConfig.from_dict(config_dict)

        assert config.a == -1.0
        assert config.b == 1.0
        assert config.c == 0.0
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

        config = LSGANConfig(
            name="test_lsgan",
            generator=generator,
            discriminator=discriminator,
            a=-1.0,
            b=1.0,
            c=0.0,
        )

        config_dict = config.to_dict()

        assert config_dict["name"] == "test_lsgan"
        assert config_dict["a"] == -1.0
        assert config_dict["b"] == 1.0
        assert config_dict["c"] == 0.0
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

        original = LSGANConfig(
            name="test_lsgan",
            generator=generator,
            discriminator=discriminator,
            a=-0.5,
            b=0.5,
            c=0.75,
        )

        # Roundtrip
        config_dict = original.to_dict()
        restored = LSGANConfig.from_dict(config_dict)

        # Compare field by field (can't use == because of metadata dict)
        assert restored.name == original.name
        assert restored.a == original.a
        assert restored.b == original.b
        assert restored.c == original.c
        assert restored.generator.latent_dim == original.generator.latent_dim
        assert restored.discriminator.input_shape == original.discriminator.input_shape


class TestLSGANConfigEdgeCases:
    """Test LSGANConfig edge cases."""

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

        config = LSGANConfig(
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

        config = LSGANConfig(
            name="large_image",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.generator.output_shape == (256, 256, 3)
        assert config.discriminator.input_shape == (256, 256, 3)

    def test_custom_network_configurations(self):
        """Test LSGAN with custom generator and discriminator configs."""
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

        config = LSGANConfig(
            name="custom_lsgan",
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
