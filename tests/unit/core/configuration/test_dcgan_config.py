"""Tests for DCGANConfig - configuration for Deep Convolutional GAN.

DCGANConfig extends GANConfig with DCGAN-specific defaults and uses
ConvGeneratorConfig and ConvDiscriminatorConfig for network configurations.
"""

import pytest

from artifex.generative_models.core.configuration.gan_config import (
    DCGANConfig,
    GANConfig,
)
from artifex.generative_models.core.configuration.network_configs import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
)


class TestDCGANConfigBasics:
    """Test basic DCGANConfig functionality."""

    @pytest.fixture
    def generator_config(self):
        """Create a standard generator config for testing."""
        return ConvGeneratorConfig(
            name="dcgan_generator",
            latent_dim=100,
            output_shape=(3, 64, 64),
            hidden_dims=(512, 256, 128, 64),
            activation="relu",
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

    @pytest.fixture
    def discriminator_config(self):
        """Create a standard discriminator config for testing."""
        return ConvDiscriminatorConfig(
            name="dcgan_discriminator",
            input_shape=(3, 64, 64),
            hidden_dims=(64, 128, 256, 512),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

    def test_instantiation_with_conv_configs(self, generator_config, discriminator_config):
        """Test that DCGANConfig can be instantiated with ConvGeneratorConfig and ConvDiscriminatorConfig."""
        config = DCGANConfig(
            name="test_dcgan",
            generator=generator_config,
            discriminator=discriminator_config,
        )
        assert config.name == "test_dcgan"
        assert config.generator == generator_config
        assert config.discriminator == discriminator_config

    def test_is_frozen_dataclass(self, generator_config, discriminator_config):
        """Test that DCGANConfig is immutable."""
        config = DCGANConfig(
            name="frozen_test",
            generator=generator_config,
            discriminator=discriminator_config,
        )
        with pytest.raises(AttributeError):
            config.name = "modified"

    def test_inherits_from_gan_config(self, generator_config, discriminator_config):
        """Test that DCGANConfig inherits from GANConfig."""
        config = DCGANConfig(
            name="inheritance_test",
            generator=generator_config,
            discriminator=discriminator_config,
        )
        assert isinstance(config, GANConfig)


class TestDCGANConfigDefaults:
    """Test default values for DCGANConfig."""

    @pytest.fixture
    def generator_config(self):
        """Create a standard generator config for testing."""
        return ConvGeneratorConfig(
            name="dcgan_generator",
            latent_dim=100,
            output_shape=(3, 64, 64),
            hidden_dims=(512, 256, 128, 64),
            activation="relu",
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

    @pytest.fixture
    def discriminator_config(self):
        """Create a standard discriminator config for testing."""
        return ConvDiscriminatorConfig(
            name="dcgan_discriminator",
            input_shape=(3, 64, 64),
            hidden_dims=(64, 128, 256, 512),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

    def test_default_loss_type(self, generator_config, discriminator_config):
        """Test default loss_type is 'vanilla' for DCGAN."""
        config = DCGANConfig(
            name="defaults_test",
            generator=generator_config,
            discriminator=discriminator_config,
        )
        assert config.loss_type == "vanilla"

    def test_default_learning_rates(self, generator_config, discriminator_config):
        """Test default learning rates for DCGAN."""
        config = DCGANConfig(
            name="defaults_test",
            generator=generator_config,
            discriminator=discriminator_config,
        )
        assert config.generator_lr == 0.0002
        assert config.discriminator_lr == 0.0002

    def test_default_beta_values(self, generator_config, discriminator_config):
        """Test default Adam beta values for DCGAN."""
        config = DCGANConfig(
            name="defaults_test",
            generator=generator_config,
            discriminator=discriminator_config,
        )
        assert config.beta1 == 0.5
        assert config.beta2 == 0.999

    def test_can_override_defaults(self, generator_config, discriminator_config):
        """Test that default values can be overridden."""
        config = DCGANConfig(
            name="override_test",
            generator=generator_config,
            discriminator=discriminator_config,
            loss_type="hinge",
            generator_lr=0.0001,
            discriminator_lr=0.0004,
            beta1=0.0,
            beta2=0.9,
        )
        assert config.loss_type == "hinge"
        assert config.generator_lr == 0.0001
        assert config.discriminator_lr == 0.0004
        assert config.beta1 == 0.0
        assert config.beta2 == 0.9


class TestDCGANConfigValidation:
    """Test validation for DCGANConfig."""

    @pytest.fixture
    def generator_config(self):
        """Create a standard generator config for testing."""
        return ConvGeneratorConfig(
            name="dcgan_generator",
            latent_dim=100,
            output_shape=(3, 64, 64),
            hidden_dims=(512, 256, 128, 64),
            activation="relu",
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

    @pytest.fixture
    def discriminator_config(self):
        """Create a standard discriminator config for testing."""
        return ConvDiscriminatorConfig(
            name="dcgan_discriminator",
            input_shape=(3, 64, 64),
            hidden_dims=(64, 128, 256, 512),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

    def test_requires_single_generator(self, generator_config, discriminator_config):
        """Test that DCGANConfig requires a single GeneratorConfig, not a dict."""
        with pytest.raises(TypeError, match="single.*GeneratorConfig"):
            DCGANConfig(
                name="invalid",
                generator={"gen1": generator_config, "gen2": generator_config},
                discriminator=discriminator_config,
            )

    def test_requires_single_discriminator(self, generator_config, discriminator_config):
        """Test that DCGANConfig requires a single DiscriminatorConfig, not a dict."""
        with pytest.raises(TypeError, match="single.*DiscriminatorConfig"):
            DCGANConfig(
                name="invalid",
                generator=generator_config,
                discriminator={"disc1": discriminator_config, "disc2": discriminator_config},
            )

    def test_accepts_conv_generator_config(self, generator_config, discriminator_config):
        """Test that DCGANConfig accepts ConvGeneratorConfig (subclass of GeneratorConfig)."""
        config = DCGANConfig(
            name="conv_test",
            generator=generator_config,
            discriminator=discriminator_config,
        )
        assert isinstance(config.generator, ConvGeneratorConfig)

    def test_accepts_conv_discriminator_config(self, generator_config, discriminator_config):
        """Test that DCGANConfig accepts ConvDiscriminatorConfig (subclass of DiscriminatorConfig)."""
        config = DCGANConfig(
            name="conv_test",
            generator=generator_config,
            discriminator=discriminator_config,
        )
        assert isinstance(config.discriminator, ConvDiscriminatorConfig)


class TestDCGANConfigSerialization:
    """Test serialization for DCGANConfig."""

    @pytest.fixture
    def generator_config(self):
        """Create a standard generator config for testing."""
        return ConvGeneratorConfig(
            name="dcgan_generator",
            latent_dim=100,
            output_shape=(3, 64, 64),
            hidden_dims=(512, 256, 128, 64),
            activation="relu",
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

    @pytest.fixture
    def discriminator_config(self):
        """Create a standard discriminator config for testing."""
        return ConvDiscriminatorConfig(
            name="dcgan_discriminator",
            input_shape=(3, 64, 64),
            hidden_dims=(64, 128, 256, 512),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

    def test_to_dict(self, generator_config, discriminator_config):
        """Test converting config to dictionary."""
        config = DCGANConfig(
            name="serialization_test",
            generator=generator_config,
            discriminator=discriminator_config,
        )
        data = config.to_dict()
        assert data["name"] == "serialization_test"
        assert data["loss_type"] == "vanilla"
        assert "generator" in data
        assert "discriminator" in data

    def test_from_dict(self, generator_config, discriminator_config):
        """Test creating config from dictionary."""
        original = DCGANConfig(
            name="from_dict_test",
            generator=generator_config,
            discriminator=discriminator_config,
        )
        data = original.to_dict()
        restored = DCGANConfig.from_dict(data)
        assert restored.name == "from_dict_test"
        assert restored.loss_type == "vanilla"

    def test_roundtrip_serialization(self, generator_config, discriminator_config):
        """Test that to_dict -> from_dict preserves all values."""
        original = DCGANConfig(
            name="roundtrip_test",
            generator=generator_config,
            discriminator=discriminator_config,
            generator_lr=0.0001,
            discriminator_lr=0.0004,
            beta1=0.0,
            beta2=0.9,
        )
        data = original.to_dict()
        restored = DCGANConfig.from_dict(data)
        assert restored == original


class TestDCGANConfigIntegration:
    """Test DCGANConfig integration with the config system."""

    def test_creates_valid_config_for_mnist(self):
        """Test creating a valid DCGANConfig for MNIST-like images."""
        generator = ConvGeneratorConfig(
            name="mnist_generator",
            latent_dim=100,
            output_shape=(1, 28, 28),
            hidden_dims=(256, 128, 64),
            activation="relu",
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        discriminator = ConvDiscriminatorConfig(
            name="mnist_discriminator",
            input_shape=(1, 28, 28),
            hidden_dims=(64, 128, 256),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = DCGANConfig(
            name="mnist_dcgan",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.generator.latent_dim == 100
        assert config.generator.output_shape == (1, 28, 28)
        assert config.discriminator.input_shape == (1, 28, 28)

    def test_creates_valid_config_for_cifar10(self):
        """Test creating a valid DCGANConfig for CIFAR-10 images."""
        generator = ConvGeneratorConfig(
            name="cifar_generator",
            latent_dim=128,
            output_shape=(3, 32, 32),
            hidden_dims=(512, 256, 128),
            activation="relu",
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        discriminator = ConvDiscriminatorConfig(
            name="cifar_discriminator",
            input_shape=(3, 32, 32),
            hidden_dims=(128, 256, 512),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = DCGANConfig(
            name="cifar_dcgan",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.generator.latent_dim == 128
        assert config.generator.output_shape == (3, 32, 32)
        assert config.discriminator.input_shape == (3, 32, 32)

    def test_generator_output_matches_discriminator_input(self):
        """Test that generator output shape should match discriminator input shape."""
        output_shape = (3, 64, 64)

        generator = ConvGeneratorConfig(
            name="matched_generator",
            latent_dim=100,
            output_shape=output_shape,
            hidden_dims=(512, 256, 128, 64),
            activation="relu",
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        discriminator = ConvDiscriminatorConfig(
            name="matched_discriminator",
            input_shape=output_shape,  # Same as generator output
            hidden_dims=(64, 128, 256, 512),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=True,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = DCGANConfig(
            name="matched_dcgan",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.generator.output_shape == config.discriminator.input_shape
