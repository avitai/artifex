"""Tests for ConditionalGANConfig frozen dataclass.

Following TDD principles: Tests define the expected behavior BEFORE implementation.
ConditionalGANConfig uses composition pattern with ConditionalParams embedded
in ConditionalGeneratorConfig and ConditionalDiscriminatorConfig.
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration.gan_config import (
    ConditionalGANConfig,
    GANConfig,
)
from artifex.generative_models.core.configuration.network_configs import (
    ConditionalDiscriminatorConfig,
    ConditionalGeneratorConfig,
    ConditionalParams,
    GeneratorConfig,
)


class TestConditionalGANConfigBasics:
    """Test basic ConditionalGANConfig creation and properties."""

    def test_conditional_gan_config_creation_minimal(self):
        """Test creating ConditionalGANConfig with minimal required parameters."""
        cond_params = ConditionalParams(num_classes=10, embedding_dim=50)

        generator = ConditionalGeneratorConfig(
            name="cgan_gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512, 256, 128),
            activation="relu",
            conditional=cond_params,
        )
        discriminator = ConditionalDiscriminatorConfig(
            name="cgan_disc",
            input_shape=(32, 32, 3),
            hidden_dims=(128, 256, 512),
            activation="leaky_relu",
            conditional=cond_params,
        )

        config = ConditionalGANConfig(
            name="test_cgan",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.name == "test_cgan"
        # Access conditional params via composition
        assert config.generator.conditional.num_classes == 10
        assert config.generator.conditional.embedding_dim == 50

    def test_conditional_gan_config_inherits_from_gan_config(self):
        """Test that ConditionalGANConfig properly inherits from GANConfig."""
        cond_params = ConditionalParams(num_classes=10)

        generator = ConditionalGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            conditional=cond_params,
        )
        discriminator = ConditionalDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            conditional=cond_params,
        )

        config = ConditionalGANConfig(
            name="test",
            generator=generator,
            discriminator=discriminator,
        )

        assert isinstance(config, GANConfig)
        assert isinstance(config, ConditionalGANConfig)

    def test_conditional_gan_config_is_frozen(self):
        """Test that ConditionalGANConfig is immutable."""
        cond_params = ConditionalParams(num_classes=10)

        generator = ConditionalGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            conditional=cond_params,
        )
        discriminator = ConditionalDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            conditional=cond_params,
        )

        config = ConditionalGANConfig(
            name="frozen_test",
            generator=generator,
            discriminator=discriminator,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "modified"  # type: ignore

    def test_conditional_gan_config_with_all_parameters(self):
        """Test creating ConditionalGANConfig with all parameters."""
        cond_params = ConditionalParams(num_classes=100, embedding_dim=100)

        generator = ConditionalGeneratorConfig(
            name="full_gen",
            latent_dim=128,
            output_shape=(64, 64, 3),
            hidden_dims=(512, 256),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
            conditional=cond_params,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        discriminator = ConditionalDiscriminatorConfig(
            name="full_disc",
            input_shape=(64, 64, 3),
            hidden_dims=(256, 512),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            dropout_rate=0.0,
            conditional=cond_params,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        config = ConditionalGANConfig(
            name="full_cgan",
            description="Full Conditional GAN config",
            generator=generator,
            discriminator=discriminator,
            generator_lr=0.0002,
            discriminator_lr=0.0002,
            beta1=0.5,
            beta2=0.999,
            tags=("test", "cgan"),
        )

        assert config.name == "full_cgan"
        assert config.generator.conditional.num_classes == 100
        assert config.generator.conditional.embedding_dim == 100


class TestConditionalGANConfigValidation:
    """Test ConditionalGANConfig validation logic."""

    def test_cgan_requires_single_generator(self):
        """Test that ConditionalGANConfig requires single generator config, not dict."""
        cond_params = ConditionalParams(num_classes=10)

        discriminator = ConditionalDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            conditional=cond_params,
        )

        with pytest.raises(
            TypeError,
            match="ConditionalGANConfig requires a single ConditionalGeneratorConfig",
        ):
            ConditionalGANConfig(
                name="test",
                generator={  # type: ignore
                    "gen1": ConditionalGeneratorConfig(
                        name="g1",
                        latent_dim=100,
                        output_shape=(32, 32, 3),
                        hidden_dims=(512,),
                        activation="relu",
                        conditional=cond_params,
                    )
                },
                discriminator=discriminator,
            )

    def test_cgan_requires_single_discriminator(self):
        """Test that ConditionalGANConfig requires single discriminator config, not dict."""
        cond_params = ConditionalParams(num_classes=10)

        generator = ConditionalGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            conditional=cond_params,
        )

        with pytest.raises(
            TypeError,
            match="ConditionalGANConfig requires a single ConditionalDiscriminatorConfig",
        ):
            ConditionalGANConfig(
                name="test",
                generator=generator,
                discriminator={  # type: ignore
                    "disc1": ConditionalDiscriminatorConfig(
                        name="d1",
                        input_shape=(32, 32, 3),
                        hidden_dims=(512,),
                        activation="leaky_relu",
                        conditional=cond_params,
                    )
                },
            )

    def test_cgan_requires_conditional_generator_config(self):
        """Test that ConditionalGANConfig requires ConditionalGeneratorConfig type."""
        cond_params = ConditionalParams(num_classes=10)

        # Use base GeneratorConfig instead of ConditionalGeneratorConfig
        generator = GeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
        )
        discriminator = ConditionalDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            conditional=cond_params,
        )

        with pytest.raises(
            TypeError, match="ConditionalGANConfig requires ConditionalGeneratorConfig"
        ):
            ConditionalGANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
            )

    def test_num_classes_must_be_positive(self):
        """Test that num_classes in ConditionalParams must be positive."""
        with pytest.raises(ValueError, match="num_classes.*positive"):
            ConditionalParams(num_classes=0)

        with pytest.raises(ValueError, match="num_classes.*positive"):
            ConditionalParams(num_classes=-10)

    def test_embedding_dim_must_be_positive(self):
        """Test that embedding_dim in ConditionalParams must be positive."""
        with pytest.raises(ValueError, match="embedding_dim.*positive"):
            ConditionalParams(num_classes=10, embedding_dim=0)

        with pytest.raises(ValueError, match="embedding_dim.*positive"):
            ConditionalParams(num_classes=10, embedding_dim=-50)


class TestConditionalGANConfigDefaults:
    """Test ConditionalGANConfig default values."""

    def test_default_values(self):
        """Test that default values are applied correctly."""
        cond_params = ConditionalParams(num_classes=10)

        generator = ConditionalGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            conditional=cond_params,
        )
        discriminator = ConditionalDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            conditional=cond_params,
        )

        config = ConditionalGANConfig(
            name="defaults_test",
            generator=generator,
            discriminator=discriminator,
        )

        # Check default ConditionalParams values
        assert config.generator.conditional.embedding_dim == 100  # Default

        # Check GANConfig defaults
        assert config.generator_lr == 0.0002
        assert config.discriminator_lr == 0.0002
        assert config.beta1 == 0.5
        assert config.beta2 == 0.999


class TestConditionalGANConfigSerialization:
    """Test ConditionalGANConfig serialization to/from dict."""

    def test_to_dict_preserves_structure(self):
        """Test that to_dict preserves the nested structure."""
        cond_params = ConditionalParams(num_classes=10, embedding_dim=50)

        generator = ConditionalGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512, 256),
            activation="relu",
            conditional=cond_params,
        )
        discriminator = ConditionalDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(256, 512),
            activation="leaky_relu",
            conditional=cond_params,
        )

        config = ConditionalGANConfig(
            name="test_cgan",
            generator=generator,
            discriminator=discriminator,
        )

        config_dict = config.to_dict()

        assert config_dict["name"] == "test_cgan"
        assert isinstance(config_dict["generator"], dict)
        assert isinstance(config_dict["discriminator"], dict)
        assert config_dict["generator"]["latent_dim"] == 100
        assert config_dict["generator"]["conditional"]["num_classes"] == 10


class TestConditionalGANConfigEdgeCases:
    """Test ConditionalGANConfig edge cases."""

    def test_binary_classification(self):
        """Test config with 2 classes (binary)."""
        cond_params = ConditionalParams(num_classes=2)

        generator = ConditionalGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            conditional=cond_params,
        )
        discriminator = ConditionalDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            conditional=cond_params,
        )

        config = ConditionalGANConfig(
            name="binary",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.generator.conditional.num_classes == 2

    def test_many_classes(self):
        """Test config with many classes."""
        cond_params = ConditionalParams(num_classes=1000)

        generator = ConditionalGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            conditional=cond_params,
        )
        discriminator = ConditionalDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            conditional=cond_params,
        )

        config = ConditionalGANConfig(
            name="many_classes",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.generator.conditional.num_classes == 1000

    def test_small_embedding_dim(self):
        """Test config with small embedding dimension."""
        cond_params = ConditionalParams(num_classes=10, embedding_dim=5)

        generator = ConditionalGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            conditional=cond_params,
        )
        discriminator = ConditionalDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            conditional=cond_params,
        )

        config = ConditionalGANConfig(
            name="small_embed",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.generator.conditional.embedding_dim == 5

    def test_large_embedding_dim(self):
        """Test config with large embedding dimension."""
        cond_params = ConditionalParams(num_classes=10, embedding_dim=512)

        generator = ConditionalGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="relu",
            conditional=cond_params,
        )
        discriminator = ConditionalDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(512,),
            activation="leaky_relu",
            conditional=cond_params,
        )

        config = ConditionalGANConfig(
            name="large_embed",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.generator.conditional.embedding_dim == 512

    def test_different_gen_disc_architectures(self):
        """Test config with different generator and discriminator architectures."""
        cond_params = ConditionalParams(num_classes=10)

        generator = ConditionalGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(32, 32, 3),
            hidden_dims=(1024, 512, 256),
            activation="relu",
            conditional=cond_params,
        )
        discriminator = ConditionalDiscriminatorConfig(
            name="disc",
            input_shape=(32, 32, 3),
            hidden_dims=(64, 128, 256, 512),
            activation="leaky_relu",
            conditional=cond_params,
        )

        config = ConditionalGANConfig(
            name="asymmetric",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.generator.hidden_dims == (1024, 512, 256)
        assert config.discriminator.hidden_dims == (64, 128, 256, 512)

    def test_grayscale_conditional_gan(self):
        """Test conditional GAN for grayscale images (e.g., MNIST)."""
        cond_params = ConditionalParams(num_classes=10)

        generator = ConditionalGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(512, 256),
            activation="relu",
            conditional=cond_params,
        )
        discriminator = ConditionalDiscriminatorConfig(
            name="disc",
            input_shape=(28, 28, 1),
            hidden_dims=(256, 512),
            activation="leaky_relu",
            conditional=cond_params,
        )

        config = ConditionalGANConfig(
            name="mnist_cgan",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.generator.output_shape == (28, 28, 1)
        assert config.discriminator.input_shape == (28, 28, 1)
        assert config.generator.conditional.num_classes == 10
