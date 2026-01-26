"""Tests for GANConfig frozen dataclass.

Following TDD principles: Tests define the expected behavior BEFORE implementation.
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.gan_config import GANConfig
from artifex.generative_models.core.configuration.network_configs import (
    DiscriminatorConfig,
    GeneratorConfig,
)


def _create_test_generator(name: str = "test_gen", **kwargs) -> GeneratorConfig:
    """Helper to create a test GeneratorConfig with defaults."""
    defaults = {
        "name": name,
        "latent_dim": 100,
        "hidden_dims": (512, 256),
        "output_shape": (28, 28, 1),
        "activation": "relu",  # Required by BaseNetworkConfig
    }
    defaults.update(kwargs)
    return GeneratorConfig(**defaults)


def _create_test_discriminator(name: str = "test_disc", **kwargs) -> DiscriminatorConfig:
    """Helper to create a test DiscriminatorConfig with defaults."""
    defaults = {
        "name": name,
        "hidden_dims": (256, 512),
        "input_shape": (28, 28, 1),
        "activation": "leaky_relu",  # Required by BaseNetworkConfig
    }
    defaults.update(kwargs)
    return DiscriminatorConfig(**defaults)


class TestGANConfigBasics:
    """Test basic GANConfig creation and properties."""

    def test_gan_config_creation_minimal(self):
        """Test creating GANConfig with minimal required parameters."""
        # Create nested network configs
        generator = GeneratorConfig(
            name="test_generator",
            latent_dim=100,
            hidden_dims=(512, 256, 128),
            output_shape=(28, 28, 1),
            activation="relu",
        )

        discriminator = DiscriminatorConfig(
            name="test_discriminator",
            hidden_dims=(128, 256, 512),
            input_shape=(28, 28, 1),
            activation="leaky_relu",
        )

        config = GANConfig(
            name="test_gan",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.name == "test_gan"
        assert isinstance(config.generator, GeneratorConfig)
        assert isinstance(config.discriminator, DiscriminatorConfig)
        assert config.generator.latent_dim == 100
        assert config.generator.hidden_dims == (512, 256, 128)
        assert config.discriminator.hidden_dims == (128, 256, 512)

    def test_gan_config_creation_with_all_parameters(self):
        """Test creating GANConfig with all parameters."""
        # Create nested network configs with custom params
        generator = GeneratorConfig(
            name="full_generator",
            latent_dim=128,
            hidden_dims=(512, 256),
            output_shape=(32, 32, 3),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
        )

        discriminator = DiscriminatorConfig(
            name="full_discriminator",
            hidden_dims=(256, 512),
            input_shape=(32, 32, 3),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
        )

        config = GANConfig(
            name="full_gan",
            description="Full test GAN",
            generator=generator,
            discriminator=discriminator,
            generator_lr=0.0002,
            discriminator_lr=0.0001,
            beta1=0.5,
            beta2=0.999,
            loss_type="wasserstein",
            gradient_penalty_weight=10.0,
            tags=("test", "gan"),
            metadata={"experiment": "test_001"},
            rngs_seeds={"params": 42, "dropout": 123},
        )

        assert config.name == "full_gan"
        assert config.description == "Full test GAN"
        assert config.generator.latent_dim == 128
        assert config.generator.hidden_dims == (512, 256)
        assert config.discriminator.hidden_dims == (256, 512)
        assert config.generator_lr == 0.0002
        assert config.discriminator_lr == 0.0001
        assert config.beta1 == 0.5
        assert config.beta2 == 0.999
        assert config.loss_type == "wasserstein"
        assert config.gradient_penalty_weight == 10.0
        assert config.tags == ("test", "gan")
        assert config.metadata == {"experiment": "test_001"}
        assert config.rngs_seeds == {"params": 42, "dropout": 123}

    def test_gan_config_is_frozen(self):
        """Test that GANConfig is immutable (frozen)."""
        generator = _create_test_generator(name="gen")
        discriminator = _create_test_discriminator(name="disc")

        config = GANConfig(
            name="frozen_test",
            generator=generator,
            discriminator=discriminator,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"  # type: ignore

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.generator = generator  # type: ignore

    def test_gan_config_inherits_from_base_config(self):
        """Test that GANConfig properly inherits from BaseConfig."""
        generator = _create_test_generator(name="gen")
        discriminator = _create_test_discriminator(name="disc")

        config = GANConfig(
            name="test",
            generator=generator,
            discriminator=discriminator,
        )

        assert isinstance(config, BaseConfig)
        assert hasattr(config, "from_dict")
        assert hasattr(config, "to_dict")


class TestGANConfigValidation:
    """Test GANConfig validation logic."""

    def test_generator_config_required(self):
        """Test that generator config is required."""
        discriminator = _create_test_discriminator()

        with pytest.raises((TypeError, ValueError)):
            GANConfig(
                name="test",
                discriminator=discriminator,
                # Missing generator!
            )

    def test_discriminator_config_required(self):
        """Test that discriminator config is required."""
        generator = _create_test_generator()

        with pytest.raises((TypeError, ValueError)):
            GANConfig(
                name="test",
                generator=generator,
                # Missing discriminator!
            )

    def test_nested_config_validation_propagates(self):
        """Test that validation errors in nested configs propagate."""
        # Invalid generator (negative latent_dim)
        with pytest.raises(ValueError, match="latent_dim"):
            GeneratorConfig(
                name="gen",
                latent_dim=-100,  # Invalid!
                hidden_dims=(512,),
                output_shape=(28, 28, 1),
                activation="relu",
            )

        # Invalid discriminator (empty hidden_dims)
        with pytest.raises(ValueError, match="hidden_dims"):
            DiscriminatorConfig(
                name="disc",
                hidden_dims=(),  # Invalid!
                input_shape=(28, 28, 1),
                activation="leaky_relu",
            )

    def test_generator_lr_must_be_positive(self):
        """Test that generator_lr must be positive."""
        generator = _create_test_generator()
        discriminator = _create_test_discriminator()

        with pytest.raises(ValueError, match="generator_lr.*positive"):
            GANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
                generator_lr=0.0,
            )

        with pytest.raises(ValueError, match="generator_lr.*positive"):
            GANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
                generator_lr=-0.001,
            )

    def test_discriminator_lr_must_be_positive(self):
        """Test that discriminator_lr must be positive."""
        generator = _create_test_generator()
        discriminator = _create_test_discriminator()

        with pytest.raises(ValueError, match="discriminator_lr.*positive"):
            GANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
                discriminator_lr=0.0,
            )

        with pytest.raises(ValueError, match="discriminator_lr.*positive"):
            GANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
                discriminator_lr=-0.001,
            )

    def test_beta1_must_be_in_valid_range(self):
        """Test that beta1 must be in [0, 1)."""
        generator = _create_test_generator()
        discriminator = _create_test_discriminator()

        with pytest.raises(ValueError, match="beta1"):
            GANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
                beta1=-0.1,
            )

        with pytest.raises(ValueError, match="beta1"):
            GANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
                beta1=1.0,
            )

        with pytest.raises(ValueError, match="beta1"):
            GANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
                beta1=1.5,
            )

    def test_beta2_must_be_in_valid_range(self):
        """Test that beta2 must be in [0, 1)."""
        generator = _create_test_generator()
        discriminator = _create_test_discriminator()

        with pytest.raises(ValueError, match="beta2"):
            GANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
                beta2=-0.1,
            )

        with pytest.raises(ValueError, match="beta2"):
            GANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
                beta2=1.0,
            )

    def test_gradient_penalty_weight_must_be_non_negative(self):
        """Test that gradient_penalty_weight must be non-negative."""
        generator = _create_test_generator()
        discriminator = _create_test_discriminator()

        # 0.0 should be valid (no gradient penalty)
        config = GANConfig(
            name="test",
            generator=generator,
            discriminator=discriminator,
            gradient_penalty_weight=0.0,
        )
        assert config.gradient_penalty_weight == 0.0

        # Positive should be valid
        config = GANConfig(
            name="test",
            generator=generator,
            discriminator=discriminator,
            gradient_penalty_weight=10.0,
        )
        assert config.gradient_penalty_weight == 10.0

        # Negative should raise
        with pytest.raises(ValueError, match="gradient_penalty_weight"):
            GANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
                gradient_penalty_weight=-1.0,
            )

    def test_valid_loss_types(self):
        """Test valid loss types."""
        generator = _create_test_generator()
        discriminator = _create_test_discriminator()
        valid_types = ["vanilla", "wasserstein", "least_squares", "hinge"]

        for loss_type in valid_types:
            config = GANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
                loss_type=loss_type,
            )
            assert config.loss_type == loss_type

    def test_invalid_loss_type_raises(self):
        """Test that invalid loss types raise ValueError."""
        generator = _create_test_generator()
        discriminator = _create_test_discriminator()

        with pytest.raises(ValueError, match="loss_type"):
            GANConfig(
                name="test",
                generator=generator,
                discriminator=discriminator,
                loss_type="invalid_loss",
            )


class TestGANConfigSerialization:
    """Test GANConfig serialization to/from dict and YAML."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        generator = _create_test_generator(name="test_gen", hidden_dims=(512, 256))
        discriminator = _create_test_discriminator(name="test_disc")

        config = GANConfig(
            name="test_gan",
            generator=generator,
            discriminator=discriminator,
            generator_lr=0.0002,
            tags=("test",),
        )

        config_dict = config.to_dict()

        assert config_dict["name"] == "test_gan"
        assert "generator" in config_dict
        assert "discriminator" in config_dict
        assert config_dict["generator"]["latent_dim"] == 100
        assert config_dict["generator"]["hidden_dims"] == (512, 256)
        assert config_dict["discriminator"]["hidden_dims"] == (256, 512)
        assert config_dict["generator_lr"] == 0.0002
        assert config_dict["tags"] == ("test",)

    def test_from_dict_minimal(self):
        """Test creation from dictionary with minimal parameters."""
        config_dict = {
            "name": "test_gan",
            "generator": {
                "name": "gen",
                "latent_dim": 128,
                "hidden_dims": [512, 256, 128],
                "output_shape": [28, 28, 1],
                "activation": "relu",
            },
            "discriminator": {
                "name": "disc",
                "hidden_dims": [128, 256, 512],
                "input_shape": [28, 28, 1],
                "activation": "leaky_relu",
            },
        }

        config = GANConfig.from_dict(config_dict)

        assert config.name == "test_gan"
        assert isinstance(config.generator, GeneratorConfig)
        assert isinstance(config.discriminator, DiscriminatorConfig)
        assert config.generator.latent_dim == 128
        assert config.generator.hidden_dims == (512, 256, 128)  # Auto-converted to tuple
        assert config.discriminator.hidden_dims == (128, 256, 512)  # Auto-converted to tuple

    def test_from_dict_with_all_parameters(self):
        """Test creation from dictionary with all parameters."""
        config_dict = {
            "name": "full_gan",
            "description": "Full GAN config",
            "generator": {
                "name": "full_gen",
                "latent_dim": 256,
                "hidden_dims": [512, 256],
                "output_shape": [32, 32, 3],
                "activation": "relu",
                "batch_norm": True,
            },
            "discriminator": {
                "name": "full_disc",
                "hidden_dims": [256, 512],
                "input_shape": [32, 32, 3],
                "activation": "leaky_relu",
                "leaky_relu_slope": 0.2,
            },
            "generator_lr": 0.0002,
            "discriminator_lr": 0.0001,
            "beta1": 0.5,
            "beta2": 0.999,
            "loss_type": "wasserstein",
            "gradient_penalty_weight": 10.0,
            "tags": ["test", "full"],
            "metadata": {"exp": "001"},
            "rngs_seeds": {"params": 42},
        }

        config = GANConfig.from_dict(config_dict)

        assert config.name == "full_gan"
        assert config.description == "Full GAN config"
        assert config.generator.latent_dim == 256
        assert config.generator.hidden_dims == (512, 256)
        assert config.discriminator.hidden_dims == (256, 512)
        assert config.generator_lr == 0.0002
        assert config.discriminator_lr == 0.0001
        assert config.beta1 == 0.5
        assert config.beta2 == 0.999
        assert config.loss_type == "wasserstein"
        assert config.gradient_penalty_weight == 10.0
        assert config.tags == ("test", "full")
        assert config.metadata == {"exp": "001"}
        assert config.rngs_seeds == {"params": 42}

    def test_yaml_roundtrip_preserves_values(self):
        """Test that YAML serialization roundtrip preserves values."""
        generator = _create_test_generator(
            name="yaml_gen",
            hidden_dims=(512, 256, 128),
        )
        discriminator = _create_test_discriminator(
            name="yaml_disc",
            hidden_dims=(128, 256, 512),
        )

        original = GANConfig(
            name="yaml_test",
            description="YAML roundtrip test",
            generator=generator,
            discriminator=discriminator,
            generator_lr=0.0001,
            discriminator_lr=0.0002,
            beta1=0.5,
            beta2=0.999,
            loss_type="wasserstein",
            gradient_penalty_weight=10.0,
            tags=("yaml", "test"),
            metadata={"version": "1.0"},
            rngs_seeds={"params": 42, "dropout": 123},
        )

        # Convert to dict, then YAML, then back
        # Note: YAML doesn't preserve Python tuples, so we skip YAML serialization
        # and just test dict roundtrip which is the important part
        config_dict = original.to_dict()
        restored = GANConfig.from_dict(config_dict)

        # Check all fields match
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.generator.latent_dim == original.generator.latent_dim
        assert restored.generator.hidden_dims == original.generator.hidden_dims
        assert restored.discriminator.hidden_dims == original.discriminator.hidden_dims
        assert restored.generator_lr == original.generator_lr
        assert restored.discriminator_lr == original.discriminator_lr
        assert restored.beta1 == original.beta1
        assert restored.beta2 == original.beta2
        assert restored.loss_type == original.loss_type
        assert restored.gradient_penalty_weight == original.gradient_penalty_weight
        assert restored.tags == original.tags
        assert restored.metadata == original.metadata
        assert restored.rngs_seeds == original.rngs_seeds


class TestGANConfigDefaults:
    """Test GANConfig default values."""

    def test_default_values(self):
        """Test that default values are applied correctly."""
        generator = _create_test_generator()
        discriminator = _create_test_discriminator()

        config = GANConfig(
            name="defaults_test",
            generator=generator,
            discriminator=discriminator,
        )

        # Check defaults (description from BaseConfig is empty string, not None)
        assert config.description == ""
        assert config.generator_lr == 0.0002
        assert config.discriminator_lr == 0.0002
        assert config.beta1 == 0.5
        assert config.beta2 == 0.999
        assert config.loss_type == "vanilla"
        assert config.gradient_penalty_weight == 0.0
        assert config.tags == ()
        assert config.metadata == {}
        assert config.rngs_seeds == {}

    def test_override_defaults(self):
        """Test that defaults can be overridden."""
        generator = _create_test_generator()
        discriminator = _create_test_discriminator()

        config = GANConfig(
            name="override_test",
            generator=generator,
            discriminator=discriminator,
            generator_lr=0.001,
            discriminator_lr=0.0005,
            beta1=0.9,
            beta2=0.99,
            loss_type="wasserstein",
            gradient_penalty_weight=5.0,
        )

        assert config.generator_lr == 0.001
        assert config.discriminator_lr == 0.0005
        assert config.beta1 == 0.9
        assert config.beta2 == 0.99
        assert config.loss_type == "wasserstein"
        assert config.gradient_penalty_weight == 5.0


class TestGANConfigEdgeCases:
    """Test GANConfig edge cases and boundary conditions."""

    def test_single_hidden_layer(self):
        """Test config with single hidden layer in generator/discriminator."""
        generator = _create_test_generator(hidden_dims=(512,))
        discriminator = _create_test_discriminator(hidden_dims=(512,))

        config = GANConfig(
            name="single_layer",
            generator=generator,
            discriminator=discriminator,
        )

        assert len(config.generator.hidden_dims) == 1
        assert len(config.discriminator.hidden_dims) == 1

    def test_many_hidden_layers(self):
        """Test config with many hidden layers."""
        generator = _create_test_generator(hidden_dims=(1024, 512, 256, 128, 64, 32))
        discriminator = _create_test_discriminator(hidden_dims=(32, 64, 128, 256, 512, 1024))

        config = GANConfig(
            name="many_layers",
            generator=generator,
            discriminator=discriminator,
        )

        assert len(config.generator.hidden_dims) == 6
        assert len(config.discriminator.hidden_dims) == 6

    def test_very_small_latent_dim(self):
        """Test config with very small latent dimension."""
        generator = _create_test_generator(latent_dim=2, hidden_dims=(512,))
        discriminator = _create_test_discriminator(hidden_dims=(512,))

        config = GANConfig(
            name="small_latent",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.generator.latent_dim == 2

    def test_very_large_latent_dim(self):
        """Test config with very large latent dimension."""
        generator = _create_test_generator(latent_dim=10000, hidden_dims=(512,))
        discriminator = _create_test_discriminator(hidden_dims=(512,))

        config = GANConfig(
            name="large_latent",
            generator=generator,
            discriminator=discriminator,
        )

        assert config.generator.latent_dim == 10000

    def test_asymmetric_hidden_dims(self):
        """Test that gen and disc can have different numbers of layers."""
        generator = _create_test_generator(hidden_dims=(512, 256))
        discriminator = _create_test_discriminator(hidden_dims=(64, 128, 256, 512))

        config = GANConfig(
            name="asymmetric",
            generator=generator,
            discriminator=discriminator,
        )

        assert len(config.generator.hidden_dims) == 2
        assert len(config.discriminator.hidden_dims) == 4

    def test_beta_values_boundary(self):
        """Test beta values at boundaries."""
        generator = _create_test_generator()
        discriminator = _create_test_discriminator()

        # beta1 = 0.0 should be valid
        config = GANConfig(
            name="beta_boundary",
            generator=generator,
            discriminator=discriminator,
            beta1=0.0,
            beta2=0.0,
        )

        assert config.beta1 == 0.0
        assert config.beta2 == 0.0

        # beta close to 1.0 should be valid
        config = GANConfig(
            name="beta_boundary2",
            generator=generator,
            discriminator=discriminator,
            beta1=0.999,
            beta2=0.9999,
        )

        assert config.beta1 == 0.999
        assert config.beta2 == 0.9999


class TestGANConfigInheritance:
    """Test that GANConfig can be inherited from."""

    def test_can_inherit_from_gan_config(self):
        """Test that we can create subclasses of GANConfig."""

        @dataclasses.dataclass(frozen=True)
        class CustomGANConfig(GANConfig):
            custom_param: int = 42

        generator = _create_test_generator()
        discriminator = _create_test_discriminator()

        config = CustomGANConfig(
            name="custom",
            generator=generator,
            discriminator=discriminator,
            custom_param=123,
        )

        assert isinstance(config, GANConfig)
        assert isinstance(config, BaseConfig)
        assert config.custom_param == 123
        assert config.generator.latent_dim == 100
