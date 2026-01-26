"""Tests for CycleGANConfig frozen dataclass.

Following TDD principles: Tests define the expected behavior BEFORE implementation.
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration.gan_config import (
    CycleGANConfig,
    GANConfig,
)
from artifex.generative_models.core.configuration.network_configs import (
    DiscriminatorConfig,
    GeneratorConfig,
)


def create_cyclegan_networks(
    input_shape_a=(256, 256, 3),
    input_shape_b=(256, 256, 3),
    latent_dim=100,
    hidden_dims=(64, 128, 256),
):
    """Helper to create generator and discriminator dicts for CycleGAN tests."""
    return {
        "generator": {
            "a_to_b": GeneratorConfig(
                name="gen_a_to_b",
                latent_dim=latent_dim,
                output_shape=input_shape_b,
                hidden_dims=hidden_dims,
                activation="relu",
            ),
            "b_to_a": GeneratorConfig(
                name="gen_b_to_a",
                latent_dim=latent_dim,
                output_shape=input_shape_a,
                hidden_dims=hidden_dims,
                activation="relu",
            ),
        },
        "discriminator": {
            "disc_a": DiscriminatorConfig(
                name="disc_a",
                input_shape=input_shape_a,
                hidden_dims=hidden_dims,
                activation="leaky_relu",
            ),
            "disc_b": DiscriminatorConfig(
                name="disc_b",
                input_shape=input_shape_b,
                hidden_dims=hidden_dims,
                activation="leaky_relu",
            ),
        },
    }


class TestCycleGANConfigBasics:
    """Test basic CycleGANConfig creation and properties."""

    def test_cyclegan_config_creation_minimal(self):
        """Test creating CycleGANConfig with minimal required parameters."""
        networks = create_cyclegan_networks()

        config = CycleGANConfig(
            name="test_cyclegan",
            generator=networks["generator"],
            discriminator=networks["discriminator"],
            input_shape_a=(256, 256, 3),
            input_shape_b=(256, 256, 3),
        )

        assert config.name == "test_cyclegan"
        assert config.input_shape_a == (256, 256, 3)
        assert config.input_shape_b == (256, 256, 3)
        assert len(config.generator) == 2
        assert len(config.discriminator) == 2

    def test_cyclegan_config_inherits_from_gan_config(self):
        """Test that CycleGANConfig properly inherits from GANConfig."""
        networks = create_cyclegan_networks(hidden_dims=(64,))

        config = CycleGANConfig(
            name="test",
            generator=networks["generator"],
            discriminator=networks["discriminator"],
            input_shape_a=(256, 256, 3),
            input_shape_b=(256, 256, 3),
        )

        assert isinstance(config, GANConfig)
        assert isinstance(config, CycleGANConfig)

    def test_cyclegan_config_is_frozen(self):
        """Test that CycleGANConfig is immutable."""
        networks = create_cyclegan_networks(hidden_dims=(64,))

        config = CycleGANConfig(
            name="frozen_test",
            generator=networks["generator"],
            discriminator=networks["discriminator"],
            input_shape_a=(256, 256, 3),
            input_shape_b=(256, 256, 3),
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.lambda_cycle = 20.0  # type: ignore

    def test_cyclegan_config_with_all_parameters(self):
        """Test creating CycleGANConfig with all parameters."""
        networks = create_cyclegan_networks(latent_dim=128)

        config = CycleGANConfig(
            name="full_cyclegan",
            description="Full CycleGAN config",
            generator=networks["generator"],
            discriminator=networks["discriminator"],
            input_shape_a=(256, 256, 3),
            input_shape_b=(256, 256, 3),
            lambda_cycle=15.0,
            lambda_identity=0.5,
            generator_lr=0.0002,
            discriminator_lr=0.0002,
            beta1=0.5,
            beta2=0.999,
        )

        assert config.name == "full_cyclegan"
        assert config.description == "Full CycleGAN config"
        assert config.lambda_cycle == 15.0
        assert config.lambda_identity == 0.5
        assert config.generator_lr == 0.0002
        assert config.discriminator_lr == 0.0002
        assert config.beta1 == 0.5
        assert config.beta2 == 0.999


class TestCycleGANConfigValidation:
    """Test CycleGANConfig validation rules."""

    def test_cyclegan_requires_dict_of_generators(self):
        """Test that CycleGAN requires dict of generators, not single GeneratorConfig."""
        discriminator_dict = {
            "disc_a": DiscriminatorConfig(
                name="disc_a", input_shape=(256, 256, 3), hidden_dims=(64,), activation="leaky_relu"
            ),
            "disc_b": DiscriminatorConfig(
                name="disc_b", input_shape=(256, 256, 3), hidden_dims=(64,), activation="leaky_relu"
            ),
        }

        with pytest.raises(
            TypeError,
            match="CycleGAN requires generator to be a dict with 2 generators",
        ):
            CycleGANConfig(
                name="test",
                generator=GeneratorConfig(  # type: ignore
                    name="single_gen",
                    latent_dim=100,
                    output_shape=(256, 256, 3),
                    hidden_dims=(64,),
                    activation="relu",
                ),
                discriminator=discriminator_dict,
                input_shape_a=(256, 256, 3),
                input_shape_b=(256, 256, 3),
            )

    def test_cyclegan_requires_dict_of_discriminators(self):
        """Test that CycleGAN requires dict of discriminators, not single DiscriminatorConfig."""
        generator_dict = {
            "a_to_b": GeneratorConfig(
                name="gen_a_to_b",
                latent_dim=100,
                output_shape=(256, 256, 3),
                hidden_dims=(64,),
                activation="relu",
            ),
            "b_to_a": GeneratorConfig(
                name="gen_b_to_a",
                latent_dim=100,
                output_shape=(256, 256, 3),
                hidden_dims=(64,),
                activation="relu",
            ),
        }

        with pytest.raises(
            TypeError,
            match="CycleGAN requires discriminator to be a dict with 2 discriminators",
        ):
            CycleGANConfig(
                name="test",
                generator=generator_dict,
                discriminator=DiscriminatorConfig(  # type: ignore
                    name="single_disc",
                    input_shape=(256, 256, 3),
                    hidden_dims=(64,),
                    activation="leaky_relu",
                ),
                input_shape_a=(256, 256, 3),
                input_shape_b=(256, 256, 3),
            )

    def test_cyclegan_requires_exactly_two_generators(self):
        """Test that CycleGAN requires exactly 2 generators."""
        discriminator_dict = {
            "disc_a": DiscriminatorConfig(
                name="disc_a", input_shape=(256, 256, 3), hidden_dims=(64,), activation="leaky_relu"
            ),
            "disc_b": DiscriminatorConfig(
                name="disc_b", input_shape=(256, 256, 3), hidden_dims=(64,), activation="leaky_relu"
            ),
        }

        # Test with only 1 generator
        with pytest.raises(ValueError, match="CycleGAN requires exactly 2 generators"):
            CycleGANConfig(
                name="test",
                generator={
                    "a_to_b": GeneratorConfig(
                        name="gen",
                        latent_dim=100,
                        output_shape=(256, 256, 3),
                        hidden_dims=(64,),
                        activation="relu",
                    ),
                },
                discriminator=discriminator_dict,
                input_shape_a=(256, 256, 3),
                input_shape_b=(256, 256, 3),
            )

        # Test with 3 generators
        with pytest.raises(ValueError, match="CycleGAN requires exactly 2 generators"):
            CycleGANConfig(
                name="test",
                generator={
                    "a_to_b": GeneratorConfig(
                        name="g1",
                        latent_dim=100,
                        output_shape=(256, 256, 3),
                        hidden_dims=(64,),
                        activation="relu",
                    ),
                    "b_to_a": GeneratorConfig(
                        name="g2",
                        latent_dim=100,
                        output_shape=(256, 256, 3),
                        hidden_dims=(64,),
                        activation="relu",
                    ),
                    "c_to_d": GeneratorConfig(
                        name="g3",
                        latent_dim=100,
                        output_shape=(256, 256, 3),
                        hidden_dims=(64,),
                        activation="relu",
                    ),
                },
                discriminator=discriminator_dict,
                input_shape_a=(256, 256, 3),
                input_shape_b=(256, 256, 3),
            )

    def test_cyclegan_requires_exactly_two_discriminators(self):
        """Test that CycleGAN requires exactly 2 discriminators."""
        generator_dict = {
            "a_to_b": GeneratorConfig(
                name="gen_a_to_b",
                latent_dim=100,
                output_shape=(256, 256, 3),
                hidden_dims=(64,),
                activation="relu",
            ),
            "b_to_a": GeneratorConfig(
                name="gen_b_to_a",
                latent_dim=100,
                output_shape=(256, 256, 3),
                hidden_dims=(64,),
                activation="relu",
            ),
        }

        # Test with only 1 discriminator
        with pytest.raises(ValueError, match="CycleGAN requires exactly 2 discriminators"):
            CycleGANConfig(
                name="test",
                generator=generator_dict,
                discriminator={
                    "disc_a": DiscriminatorConfig(
                        name="disc",
                        input_shape=(256, 256, 3),
                        hidden_dims=(64,),
                        activation="leaky_relu",
                    ),
                },
                input_shape_a=(256, 256, 3),
                input_shape_b=(256, 256, 3),
            )

    def test_input_shape_a_must_have_three_elements(self):
        """Test that input_shape_a must be a 3-tuple."""
        networks = create_cyclegan_networks()

        with pytest.raises(
            ValueError, match="input_shape_a must be a tuple of 3 positive integers"
        ):
            CycleGANConfig(
                name="test",
                generator=networks["generator"],
                discriminator=networks["discriminator"],
                input_shape_a=(256, 256),  # Only 2 elements
                input_shape_b=(256, 256, 3),
            )

    def test_input_shape_b_must_have_three_elements(self):
        """Test that input_shape_b must be a 3-tuple."""
        networks = create_cyclegan_networks()

        with pytest.raises(
            ValueError, match="input_shape_b must be a tuple of 3 positive integers"
        ):
            CycleGANConfig(
                name="test",
                generator=networks["generator"],
                discriminator=networks["discriminator"],
                input_shape_a=(256, 256, 3),
                input_shape_b=(256, 256, 3, 1),  # 4 elements
            )

    def test_input_shape_a_must_be_positive(self):
        """Test that all elements of input_shape_a must be positive."""
        networks = create_cyclegan_networks()

        with pytest.raises(
            ValueError, match="input_shape_a must be a tuple of 3 positive integers"
        ):
            CycleGANConfig(
                name="test",
                generator=networks["generator"],
                discriminator=networks["discriminator"],
                input_shape_a=(256, 0, 3),  # Zero element
                input_shape_b=(256, 256, 3),
            )

    def test_input_shape_b_must_be_positive(self):
        """Test that all elements of input_shape_b must be positive."""
        networks = create_cyclegan_networks()

        with pytest.raises(
            ValueError, match="input_shape_b must be a tuple of 3 positive integers"
        ):
            CycleGANConfig(
                name="test",
                generator=networks["generator"],
                discriminator=networks["discriminator"],
                input_shape_a=(256, 256, 3),
                input_shape_b=(-1, 256, 3),  # Negative element
            )

    def test_lambda_cycle_must_be_non_negative(self):
        """Test that lambda_cycle must be non-negative."""
        networks = create_cyclegan_networks()

        with pytest.raises(ValueError, match="lambda_cycle must be non-negative"):
            CycleGANConfig(
                name="test",
                generator=networks["generator"],
                discriminator=networks["discriminator"],
                input_shape_a=(256, 256, 3),
                input_shape_b=(256, 256, 3),
                lambda_cycle=-1.0,
            )

    def test_lambda_identity_must_be_non_negative(self):
        """Test that lambda_identity must be non-negative."""
        networks = create_cyclegan_networks()

        with pytest.raises(ValueError, match="lambda_identity must be non-negative"):
            CycleGANConfig(
                name="test",
                generator=networks["generator"],
                discriminator=networks["discriminator"],
                input_shape_a=(256, 256, 3),
                input_shape_b=(256, 256, 3),
                lambda_identity=-0.5,
            )


class TestCycleGANConfigDefaults:
    """Test CycleGANConfig default values."""

    def test_default_lambda_cycle(self):
        """Test default lambda_cycle is 10.0."""
        networks = create_cyclegan_networks(hidden_dims=(64,))

        config = CycleGANConfig(
            name="test",
            generator=networks["generator"],
            discriminator=networks["discriminator"],
            input_shape_a=(256, 256, 3),
            input_shape_b=(256, 256, 3),
        )
        assert config.lambda_cycle == 10.0

    def test_default_lambda_identity(self):
        """Test default lambda_identity is 0.5."""
        networks = create_cyclegan_networks(hidden_dims=(64,))

        config = CycleGANConfig(
            name="test",
            generator=networks["generator"],
            discriminator=networks["discriminator"],
            input_shape_a=(256, 256, 3),
            input_shape_b=(256, 256, 3),
        )
        assert config.lambda_identity == 0.5

    def test_default_generator_lr(self):
        """Test default generator_lr is 0.0002."""
        networks = create_cyclegan_networks(hidden_dims=(64,))

        config = CycleGANConfig(
            name="test",
            generator=networks["generator"],
            discriminator=networks["discriminator"],
            input_shape_a=(256, 256, 3),
            input_shape_b=(256, 256, 3),
        )
        assert config.generator_lr == 0.0002

    def test_default_discriminator_lr(self):
        """Test default discriminator_lr is 0.0002."""
        networks = create_cyclegan_networks(hidden_dims=(64,))

        config = CycleGANConfig(
            name="test",
            generator=networks["generator"],
            discriminator=networks["discriminator"],
            input_shape_a=(256, 256, 3),
            input_shape_b=(256, 256, 3),
        )
        assert config.discriminator_lr == 0.0002

    def test_default_beta1(self):
        """Test default beta1 is 0.5."""
        networks = create_cyclegan_networks(hidden_dims=(64,))

        config = CycleGANConfig(
            name="test",
            generator=networks["generator"],
            discriminator=networks["discriminator"],
            input_shape_a=(256, 256, 3),
            input_shape_b=(256, 256, 3),
        )
        assert config.beta1 == 0.5

    def test_default_beta2(self):
        """Test default beta2 is 0.999."""
        networks = create_cyclegan_networks(hidden_dims=(64,))

        config = CycleGANConfig(
            name="test",
            generator=networks["generator"],
            discriminator=networks["discriminator"],
            input_shape_a=(256, 256, 3),
            input_shape_b=(256, 256, 3),
        )
        assert config.beta2 == 0.999


class TestCycleGANConfigSerialization:
    """Test CycleGANConfig serialization and deserialization."""

    def test_from_dict_minimal(self):
        """Test creating CycleGANConfig from dictionary with minimal fields."""
        config_dict = {
            "name": "test",
            "generator": {
                "a_to_b": {
                    "name": "gen_a_to_b",
                    "latent_dim": 100,
                    "output_shape": [256, 256, 3],
                    "hidden_dims": [64, 128, 256],
                    "activation": "relu",
                },
                "b_to_a": {
                    "name": "gen_b_to_a",
                    "latent_dim": 100,
                    "output_shape": [256, 256, 3],
                    "hidden_dims": [64, 128, 256],
                    "activation": "relu",
                },
            },
            "discriminator": {
                "disc_a": {
                    "name": "disc_a",
                    "input_shape": [256, 256, 3],
                    "hidden_dims": [64, 128, 256],
                    "activation": "leaky_relu",
                },
                "disc_b": {
                    "name": "disc_b",
                    "input_shape": [256, 256, 3],
                    "hidden_dims": [64, 128, 256],
                    "activation": "leaky_relu",
                },
            },
            "input_shape_a": [256, 256, 3],
            "input_shape_b": [256, 256, 3],
        }

        config = CycleGANConfig.from_dict(config_dict)

        assert config.name == "test"
        assert config.input_shape_a == (256, 256, 3)
        assert config.input_shape_b == (256, 256, 3)
        assert len(config.generator) == 2
        assert len(config.discriminator) == 2

    def test_from_dict_with_all_fields(self):
        """Test creating CycleGANConfig from dictionary with all fields."""
        config_dict = {
            "name": "full_cyclegan",
            "description": "Full config",
            "generator": {
                "a_to_b": {
                    "name": "gen_a_to_b",
                    "latent_dim": 128,
                    "output_shape": [256, 256, 3],
                    "hidden_dims": [64, 128, 256],
                    "activation": "relu",
                },
                "b_to_a": {
                    "name": "gen_b_to_a",
                    "latent_dim": 128,
                    "output_shape": [256, 256, 3],
                    "hidden_dims": [64, 128, 256],
                    "activation": "relu",
                },
            },
            "discriminator": {
                "disc_a": {
                    "name": "disc_a",
                    "input_shape": [256, 256, 3],
                    "hidden_dims": [64, 128, 256],
                    "activation": "leaky_relu",
                },
                "disc_b": {
                    "name": "disc_b",
                    "input_shape": [256, 256, 3],
                    "hidden_dims": [64, 128, 256],
                    "activation": "leaky_relu",
                },
            },
            "input_shape_a": [256, 256, 3],
            "input_shape_b": [256, 256, 3],
            "lambda_cycle": 15.0,
            "lambda_identity": 0.5,
            "generator_lr": 0.0002,
            "discriminator_lr": 0.0002,
            "beta1": 0.5,
            "beta2": 0.999,
        }

        config = CycleGANConfig.from_dict(config_dict)

        assert config.name == "full_cyclegan"
        assert config.description == "Full config"
        assert config.lambda_cycle == 15.0
        assert config.lambda_identity == 0.5

    def test_to_dict_roundtrip(self):
        """Test that to_dict/from_dict roundtrip preserves config."""
        networks = create_cyclegan_networks(latent_dim=100)

        original = CycleGANConfig(
            name="roundtrip",
            generator=networks["generator"],
            discriminator=networks["discriminator"],
            input_shape_a=(256, 256, 3),
            input_shape_b=(256, 256, 3),
            lambda_cycle=15.0,
        )

        config_dict = original.to_dict()
        restored = CycleGANConfig.from_dict(config_dict)

        assert restored.name == original.name
        assert restored.input_shape_a == original.input_shape_a
        assert restored.input_shape_b == original.input_shape_b
        assert restored.lambda_cycle == original.lambda_cycle
        assert len(restored.generator) == 2
        assert len(restored.discriminator) == 2


class TestCycleGANConfigEdgeCases:
    """Test CycleGANConfig edge cases."""

    def test_different_domain_shapes(self):
        """Test CycleGAN with different domain shapes (e.g., photos to paintings)."""
        networks = create_cyclegan_networks(
            input_shape_a=(256, 256, 3),
            input_shape_b=(128, 128, 1),
            hidden_dims=(64,),
        )

        config = CycleGANConfig(
            name="test",
            generator=networks["generator"],
            discriminator=networks["discriminator"],
            input_shape_a=(256, 256, 3),  # RGB photos
            input_shape_b=(128, 128, 1),  # Grayscale paintings
        )

        assert config.input_shape_a == (256, 256, 3)
        assert config.input_shape_b == (128, 128, 1)

    def test_zero_lambda_identity(self):
        """Test that lambda_identity can be zero (no identity loss)."""
        networks = create_cyclegan_networks(hidden_dims=(64,))

        config = CycleGANConfig(
            name="test",
            generator=networks["generator"],
            discriminator=networks["discriminator"],
            input_shape_a=(256, 256, 3),
            input_shape_b=(256, 256, 3),
            lambda_identity=0.0,
        )

        assert config.lambda_identity == 0.0

    def test_very_high_lambda_cycle(self):
        """Test that lambda_cycle can be very high for strong cycle consistency."""
        networks = create_cyclegan_networks(hidden_dims=(64,))

        config = CycleGANConfig(
            name="test",
            generator=networks["generator"],
            discriminator=networks["discriminator"],
            input_shape_a=(256, 256, 3),
            input_shape_b=(256, 256, 3),
            lambda_cycle=100.0,
        )

        assert config.lambda_cycle == 100.0
