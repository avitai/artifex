"""Tests for core factory functionality with dataclass configs.

This module tests the ModelFactory with the new dataclass-based configuration system.
Model type is determined by config type (VAEConfig, GANConfig, etc.), not by model_class strings.
"""

import jax
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
    DCGANConfig,
    DecoderConfig,
    EncoderConfig,
    VAEConfig,
)
from artifex.generative_models.factory.core import create_model, ModelFactory
from artifex.generative_models.modalities.registry import register_modality


class MockModality:
    """Mock modality for testing."""

    def __init__(self, **kwargs):
        """Initialize mock modality, accepting any kwargs for compatibility."""
        pass

    @classmethod
    def get_adapter(cls, model_type: str):
        """Get adapter for model type."""
        return MockAdapter()


class MockAdapter:
    """Mock adapter for testing."""

    def adapt(self, model, config):
        """Adapt a model."""
        model.adapted = True
        return model


class TestModelFactory:
    """Test model factory functionality with dataclass configs."""

    @pytest.fixture
    def factory(self):
        """Create factory instance."""
        return ModelFactory()

    @pytest.fixture
    def rngs(self):
        """Create RNGs for testing."""
        return nnx.Rngs(params=jax.random.PRNGKey(42))

    @pytest.fixture
    def vae_config(self):
        """Create a VAE config for testing."""
        encoder = EncoderConfig(
            name="encoder",
            input_shape=(28, 28, 1),
            latent_dim=64,
            hidden_dims=(512, 256),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="decoder",
            latent_dim=64,
            output_shape=(28, 28, 1),
            hidden_dims=(256, 512),
            activation="relu",
        )
        return VAEConfig(
            name="test_vae",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
        )

    @pytest.fixture(autouse=True)
    def setup_modality(self):
        """Setup and cleanup mock modality."""
        from artifex.generative_models.modalities.registry import _MODALITY_REGISTRY

        original = _MODALITY_REGISTRY.copy()
        register_modality("mock", MockModality)
        yield
        _MODALITY_REGISTRY.clear()
        _MODALITY_REGISTRY.update(original)

    def test_create_model_without_modality(self, factory, vae_config, rngs):
        """Test creating a model without modality."""
        model = factory.create(vae_config, rngs=rngs)

        assert model is not None
        assert not hasattr(model, "adapted")

    def test_create_model_with_modality(self, factory, vae_config, rngs):
        """Test creating a model with modality adaptation."""
        model = factory.create(vae_config, modality="mock", rngs=rngs)

        assert model is not None
        assert hasattr(model, "adapted")
        assert model.adapted is True

    def test_extract_model_type_from_config(self, factory):
        """Test model type extraction from config type."""
        encoder = EncoderConfig(
            name="encoder",
            input_shape=(28, 28, 1),
            latent_dim=32,
            hidden_dims=(64,),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="decoder",
            latent_dim=32,
            output_shape=(28, 28, 1),
            hidden_dims=(64,),
            activation="relu",
        )
        vae_config = VAEConfig(
            name="test",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
        )

        assert factory._extract_model_type(vae_config) == "vae"

    def test_validate_config_accepts_dataclass(self, factory, vae_config):
        """Test configuration validation accepts dataclass configs."""
        # Should not raise for valid dataclass config
        factory._validate_config(vae_config)

    def test_validate_config_rejects_dict(self, factory):
        """Test configuration validation rejects dict configs."""
        with pytest.raises(TypeError, match="Expected dataclass config.*got dict"):
            factory._validate_config({"latent_dim": 32})

    def test_validate_config_rejects_none(self, factory):
        """Test configuration validation rejects None."""
        with pytest.raises(TypeError, match="config cannot be None"):
            factory._validate_config(None)

    def test_create_with_unknown_config_type(self, factory, rngs):
        """Test error handling for unknown config types."""

        class UnknownConfig:
            name = "test"

        with pytest.raises(TypeError, match="Expected dataclass config"):
            factory.create(UnknownConfig(), rngs=rngs)

    def test_create_model_function(self, vae_config, rngs):
        """Test the global create_model function."""
        model = create_model(vae_config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "encode")
        assert hasattr(model, "decode")


class TestModelFactoryWithGAN:
    """Test model factory with GAN configs."""

    @pytest.fixture
    def rngs(self):
        """Create RNGs for testing."""
        return nnx.Rngs(
            params=jax.random.PRNGKey(42),
            dropout=jax.random.PRNGKey(43),
            sample=jax.random.PRNGKey(44),
        )

    @pytest.fixture
    def dcgan_config(self):
        """Create a DCGAN config for testing."""
        generator = ConvGeneratorConfig(
            name="generator",
            latent_dim=32,
            output_shape=(3, 32, 32),
            hidden_dims=(64, 32),
            activation="relu",
        )
        discriminator = ConvDiscriminatorConfig(
            name="discriminator",
            input_shape=(3, 32, 32),
            hidden_dims=(32, 64),
            activation="leaky_relu",
        )
        return DCGANConfig(
            name="test_dcgan",
            generator=generator,
            discriminator=discriminator,
        )

    def test_create_gan_model(self, dcgan_config, rngs):
        """Test creating a GAN model."""
        model = create_model(dcgan_config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "generator")
        assert hasattr(model, "discriminator")


class TestFactoryTypeDispatch:
    """Test that factory correctly dispatches based on config type."""

    @pytest.fixture
    def rngs(self):
        """Create RNGs for testing."""
        return nnx.Rngs(params=jax.random.PRNGKey(42))

    def test_vae_config_dispatches_to_vae_builder(self, rngs):
        """Test VAEConfig dispatches to VAE builder."""
        encoder = EncoderConfig(
            name="encoder",
            input_shape=(28, 28, 1),
            latent_dim=16,
            hidden_dims=(32,),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="decoder",
            latent_dim=16,
            output_shape=(28, 28, 1),
            hidden_dims=(32,),
            activation="relu",
        )
        config = VAEConfig(
            name="test",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
        )

        model = create_model(config, rngs=rngs)

        # VAE should have encode/decode methods
        assert hasattr(model, "encode")
        assert hasattr(model, "decode")
        assert hasattr(model, "sample")


class TestFactoryErrorHandling:
    """Test factory error handling."""

    @pytest.fixture
    def rngs(self):
        """Create RNGs for testing."""
        return nnx.Rngs(params=jax.random.PRNGKey(42))

    def test_dict_config_raises_type_error(self, rngs):
        """Test that dict config raises TypeError."""
        with pytest.raises(TypeError, match="Expected dataclass config.*got dict"):
            create_model({"latent_dim": 32}, rngs=rngs)

    def test_string_config_raises_type_error(self, rngs):
        """Test that string config raises TypeError."""
        with pytest.raises(TypeError, match="Expected dataclass config.*got str"):
            create_model("invalid", rngs=rngs)

    def test_none_config_raises_type_error(self, rngs):
        """Test that None config raises TypeError."""
        with pytest.raises(TypeError, match="config cannot be None"):
            create_model(None, rngs=rngs)

    def test_legacy_class_config_raises_type_error(self, rngs):
        """Test that legacy config classes are rejected."""

        class LegacyVAEConfig:
            def __init__(self):
                self.latent_dim = 32
                self.hidden_dims = [256, 128]

        with pytest.raises(TypeError, match="Expected dataclass config"):
            create_model(LegacyVAEConfig(), rngs=rngs)
