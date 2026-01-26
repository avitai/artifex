"""Tests for factory extension integration."""

import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    DecoderConfig,
    EncoderConfig,
    ExtensionConfig,
    VAEConfig,
)
from artifex.generative_models.extensions.base import ModelExtension
from artifex.generative_models.factory.core import create_model, create_model_with_extensions


class TestFactoryExtensionIntegration:
    """Test extension integration with model factory."""

    @pytest.fixture
    def rngs(self):
        """Create test rngs."""
        return nnx.Rngs(42)

    @pytest.fixture
    def vae_config(self):
        """Create a simple VAE config."""
        encoder = EncoderConfig(
            name="test_encoder",
            input_shape=(32, 32, 3),
            latent_dim=16,
            hidden_dims=(64, 32),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="test_decoder",
            latent_dim=16,
            output_shape=(32, 32, 3),
            hidden_dims=(32, 64),
            activation="relu",
        )
        return VAEConfig(
            name="test_vae",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
        )

    @pytest.fixture
    def extension_configs(self):
        """Create extension configs for testing."""
        return {
            "test_ext_1": ExtensionConfig(name="test_ext_1", weight=1.0, enabled=True),
            "test_ext_2": ExtensionConfig(name="test_ext_2", weight=0.5, enabled=True),
        }

    def test_create_model_with_extensions_returns_tuple(self, vae_config, extension_configs, rngs):
        """Test that create_model_with_extensions returns model and extensions."""

        # Create a mock extension class
        class MockExtension(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

        # Register mock extension in registry
        from artifex.generative_models.extensions.registry import get_extensions_registry

        registry = get_extensions_registry()
        registry.register_extension(
            "test_ext_1",
            MockExtension,
            modalities=["image"],
            capabilities=["test"],
        )
        registry.register_extension(
            "test_ext_2",
            MockExtension,
            modalities=["image"],
            capabilities=["test"],
        )

        model, extensions = create_model_with_extensions(
            vae_config,
            extensions_config=extension_configs,
            rngs=rngs,
        )

        assert model is not None
        assert isinstance(extensions, dict)
        assert len(extensions) == 2
        assert "test_ext_1" in extensions
        assert "test_ext_2" in extensions

    def test_create_model_with_no_extensions(self, vae_config, rngs):
        """Test that create_model_with_extensions works without extensions."""
        model, extensions = create_model_with_extensions(
            vae_config,
            extensions_config=None,
            rngs=rngs,
        )

        assert model is not None
        assert isinstance(extensions, dict)
        assert len(extensions) == 0

    def test_create_model_with_empty_extensions(self, vae_config, rngs):
        """Test that create_model_with_extensions works with empty extensions dict."""
        model, extensions = create_model_with_extensions(
            vae_config,
            extensions_config={},
            rngs=rngs,
        )

        assert model is not None
        assert isinstance(extensions, dict)
        assert len(extensions) == 0

    def test_create_model_still_works(self, vae_config, rngs):
        """Test that original create_model still works without extensions."""
        model = create_model(vae_config, rngs=rngs)

        assert model is not None
        # Should return just the model, not a tuple
        assert not isinstance(model, tuple)

    def test_extensions_have_correct_config(self, vae_config, rngs):
        """Test that created extensions have the correct configuration."""

        class MockExtension(ModelExtension):
            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

        from artifex.generative_models.extensions.registry import get_extensions_registry

        registry = get_extensions_registry()

        # Register if not already registered
        if "weighted_ext" not in registry.list_all_extensions():
            registry.register_extension(
                "weighted_ext",
                MockExtension,
                modalities=["image"],
                capabilities=["test"],
            )

        ext_config = ExtensionConfig(name="weighted_ext", weight=0.75, enabled=False)
        extension_configs = {"weighted_ext": ext_config}

        model, extensions = create_model_with_extensions(
            vae_config,
            extensions_config=extension_configs,
            rngs=rngs,
        )

        assert "weighted_ext" in extensions
        assert extensions["weighted_ext"].weight == 0.75
        assert extensions["weighted_ext"].enabled is False
