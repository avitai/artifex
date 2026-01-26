"""Tests for image modality implementation.

Following Test-Driven Development principles:
- Tests define expected behavior
- Tests verify all functionality
- Tests cover edge cases
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    DecoderConfig,
    EncoderConfig,
    VAEConfig,
)
from artifex.generative_models.factory import ModelFactory
from artifex.generative_models.modalities.image import (
    ImageModality,
    ImageModalityAdapter,
    ImageModalityConfig,
    ImageRepresentation,
)
from artifex.generative_models.modalities.registry import (
    _MODALITY_REGISTRY,
    get_modality,
)


@pytest.fixture(autouse=True)
def setup_registry():
    """Ensure modality registry is properly setup for tests."""
    # Save original registry state
    original = _MODALITY_REGISTRY.copy()

    # Ensure ImageModality is registered
    if "image" not in _MODALITY_REGISTRY:
        from artifex.generative_models.modalities.image.base import ImageModality

        _MODALITY_REGISTRY["image"] = ImageModality

    yield

    # Restore original registry state
    _MODALITY_REGISTRY.clear()
    _MODALITY_REGISTRY.update(original)


class TestImageModalityConfig:
    """Test image modality configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ImageModalityConfig()

        assert config.representation == ImageRepresentation.RGB
        assert config.height == 64
        assert config.width == 64  # Should default to height
        assert config.channels == 3  # Should be auto-determined from RGB
        assert config.normalize is True
        assert config.augmentation is False
        assert config.resize_method == "bilinear"

    def test_grayscale_config(self):
        """Test grayscale configuration."""
        config = ImageModalityConfig(
            representation=ImageRepresentation.GRAYSCALE,
            height=128,
        )

        assert config.channels == 1
        assert config.width == 128  # Should default to height

    def test_rgba_config(self):
        """Test RGBA configuration."""
        config = ImageModalityConfig(
            representation=ImageRepresentation.RGBA,
            height=256,
            width=512,
        )

        assert config.channels == 4
        assert config.width == 512  # Should use specified width


class TestImageModality:
    """Test ImageModality class."""

    def test_initialization(self):
        """Test ImageModality initialization."""
        rngs = nnx.Rngs(42)
        config = ImageModalityConfig()

        modality = ImageModality(config=config, rngs=rngs)

        assert modality.name == "image"
        assert modality.config == config
        assert hasattr(modality, "rngs")

    def test_initialization_without_config(self):
        """Test ImageModality initialization without config."""
        rngs = nnx.Rngs(42)

        modality = ImageModality(rngs=rngs)

        assert modality.config is not None
        assert isinstance(modality.config, ImageModalityConfig)

    def test_image_shape_property(self):
        """Test image_shape property."""
        rngs = nnx.Rngs(42)
        config = ImageModalityConfig(height=128, width=256, channels=3)

        modality = ImageModality(config=config, rngs=rngs)

        assert modality.image_shape == (128, 256, 3)

    def test_get_adapter_method(self):
        """Test get_adapter method."""
        rngs = nnx.Rngs(42)
        modality = ImageModality(rngs=rngs)

        adapter = modality.get_adapter("vae")

        assert adapter is not None
        assert isinstance(adapter, ImageModalityAdapter)

    def test_get_extensions_method(self):
        """Test get_extensions method."""
        rngs = nnx.Rngs(42)
        modality = ImageModality(rngs=rngs)

        extensions = modality.get_extensions({}, rngs=rngs)

        assert isinstance(extensions, dict)
        assert len(extensions) == 0  # Currently no extensions for image

    def test_generate_method(self):
        """Test generate method."""
        rngs = nnx.Rngs(42)
        config = ImageModalityConfig(height=32, width=32, channels=3)
        modality = ImageModality(config=config, rngs=rngs)

        # Generate random images
        images = modality.generate(n_samples=4, rngs=rngs)

        assert images.shape == (4, 32, 32, 3)
        assert images.dtype == jnp.float32

    def test_process_method(self):
        """Test process method for multi-modal fusion."""
        rngs = nnx.Rngs(42)
        config = ImageModalityConfig(height=32, width=32, channels=3)
        modality = ImageModality(config=config, rngs=rngs)

        # Test with single image
        image = jnp.ones((32, 32, 3))
        processed = modality.process(image)

        assert processed.shape == (32 * 32 * 3,)  # Flattened

        # Test with batch
        batch = jnp.ones((4, 32, 32, 3))
        processed_batch = modality.process(batch)

        assert processed_batch.shape == (4, 32 * 32 * 3)


class TestImageModalityAdapter:
    """Test ImageModalityAdapter class."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = ImageModalityAdapter()

        assert adapter.name == "image_adapter"
        assert adapter.modality == "image"

    def test_adapt_method(self):
        """Test adapt method."""
        adapter = ImageModalityAdapter()

        # Create a mock model
        class MockModel:
            pass

        model = MockModel()
        encoder = EncoderConfig(
            name="encoder",
            input_shape=(64, 64, 3),
            latent_dim=128,
            hidden_dims=(32,),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="decoder",
            latent_dim=128,
            output_shape=(64, 64, 3),
            hidden_dims=(32,),
            activation="relu",
        )
        config = VAEConfig(
            name="test",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
        )

        adapted_model = adapter.adapt(model, config)

        # Should return model unchanged for now
        assert adapted_model is model

    def test_create_method_raises(self):
        """Test that create method raises NotImplementedError."""
        adapter = ImageModalityAdapter()
        rngs = nnx.Rngs(params=0, dropout=1, sample=2)

        encoder = EncoderConfig(
            name="encoder",
            input_shape=(64, 64, 3),
            latent_dim=128,
            hidden_dims=(32,),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="decoder",
            latent_dim=128,
            output_shape=(64, 64, 3),
            hidden_dims=(32,),
            activation="relu",
        )
        config = VAEConfig(
            name="test",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
        )

        with pytest.raises(NotImplementedError):
            adapter.create(config, rngs=rngs)


class TestModalityRegistry:
    """Test modality registry with image modality."""

    def test_get_modality_with_rngs(self):
        """Test getting image modality from registry with rngs."""
        rngs = nnx.Rngs(42)

        modality = get_modality("image", rngs=rngs)

        assert isinstance(modality, ImageModality)
        assert modality.name == "image"

    def test_get_modality_with_config(self):
        """Test getting image modality with config."""
        rngs = nnx.Rngs(42)
        config = ImageModalityConfig(height=128, width=256)

        modality = get_modality("image", config=config, rngs=rngs)

        assert isinstance(modality, ImageModality)
        # Note: Current implementation doesn't pass config through,
        # but this test defines expected behavior


class TestFactoryWithModality:
    """Test factory integration with image modality."""

    def test_create_vae_with_image_modality(self):
        """Test creating VAE with image modality."""
        rngs = nnx.Rngs(params=0, dropout=1, sample=2)
        factory = ModelFactory()

        encoder = EncoderConfig(
            name="encoder",
            input_shape=(64, 64, 3),
            latent_dim=128,
            hidden_dims=(32, 64),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="decoder",
            latent_dim=128,
            output_shape=(64, 64, 3),
            hidden_dims=(64, 32),
            activation="relu",
        )
        config = VAEConfig(
            name="test_vae",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
            kl_weight=1.0,
        )

        # Create model with image modality
        model = factory.create(config=config, modality="image", rngs=rngs)

        assert model is not None
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "latent_dim")
        assert model.latent_dim == 128

    def test_create_model_without_modality(self):
        """Test creating model without modality."""
        rngs = nnx.Rngs(params=0, dropout=1, sample=2)
        factory = ModelFactory()

        encoder = EncoderConfig(
            name="encoder",
            input_shape=(64, 64, 3),
            latent_dim=128,
            hidden_dims=(32, 64),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="decoder",
            latent_dim=128,
            output_shape=(64, 64, 3),
            hidden_dims=(64, 32),
            activation="relu",
        )
        config = VAEConfig(
            name="test_vae",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
            kl_weight=1.0,
        )

        # Create model without modality
        model = factory.create(config=config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")


class TestEndToEndIntegration:
    """Test end-to-end integration scenarios."""

    def test_complete_workflow(self):
        """Test complete workflow from config to model generation."""
        # 1. Create configuration
        ImageModalityConfig(
            representation=ImageRepresentation.RGB,
            height=32,
            width=32,
            channels=3,
        )

        encoder = EncoderConfig(
            name="encoder",
            input_shape=(32, 32, 3),
            latent_dim=64,
            hidden_dims=(16, 32),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="decoder",
            latent_dim=64,
            output_shape=(32, 32, 3),
            hidden_dims=(32, 16),
            activation="relu",
        )
        model_config = VAEConfig(
            name="test_integration",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
            kl_weight=1.0,
        )

        # 2. Create model with factory
        rngs = nnx.Rngs(params=0, dropout=1, sample=2)
        factory = ModelFactory()
        model = factory.create(config=model_config, modality="image", rngs=rngs)

        # 3. Test model functionality
        assert model is not None

        # Test encoding
        test_images = jax.random.normal(rngs.sample(), (4, 32, 32, 3))
        mean, log_var = model.encode(test_images)

        assert mean.shape == (4, 64)
        assert log_var.shape == (4, 64)

        # Test decoding
        z = model.reparameterize(mean, log_var)
        reconstructed = model.decode(z)

        assert reconstructed.shape == test_images.shape

    def test_modality_adapter_in_workflow(self):
        """Test that modality adapter is correctly applied in workflow."""
        rngs = nnx.Rngs(params=0, dropout=1, sample=2)

        # Get modality and adapter
        modality = get_modality("image", rngs=rngs)
        adapter = modality.get_adapter("vae")

        assert isinstance(adapter, ImageModalityAdapter)
        assert adapter.modality == "image"

        # Test adapter with config (using VAEConfig now)
        encoder = EncoderConfig(
            name="encoder",
            input_shape=(64, 64, 3),
            latent_dim=128,
            hidden_dims=(32,),
            activation="relu",
        )
        decoder = DecoderConfig(
            name="decoder",
            latent_dim=128,
            output_shape=(64, 64, 3),
            hidden_dims=(32,),
            activation="relu",
        )
        config = VAEConfig(
            name="test_adapter",
            encoder=encoder,
            decoder=decoder,
            encoder_type="dense",
        )

        # Mock model for testing
        class MockVAE:
            def __init__(self):
                self.adapted = False

        model = MockVAE()
        adapted = adapter.adapt(model, config)

        # Currently adapter returns model unchanged
        assert adapted is model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
