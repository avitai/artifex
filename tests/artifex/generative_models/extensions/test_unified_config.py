"""Tests for extension system with unified configuration."""

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from artifex.generative_models.core.configuration import (
    AugmentationExtensionConfig,
    ExtensionConfig,
    ModalityConfig,
)
from artifex.generative_models.extensions.base import (
    ConstraintExtension,
    ModelExtension,
)
from artifex.generative_models.extensions.registry import (
    ExtensionsRegistry,
    get_extensions_registry,
)
from artifex.generative_models.extensions.vision.augmentation import (
    AdvancedImageAugmentation,
)


class TestExtensionConfig:
    """Test ExtensionConfig with unified configuration."""

    def test_extension_config_creation(self):
        """Test creating extension configuration."""
        config = ExtensionConfig(
            name="test_config",
            weight=0.5,
            enabled=False,
        )

        assert config.weight == 0.5
        assert config.enabled is False

    def test_extension_config_defaults(self):
        """Test extension config defaults."""
        config = ExtensionConfig(name="test_defaults")

        assert config.weight == 1.0
        assert config.enabled is True


class TestModelExtension:
    """Test ModelExtension with unified configuration."""

    @pytest.fixture
    def rngs(self):
        """Create RNG for tests."""
        return nnx.Rngs(42)

    @pytest.fixture
    def extension_config(self):
        """Create extension configuration."""
        return ExtensionConfig(
            name="test_extension",
            weight=0.8,
            enabled=True,
        )

    def test_model_extension_with_typed_config(self, extension_config, rngs):
        """Test ModelExtension with typed configuration."""

        # Create a test extension class
        class TestExtension(ModelExtension):
            def __init__(self, config: ExtensionConfig, *, rngs: nnx.Rngs):
                super().__init__(config, rngs=rngs)
                self.test_value = 42

            def __call__(self, inputs, model_outputs, **kwargs):
                return {"test": self.test_value * self.weight}

        # Create extension
        extension = TestExtension(extension_config, rngs=rngs)

        assert extension.weight == 0.8
        assert extension.enabled is True
        assert extension.is_enabled() is True

        # Test call
        result = extension(None, None)
        assert result["test"] == 42 * 0.8

    def test_extension_rejects_dict_config(self, rngs):
        """Test that extension rejects dict config."""

        class StrictExtension(ModelExtension):
            def __init__(self, config: ExtensionConfig, *, rngs: nnx.Rngs):
                super().__init__(config, rngs=rngs)

            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

        # Should reject dict config
        with pytest.raises(TypeError, match="config must be ExtensionConfig"):
            StrictExtension({"weight": 0.5}, rngs=rngs)

    def test_extension_loss_fn(self, extension_config, rngs):
        """Test extension loss function."""

        class TestExtension(ModelExtension):
            def __init__(self, config: ExtensionConfig, *, rngs: nnx.Rngs):
                super().__init__(config, rngs=rngs)

            def __call__(self, inputs, model_outputs, **kwargs):
                return {}

            def loss_fn(self, batch, model_outputs, **kwargs):
                if not self.enabled:
                    return jnp.array(0.0)
                return jnp.array(1.0) * self.weight

        extension = TestExtension(extension_config, rngs=rngs)
        loss = extension.loss_fn({}, {})
        assert loss == 0.8  # weight * 1.0

        # Test disabled extension
        extension.enabled = False
        loss = extension.loss_fn({}, {})
        assert loss == 0.0


class TestConstraintExtension:
    """Test ConstraintExtension with unified configuration."""

    @pytest.fixture
    def rngs(self):
        """Create RNG for tests."""
        return nnx.Rngs(42)

    def test_constraint_extension(self, rngs):
        """Test constraint extension with typed config."""
        config = ExtensionConfig(name="test_constraint", weight=0.5, enabled=True)

        class TestConstraint(ConstraintExtension):
            def __init__(self, config: ExtensionConfig, *, rngs: nnx.Rngs):
                super().__init__(config, rngs=rngs)

            def __call__(self, inputs, model_outputs, **kwargs):
                return {"constrained": True}

            def validate(self, outputs):
                return {"valid": jnp.array(1.0)}

            def project(self, outputs):
                if not self.enabled:
                    return outputs
                return outputs * 0.5  # Simple projection

        constraint = TestConstraint(config, rngs=rngs)

        # Test validation
        validation = constraint.validate(jnp.ones((2, 2)))
        assert "valid" in validation

        # Test projection
        projected = constraint.project(jnp.ones((2, 2)))
        assert jnp.allclose(projected, 0.5)

        # Test disabled projection
        constraint.enabled = False
        projected = constraint.project(jnp.ones((2, 2)))
        assert jnp.allclose(projected, 1.0)


class TestAdvancedImageAugmentation:
    """Test AdvancedImageAugmentation with unified configuration."""

    @pytest.fixture
    def rngs(self):
        """Create RNG for tests."""
        return nnx.Rngs(42)

    @pytest.fixture
    def augmentation_config(self):
        """Create augmentation configuration."""
        return ModalityConfig(
            name="image_modality",
            modality_name="image",
            supported_models=["vae", "gan"],
            preprocessing_steps=[],
            default_metrics=["fid", "is"],
            extensions={
                "augmentation": {
                    "rotation_range": 30.0,
                    "translation_range": 0.2,
                    "scale_range": (0.7, 1.3),
                    "brightness_range": 0.3,
                    "noise_level": 0.05,
                }
            },
        )

    def test_augmentation_with_typed_config(self, augmentation_config, rngs):
        """Test image augmentation with typed configuration."""
        # Create AugmentationExtensionConfig with custom parameters
        config = AugmentationExtensionConfig(
            name="image_augmentation",
            weight=1.0,
            enabled=True,
            probability=1.0,
        )

        augmentation = AdvancedImageAugmentation(config, rngs=rngs)

        # Check basic configuration was applied
        assert augmentation.weight == 1.0
        assert augmentation.enabled is True

    def test_augmentation_call(self, rngs):
        """Test augmentation forward pass."""
        # Create with default config
        config = AugmentationExtensionConfig(name="test_augmentation")
        augmentation = AdvancedImageAugmentation(config, rngs=rngs)

        # Create test images
        images = jnp.ones((2, 32, 32, 3))

        # Test non-deterministic augmentation
        augmented = augmentation(images, deterministic=False)
        assert augmented.shape == images.shape

        # Test deterministic mode (no augmentation)
        augmented_det = augmentation(images, deterministic=True)
        assert jnp.allclose(augmented_det, images)

    def test_augmentation_methods(self, rngs):
        """Test individual augmentation methods."""
        config = AugmentationExtensionConfig(name="test_augmentation_methods")
        augmentation = AdvancedImageAugmentation(config, rngs=rngs)
        images = jnp.ones((2, 32, 32, 3))

        # Test horizontal flip
        flipped = augmentation.apply_horizontal_flip(images)
        assert flipped.shape == images.shape

        # Test cutout
        cutout = augmentation.apply_cutout(images, cutout_size=8, num_cutouts=2)
        assert cutout.shape == images.shape

        # Test augmentation sequence creation
        sequence = augmentation.create_augmentation_sequence(
            ["geometric", "color", "invalid", "noise"]
        )
        assert sequence == ["geometric", "color", "noise"]


class TestExtensionsRegistry:
    """Test ExtensionsRegistry with unified configuration."""

    @pytest.fixture
    def rngs(self):
        """Create RNG for tests."""
        return nnx.Rngs(42)

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for testing."""
        return ExtensionsRegistry()

    def test_register_extension(self, registry):
        """Test registering extensions."""

        class TestExtension(nnx.Module):
            def __init__(self, config: ExtensionConfig, *, rngs: nnx.Rngs):
                super().__init__()
                self.config = config

        registry.register_extension(
            "test_ext",
            TestExtension,
            modalities=["image", "text"],
            capabilities=["augmentation"],
            description="Test extension",
        )

        # Check registration
        assert "test_ext" in registry.list_all_extensions()
        info = registry.get_extension_info("test_ext")
        assert info["modalities"] == ["image", "text"]
        assert info["capabilities"] == ["augmentation"]

    def test_registry_queries(self, registry):
        """Test registry query methods."""

        # Register test extensions
        class ExtA(nnx.Module):
            def __init__(self, config: ExtensionConfig, *, rngs: nnx.Rngs):
                super().__init__()

        class ExtB(nnx.Module):
            def __init__(self, config: ExtensionConfig, *, rngs: nnx.Rngs):
                super().__init__()

        registry.register_extension(
            "ext_a",
            ExtA,
            modalities=["test_modality"],
            capabilities=["test_augmentation", "preprocessing"],
        )

        registry.register_extension(
            "ext_b",
            ExtB,
            modalities=["test_text", "test_modality"],
            capabilities=["test_tokenization"],
        )

        # Test modality queries
        test_exts = registry.get_extensions_for_modality("test_modality")
        assert set(test_exts) == {"ext_a", "ext_b"}

        text_exts = registry.get_extensions_for_modality("test_text")
        assert text_exts == ["ext_b"]

        # Test capability queries
        aug_exts = registry.get_extensions_by_capability("test_augmentation")
        assert aug_exts == ["ext_a"]

        # Test search
        results = registry.search_extensions(
            modality="test_modality", capability="test_augmentation"
        )
        assert results == ["ext_a"]

    def test_create_extension_with_typed_config(self, registry, rngs):
        """Test creating extensions with typed configuration."""

        class TestExtension(nnx.Module):
            def __init__(self, config: ExtensionConfig, *, rngs: nnx.Rngs):
                super().__init__()
                if not isinstance(config, ExtensionConfig):
                    raise TypeError(f"config must be ExtensionConfig, got {type(config).__name__}")
                self.config = config
                self.rngs = rngs

        registry.register_extension(
            "typed_ext",
            TestExtension,
            modalities=["test"],
            capabilities=["test"],
        )

        # Create with typed config
        config = ExtensionConfig(name="typed_ext", weight=0.7, enabled=False)
        extension = registry.create_extension("typed_ext", config, rngs=rngs)

        assert extension.config.weight == 0.7
        assert extension.config.enabled is False

    def test_extension_pipeline(self, registry, rngs):
        """Test creating extension pipeline with typed configs."""

        class ExtA(nnx.Module):
            def __init__(self, config: ExtensionConfig, *, rngs: nnx.Rngs):
                super().__init__()
                self.name = "A"
                self.config = config

        class ExtB(nnx.Module):
            def __init__(self, config: ExtensionConfig, *, rngs: nnx.Rngs):
                super().__init__()
                self.name = "B"
                self.config = config

        registry.register_extension("ext_a", ExtA, ["test"], ["test"])
        registry.register_extension("ext_b", ExtB, ["test"], ["test"])

        # Create pipeline with typed configs as list of tuples
        pipeline_configs = [
            ("ext_a", ExtensionConfig(name="ext_a", weight=0.5)),
            ("ext_b", ExtensionConfig(name="ext_b", weight=0.8)),
        ]

        pipeline = registry.create_extension_pipeline(pipeline_configs, rngs=rngs)

        assert len(pipeline) == 2
        assert pipeline[0].name == "A"
        assert pipeline[0].config.weight == 0.5
        assert pipeline[1].name == "B"
        assert pipeline[1].config.weight == 0.8

    def test_registry_validation(self, registry):
        """Test extension compatibility validation."""

        class TestExt(nnx.Module):
            pass

        registry.register_extension(
            "image_ext",
            TestExt,
            modalities=["image"],
            capabilities=["test"],
        )

        # Test validation
        compatibility = registry.validate_extension_compatibility(
            ["image_ext", "nonexistent"], "image"
        )

        assert compatibility["image_ext"] is True
        assert compatibility["nonexistent"] is False

        # Test wrong modality
        compatibility = registry.validate_extension_compatibility(["image_ext"], "text")
        assert compatibility["image_ext"] is False

    def test_global_registry(self):
        """Test global registry singleton."""
        registry1 = get_extensions_registry()
        registry2 = get_extensions_registry()

        assert registry1 is registry2
