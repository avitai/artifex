"""Tests for ModalityConfig frozen dataclass.

This module tests the ModalityConfig frozen dataclass which replaces the
Pydantic-based ModalityConfig. Tests verify:
1. Basic instantiation and frozen behavior
2. Field validation (types, required fields)
3. Default values
4. Serialization (to_dict, from_dict)
5. Edge cases and error handling
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration.modality_config import ModalityConfig


class TestModalityConfigBasics:
    """Test basic ModalityConfig functionality."""

    def test_create_minimal(self):
        """Test creating ModalityConfig with minimal required fields."""
        config = ModalityConfig(
            name="image-modality",
            modality_name="image",
        )
        assert config.name == "image-modality"
        assert config.modality_name == "image"
        assert config.supported_models == ()  # default empty tuple
        assert config.preprocessing_steps == ()  # default empty tuple
        assert config.default_metrics == ()  # default empty tuple
        assert config.extensions == {}  # default empty dict

    def test_create_full(self):
        """Test creating ModalityConfig with all fields."""
        config = ModalityConfig(
            name="image-modality",
            modality_name="image",
            supported_models=("vae", "gan", "diffusion"),
            preprocessing_steps=(
                {"type": "resize", "size": 256},
                {"type": "normalize", "mean": 0.5, "std": 0.5},
            ),
            default_metrics=("fid", "inception_score", "ssim"),
            extensions={"color_space": "rgb", "channels": 3},
            description="Image modality configuration",
            tags=("vision", "2d"),
        )
        assert config.name == "image-modality"
        assert config.modality_name == "image"
        assert config.supported_models == ("vae", "gan", "diffusion")
        assert len(config.preprocessing_steps) == 2
        assert config.preprocessing_steps[0] == {"type": "resize", "size": 256}
        assert config.default_metrics == ("fid", "inception_score", "ssim")
        assert config.extensions == {"color_space": "rgb", "channels": 3}
        assert config.description == "Image modality configuration"
        assert config.tags == ("vision", "2d")

    def test_frozen(self):
        """Test that ModalityConfig is frozen (immutable)."""
        config = ModalityConfig(name="test", modality_name="image")
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.modality_name = "text"

    def test_hash(self):
        """Test that ModalityConfig instances are not hashable due to mutable fields."""
        config1 = ModalityConfig(name="test", modality_name="image")
        # Configs with dict/tuple fields containing dicts are not hashable
        with pytest.raises(TypeError, match="unhashable type"):
            hash(config1)


class TestModalityConfigValidation:
    """Test ModalityConfig validation."""

    def test_modality_name_required(self):
        """Test that modality_name is required (validated in __post_init__)."""
        with pytest.raises(ValueError, match="modality_name cannot be empty"):
            ModalityConfig(name="test")

    def test_modality_name_empty_string(self):
        """Test that empty modality_name is caught."""
        with pytest.raises(ValueError, match="modality_name cannot be empty"):
            ModalityConfig(name="test", modality_name="")

    def test_modality_name_whitespace(self):
        """Test that whitespace-only modality_name is caught."""
        with pytest.raises(ValueError, match="modality_name cannot be empty"):
            ModalityConfig(name="test", modality_name="   ")

    def test_supported_models_accepts_tuple(self):
        """Test that supported_models accepts tuple."""
        config = ModalityConfig(
            name="test",
            modality_name="image",
            supported_models=("vae", "gan"),
        )
        assert config.supported_models == ("vae", "gan")

    def test_preprocessing_steps_accepts_tuple_of_dicts(self):
        """Test that preprocessing_steps accepts tuple of dicts."""
        steps = ({"type": "resize"}, {"type": "normalize"})
        config = ModalityConfig(
            name="test",
            modality_name="image",
            preprocessing_steps=steps,
        )
        assert config.preprocessing_steps == steps

    def test_default_metrics_accepts_tuple(self):
        """Test that default_metrics accepts tuple."""
        config = ModalityConfig(
            name="test",
            modality_name="image",
            default_metrics=("fid", "ssim"),
        )
        assert config.default_metrics == ("fid", "ssim")


class TestModalityConfigDefaults:
    """Test ModalityConfig default values."""

    def test_default_supported_models(self):
        """Test default supported_models is empty tuple."""
        config = ModalityConfig(name="test", modality_name="image")
        assert config.supported_models == ()

    def test_default_preprocessing_steps(self):
        """Test default preprocessing_steps is empty tuple."""
        config = ModalityConfig(name="test", modality_name="image")
        assert config.preprocessing_steps == ()

    def test_default_metrics(self):
        """Test default default_metrics is empty tuple."""
        config = ModalityConfig(name="test", modality_name="image")
        assert config.default_metrics == ()

    def test_default_extensions(self):
        """Test default extensions is empty dict."""
        config = ModalityConfig(name="test", modality_name="image")
        assert config.extensions == {}

    def test_default_description(self):
        """Test default description from BaseConfig."""
        config = ModalityConfig(name="test", modality_name="image")
        assert config.description == ""

    def test_default_tags(self):
        """Test default tags from BaseConfig."""
        config = ModalityConfig(name="test", modality_name="image")
        assert config.tags == ()


class TestModalityConfigSerialization:
    """Test ModalityConfig serialization."""

    def test_to_dict_minimal(self):
        """Test converting minimal config to dict."""
        config = ModalityConfig(name="test", modality_name="image")
        data = config.to_dict()
        assert data["name"] == "test"
        assert data["modality_name"] == "image"
        assert data["supported_models"] == ()
        assert data["preprocessing_steps"] == ()
        assert data["default_metrics"] == ()
        assert data["extensions"] == {}

    def test_to_dict_full(self):
        """Test converting full config to dict."""
        config = ModalityConfig(
            name="image-modality",
            modality_name="image",
            supported_models=("vae", "gan"),
            preprocessing_steps=({"type": "resize", "size": 256},),
            default_metrics=("fid", "ssim"),
            extensions={"channels": 3},
        )
        data = config.to_dict()
        assert data["modality_name"] == "image"
        assert data["supported_models"] == ("vae", "gan")
        assert data["preprocessing_steps"] == ({"type": "resize", "size": 256},)
        assert data["default_metrics"] == ("fid", "ssim")
        assert data["extensions"] == {"channels": 3}

    def test_from_dict_minimal(self):
        """Test creating config from minimal dict."""
        data = {"name": "test", "modality_name": "image"}
        config = ModalityConfig.from_dict(data)
        assert config.name == "test"
        assert config.modality_name == "image"
        assert config.supported_models == ()

    def test_from_dict_full(self):
        """Test creating config from full dict."""
        data = {
            "name": "image-modality",
            "modality_name": "image",
            "supported_models": ["vae", "gan"],  # lists converted to tuples
            "preprocessing_steps": [{"type": "resize"}],
            "default_metrics": ["fid", "ssim"],
            "extensions": {"channels": 3},
        }
        config = ModalityConfig.from_dict(data)
        assert config.modality_name == "image"
        assert config.supported_models == ("vae", "gan")  # converted to tuple
        assert config.preprocessing_steps == ({"type": "resize"},)
        assert config.default_metrics == ("fid", "ssim")

    def test_from_dict_converts_lists_to_tuples(self):
        """Test that from_dict converts lists to tuples."""
        data = {
            "name": "test",
            "modality_name": "image",
            "supported_models": ["vae", "gan"],
            "default_metrics": ["fid"],
        }
        config = ModalityConfig.from_dict(data)
        assert isinstance(config.supported_models, tuple)
        assert isinstance(config.default_metrics, tuple)
        assert config.supported_models == ("vae", "gan")

    def test_roundtrip(self):
        """Test that to_dict -> from_dict roundtrip works."""
        original = ModalityConfig(
            name="image-modality",
            modality_name="image",
            supported_models=("vae", "gan"),
            default_metrics=("fid",),
            extensions={"channels": 3},
        )
        data = original.to_dict()
        restored = ModalityConfig.from_dict(data)
        assert restored.name == original.name
        assert restored.modality_name == original.modality_name
        assert restored.supported_models == original.supported_models
        assert restored.default_metrics == original.default_metrics
        assert restored.extensions == original.extensions


class TestModalityConfigEdgeCases:
    """Test ModalityConfig edge cases."""

    def test_empty_supported_models(self):
        """Test that empty supported_models is valid."""
        config = ModalityConfig(
            name="test",
            modality_name="image",
            supported_models=(),
        )
        assert config.supported_models == ()

    def test_empty_preprocessing_steps(self):
        """Test that empty preprocessing_steps is valid."""
        config = ModalityConfig(
            name="test",
            modality_name="image",
            preprocessing_steps=(),
        )
        assert config.preprocessing_steps == ()

    def test_empty_default_metrics(self):
        """Test that empty default_metrics is valid."""
        config = ModalityConfig(
            name="test",
            modality_name="image",
            default_metrics=(),
        )
        assert config.default_metrics == ()

    def test_empty_extensions(self):
        """Test that empty extensions is valid."""
        config = ModalityConfig(
            name="test",
            modality_name="image",
            extensions={},
        )
        assert config.extensions == {}

    def test_complex_preprocessing_steps(self):
        """Test that complex preprocessing_steps are handled."""
        steps = (
            {"type": "resize", "size": 256, "interpolation": "bilinear"},
            {"type": "normalize", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            {"type": "augment", "rotation": 15, "flip": True},
        )
        config = ModalityConfig(
            name="test",
            modality_name="image",
            preprocessing_steps=steps,
        )
        assert config.preprocessing_steps == steps
        assert len(config.preprocessing_steps) == 3

    def test_complex_extensions(self):
        """Test that complex extensions are handled."""
        extensions = {
            "color_space": "rgb",
            "channels": 3,
            "resolution": {"width": 256, "height": 256},
            "supported_formats": ["jpg", "png", "webp"],
        }
        config = ModalityConfig(
            name="test",
            modality_name="image",
            extensions=extensions,
        )
        assert config.extensions == extensions

    def test_single_model_in_supported_models(self):
        """Test single model in supported_models."""
        config = ModalityConfig(
            name="test",
            modality_name="text",
            supported_models=("transformer",),
        )
        assert config.supported_models == ("transformer",)
        assert len(config.supported_models) == 1

    def test_many_metrics(self):
        """Test many metrics in default_metrics."""
        metrics = ("fid", "inception_score", "ssim", "psnr", "lpips", "kid")
        config = ModalityConfig(
            name="test",
            modality_name="image",
            default_metrics=metrics,
        )
        assert config.default_metrics == metrics
        assert len(config.default_metrics) == 6
