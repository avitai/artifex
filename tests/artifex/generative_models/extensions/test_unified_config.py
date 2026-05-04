"""Tests for extension system with unified configuration."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from artifex.generative_models.core.configuration import (
    ExtensionConfig,
    ImageAugmentationConfig,
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

    def test_augmentation_with_typed_config(self, rngs):
        """Test image augmentation with typed configuration."""
        config = ImageAugmentationConfig(
            name="image_augmentation",
            weight=1.0,
            enabled=True,
            probability=1.0,
            color_jitter=True,
            brightness_range=(0.8, 1.2),
            contrast_range=(0.9, 1.1),
        )

        augmentation = AdvancedImageAugmentation(config, rngs=rngs)

        # Check basic configuration was applied
        assert augmentation.weight == 1.0
        assert augmentation.enabled is True
        assert isinstance(augmentation.config, ImageAugmentationConfig)

    def test_augmentation_call(self, rngs):
        """Test augmentation forward pass."""
        # Create with default config
        config = ImageAugmentationConfig(name="test_augmentation")
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
        config = ImageAugmentationConfig(name="test_augmentation_methods")
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

    def test_augmentation_rejects_modality_config_entrypoint(self, rngs):
        """The runtime augmentation surface should not accept modality configs directly."""
        config = ModalityConfig(name="image", modality_name="image")

        with pytest.raises(TypeError, match="ImageAugmentationConfig"):
            AdvancedImageAugmentation(config, rngs=rngs)

    def test_augmentation_probability_zero_skips_runtime_changes(self, rngs):
        """Probability should control whether augmentation is applied at all."""
        config = ImageAugmentationConfig(
            name="zero_probability",
            probability=0.0,
            color_jitter=True,
            brightness_range=(0.5, 1.5),
            contrast_range=(0.5, 1.5),
        )
        augmentation = AdvancedImageAugmentation(config, rngs=rngs)
        images = jnp.arange(2 * 8 * 8 * 3, dtype=jnp.float32).reshape(2, 8, 8, 3) / 255.0

        augmented = augmentation(images)

        assert jnp.allclose(augmented, images)

    def test_augmentation_config_level_deterministic_flag_skips_runtime_changes(self, rngs):
        """The typed deterministic field should bypass stochastic augmentation."""
        config = ImageAugmentationConfig(
            name="deterministic",
            deterministic=True,
            color_jitter=True,
            brightness_range=(0.5, 1.5),
            contrast_range=(0.5, 1.5),
        )
        augmentation = AdvancedImageAugmentation(config, rngs=rngs)
        images = jnp.arange(2 * 8 * 8 * 3, dtype=jnp.float32).reshape(2, 8, 8, 3) / 255.0

        augmented = augmentation(images)

        assert jnp.allclose(augmented, images)

    def test_vertical_flip_with_probability_one_flips_every_image(self, rngs):
        """Vertical flip should flip along the height axis when probability is certain."""
        config = ImageAugmentationConfig(name="vertical_flip", probability=1.0)
        augmentation = AdvancedImageAugmentation(config, rngs=rngs)
        images = jnp.arange(2 * 3 * 2 * 1, dtype=jnp.float32).reshape(2, 3, 2, 1)

        flipped = augmentation.apply_vertical_flip(images)

        assert jnp.allclose(flipped, jnp.flip(images, axis=1))

    def test_color_jitter_with_identity_ranges_preserves_images(self, rngs):
        """Identity brightness and contrast ranges should preserve image values."""
        config = ImageAugmentationConfig(
            name="identity_jitter",
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
        )
        augmentation = AdvancedImageAugmentation(config, rngs=rngs)
        images = jnp.linspace(0.0, 1.0, 2 * 4 * 4 * 3).reshape(2, 4, 4, 3)

        jittered = augmentation._apply_color_jitter(images)

        assert jnp.allclose(jittered, images)

    def test_zero_strength_optional_augmentations_preserve_images(self, rngs):
        """Optional saturation and noise helpers should honor zero-strength metadata."""
        config = ImageAugmentationConfig(
            name="zero_strength",
            metadata={"saturation_range": 0.0, "noise_level": 0.0},
        )
        augmentation = AdvancedImageAugmentation(config, rngs=rngs)
        images = jnp.linspace(0.0, 1.0, 2 * 4 * 4 * 3).reshape(2, 4, 4, 3)

        assert augmentation._metadata_float("missing", 0.75) == 0.75
        assert jnp.allclose(augmentation._apply_saturation(images), images)
        assert jnp.allclose(augmentation._apply_noise_injection(images), images)

    def test_affine_identity_and_zero_rotation_preserve_images(self, rngs):
        """Identity affine transforms and zero rotation should preserve image values."""
        config = ImageAugmentationConfig(name="identity_geometric")
        augmentation = AdvancedImageAugmentation(config, rngs=rngs)
        image = jnp.arange(4 * 4 * 1, dtype=jnp.float32).reshape(4, 4, 1) / 16.0
        transform = jnp.eye(3)

        transformed = augmentation._apply_affine_transform(image, transform)
        rotated = augmentation._apply_random_rotations(image[None, ...], max_rotation=0.0)

        assert jnp.allclose(transformed, image)
        assert jnp.allclose(rotated[0], image)

    def test_bilinear_interpolation_supports_grayscale_images(self, rngs):
        """Bilinear interpolation should return grayscale arrays without a channel axis."""
        config = ImageAugmentationConfig(name="grayscale_interpolation")
        augmentation = AdvancedImageAugmentation(config, rngs=rngs)
        image = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
        coords = jnp.array([[0.0, 1.0], [1.0, 2.0]])

        interpolated = augmentation._bilinear_interpolate(image, coords, coords)

        assert interpolated.shape == (2, 2)
        assert jnp.allclose(interpolated, jnp.array([[0.0, 4.0], [4.0, 8.0]]))

    def test_blur_helpers_preserve_shape_for_color_and_grayscale(self, rngs):
        """Blur helpers should support both color batches and grayscale images."""
        config = ImageAugmentationConfig(
            name="blur",
            metadata={"blur_probability": 1.0},
        )
        augmentation = AdvancedImageAugmentation(config, rngs=rngs)
        color = jnp.zeros((2, 5, 5, 3), dtype=jnp.float32).at[:, 2, 2, :].set(1.0)
        grayscale = jnp.zeros((5, 5), dtype=jnp.float32).at[2, 2].set(1.0)

        blurred_color = augmentation._apply_blur(color)
        blurred_grayscale = augmentation._gaussian_blur(grayscale)

        assert blurred_color.shape == color.shape
        assert blurred_grayscale.shape == grayscale.shape
        assert jnp.isfinite(blurred_color).all()
        assert jnp.isfinite(blurred_grayscale).all()

    def test_deterministic_augment_is_jittable_and_differentiable(self, rngs):
        """The pure deterministic augmentation bypass should compile under JAX transforms."""
        config = ImageAugmentationConfig(name="deterministic_jit")
        augmentation = AdvancedImageAugmentation(config, rngs=rngs)
        images = jnp.linspace(0.0, 1.0, 2 * 4 * 4 * 3).reshape(2, 4, 4, 3)

        def loss_fn(values):
            return jnp.sum(augmentation.augment(values, deterministic=True))

        compiled_value = jax.jit(loss_fn)(images)
        gradients = jax.grad(loss_fn)(images)

        assert compiled_value.shape == ()
        assert jnp.isfinite(compiled_value)
        assert jnp.allclose(gradients, jnp.ones_like(images))

    @pytest.mark.parametrize(
        "method_name",
        [
            "_apply_color_jitter",
            "_apply_saturation",
            "_apply_noise_injection",
            "_apply_blur",
            "apply_horizontal_flip",
            "apply_vertical_flip",
        ],
    )
    def test_rng_backed_augmentation_methods_are_nnx_jittable_and_differentiable(self, method_name):
        """RNG-backed augmentation helpers should compile through the NNX transform path."""
        config = ImageAugmentationConfig(
            name=f"{method_name}_jit",
            probability=1.0,
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            metadata={
                "blur_probability": 1.0,
                "noise_level": 0.0,
                "saturation_range": 0.0,
            },
        )
        augmentation = AdvancedImageAugmentation(config, rngs=nnx.Rngs(42))
        images = jnp.linspace(0.0, 1.0, 2 * 4 * 4 * 3).reshape(2, 4, 4, 3)

        def apply_method(module, values):
            return getattr(module, method_name)(values)

        compiled = nnx.jit(apply_method)
        gradients_fn = nnx.grad(
            lambda module, values: jnp.sum(apply_method(module, values)), argnums=1
        )

        transformed = compiled(augmentation, images)
        gradients = gradients_fn(augmentation, images)

        assert transformed.shape == images.shape
        assert jnp.isfinite(transformed).all()
        assert jnp.isfinite(gradients).all()

    def test_cutout_is_nnx_jittable_and_differentiable(self):
        """Cutout should compile through NNX transforms and expose finite mask gradients."""
        config = ImageAugmentationConfig(name="cutout_jit", probability=1.0)
        augmentation = AdvancedImageAugmentation(config, rngs=nnx.Rngs(42))
        images = jnp.ones((2, 4, 4, 1), dtype=jnp.float32)

        def apply_cutout(module, values):
            return module.apply_cutout(values, cutout_size=2, num_cutouts=1)

        compiled = nnx.jit(apply_cutout)
        gradients_fn = nnx.grad(
            lambda module, values: jnp.sum(apply_cutout(module, values)), argnums=1
        )

        transformed = compiled(augmentation, images)
        gradients = gradients_fn(augmentation, images)

        assert transformed.shape == images.shape
        assert jnp.isfinite(transformed).all()
        assert jnp.isfinite(gradients).all()

    @pytest.mark.parametrize(
        ("method_name", "input_shape"),
        [
            ("_gaussian_blur", (5, 5)),
            ("_gaussian_blur", (5, 5, 3)),
            ("_apply_conv2d", (5, 5)),
        ],
    )
    def test_pure_blur_helpers_are_jittable_and_differentiable(
        self, method_name, input_shape, rngs
    ):
        """Pure convolution helpers should compile and expose finite image gradients."""
        config = ImageAugmentationConfig(name=f"{method_name}_pure")
        augmentation = AdvancedImageAugmentation(config, rngs=rngs)
        image = jnp.linspace(0.0, 1.0, int(jnp.prod(jnp.array(input_shape)))).reshape(input_shape)
        kernel = jnp.ones((3, 3), dtype=jnp.float32) / 9.0

        def apply_helper(values):
            if method_name == "_apply_conv2d":
                return augmentation._apply_conv2d(values, kernel)
            return augmentation._gaussian_blur(values)

        compiled_value = jax.jit(apply_helper)(image)
        gradients = jax.grad(lambda values: jnp.sum(apply_helper(values)))(image)

        assert compiled_value.shape == image.shape
        assert jnp.isfinite(compiled_value).all()
        assert jnp.isfinite(gradients).all()

    def test_affine_and_interpolation_helpers_are_jittable_and_differentiable(self, rngs):
        """Pure geometric helpers should compile and expose finite image gradients."""
        config = ImageAugmentationConfig(name="geometric_jit")
        augmentation = AdvancedImageAugmentation(config, rngs=rngs)
        image = jnp.arange(4 * 4 * 1, dtype=jnp.float32).reshape(4, 4, 1) / 16.0
        transform = jnp.eye(3)
        coords = jnp.array([[0.0, 1.0], [1.0, 2.0]])

        affine = jax.jit(lambda values: augmentation._apply_affine_transform(values, transform))
        single_transform = jax.jit(
            lambda values: augmentation._transform_single_image(
                values,
                jnp.array(0.0),
                jnp.array(0.0),
                jnp.array(0.0),
                jnp.array(1.0),
            )
        )
        interpolation = jax.jit(
            lambda values: augmentation._bilinear_interpolate(values, coords, coords)
        )

        affine_grad = jax.grad(lambda values: jnp.sum(affine(values)))(image)
        single_transform_grad = jax.grad(lambda values: jnp.sum(single_transform(values)))(image)
        interpolation_grad = jax.grad(lambda values: jnp.sum(interpolation(values)))(image)

        assert affine(image).shape == image.shape
        assert single_transform(image).shape == image.shape
        assert interpolation(image).shape == (*coords.shape, 1)
        assert jnp.isfinite(affine_grad).all()
        assert jnp.isfinite(single_transform_grad).all()
        assert jnp.isfinite(interpolation_grad).all()


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

    def test_registry_default_augmentation_extension_config_is_typed(self, registry, rngs):
        """The image augmentation registry entry should materialize the typed image config."""
        extension = registry.create_extension("image_augmentation", rngs=rngs)

        assert isinstance(extension, AdvancedImageAugmentation)
        assert isinstance(extension.config, ImageAugmentationConfig)

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
