"""Tests for diffusion guidance techniques.

This module provides comprehensive tests for guidance methods including
classifier-free guidance, classifier guidance, and conditional diffusion.

DiffusionModel uses the (config, *, rngs) signature pattern.
Backbone is created internally via create_backbone factory from config.backbone.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    DiffusionConfig,
    NoiseScheduleConfig,
    UNet2DConditionBackboneConfig,
    UNetBackboneConfig,
)
from artifex.generative_models.core.configuration.diffusion_config import (
    ConditionalDiffusionConfig,
)
from artifex.generative_models.models.diffusion.base import DiffusionModel
from artifex.generative_models.models.diffusion.guidance import (
    apply_guidance,
    ClassifierFreeGuidance,
    ClassifierGuidance,
    ConditionalDiffusionMixin,
    cosine_guidance_schedule,
    GuidedDiffusionModel,
    linear_guidance_schedule,
)


class SimpleClassifier(nnx.Module):
    """Simple classifier for testing."""

    def __init__(self, num_classes: int = 10, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear = nnx.Linear(784, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Flatten input
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        return self.linear(x_flat)


def create_test_diffusion_config() -> DiffusionConfig:
    """Create a test DiffusionConfig for testing."""
    backbone = UNetBackboneConfig(
        name="test_unet",
        hidden_dims=(32, 64),
        activation="relu",
        in_channels=1,
        out_channels=1,
        channel_mult=(1, 2),
        num_res_blocks=1,
    )
    noise_schedule = NoiseScheduleConfig(
        name="test_schedule",
        num_timesteps=100,
        schedule_type="linear",
        beta_start=1e-4,
        beta_end=0.02,
    )
    return DiffusionConfig(
        name="test_diffusion",
        backbone=backbone,
        noise_schedule=noise_schedule,
        input_shape=(1, 28, 28),  # (C, H, W) format
    )


def create_conditional_diffusion_config(conditioning_dim: int = 128) -> ConditionalDiffusionConfig:
    """Create a test ConditionalDiffusionConfig for testing."""
    backbone = UNetBackboneConfig(
        name="test_unet",
        hidden_dims=(32, 64),
        activation="relu",
        in_channels=1,
        out_channels=1,
        channel_mult=(1, 2),
        num_res_blocks=1,
    )
    noise_schedule = NoiseScheduleConfig(
        name="test_schedule",
        num_timesteps=100,
        schedule_type="linear",
        beta_start=1e-4,
        beta_end=0.02,
    )
    return ConditionalDiffusionConfig(
        name="test_conditional_diffusion",
        backbone=backbone,
        noise_schedule=noise_schedule,
        input_shape=(1, 28, 28),  # (C, H, W) format
        conditioning_dim=conditioning_dim,
    )


def create_cfg_test_config() -> DiffusionConfig:
    """Create a DiffusionConfig with UNet2DCondition backbone for CFG testing.

    The UNet2DCondition backbone actually uses conditioning (cross-attention),
    unlike the base UNet which ignores conditioning. This is required for
    classifier-free guidance tests where different conditioning should produce
    different outputs.
    """
    backbone = UNet2DConditionBackboneConfig(
        name="test_unet2d_condition",
        hidden_dims=(32, 64),
        activation="silu",
        in_channels=1,
        out_channels=1,
        cross_attention_dim=8,  # Must be divisible by num_heads
        num_heads=2,  # Must divide cross_attention_dim
        num_res_blocks=1,
        attention_levels=(0,),  # Only first level to keep test fast
        time_embedding_dim=32,
    )
    noise_schedule = NoiseScheduleConfig(
        name="test_schedule",
        num_timesteps=100,
        schedule_type="linear",
        beta_start=1e-4,
        beta_end=0.02,
    )
    return DiffusionConfig(
        name="test_cfg_diffusion",
        backbone=backbone,
        noise_schedule=noise_schedule,
        input_shape=(1, 28, 28),  # (C, H, W) format
    )


class TestClassifierFreeGuidance:
    """Tests for classifier-free guidance."""

    @pytest.fixture
    def base_model(self, base_rngs):
        """Create a simple diffusion model for testing."""
        config = create_test_diffusion_config()
        # DiffusionModel uses (config, *, rngs) signature
        # Backbone is created internally via create_backbone factory
        return DiffusionModel(config, rngs=base_rngs)

    @pytest.fixture
    def cfg_model(self, base_rngs):
        """Create a conditioning-aware diffusion model for CFG testing.

        Uses UNet2DCondition backbone which actually responds to conditioning,
        unlike the base UNet which ignores it.
        """
        config = create_cfg_test_config()
        return DiffusionModel(config, rngs=base_rngs)

    def test_cfg_initialization(self):
        """Test ClassifierFreeGuidance initialization."""
        guidance = ClassifierFreeGuidance(guidance_scale=7.5)
        assert guidance.guidance_scale == 7.5
        assert guidance.unconditional_conditioning is None

        # With unconditional conditioning
        uncond = jnp.zeros((1, 10))
        guidance = ClassifierFreeGuidance(guidance_scale=3.0, unconditional_conditioning=uncond)
        assert guidance.guidance_scale == 3.0
        assert jnp.array_equal(guidance.unconditional_conditioning, uncond)

    def test_cfg_guidance_application(self, base_model, base_rngs):
        """Test applying classifier-free guidance."""
        guidance = ClassifierFreeGuidance(guidance_scale=7.5)

        # Create test inputs
        x = jnp.ones((4, 28, 28, 1))
        t = jnp.array([50, 50, 50, 50])
        conditioning = jnp.ones((4, 10))

        # Apply guidance (NNX stores rngs at init, no need to pass)
        guided_noise = guidance(base_model, x, t, conditioning)

        # Should return an array
        assert isinstance(guided_noise, jax.Array)
        assert guided_noise.shape == x.shape
        assert jnp.all(jnp.isfinite(guided_noise))

    def test_cfg_different_scales(self, cfg_model, base_rngs):
        """Test different guidance scales produce different outputs.

        This test uses a conditioning-aware model (UNet2DCondition) to verify
        that different guidance scales produce different outputs.
        """
        x = jnp.ones((2, 28, 28, 1))
        t = jnp.array([50, 50])
        # Conditioning with shape matching cross_attention_dim (8)
        # Adding seq_len dimension for text embedding: [batch, seq_len, cross_attention_dim]
        conditioning = jnp.ones((2, 4, 8))

        # Test different scales
        guidance_weak = ClassifierFreeGuidance(guidance_scale=1.0)
        guidance_strong = ClassifierFreeGuidance(guidance_scale=10.0)

        # NNX stores rngs at init, no need to pass
        output_weak = guidance_weak(cfg_model, x, t, conditioning)
        output_strong = guidance_strong(cfg_model, x, t, conditioning)

        # Different scales should produce different outputs
        assert not jnp.allclose(output_weak, output_strong, atol=1e-5)


class TestClassifierGuidance:
    """Tests for classifier guidance."""

    @pytest.fixture
    def classifier(self, base_rngs):
        """Create a simple classifier for testing."""
        return SimpleClassifier(num_classes=10, rngs=base_rngs)

    @pytest.fixture
    def base_model(self, base_rngs):
        """Create a simple diffusion model for testing."""
        config = create_test_diffusion_config()
        # DiffusionModel uses (config, *, rngs) signature
        # Backbone is created internally via create_backbone factory
        return DiffusionModel(config, rngs=base_rngs)

    def test_classifier_guidance_initialization(self, classifier):
        """Test ClassifierGuidance initialization."""
        guidance = ClassifierGuidance(classifier, guidance_scale=1.0, class_label=5)
        assert guidance.guidance_scale == 1.0
        assert guidance.class_label == 5
        assert guidance.classifier is classifier

    def test_classifier_guidance_scaling(self, classifier):
        """Test input scaling for classifier."""
        guidance = ClassifierGuidance(classifier, guidance_scale=1.0)

        # Input in [-1, 1] range
        x = jnp.array([[-1.0, 0.0, 1.0]])
        scaled = guidance._scale_for_classifier(x)

        # Should be scaled to [0, 1]
        assert jnp.allclose(scaled, jnp.array([[0.0, 0.5, 1.0]]))

    def test_classifier_guidance_requires_class_label(self, classifier, base_model, base_rngs):
        """Test that classifier guidance requires a class label."""
        guidance = ClassifierGuidance(classifier, guidance_scale=1.0)

        x = jnp.ones((2, 28, 28, 1))
        t = jnp.array([50, 50])

        # Should raise error without class label (NNX stores rngs at init)
        with pytest.raises(ValueError, match="No target class specified"):
            guidance(base_model, x, t)

    def test_classifier_guidance_with_class_label(self, classifier, base_model, base_rngs):
        """Test classifier guidance with specified class."""
        guidance = ClassifierGuidance(classifier, guidance_scale=1.0, class_label=3)

        x = jnp.ones((2, 28, 28, 1))
        t = jnp.array([50, 50])

        # Should work with class label (NNX stores rngs at init)
        guided_noise = guidance(base_model, x, t)

        assert isinstance(guided_noise, jax.Array)
        assert guided_noise.shape == x.shape
        assert jnp.all(jnp.isfinite(guided_noise))


class TestGuidedDiffusionModel:
    """Tests for guided diffusion model."""

    @pytest.fixture
    def guided_config(self):
        """Configuration for guided diffusion model."""
        backbone = UNetBackboneConfig(
            name="test_unet_guided",
            hidden_dims=(32, 64),
            activation="relu",
            in_channels=1,
            out_channels=1,
            channel_mult=(1, 2),
            num_res_blocks=1,
        )
        noise_schedule = NoiseScheduleConfig(
            name="test_schedule_guided",
            num_timesteps=50,
            schedule_type="linear",
            beta_start=1e-4,
            beta_end=0.02,
        )
        return DiffusionConfig(
            name="test_guided_diffusion",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(1, 28, 28),  # (C, H, W) format
        )

    def test_guided_model_initialization_no_guidance(self, guided_config, base_rngs):
        """Test initialization without guidance."""
        # GuidedDiffusionModel uses (config, *, rngs) signature
        model = GuidedDiffusionModel(guided_config, rngs=base_rngs, guidance_method=None)

        assert model.guidance_method is None
        assert model.guidance is None

    def test_guided_model_initialization_cfg(self, guided_config, base_rngs):
        """Test initialization with classifier-free guidance."""
        model = GuidedDiffusionModel(
            guided_config,
            rngs=base_rngs,
            guidance_method="classifier_free",
            guidance_scale=7.5,
        )

        assert model.guidance_method == "classifier_free"
        assert isinstance(model.guidance, ClassifierFreeGuidance)
        assert model.guidance.guidance_scale == 7.5

    def test_guided_model_initialization_classifier(self, guided_config, base_rngs):
        """Test initialization with classifier guidance."""
        classifier = SimpleClassifier(num_classes=10, rngs=base_rngs)

        model = GuidedDiffusionModel(
            guided_config,
            rngs=base_rngs,
            guidance_method="classifier",
            classifier=classifier,
            guidance_scale=2.0,
        )

        assert model.guidance_method == "classifier"
        assert isinstance(model.guidance, ClassifierGuidance)
        assert model.guidance.guidance_scale == 2.0

    def test_guided_model_requires_classifier(self, guided_config, base_rngs):
        """Test that classifier guidance requires a classifier."""
        with pytest.raises(ValueError, match="Classifier required"):
            GuidedDiffusionModel(
                guided_config,
                rngs=base_rngs,
                guidance_method="classifier",
                classifier=None,
            )

    def test_guided_sample_step_no_guidance(self, guided_config, base_rngs):
        """Test sampling step without guidance."""
        model = GuidedDiffusionModel(guided_config, rngs=base_rngs, guidance_method=None)

        x = jnp.ones((2, 28, 28, 1))
        t = jnp.array([25, 25])

        # NNX stores rngs at init, no need to pass
        output = model.guided_sample_step(x, t)
        assert isinstance(output, jax.Array)
        assert jnp.all(jnp.isfinite(output))


class TestConditionalDiffusionMixin:
    """Tests for conditional diffusion mixin."""

    def test_conditional_mixin_initialization(self, base_rngs):
        """Test ConditionalDiffusionMixin initialization."""

        # Create a simple test class that uses the mixin
        # ConditionalDiffusionMixin uses (config, *, rngs) signature
        # Config must have conditioning_dim field (ConditionalDiffusionConfig)
        class TestModel(ConditionalDiffusionMixin, DiffusionModel):
            def __init__(self, config, *, rngs):
                super().__init__(config, rngs=rngs)

        config = create_conditional_diffusion_config(conditioning_dim=128)
        model = TestModel(config, rngs=base_rngs)
        assert model.conditioning_dim == 128

    def test_conditional_forward_with_conditioning(self, base_rngs):
        """Test forward pass with conditioning."""

        class TestModel(ConditionalDiffusionMixin, DiffusionModel):
            def __init__(self, config, *, rngs):
                super().__init__(config, rngs=rngs)

        config = create_conditional_diffusion_config(conditioning_dim=128)
        model = TestModel(config, rngs=base_rngs)

        x = jnp.ones((2, 28, 28, 1))
        t = jnp.array([25, 25])
        conditioning = jnp.ones((2, 128))

        # Should accept conditioning (NNX stores rngs at init)
        output = model(x, t, conditioning=conditioning)
        assert isinstance(output, dict)


class TestGuidanceHelperFunctions:
    """Tests for guidance helper functions."""

    def test_apply_guidance(self):
        """Test apply_guidance function."""
        noise_cond = jnp.ones((4, 28, 28, 1))
        noise_uncond = jnp.zeros((4, 28, 28, 1))

        # Test with different guidance scales
        guided_weak = apply_guidance(noise_cond, noise_uncond, guidance_scale=1.0)
        guided_strong = apply_guidance(noise_cond, noise_uncond, guidance_scale=7.5)

        # Weak guidance should be closer to conditional
        assert jnp.allclose(guided_weak, noise_cond)

        # Strong guidance should amplify difference
        expected_strong = noise_uncond + 7.5 * (noise_cond - noise_uncond)
        assert jnp.allclose(guided_strong, expected_strong)

    def test_linear_guidance_schedule(self):
        """Test linear guidance schedule."""
        # At start
        scale_start = linear_guidance_schedule(0, 100, start_scale=1.0, end_scale=7.5)
        assert jnp.isclose(scale_start, 1.0)

        # At end
        scale_end = linear_guidance_schedule(100, 100, start_scale=1.0, end_scale=7.5)
        assert jnp.isclose(scale_end, 7.5)

        # At middle
        scale_mid = linear_guidance_schedule(50, 100, start_scale=1.0, end_scale=7.5)
        assert jnp.isclose(scale_mid, 4.25)  # (1.0 + 7.5) / 2

        # Should be monotonically increasing
        scales = [linear_guidance_schedule(i, 100, 1.0, 7.5) for i in range(0, 101, 10)]
        assert all(scales[i] <= scales[i + 1] for i in range(len(scales) - 1))

    def test_cosine_guidance_schedule(self):
        """Test cosine guidance schedule."""
        # At start (cosine starts high and decreases)
        scale_start = cosine_guidance_schedule(0, 100, start_scale=1.0, end_scale=7.5)
        # At step 0: alpha = 0.5 * (1 + cos(0)) = 0.5 * 2 = 1.0
        # scale = 7.5 + 1.0 * (1.0 - 7.5) = 7.5 - 6.5 = 1.0
        assert jnp.isclose(scale_start, 1.0, atol=0.01)

        # At end (cosine ends low)
        scale_end = cosine_guidance_schedule(100, 100, start_scale=1.0, end_scale=7.5)
        # At step 100: alpha = 0.5 * (1 + cos(pi)) = 0.5 * 0 = 0.0
        # scale = 7.5 + 0.0 * (1.0 - 7.5) = 7.5
        assert jnp.isclose(scale_end, 7.5, atol=0.01)

        # Should be monotonically increasing for cosine schedule (from 1.0 to 7.5)
        scales = [cosine_guidance_schedule(i, 100, 1.0, 7.5) for i in range(0, 101, 10)]
        assert all(scales[i] <= scales[i + 1] for i in range(len(scales) - 1))

    def test_guidance_schedules_valid_range(self):
        """Test that guidance schedules stay within valid range."""
        total_steps = 50

        for step in range(total_steps + 1):
            linear_scale = linear_guidance_schedule(step, total_steps, 1.0, 7.5)
            cosine_scale = cosine_guidance_schedule(step, total_steps, 1.0, 7.5)

            # Both should be within [1.0, 7.5]
            assert 1.0 <= linear_scale <= 7.5
            assert 1.0 <= cosine_scale <= 7.5


class TestGuidanceIntegration:
    """Integration tests for guidance with diffusion models."""

    def test_training_eval_mode_compatibility(self, base_rngs):
        """Test that guidance works with train/eval mode switching."""
        config = create_test_diffusion_config()

        # GuidedDiffusionModel uses (config, *, rngs) signature
        model = GuidedDiffusionModel(
            config,
            rngs=base_rngs,
            guidance_method="classifier_free",
            guidance_scale=7.5,
        )

        x = jnp.ones((2, 28, 28, 1))
        t = jnp.array([25, 25])
        conditioning = jnp.ones((2, 10))

        # Should work in both modes (NNX stores rngs at init)
        model.train()
        output_train = model.guided_sample_step(x, t, conditioning=conditioning)
        assert jnp.all(jnp.isfinite(output_train))

        model.eval()
        output_eval = model.guided_sample_step(x, t, conditioning=conditioning)
        assert jnp.all(jnp.isfinite(output_eval))
