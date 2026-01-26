"""Tests for DiffusionBuilder with dataclass configs.

These tests define the expected behavior for DiffusionBuilder following TDD principles.
The builder should accept dataclass-based diffusion configs (DDPMConfig, ScoreDiffusionConfig, etc.)
with polymorphic BackboneConfig.

The signature should be:
    DiffusionBuilder.build(config: DiffusionConfig | DDPMConfig | ..., rngs: nnx.Rngs) -> DiffusionModel

NOT:
    DiffusionBuilder.build(config: ModelConfig, rngs: nnx.Rngs) -> DiffusionModel

Following Principle #4: Methods Take Configs, NOT Individual Parameters
"""

import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    DDPMConfig,
    DiffusionConfig,
    NoiseScheduleConfig,
    ScoreDiffusionConfig,
)
from artifex.generative_models.core.configuration.backbone_config import (
    DiTBackboneConfig,
    UNetBackboneConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def rngs():
    """Create nnx.Rngs for testing."""
    return nnx.Rngs(42)


@pytest.fixture
def unet_backbone_config():
    """Create a UNetBackboneConfig for testing."""
    return UNetBackboneConfig(
        name="test_unet",
        hidden_dims=(32, 64),
        activation="gelu",
        in_channels=3,
        out_channels=3,
        time_embedding_dim=64,
    )


@pytest.fixture
def noise_schedule_config():
    """Create a NoiseScheduleConfig for testing."""
    return NoiseScheduleConfig(
        name="test_schedule",
        schedule_type="linear",
        num_timesteps=10,  # Small for fast testing
        beta_start=1e-4,
        beta_end=2e-2,
    )


@pytest.fixture
def ddpm_config(unet_backbone_config, noise_schedule_config):
    """Create a DDPMConfig for testing."""
    return DDPMConfig(
        name="test_ddpm",
        input_shape=(8, 8, 3),  # Small for testing
        backbone=unet_backbone_config,
        noise_schedule=noise_schedule_config,
        loss_type="mse",
        clip_denoised=True,
    )


@pytest.fixture
def diffusion_config(unet_backbone_config, noise_schedule_config):
    """Create a DiffusionConfig for testing."""
    return DiffusionConfig(
        name="test_diffusion",
        input_shape=(8, 8, 3),
        backbone=unet_backbone_config,
        noise_schedule=noise_schedule_config,
    )


@pytest.fixture
def score_config(unet_backbone_config, noise_schedule_config):
    """Create a ScoreDiffusionConfig for testing."""
    return ScoreDiffusionConfig(
        name="test_score",
        input_shape=(8, 8, 3),
        backbone=unet_backbone_config,
        noise_schedule=noise_schedule_config,
        sigma_min=0.01,
        sigma_max=50.0,
        score_scaling=1.0,
    )


# =============================================================================
# Builder Existence Tests
# =============================================================================


class TestDiffusionBuilderExists:
    """Test that DiffusionBuilder exists and has required methods."""

    def test_builder_class_exists(self):
        """Test that DiffusionBuilder class exists."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder

        assert DiffusionBuilder is not None

    def test_builder_has_build_method(self):
        """Test that DiffusionBuilder has build method."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder

        builder = DiffusionBuilder()
        assert hasattr(builder, "build")
        assert callable(builder.build)


# =============================================================================
# Build DDPM Tests
# =============================================================================


class TestBuildDDPM:
    """Test building DDPMModel from DDPMConfig."""

    def test_build_ddpm_from_config(self, ddpm_config, rngs):
        """Test building DDPMModel from DDPMConfig."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder
        from artifex.generative_models.models.diffusion.ddpm import DDPMModel

        builder = DiffusionBuilder()
        model = builder.build(ddpm_config, rngs=rngs)

        assert model is not None
        assert isinstance(model, DDPMModel)

    def test_ddpm_has_required_attributes(self, ddpm_config, rngs):
        """Test that built DDPM has required attributes."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder

        builder = DiffusionBuilder()
        model = builder.build(ddpm_config, rngs=rngs)

        # From DDPMModel
        assert hasattr(model, "loss_type")
        assert hasattr(model, "clip_denoised")
        assert hasattr(model, "noise_steps")

        # From DiffusionModel base
        assert hasattr(model, "backbone")
        assert hasattr(model, "noise_schedule")
        assert hasattr(model, "q_sample")
        assert hasattr(model, "p_sample")
        assert hasattr(model, "generate")

    def test_ddpm_config_values_preserved(self, ddpm_config, rngs):
        """Test that config values are preserved in built model."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder

        builder = DiffusionBuilder()
        model = builder.build(ddpm_config, rngs=rngs)

        assert model.loss_type == ddpm_config.loss_type
        assert model.clip_denoised == ddpm_config.clip_denoised
        assert model.noise_steps == ddpm_config.noise_schedule.num_timesteps


# =============================================================================
# Build Base DiffusionModel Tests
# =============================================================================


class TestBuildBaseDiffusion:
    """Test building base DiffusionModel from DiffusionConfig."""

    def test_build_diffusion_from_config(self, diffusion_config, rngs):
        """Test building DiffusionModel from DiffusionConfig."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder
        from artifex.generative_models.models.diffusion.base import DiffusionModel

        builder = DiffusionBuilder()
        model = builder.build(diffusion_config, rngs=rngs)

        assert model is not None
        assert isinstance(model, DiffusionModel)

    def test_diffusion_has_backbone(self, diffusion_config, rngs):
        """Test that built model has backbone."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder
        from artifex.generative_models.models.backbones.unet import UNet

        builder = DiffusionBuilder()
        model = builder.build(diffusion_config, rngs=rngs)

        assert model.backbone is not None
        assert isinstance(model.backbone, UNet)


# =============================================================================
# Build ScoreDiffusion Tests
# =============================================================================


class TestBuildScoreDiffusion:
    """Test building ScoreDiffusionModel from ScoreDiffusionConfig."""

    def test_build_score_from_config(self, score_config, rngs):
        """Test building ScoreDiffusionModel from ScoreDiffusionConfig."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder
        from artifex.generative_models.models.diffusion.score import ScoreDiffusionModel

        builder = DiffusionBuilder()
        model = builder.build(score_config, rngs=rngs)

        assert model is not None
        assert isinstance(model, ScoreDiffusionModel)

    def test_score_config_values_preserved(self, score_config, rngs):
        """Test that config values are preserved in built model."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder

        builder = DiffusionBuilder()
        model = builder.build(score_config, rngs=rngs)

        assert model.sigma_min == score_config.sigma_min
        assert model.sigma_max == score_config.sigma_max
        assert model.score_scaling == score_config.score_scaling


# =============================================================================
# Polymorphic Backbone Tests
# =============================================================================


class TestPolymorphicBackbone:
    """Test that builder works with different backbone types."""

    def test_build_with_unet_backbone(self, noise_schedule_config, rngs):
        """Test building with UNetBackboneConfig."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder
        from artifex.generative_models.models.backbones.unet import UNet

        unet_config = UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(32, 64),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )
        config = DDPMConfig(
            name="test_ddpm_unet",
            input_shape=(8, 8, 3),
            backbone=unet_config,
            noise_schedule=noise_schedule_config,
        )

        builder = DiffusionBuilder()
        model = builder.build(config, rngs=rngs)

        assert isinstance(model.backbone, UNet)

    def test_build_with_dit_backbone(self, noise_schedule_config, rngs):
        """Test building with DiTBackboneConfig."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder
        from artifex.generative_models.models.backbones.dit import DiffusionTransformer

        dit_config = DiTBackboneConfig(
            name="test_dit",
            hidden_dims=(128,),  # Required by BaseNetworkConfig
            activation="gelu",
            img_size=8,
            patch_size=2,
            hidden_size=128,
            depth=2,
            num_heads=4,
        )
        config = DDPMConfig(
            name="test_ddpm_dit",
            input_shape=(8, 8, 3),
            backbone=dit_config,
            noise_schedule=noise_schedule_config,
        )

        builder = DiffusionBuilder()
        model = builder.build(config, rngs=rngs)

        assert isinstance(model.backbone, DiffusionTransformer)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestBuilderErrorHandling:
    """Test error handling in DiffusionBuilder."""

    def test_reject_none_config(self, rngs):
        """Test that None config raises error."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder

        builder = DiffusionBuilder()

        with pytest.raises((TypeError, ValueError)):
            builder.build(None, rngs=rngs)

    def test_reject_dict_config(self, rngs):
        """Test that dict config raises error."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder

        dict_config = {
            "name": "test",
            "input_shape": (8, 8, 3),
        }

        builder = DiffusionBuilder()

        with pytest.raises(TypeError):
            builder.build(dict_config, rngs=rngs)


# =============================================================================
# Model Class Selection Tests
# =============================================================================


class TestModelClassSelection:
    """Test that builder selects correct model class based on config type."""

    def test_ddpm_config_creates_ddpm_model(self, ddpm_config, rngs):
        """Test DDPMConfig creates DDPMModel."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder

        builder = DiffusionBuilder()
        model = builder.build(ddpm_config, rngs=rngs)

        assert type(model).__name__ == "DDPMModel"

    def test_diffusion_config_creates_diffusion_model(self, diffusion_config, rngs):
        """Test DiffusionConfig creates DiffusionModel."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder

        builder = DiffusionBuilder()
        model = builder.build(diffusion_config, rngs=rngs)

        assert type(model).__name__ == "DiffusionModel"

    def test_score_config_creates_score_model(self, score_config, rngs):
        """Test ScoreDiffusionConfig creates ScoreDiffusionModel."""
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder

        builder = DiffusionBuilder()
        model = builder.build(score_config, rngs=rngs)

        assert type(model).__name__ == "ScoreDiffusionModel"
