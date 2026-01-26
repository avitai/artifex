"""Tests for diffusion model configuration dataclasses.

This module tests the diffusion configuration hierarchy:
- NoiseScheduleConfig
- UNetBackboneConfig (polymorphic backbone config)
- DiffusionConfig (base)
- DDPMConfig
- DDIMConfig
- ScoreDiffusionConfig
- LatentDiffusionConfig
- DiTConfig
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration import (
    BaseConfig,
    DecoderConfig,
    EncoderConfig,
    UNetBackboneConfig,
)
from artifex.generative_models.core.configuration.diffusion_config import (
    ConditionalDiffusionConfig,
    DDIMConfig,
    DDPMConfig,
    DiffusionConfig,
    DiTConfig,
    LatentDiffusionConfig,
    NoiseScheduleConfig,
    ScoreDiffusionConfig,
)


# =============================================================================
# NoiseScheduleConfig Tests
# =============================================================================


class TestNoiseScheduleConfigBasics:
    """Test basic functionality of NoiseScheduleConfig."""

    def test_create_default(self):
        """Test creating NoiseScheduleConfig with default values."""
        config = NoiseScheduleConfig(name="schedule")
        assert config.name == "schedule"
        assert config.schedule_type == "linear"
        assert config.num_timesteps == 1000
        assert config.beta_start == 1e-4
        assert config.beta_end == 2e-2
        assert config.clip_min == 1e-20

    def test_create_with_custom_values(self):
        """Test creating NoiseScheduleConfig with custom values."""
        config = NoiseScheduleConfig(
            name="cosine_schedule",
            schedule_type="cosine",
            num_timesteps=500,
            beta_start=1e-5,
            beta_end=1e-2,
            clip_min=1e-15,
        )
        assert config.schedule_type == "cosine"
        assert config.num_timesteps == 500
        assert config.beta_start == 1e-5
        assert config.beta_end == 1e-2
        assert config.clip_min == 1e-15

    def test_frozen(self):
        """Test that NoiseScheduleConfig is frozen."""
        config = NoiseScheduleConfig(name="schedule")
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.num_timesteps = 500  # type: ignore

    def test_inheritance(self):
        """Test that NoiseScheduleConfig inherits from BaseConfig."""
        config = NoiseScheduleConfig(name="schedule")
        assert isinstance(config, BaseConfig)
        # Should have BaseConfig fields
        assert hasattr(config, "description")
        assert hasattr(config, "tags")
        assert hasattr(config, "metadata")


class TestNoiseScheduleConfigValidation:
    """Test validation of NoiseScheduleConfig."""

    def test_invalid_schedule_type(self):
        """Test that invalid schedule_type raises ValueError."""
        with pytest.raises(ValueError, match="schedule_type must be one of"):
            NoiseScheduleConfig(name="schedule", schedule_type="invalid")

    def test_valid_schedule_types(self):
        """Test all valid schedule types."""
        for schedule_type in ["linear", "cosine", "quadratic"]:
            config = NoiseScheduleConfig(name="schedule", schedule_type=schedule_type)
            assert config.schedule_type == schedule_type

    def test_invalid_num_timesteps_zero(self):
        """Test that zero num_timesteps raises ValueError."""
        with pytest.raises(ValueError, match="num_timesteps must be positive"):
            NoiseScheduleConfig(name="schedule", num_timesteps=0)

    def test_invalid_num_timesteps_negative(self):
        """Test that negative num_timesteps raises ValueError."""
        with pytest.raises(ValueError, match="num_timesteps must be positive"):
            NoiseScheduleConfig(name="schedule", num_timesteps=-1)

    def test_invalid_beta_start_negative(self):
        """Test that negative beta_start raises ValueError."""
        with pytest.raises(ValueError, match="beta_start must be positive"):
            NoiseScheduleConfig(name="schedule", beta_start=-0.001)

    def test_invalid_beta_start_zero(self):
        """Test that zero beta_start raises ValueError."""
        with pytest.raises(ValueError, match="beta_start must be positive"):
            NoiseScheduleConfig(name="schedule", beta_start=0.0)

    def test_invalid_beta_end_negative(self):
        """Test that negative beta_end raises ValueError."""
        with pytest.raises(ValueError, match="beta_end must be positive"):
            NoiseScheduleConfig(name="schedule", beta_end=-0.001)

    def test_beta_start_greater_than_end(self):
        """Test that beta_start > beta_end raises ValueError."""
        with pytest.raises(ValueError, match="beta_start must be less than beta_end"):
            NoiseScheduleConfig(name="schedule", beta_start=0.1, beta_end=0.01)

    def test_invalid_clip_min_negative(self):
        """Test that negative clip_min raises ValueError."""
        with pytest.raises(ValueError, match="clip_min must be non-negative"):
            NoiseScheduleConfig(name="schedule", clip_min=-1e-20)


class TestNoiseScheduleConfigSerialization:
    """Test serialization of NoiseScheduleConfig."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        config = NoiseScheduleConfig(
            name="schedule",
            schedule_type="cosine",
            num_timesteps=500,
        )
        d = config.to_dict()
        assert d["name"] == "schedule"
        assert d["schedule_type"] == "cosine"
        assert d["num_timesteps"] == 500

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "name": "schedule",
            "schedule_type": "cosine",
            "num_timesteps": 500,
            "beta_start": 1e-5,
            "beta_end": 1e-2,
        }
        config = NoiseScheduleConfig.from_dict(data)
        assert config.name == "schedule"
        assert config.schedule_type == "cosine"
        assert config.num_timesteps == 500

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = NoiseScheduleConfig(
            name="schedule",
            schedule_type="quadratic",
            num_timesteps=2000,
            beta_start=1e-5,
            beta_end=5e-2,
        )
        d = original.to_dict()
        restored = NoiseScheduleConfig.from_dict(d)
        assert restored == original


# =============================================================================
# DiffusionConfig Tests
# =============================================================================


class TestDiffusionConfigBasics:
    """Test basic functionality of DiffusionConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_create_minimal(self, backbone_config, schedule_config):
        """Test creating DiffusionConfig with minimal required fields."""
        config = DiffusionConfig(
            name="diffusion",
            backbone=backbone_config,
            noise_schedule=schedule_config,
        )
        assert config.name == "diffusion"
        assert config.backbone == backbone_config
        assert config.noise_schedule == schedule_config
        assert config.input_shape == (32, 32, 3)  # Default (H, W, C)

    def test_create_with_input_shape(self, backbone_config, schedule_config):
        """Test creating DiffusionConfig with custom input shape."""
        config = DiffusionConfig(
            name="diffusion",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            input_shape=(64, 64, 3),
        )
        assert config.input_shape == (64, 64, 3)

    def test_frozen(self, backbone_config, schedule_config):
        """Test that DiffusionConfig is frozen."""
        config = DiffusionConfig(
            name="diffusion",
            backbone=backbone_config,
            noise_schedule=schedule_config,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.input_shape = (64, 64, 3)  # type: ignore


class TestDiffusionConfigValidation:
    """Test validation of DiffusionConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_missing_backbone(self, schedule_config):
        """Test that missing backbone raises ValueError."""
        with pytest.raises(ValueError, match="backbone config is required"):
            DiffusionConfig(
                name="diffusion",
                noise_schedule=schedule_config,
            )

    def test_missing_noise_schedule(self, backbone_config):
        """Test that missing noise_schedule raises ValueError."""
        with pytest.raises(ValueError, match="noise_schedule config is required"):
            DiffusionConfig(
                name="diffusion",
                backbone=backbone_config,
            )

    def test_wrong_backbone_type(self, schedule_config):
        """Test that wrong backbone type raises TypeError."""
        with pytest.raises(TypeError, match="backbone must be a BackboneConfig type"):
            DiffusionConfig(
                name="diffusion",
                backbone={"hidden_dims": [64, 128]},  # type: ignore
                noise_schedule=schedule_config,
            )

    def test_wrong_schedule_type(self, backbone_config):
        """Test that wrong noise_schedule type raises TypeError."""
        with pytest.raises(TypeError, match="noise_schedule must be NoiseScheduleConfig"):
            DiffusionConfig(
                name="diffusion",
                backbone=backbone_config,
                noise_schedule={"num_timesteps": 1000},  # type: ignore
            )

    def test_empty_input_shape(self, backbone_config, schedule_config):
        """Test that empty input_shape raises ValueError."""
        with pytest.raises(ValueError, match="input_shape cannot be empty"):
            DiffusionConfig(
                name="diffusion",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                input_shape=(),
            )


class TestDiffusionConfigSerialization:
    """Test serialization of DiffusionConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_to_dict(self, backbone_config, schedule_config):
        """Test converting to dictionary."""
        config = DiffusionConfig(
            name="diffusion",
            backbone=backbone_config,
            noise_schedule=schedule_config,
        )
        d = config.to_dict()
        assert d["name"] == "diffusion"
        assert "backbone" in d
        assert "noise_schedule" in d

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "name": "diffusion",
            "backbone": {
                "backbone_type": "unet",  # Required discriminator field
                "name": "unet",
                "hidden_dims": [64, 128],
                "activation": "gelu",
                "in_channels": 3,
                "out_channels": 3,
            },
            "noise_schedule": {
                "name": "schedule",
            },
            "input_shape": [64, 64, 3],
        }
        config = DiffusionConfig.from_dict(data)
        assert config.name == "diffusion"
        assert config.backbone.hidden_dims == (64, 128)
        assert config.input_shape == (64, 64, 3)

    def test_roundtrip(self, backbone_config, schedule_config):
        """Test serialization roundtrip."""
        original = DiffusionConfig(
            name="diffusion",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            input_shape=(64, 64, 3),
        )
        d = original.to_dict()
        restored = DiffusionConfig.from_dict(d)
        assert restored == original


# =============================================================================
# DDPMConfig Tests
# =============================================================================


class TestDDPMConfigBasics:
    """Test basic functionality of DDPMConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_create_default(self, backbone_config, schedule_config):
        """Test creating DDPMConfig with default values."""
        config = DDPMConfig(
            name="ddpm",
            backbone=backbone_config,
            noise_schedule=schedule_config,
        )
        assert config.loss_type == "mse"
        assert config.clip_denoised is True

    def test_create_with_custom_values(self, backbone_config, schedule_config):
        """Test creating DDPMConfig with custom values."""
        config = DDPMConfig(
            name="ddpm",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            loss_type="l1",
            clip_denoised=False,
        )
        assert config.loss_type == "l1"
        assert config.clip_denoised is False

    def test_inherits_from_diffusion_config(self, backbone_config, schedule_config):
        """Test that DDPMConfig inherits from DiffusionConfig."""
        config = DDPMConfig(
            name="ddpm",
            backbone=backbone_config,
            noise_schedule=schedule_config,
        )
        assert isinstance(config, DiffusionConfig)


class TestDDPMConfigValidation:
    """Test validation of DDPMConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_valid_loss_types(self, backbone_config, schedule_config):
        """Test all valid loss types."""
        for loss_type in ["mse", "l1", "huber"]:
            config = DDPMConfig(
                name="ddpm",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                loss_type=loss_type,
            )
            assert config.loss_type == loss_type

    def test_invalid_loss_type(self, backbone_config, schedule_config):
        """Test that invalid loss_type raises ValueError."""
        with pytest.raises(ValueError, match="loss_type must be one of"):
            DDPMConfig(
                name="ddpm",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                loss_type="invalid",
            )


class TestDDPMConfigSerialization:
    """Test serialization of DDPMConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_roundtrip(self, backbone_config, schedule_config):
        """Test serialization roundtrip."""
        original = DDPMConfig(
            name="ddpm",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            loss_type="l1",
            clip_denoised=False,
        )
        d = original.to_dict()
        restored = DDPMConfig.from_dict(d)
        assert restored == original


# =============================================================================
# DDIMConfig Tests
# =============================================================================


class TestDDIMConfigBasics:
    """Test basic functionality of DDIMConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_create_default(self, backbone_config, schedule_config):
        """Test creating DDIMConfig with default values."""
        config = DDIMConfig(
            name="ddim",
            backbone=backbone_config,
            noise_schedule=schedule_config,
        )
        assert config.eta == 0.0
        assert config.num_inference_steps == 50
        assert config.skip_type == "uniform"

    def test_create_with_custom_values(self, backbone_config, schedule_config):
        """Test creating DDIMConfig with custom values."""
        config = DDIMConfig(
            name="ddim",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            eta=0.5,
            num_inference_steps=100,
            skip_type="quadratic",
        )
        assert config.eta == 0.5
        assert config.num_inference_steps == 100
        assert config.skip_type == "quadratic"

    def test_inherits_from_diffusion_config(self, backbone_config, schedule_config):
        """Test that DDIMConfig inherits from DiffusionConfig."""
        config = DDIMConfig(
            name="ddim",
            backbone=backbone_config,
            noise_schedule=schedule_config,
        )
        assert isinstance(config, DiffusionConfig)


class TestDDIMConfigValidation:
    """Test validation of DDIMConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_invalid_eta_negative(self, backbone_config, schedule_config):
        """Test that negative eta raises ValueError."""
        with pytest.raises(ValueError, match="eta must be in range"):
            DDIMConfig(
                name="ddim",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                eta=-0.1,
            )

    def test_invalid_eta_greater_than_one(self, backbone_config, schedule_config):
        """Test that eta > 1 raises ValueError."""
        with pytest.raises(ValueError, match="eta must be in range"):
            DDIMConfig(
                name="ddim",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                eta=1.5,
            )

    def test_valid_eta_boundary(self, backbone_config, schedule_config):
        """Test valid eta boundary values."""
        config_zero = DDIMConfig(
            name="ddim",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            eta=0.0,
        )
        assert config_zero.eta == 0.0

        config_one = DDIMConfig(
            name="ddim",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            eta=1.0,
        )
        assert config_one.eta == 1.0

    def test_invalid_num_inference_steps_zero(self, backbone_config, schedule_config):
        """Test that zero num_inference_steps raises ValueError."""
        with pytest.raises(ValueError, match="num_inference_steps must be positive"):
            DDIMConfig(
                name="ddim",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                num_inference_steps=0,
            )

    def test_valid_skip_types(self, backbone_config, schedule_config):
        """Test all valid skip types."""
        for skip_type in ["uniform", "quadratic"]:
            config = DDIMConfig(
                name="ddim",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                skip_type=skip_type,
            )
            assert config.skip_type == skip_type

    def test_invalid_skip_type(self, backbone_config, schedule_config):
        """Test that invalid skip_type raises ValueError."""
        with pytest.raises(ValueError, match="skip_type must be one of"):
            DDIMConfig(
                name="ddim",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                skip_type="invalid",
            )


class TestDDIMConfigSerialization:
    """Test serialization of DDIMConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_roundtrip(self, backbone_config, schedule_config):
        """Test serialization roundtrip."""
        original = DDIMConfig(
            name="ddim",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            eta=0.5,
            num_inference_steps=100,
        )
        d = original.to_dict()
        restored = DDIMConfig.from_dict(d)
        assert restored == original


# =============================================================================
# ScoreDiffusionConfig Tests
# =============================================================================


class TestScoreDiffusionConfigBasics:
    """Test basic functionality of ScoreDiffusionConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_create_default(self, backbone_config, schedule_config):
        """Test creating ScoreDiffusionConfig with default values."""
        config = ScoreDiffusionConfig(
            name="score",
            backbone=backbone_config,
            noise_schedule=schedule_config,
        )
        assert config.sigma_min == 0.01
        assert config.sigma_max == 50.0
        assert config.score_scaling == 1.0

    def test_create_with_custom_values(self, backbone_config, schedule_config):
        """Test creating ScoreDiffusionConfig with custom values."""
        config = ScoreDiffusionConfig(
            name="score",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            sigma_min=0.001,
            sigma_max=100.0,
            score_scaling=2.0,
        )
        assert config.sigma_min == 0.001
        assert config.sigma_max == 100.0
        assert config.score_scaling == 2.0


class TestScoreDiffusionConfigValidation:
    """Test validation of ScoreDiffusionConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_invalid_sigma_min_negative(self, backbone_config, schedule_config):
        """Test that negative sigma_min raises ValueError."""
        with pytest.raises(ValueError, match="sigma_min must be positive"):
            ScoreDiffusionConfig(
                name="score",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                sigma_min=-0.01,
            )

    def test_invalid_sigma_max_negative(self, backbone_config, schedule_config):
        """Test that negative sigma_max raises ValueError."""
        with pytest.raises(ValueError, match="sigma_max must be positive"):
            ScoreDiffusionConfig(
                name="score",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                sigma_max=-1.0,
            )

    def test_sigma_min_greater_than_max(self, backbone_config, schedule_config):
        """Test that sigma_min > sigma_max raises ValueError."""
        with pytest.raises(ValueError, match="sigma_min must be less than sigma_max"):
            ScoreDiffusionConfig(
                name="score",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                sigma_min=100.0,
                sigma_max=1.0,
            )

    def test_invalid_score_scaling_zero(self, backbone_config, schedule_config):
        """Test that zero score_scaling raises ValueError."""
        with pytest.raises(ValueError, match="score_scaling must be positive"):
            ScoreDiffusionConfig(
                name="score",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                score_scaling=0.0,
            )


# =============================================================================
# LatentDiffusionConfig Tests
# =============================================================================


class TestLatentDiffusionConfigBasics:
    """Test basic functionality of LatentDiffusionConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=4,  # Latent channels
            out_channels=4,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    @pytest.fixture
    def encoder_config(self):
        """Create an EncoderConfig for testing."""
        return EncoderConfig(
            name="encoder",
            input_shape=(64, 64, 3),
            latent_dim=4,
            hidden_dims=(32, 64),
            activation="gelu",
        )

    @pytest.fixture
    def decoder_config(self):
        """Create a DecoderConfig for testing."""
        return DecoderConfig(
            name="decoder",
            latent_dim=4,
            output_shape=(64, 64, 3),
            hidden_dims=(64, 32),
            activation="gelu",
        )

    def test_create_default(self, backbone_config, schedule_config, encoder_config, decoder_config):
        """Test creating LatentDiffusionConfig with default values."""
        config = LatentDiffusionConfig(
            name="ldm",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            encoder=encoder_config,
            decoder=decoder_config,
        )
        assert config.latent_scale_factor == 0.18215

    def test_create_with_custom_values(
        self, backbone_config, schedule_config, encoder_config, decoder_config
    ):
        """Test creating LatentDiffusionConfig with custom values."""
        config = LatentDiffusionConfig(
            name="ldm",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            encoder=encoder_config,
            decoder=decoder_config,
            latent_scale_factor=0.5,
        )
        assert config.latent_scale_factor == 0.5


class TestLatentDiffusionConfigValidation:
    """Test validation of LatentDiffusionConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=4,
            out_channels=4,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    @pytest.fixture
    def encoder_config(self):
        """Create an EncoderConfig for testing."""
        return EncoderConfig(
            name="encoder",
            input_shape=(64, 64, 3),
            latent_dim=4,
            hidden_dims=(32, 64),
            activation="gelu",
        )

    @pytest.fixture
    def decoder_config(self):
        """Create a DecoderConfig for testing."""
        return DecoderConfig(
            name="decoder",
            latent_dim=4,
            output_shape=(64, 64, 3),
            hidden_dims=(64, 32),
            activation="gelu",
        )

    def test_missing_encoder(self, backbone_config, schedule_config, decoder_config):
        """Test that missing encoder raises ValueError."""
        with pytest.raises(ValueError, match="encoder config is required"):
            LatentDiffusionConfig(
                name="ldm",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                decoder=decoder_config,
            )

    def test_missing_decoder(self, backbone_config, schedule_config, encoder_config):
        """Test that missing decoder raises ValueError."""
        with pytest.raises(ValueError, match="decoder config is required"):
            LatentDiffusionConfig(
                name="ldm",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                encoder=encoder_config,
            )

    def test_wrong_encoder_type(self, backbone_config, schedule_config, decoder_config):
        """Test that wrong encoder type raises TypeError."""
        with pytest.raises(TypeError, match="encoder must be EncoderConfig"):
            LatentDiffusionConfig(
                name="ldm",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                encoder={"hidden_dims": [32, 64]},  # type: ignore
                decoder=decoder_config,
            )

    def test_latent_dim_mismatch(self, backbone_config, schedule_config):
        """Test that latent_dim mismatch raises ValueError."""
        encoder = EncoderConfig(
            name="encoder",
            input_shape=(64, 64, 3),
            latent_dim=4,
            hidden_dims=(32, 64),
            activation="gelu",
        )
        decoder = DecoderConfig(
            name="decoder",
            latent_dim=8,  # Different from encoder
            output_shape=(64, 64, 3),
            hidden_dims=(64, 32),
            activation="gelu",
        )
        with pytest.raises(ValueError, match="encoder latent_dim.*must match.*decoder latent_dim"):
            LatentDiffusionConfig(
                name="ldm",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                encoder=encoder,
                decoder=decoder,
            )

    def test_invalid_latent_scale_factor(
        self, backbone_config, schedule_config, encoder_config, decoder_config
    ):
        """Test that zero latent_scale_factor raises ValueError."""
        with pytest.raises(ValueError, match="latent_scale_factor must be positive"):
            LatentDiffusionConfig(
                name="ldm",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                encoder=encoder_config,
                decoder=decoder_config,
                latent_scale_factor=0.0,
            )


# =============================================================================
# DiTConfig Tests
# =============================================================================


class TestDiTConfigBasics:
    """Test basic functionality of DiTConfig."""

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_create_default(self, schedule_config):
        """Test creating DiTConfig with default values."""
        config = DiTConfig(
            name="dit",
            noise_schedule=schedule_config,
        )
        assert config.patch_size == 2
        assert config.hidden_size == 512
        assert config.depth == 12
        assert config.num_heads == 8
        assert config.mlp_ratio == 4.0
        assert config.learn_sigma is False
        assert config.num_classes is None
        assert config.cfg_scale == 1.0

    def test_create_with_custom_values(self, schedule_config):
        """Test creating DiTConfig with custom values."""
        config = DiTConfig(
            name="dit_large",
            noise_schedule=schedule_config,
            patch_size=4,
            hidden_size=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True,
            num_classes=1000,
            cfg_scale=4.0,
        )
        assert config.patch_size == 4
        assert config.hidden_size == 1024
        assert config.depth == 24
        assert config.num_heads == 16
        assert config.learn_sigma is True
        assert config.num_classes == 1000
        assert config.cfg_scale == 4.0

    def test_inherits_from_base_config(self, schedule_config):
        """Test that DiTConfig inherits from BaseConfig."""
        config = DiTConfig(name="dit", noise_schedule=schedule_config)
        assert isinstance(config, BaseConfig)


class TestDiTConfigValidation:
    """Test validation of DiTConfig."""

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_invalid_patch_size_zero(self, schedule_config):
        """Test that zero patch_size raises ValueError."""
        with pytest.raises(ValueError, match="patch_size must be positive"):
            DiTConfig(name="dit", noise_schedule=schedule_config, patch_size=0)

    def test_invalid_hidden_size_zero(self, schedule_config):
        """Test that zero hidden_size raises ValueError."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            DiTConfig(name="dit", noise_schedule=schedule_config, hidden_size=0)

    def test_invalid_depth_zero(self, schedule_config):
        """Test that zero depth raises ValueError."""
        with pytest.raises(ValueError, match="depth must be positive"):
            DiTConfig(name="dit", noise_schedule=schedule_config, depth=0)

    def test_invalid_num_heads_zero(self, schedule_config):
        """Test that zero num_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            DiTConfig(name="dit", noise_schedule=schedule_config, num_heads=0)

    def test_invalid_mlp_ratio_zero(self, schedule_config):
        """Test that zero mlp_ratio raises ValueError."""
        with pytest.raises(ValueError, match="mlp_ratio must be positive"):
            DiTConfig(name="dit", noise_schedule=schedule_config, mlp_ratio=0.0)

    def test_invalid_num_classes_negative(self, schedule_config):
        """Test that negative num_classes raises ValueError."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            DiTConfig(name="dit", noise_schedule=schedule_config, num_classes=-1)

    def test_num_classes_zero_raises(self, schedule_config):
        """Test that zero num_classes raises ValueError."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            DiTConfig(name="dit", noise_schedule=schedule_config, num_classes=0)

    def test_num_classes_none_allowed(self, schedule_config):
        """Test that None num_classes is allowed."""
        config = DiTConfig(name="dit", noise_schedule=schedule_config, num_classes=None)
        assert config.num_classes is None

    def test_invalid_cfg_scale_negative(self, schedule_config):
        """Test that negative cfg_scale raises ValueError."""
        with pytest.raises(ValueError, match="cfg_scale must be non-negative"):
            DiTConfig(name="dit", noise_schedule=schedule_config, cfg_scale=-1.0)

    def test_hidden_size_divisible_by_num_heads(self, schedule_config):
        """Test that hidden_size must be divisible by num_heads."""
        with pytest.raises(ValueError, match="hidden_size must be divisible by num_heads"):
            DiTConfig(
                name="dit",
                noise_schedule=schedule_config,
                hidden_size=512,
                num_heads=7,  # 512 not divisible by 7
            )


class TestDiTConfigSerialization:
    """Test serialization of DiTConfig."""

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_roundtrip(self, schedule_config):
        """Test serialization roundtrip."""
        original = DiTConfig(
            name="dit",
            noise_schedule=schedule_config,
            patch_size=4,
            hidden_size=768,
            depth=16,
            num_heads=12,
            num_classes=1000,
        )
        d = original.to_dict()
        restored = DiTConfig.from_dict(d)
        assert restored == original

    def test_from_dict_without_backbone(self):
        """Test that from_dict works without backbone field."""
        data = {
            "name": "dit",
            "noise_schedule": {"name": "schedule"},
            "patch_size": 4,
            "hidden_size": 512,
        }
        config = DiTConfig.from_dict(data)
        assert config.patch_size == 4
        assert config.hidden_size == 512


# =============================================================================
# ConditionalDiffusionConfig Tests
# =============================================================================


class TestConditionalDiffusionConfigBasics:
    """Test basic functionality of ConditionalDiffusionConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_create_with_conditioning_dim(self, backbone_config, schedule_config):
        """Test creating ConditionalDiffusionConfig with conditioning_dim."""
        config = ConditionalDiffusionConfig(
            name="conditional",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            conditioning_dim=128,
        )
        assert config.name == "conditional"
        assert config.conditioning_dim == 128
        assert config.backbone == backbone_config
        assert config.noise_schedule == schedule_config

    def test_frozen(self, backbone_config, schedule_config):
        """Test that ConditionalDiffusionConfig is frozen."""
        config = ConditionalDiffusionConfig(
            name="conditional",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            conditioning_dim=128,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.conditioning_dim = 256  # type: ignore

    def test_inheritance(self, backbone_config, schedule_config):
        """Test that ConditionalDiffusionConfig inherits from DiffusionConfig."""
        config = ConditionalDiffusionConfig(
            name="conditional",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            conditioning_dim=128,
        )
        assert isinstance(config, DiffusionConfig)
        assert isinstance(config, BaseConfig)
        # Should have DiffusionConfig fields
        assert hasattr(config, "input_shape")


class TestConditionalDiffusionConfigValidation:
    """Test validation of ConditionalDiffusionConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_invalid_conditioning_dim_zero(self, backbone_config, schedule_config):
        """Test that zero conditioning_dim raises ValueError."""
        with pytest.raises(ValueError, match="conditioning_dim must be positive"):
            ConditionalDiffusionConfig(
                name="conditional",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                conditioning_dim=0,
            )

    def test_invalid_conditioning_dim_negative(self, backbone_config, schedule_config):
        """Test that negative conditioning_dim raises ValueError."""
        with pytest.raises(ValueError, match="conditioning_dim must be positive"):
            ConditionalDiffusionConfig(
                name="conditional",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                conditioning_dim=-1,
            )

    def test_valid_conditioning_dim_values(self, backbone_config, schedule_config):
        """Test valid conditioning_dim values."""
        for dim in [1, 64, 128, 256, 512, 1024]:
            config = ConditionalDiffusionConfig(
                name="conditional",
                backbone=backbone_config,
                noise_schedule=schedule_config,
                conditioning_dim=dim,
            )
            assert config.conditioning_dim == dim


class TestConditionalDiffusionConfigSerialization:
    """Test serialization of ConditionalDiffusionConfig."""

    @pytest.fixture
    def backbone_config(self):
        """Create a UNetBackboneConfig for testing."""
        return UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )

    @pytest.fixture
    def schedule_config(self):
        """Create a NoiseScheduleConfig for testing."""
        return NoiseScheduleConfig(name="test_schedule")

    def test_to_dict(self, backbone_config, schedule_config):
        """Test converting to dictionary."""
        config = ConditionalDiffusionConfig(
            name="conditional",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            conditioning_dim=128,
        )
        d = config.to_dict()
        assert d["name"] == "conditional"
        assert d["conditioning_dim"] == 128
        assert "backbone" in d
        assert "noise_schedule" in d

    def test_roundtrip(self, backbone_config, schedule_config):
        """Test serialization roundtrip."""
        original = ConditionalDiffusionConfig(
            name="conditional",
            backbone=backbone_config,
            noise_schedule=schedule_config,
            conditioning_dim=256,
            input_shape=(64, 64, 3),
        )
        d = original.to_dict()
        restored = ConditionalDiffusionConfig.from_dict(d)
        assert restored == original
