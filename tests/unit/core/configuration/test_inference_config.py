"""Tests for Inference configuration frozen dataclass classes.

Tests cover InferenceConfig, DiffusionInferenceConfig, and
ProteinDiffusionInferenceConfig: creation, defaults, validation,
inheritance, and serialization.
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.inference_config import (
    DiffusionInferenceConfig,
    InferenceConfig,
    ProteinDiffusionInferenceConfig,
)


# =============================================================================
# InferenceConfig Tests
# =============================================================================
class TestInferenceConfigBasics:
    """Test basic functionality of InferenceConfig."""

    def test_create_with_required_fields(self):
        """Test creation with required checkpoint_path."""
        config = InferenceConfig(name="infer", checkpoint_path="/model/ckpt")
        assert config.name == "infer"
        assert config.checkpoint_path == "/model/ckpt"

    def test_frozen(self):
        """Test that config is frozen."""
        config = InferenceConfig(name="infer", checkpoint_path="/ckpt")
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"  # type: ignore

    def test_inherits_from_base_config(self):
        """Test inheritance from BaseConfig."""
        config = InferenceConfig(name="infer", checkpoint_path="/ckpt")
        assert isinstance(config, BaseConfig)


class TestInferenceConfigDefaults:
    """Test default values of InferenceConfig."""

    def test_default_output_dir(self):
        """Test output_dir defaults to './outputs'."""
        config = InferenceConfig(name="infer", checkpoint_path="/ckpt")
        assert config.output_dir == "./outputs"

    def test_default_batch_size(self):
        """Test batch_size defaults to 1."""
        config = InferenceConfig(name="infer", checkpoint_path="/ckpt")
        assert config.batch_size == 1

    def test_default_num_samples(self):
        """Test num_samples defaults to 1."""
        config = InferenceConfig(name="infer", checkpoint_path="/ckpt")
        assert config.num_samples == 1

    def test_default_device(self):
        """Test device defaults to 'cuda'."""
        config = InferenceConfig(name="infer", checkpoint_path="/ckpt")
        assert config.device == "cuda"


class TestInferenceConfigValidation:
    """Test validation of InferenceConfig."""

    def test_missing_checkpoint_path(self):
        """Test that missing checkpoint_path raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_path.*required"):
            InferenceConfig(name="infer")

    def test_invalid_batch_size_zero(self):
        """Test that batch_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            InferenceConfig(name="infer", checkpoint_path="/ckpt", batch_size=0)

    def test_invalid_batch_size_negative(self):
        """Test that negative batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            InferenceConfig(name="infer", checkpoint_path="/ckpt", batch_size=-1)

    def test_invalid_num_samples_zero(self):
        """Test that num_samples=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_samples"):
            InferenceConfig(name="infer", checkpoint_path="/ckpt", num_samples=0)

    def test_invalid_device(self):
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="device"):
            InferenceConfig(name="infer", checkpoint_path="/ckpt", device="gpu")

    def test_valid_devices(self):
        """Test that all valid devices are accepted."""
        for device in ("cpu", "cuda", "tpu"):
            config = InferenceConfig(
                name="infer",
                checkpoint_path="/ckpt",
                device=device,
            )
            assert config.device == device


class TestInferenceConfigSerialization:
    """Test serialization of InferenceConfig."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = InferenceConfig(
            name="infer",
            checkpoint_path="/ckpt",
            batch_size=8,
            device="cpu",
        )
        data = config.to_dict()
        assert data["checkpoint_path"] == "/ckpt"
        assert data["batch_size"] == 8
        assert data["device"] == "cpu"

    def test_from_dict(self):
        """Test from_dict creation."""
        data = {
            "name": "infer",
            "checkpoint_path": "/model/ckpt",
            "batch_size": 4,
            "device": "tpu",
        }
        config = InferenceConfig.from_dict(data)
        assert config.checkpoint_path == "/model/ckpt"
        assert config.batch_size == 4
        assert config.device == "tpu"

    def test_roundtrip(self):
        """Test roundtrip serialization."""
        original = InferenceConfig(
            name="infer",
            checkpoint_path="/ckpt",
            output_dir="/out",
            batch_size=16,
            num_samples=100,
            device="cpu",
        )
        data = original.to_dict()
        restored = InferenceConfig.from_dict(data)
        assert original == restored


# =============================================================================
# DiffusionInferenceConfig Tests
# =============================================================================
class TestDiffusionInferenceConfigBasics:
    """Test basic functionality of DiffusionInferenceConfig."""

    def test_create_with_required_fields(self):
        """Test creation with checkpoint_path."""
        config = DiffusionInferenceConfig(
            name="diff_infer",
            checkpoint_path="/ckpt",
        )
        assert config.name == "diff_infer"
        assert config.checkpoint_path == "/ckpt"

    def test_inherits_from_inference_config(self):
        """Test inheritance from InferenceConfig."""
        config = DiffusionInferenceConfig(
            name="diff_infer",
            checkpoint_path="/ckpt",
        )
        assert isinstance(config, InferenceConfig)

    def test_frozen(self):
        """Test that config is frozen."""
        config = DiffusionInferenceConfig(
            name="diff_infer",
            checkpoint_path="/ckpt",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.sampler = "ddim"  # type: ignore


class TestDiffusionInferenceConfigDefaults:
    """Test default values of DiffusionInferenceConfig."""

    def test_default_sampler(self):
        """Test sampler defaults to 'ddpm'."""
        config = DiffusionInferenceConfig(
            name="diff_infer",
            checkpoint_path="/ckpt",
        )
        assert config.sampler == "ddpm"

    def test_default_timesteps(self):
        """Test timesteps defaults to 1000."""
        config = DiffusionInferenceConfig(
            name="diff_infer",
            checkpoint_path="/ckpt",
        )
        assert config.timesteps == 1000

    def test_default_temperature(self):
        """Test temperature defaults to 1.0."""
        config = DiffusionInferenceConfig(
            name="diff_infer",
            checkpoint_path="/ckpt",
        )
        assert config.temperature == 1.0

    def test_default_guidance(self):
        """Test guidance defaults."""
        config = DiffusionInferenceConfig(
            name="diff_infer",
            checkpoint_path="/ckpt",
        )
        assert config.sample_with_classifier_guidance is False
        assert config.guidance_scale == 7.5

    def test_default_intermediate_steps(self):
        """Test intermediate step defaults."""
        config = DiffusionInferenceConfig(
            name="diff_infer",
            checkpoint_path="/ckpt",
        )
        assert config.save_intermediate_steps is False
        assert config.intermediate_step_interval == 50

    def test_default_seed(self):
        """Test seed defaults to 42."""
        config = DiffusionInferenceConfig(
            name="diff_infer",
            checkpoint_path="/ckpt",
        )
        assert config.seed == 42


class TestDiffusionInferenceConfigValidation:
    """Test validation of DiffusionInferenceConfig."""

    def test_invalid_timesteps_zero(self):
        """Test that timesteps=0 raises ValueError."""
        with pytest.raises(ValueError, match="timesteps"):
            DiffusionInferenceConfig(
                name="diff_infer",
                checkpoint_path="/ckpt",
                timesteps=0,
            )

    def test_invalid_temperature_negative(self):
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            DiffusionInferenceConfig(
                name="diff_infer",
                checkpoint_path="/ckpt",
                temperature=-0.1,
            )

    def test_zero_temperature_allowed(self):
        """Test that zero temperature is allowed (deterministic)."""
        config = DiffusionInferenceConfig(
            name="diff_infer",
            checkpoint_path="/ckpt",
            temperature=0.0,
        )
        assert config.temperature == 0.0

    def test_invalid_guidance_scale_negative(self):
        """Test that negative guidance_scale raises ValueError."""
        with pytest.raises(ValueError, match="guidance_scale"):
            DiffusionInferenceConfig(
                name="diff_infer",
                checkpoint_path="/ckpt",
                guidance_scale=-1.0,
            )

    def test_zero_guidance_scale_allowed(self):
        """Test that zero guidance_scale is allowed."""
        config = DiffusionInferenceConfig(
            name="diff_infer",
            checkpoint_path="/ckpt",
            guidance_scale=0.0,
        )
        assert config.guidance_scale == 0.0

    def test_invalid_intermediate_step_interval_zero(self):
        """Test that zero intermediate_step_interval raises ValueError."""
        with pytest.raises(ValueError, match="intermediate_step_interval"):
            DiffusionInferenceConfig(
                name="diff_infer",
                checkpoint_path="/ckpt",
                intermediate_step_interval=0,
            )

    def test_none_seed_allowed(self):
        """Test that None seed is allowed."""
        config = DiffusionInferenceConfig(
            name="diff_infer",
            checkpoint_path="/ckpt",
            seed=None,
        )
        assert config.seed is None

    def test_inherits_parent_validation(self):
        """Test that parent validation still applies."""
        with pytest.raises(ValueError, match="checkpoint_path"):
            DiffusionInferenceConfig(name="diff_infer")


class TestDiffusionInferenceConfigSerialization:
    """Test serialization of DiffusionInferenceConfig."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = DiffusionInferenceConfig(
            name="diff_infer",
            checkpoint_path="/ckpt",
            sampler="ddim",
            timesteps=50,
        )
        data = config.to_dict()
        assert data["sampler"] == "ddim"
        assert data["timesteps"] == 50

    def test_roundtrip(self):
        """Test roundtrip serialization."""
        original = DiffusionInferenceConfig(
            name="diff_infer",
            checkpoint_path="/ckpt",
            sampler="ddim",
            timesteps=100,
            temperature=0.8,
            guidance_scale=3.0,
            seed=123,
        )
        data = original.to_dict()
        restored = DiffusionInferenceConfig.from_dict(data)
        assert original == restored


# =============================================================================
# ProteinDiffusionInferenceConfig Tests
# =============================================================================
class TestProteinDiffusionInferenceConfigBasics:
    """Test basic functionality of ProteinDiffusionInferenceConfig."""

    def test_create_with_required_fields(self):
        """Test creation with checkpoint_path."""
        config = ProteinDiffusionInferenceConfig(
            name="protein_infer",
            checkpoint_path="/ckpt",
        )
        assert config.name == "protein_infer"
        assert config.checkpoint_path == "/ckpt"

    def test_inherits_from_diffusion_inference_config(self):
        """Test inheritance from DiffusionInferenceConfig."""
        config = ProteinDiffusionInferenceConfig(
            name="protein_infer",
            checkpoint_path="/ckpt",
        )
        assert isinstance(config, DiffusionInferenceConfig)
        assert isinstance(config, InferenceConfig)


class TestProteinDiffusionInferenceConfigDefaults:
    """Test default values of ProteinDiffusionInferenceConfig."""

    def test_default_target_seq_length(self):
        """Test target_seq_length defaults to 128."""
        config = ProteinDiffusionInferenceConfig(
            name="protein_infer",
            checkpoint_path="/ckpt",
        )
        assert config.target_seq_length == 128

    def test_default_backbone_atom_indices(self):
        """Test backbone_atom_indices defaults to (0, 1, 2, 4)."""
        config = ProteinDiffusionInferenceConfig(
            name="protein_infer",
            checkpoint_path="/ckpt",
        )
        assert config.backbone_atom_indices == (0, 1, 2, 4)

    def test_default_flags(self):
        """Test boolean flag defaults."""
        config = ProteinDiffusionInferenceConfig(
            name="protein_infer",
            checkpoint_path="/ckpt",
        )
        assert config.calculate_metrics is True
        assert config.visualize_structures is True
        assert config.save_as_pdb is True


class TestProteinDiffusionInferenceConfigValidation:
    """Test validation of ProteinDiffusionInferenceConfig."""

    def test_invalid_target_seq_length_zero(self):
        """Test that target_seq_length=0 raises ValueError."""
        with pytest.raises(ValueError, match="target_seq_length"):
            ProteinDiffusionInferenceConfig(
                name="protein_infer",
                checkpoint_path="/ckpt",
                target_seq_length=0,
            )

    def test_invalid_target_seq_length_negative(self):
        """Test that negative target_seq_length raises ValueError."""
        with pytest.raises(ValueError, match="target_seq_length"):
            ProteinDiffusionInferenceConfig(
                name="protein_infer",
                checkpoint_path="/ckpt",
                target_seq_length=-10,
            )

    def test_inherits_all_parent_validation(self):
        """Test that all parent validation is inherited."""
        with pytest.raises(ValueError, match="checkpoint_path"):
            ProteinDiffusionInferenceConfig(name="protein_infer")


class TestProteinDiffusionInferenceConfigSerialization:
    """Test serialization of ProteinDiffusionInferenceConfig."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = ProteinDiffusionInferenceConfig(
            name="protein_infer",
            checkpoint_path="/ckpt",
            target_seq_length=256,
            save_as_pdb=False,
        )
        data = config.to_dict()
        assert data["target_seq_length"] == 256
        assert data["save_as_pdb"] is False
        assert data["backbone_atom_indices"] == (0, 1, 2, 4)

    def test_roundtrip(self):
        """Test roundtrip serialization."""
        original = ProteinDiffusionInferenceConfig(
            name="protein_infer",
            checkpoint_path="/ckpt",
            target_seq_length=64,
            backbone_atom_indices=(0, 1, 2, 3),
            calculate_metrics=False,
            sampler="ddim",
            timesteps=500,
        )
        data = original.to_dict()
        restored = ProteinDiffusionInferenceConfig.from_dict(data)
        assert original == restored
