"""Tests for extension configuration classes.

This module tests the frozen dataclass extension configurations
following TDD principles.
"""

import dataclasses
import math

import pytest

from artifex.generative_models.core.configuration import (
    ArchitectureExtensionConfig,
    AudioSpectralConfig,
    AugmentationExtensionConfig,
    CallbackExtensionConfig,
    ChemicalConstraintConfig,
    ClassifierFreeGuidanceConfig,
    ConstrainedSamplingConfig,
    ConstraintExtensionConfig,
    EvaluationExtensionConfig,
    ExtensionConfig,
    ExtensionPipelineConfig,
    ImageAugmentationConfig,
    LossExtensionConfig,
    ModalityExtensionConfig,
    ProteinDihedralConfig,
    ProteinExtensionConfig,
    SamplingExtensionConfig,
    TextEmbeddingConfig,
)


# =============================================================================
# Base ExtensionConfig Tests
# =============================================================================


class TestExtensionConfig:
    """Tests for base ExtensionConfig class."""

    def test_create_with_defaults(self):
        """Test creating ExtensionConfig with default values."""
        config = ExtensionConfig(name="test")

        assert config.name == "test"
        assert config.weight == 1.0
        assert config.enabled is True
        assert config.description == ""
        assert config.tags == ()

    def test_create_with_custom_values(self):
        """Test creating ExtensionConfig with custom values."""
        config = ExtensionConfig(
            name="custom",
            weight=0.5,
            enabled=False,
            description="Test extension",
            tags=("tag1", "tag2"),
        )

        assert config.name == "custom"
        assert config.weight == 0.5
        assert config.enabled is False
        assert config.description == "Test extension"
        assert config.tags == ("tag1", "tag2")

    def test_immutability(self):
        """Test that ExtensionConfig is immutable (frozen)."""
        config = ExtensionConfig(name="test")

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.weight = 2.0

    def test_weight_validation_negative(self):
        """Test that negative weight raises ValueError."""
        with pytest.raises(ValueError, match="weight must be non-negative"):
            ExtensionConfig(name="test", weight=-1.0)

    def test_weight_validation_zero(self):
        """Test that zero weight is valid."""
        config = ExtensionConfig(name="test", weight=0.0)
        assert config.weight == 0.0

    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name must be non-empty"):
            ExtensionConfig(name="")

    def test_whitespace_name_raises_error(self):
        """Test that whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="name must be non-empty"):
            ExtensionConfig(name="   ")

    def test_from_dict(self):
        """Test creating ExtensionConfig from dict."""
        config = ExtensionConfig.from_dict(
            {
                "name": "from_dict",
                "weight": 0.75,
                "enabled": True,
            }
        )

        assert config.name == "from_dict"
        assert config.weight == 0.75
        assert config.enabled is True

    def test_to_dict(self):
        """Test converting ExtensionConfig to dict."""
        config = ExtensionConfig(name="test", weight=0.5)
        config_dict = config.to_dict()

        assert config_dict["name"] == "test"
        assert config_dict["weight"] == 0.5
        assert config_dict["enabled"] is True


# =============================================================================
# ConstraintExtensionConfig Tests
# =============================================================================


class TestConstraintExtensionConfig:
    """Tests for ConstraintExtensionConfig class."""

    def test_create_with_defaults(self):
        """Test creating ConstraintExtensionConfig with default values."""
        config = ConstraintExtensionConfig(name="constraint")

        assert config.name == "constraint"
        assert config.tolerance == 0.01
        assert config.projection_enabled is True
        assert config.weight == 1.0

    def test_create_with_custom_values(self):
        """Test creating ConstraintExtensionConfig with custom values."""
        config = ConstraintExtensionConfig(
            name="custom_constraint",
            tolerance=0.001,
            projection_enabled=False,
            weight=2.0,
        )

        assert config.tolerance == 0.001
        assert config.projection_enabled is False
        assert config.weight == 2.0

    def test_negative_tolerance_raises_error(self):
        """Test that negative tolerance raises ValueError."""
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            ConstraintExtensionConfig(name="test", tolerance=-0.01)

    def test_inheritance(self):
        """Test that ConstraintExtensionConfig inherits from ExtensionConfig."""
        config = ConstraintExtensionConfig(name="test")
        assert isinstance(config, ExtensionConfig)


# =============================================================================
# AugmentationExtensionConfig Tests
# =============================================================================


class TestAugmentationExtensionConfig:
    """Tests for AugmentationExtensionConfig class."""

    def test_create_with_defaults(self):
        """Test creating AugmentationExtensionConfig with default values."""
        config = AugmentationExtensionConfig(name="augment")

        assert config.probability == 1.0
        assert config.deterministic is False

    def test_probability_validation_out_of_range(self):
        """Test that probability outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="probability must be in"):
            AugmentationExtensionConfig(name="test", probability=1.5)

        with pytest.raises(ValueError, match="probability must be in"):
            AugmentationExtensionConfig(name="test", probability=-0.1)

    def test_valid_probability_boundaries(self):
        """Test valid probability boundary values."""
        config_zero = AugmentationExtensionConfig(name="test", probability=0.0)
        assert config_zero.probability == 0.0

        config_one = AugmentationExtensionConfig(name="test", probability=1.0)
        assert config_one.probability == 1.0


# =============================================================================
# SamplingExtensionConfig Tests
# =============================================================================


class TestSamplingExtensionConfig:
    """Tests for SamplingExtensionConfig class."""

    def test_create_with_defaults(self):
        """Test creating SamplingExtensionConfig with default values."""
        config = SamplingExtensionConfig(name="sampling")

        assert config.guidance_scale == 1.0
        assert config.temperature == 1.0

    def test_negative_guidance_scale_raises_error(self):
        """Test that negative guidance_scale raises ValueError."""
        with pytest.raises(ValueError, match="guidance_scale must be non-negative"):
            SamplingExtensionConfig(name="test", guidance_scale=-1.0)

    def test_negative_temperature_raises_error(self):
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            SamplingExtensionConfig(name="test", temperature=-0.5)


# =============================================================================
# LossExtensionConfig Tests
# =============================================================================


class TestLossExtensionConfig:
    """Tests for LossExtensionConfig class."""

    def test_create_with_defaults(self):
        """Test creating LossExtensionConfig with default values."""
        config = LossExtensionConfig(name="loss")

        assert config.weight_schedule == "constant"
        assert config.warmup_steps == 0

    def test_valid_weight_schedules(self):
        """Test all valid weight schedule options."""
        for schedule in ["constant", "linear", "cosine", "exponential"]:
            config = LossExtensionConfig(name="test", weight_schedule=schedule)
            assert config.weight_schedule == schedule

    def test_invalid_weight_schedule_raises_error(self):
        """Test that invalid weight_schedule raises ValueError."""
        with pytest.raises(ValueError, match="weight_schedule must be one of"):
            LossExtensionConfig(name="test", weight_schedule="invalid")

    def test_negative_warmup_steps_raises_error(self):
        """Test that negative warmup_steps raises ValueError."""
        with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
            LossExtensionConfig(name="test", warmup_steps=-1)


# =============================================================================
# EvaluationExtensionConfig Tests
# =============================================================================


class TestEvaluationExtensionConfig:
    """Tests for EvaluationExtensionConfig class."""

    def test_create_with_defaults(self):
        """Test creating EvaluationExtensionConfig with default values."""
        config = EvaluationExtensionConfig(name="eval")

        assert config.compute_on_train is False
        assert config.compute_on_eval is True

    def test_create_with_custom_values(self):
        """Test creating with both train and eval enabled."""
        config = EvaluationExtensionConfig(
            name="test",
            compute_on_train=True,
            compute_on_eval=True,
        )

        assert config.compute_on_train is True
        assert config.compute_on_eval is True


# =============================================================================
# CallbackExtensionConfig Tests
# =============================================================================


class TestCallbackExtensionConfig:
    """Tests for CallbackExtensionConfig class."""

    def test_create_with_defaults(self):
        """Test creating CallbackExtensionConfig with default values."""
        config = CallbackExtensionConfig(name="callback")

        assert config.frequency == 1
        assert config.on_train is True
        assert config.on_eval is True

    def test_frequency_validation(self):
        """Test that frequency < 1 raises ValueError."""
        with pytest.raises(ValueError, match="frequency must be at least 1"):
            CallbackExtensionConfig(name="test", frequency=0)

        with pytest.raises(ValueError, match="frequency must be at least 1"):
            CallbackExtensionConfig(name="test", frequency=-1)


# =============================================================================
# ModalityExtensionConfig Tests
# =============================================================================


class TestModalityExtensionConfig:
    """Tests for ModalityExtensionConfig class."""

    def test_create_with_defaults(self):
        """Test creating ModalityExtensionConfig with default values."""
        config = ModalityExtensionConfig(name="modality")

        assert config.input_key == "input"
        assert config.output_key == "output"

    def test_create_with_custom_keys(self):
        """Test creating with custom input/output keys."""
        config = ModalityExtensionConfig(
            name="test",
            input_key="images",
            output_key="reconstructed",
        )

        assert config.input_key == "images"
        assert config.output_key == "reconstructed"


# =============================================================================
# ArchitectureExtensionConfig Tests
# =============================================================================


class TestArchitectureExtensionConfig:
    """Tests for ArchitectureExtensionConfig class."""

    def test_create_with_defaults(self):
        """Test creating ArchitectureExtensionConfig with default values."""
        config = ArchitectureExtensionConfig(name="lora")

        assert config.rank == 4
        assert config.alpha == 1.0
        assert config.target_modules == ()
        assert config.dropout == 0.0

    def test_rank_validation(self):
        """Test that rank < 1 raises ValueError."""
        with pytest.raises(ValueError, match="rank must be at least 1"):
            ArchitectureExtensionConfig(name="test", rank=0)

    def test_dropout_validation(self):
        """Test that dropout outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="dropout must be in"):
            ArchitectureExtensionConfig(name="test", dropout=1.5)

    def test_target_modules_tuple(self):
        """Test that target_modules accepts tuples."""
        config = ArchitectureExtensionConfig(
            name="test",
            target_modules=("q_proj", "v_proj", "k_proj"),
        )

        assert config.target_modules == ("q_proj", "v_proj", "k_proj")


# =============================================================================
# ProteinExtensionConfig Tests
# =============================================================================


class TestProteinExtensionConfig:
    """Tests for ProteinExtensionConfig class."""

    def test_create_with_defaults(self):
        """Test creating ProteinExtensionConfig with default values."""
        config = ProteinExtensionConfig(name="protein")

        assert config.backbone_atoms == ("N", "CA", "C", "O")
        assert config.bond_length_weight == 1.0
        assert config.bond_angle_weight == 0.5
        assert config.dihedral_weight == 0.3
        assert config.ideal_bond_lengths == {}
        assert config.ideal_bond_angles == {}

    def test_inheritance(self):
        """Test that ProteinExtensionConfig inherits correctly."""
        config = ProteinExtensionConfig(name="test")

        assert isinstance(config, ConstraintExtensionConfig)
        assert isinstance(config, ExtensionConfig)

    def test_negative_weight_validation(self):
        """Test that negative weights raise ValueError."""
        with pytest.raises(ValueError, match="bond_length_weight must be non-negative"):
            ProteinExtensionConfig(name="test", bond_length_weight=-1.0)

        with pytest.raises(ValueError, match="bond_angle_weight must be non-negative"):
            ProteinExtensionConfig(name="test", bond_angle_weight=-1.0)

        with pytest.raises(ValueError, match="dihedral_weight must be non-negative"):
            ProteinExtensionConfig(name="test", dihedral_weight=-1.0)

    def test_custom_ideal_values(self):
        """Test creating with custom ideal bond lengths and angles."""
        config = ProteinExtensionConfig(
            name="test",
            ideal_bond_lengths={"N-CA": 1.458, "CA-C": 1.523},
            ideal_bond_angles={"N-CA-C": 1.94},
        )

        assert config.ideal_bond_lengths["N-CA"] == 1.458
        assert config.ideal_bond_angles["N-CA-C"] == 1.94


# =============================================================================
# ProteinDihedralConfig Tests
# =============================================================================


class TestProteinDihedralConfig:
    """Tests for ProteinDihedralConfig class."""

    def test_create_with_defaults(self):
        """Test creating ProteinDihedralConfig with default values."""
        config = ProteinDihedralConfig(name="dihedral")

        assert config.target_secondary_structure == "alpha_helix"
        assert config.phi_weight == 0.5
        assert config.psi_weight == 0.5
        assert config.omega_weight == 1.0
        assert config.ideal_phi is None
        assert config.ideal_psi is None
        assert abs(config.ideal_omega - math.pi) < 1e-10

    def test_secondary_structure_options(self):
        """Test valid secondary structure options."""
        for ss in ["alpha_helix", "beta_sheet", "coil"]:
            config = ProteinDihedralConfig(name="test", target_secondary_structure=ss)
            assert config.target_secondary_structure == ss


# =============================================================================
# ChemicalConstraintConfig Tests
# =============================================================================


class TestChemicalConstraintConfig:
    """Tests for ChemicalConstraintConfig class."""

    def test_create_with_defaults(self):
        """Test creating ChemicalConstraintConfig with default values."""
        config = ChemicalConstraintConfig(name="chemical")

        assert config.enforce_valence is True
        assert config.enforce_bond_lengths is True
        assert config.enforce_ring_closure is True
        assert config.max_ring_size == 8

    def test_max_ring_size_validation(self):
        """Test that max_ring_size < 3 raises ValueError."""
        with pytest.raises(ValueError, match="max_ring_size must be at least 3"):
            ChemicalConstraintConfig(name="test", max_ring_size=2)


# =============================================================================
# ImageAugmentationConfig Tests
# =============================================================================


class TestImageAugmentationConfig:
    """Tests for ImageAugmentationConfig class."""

    def test_create_with_defaults(self):
        """Test creating ImageAugmentationConfig with default values."""
        config = ImageAugmentationConfig(name="image_aug")

        assert config.random_flip_horizontal is True
        assert config.random_flip_vertical is False
        assert config.random_rotation == 0.0
        assert config.color_jitter is False
        assert config.brightness_range == (0.9, 1.1)
        assert config.contrast_range == (0.9, 1.1)

    def test_inheritance(self):
        """Test that ImageAugmentationConfig inherits from AugmentationExtensionConfig."""
        config = ImageAugmentationConfig(name="test")
        assert isinstance(config, AugmentationExtensionConfig)

    def test_brightness_range_validation(self):
        """Test brightness_range validation."""
        # Wrong order
        with pytest.raises(ValueError, match="brightness_range"):
            ImageAugmentationConfig(name="test", brightness_range=(1.2, 0.8))

    def test_contrast_range_validation(self):
        """Test contrast_range validation."""
        # Wrong order
        with pytest.raises(ValueError, match="contrast_range"):
            ImageAugmentationConfig(name="test", contrast_range=(1.2, 0.8))


# =============================================================================
# AudioSpectralConfig Tests
# =============================================================================


class TestAudioSpectralConfig:
    """Tests for AudioSpectralConfig class."""

    def test_create_with_defaults(self):
        """Test creating AudioSpectralConfig with default values."""
        config = AudioSpectralConfig(name="audio")

        assert config.n_fft == 2048
        assert config.hop_length == 512
        assert config.n_mels == 128
        assert config.sample_rate == 22050
        assert config.f_min == 0.0
        assert config.f_max is None

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="n_fft must be at least 1"):
            AudioSpectralConfig(name="test", n_fft=0)

        with pytest.raises(ValueError, match="hop_length must be at least 1"):
            AudioSpectralConfig(name="test", hop_length=0)

        with pytest.raises(ValueError, match="n_mels must be at least 1"):
            AudioSpectralConfig(name="test", n_mels=0)

        with pytest.raises(ValueError, match="sample_rate must be at least 1"):
            AudioSpectralConfig(name="test", sample_rate=0)

    def test_frequency_range_validation(self):
        """Test that f_max > f_min when f_max is specified."""
        with pytest.raises(ValueError, match="f_max must be greater than f_min"):
            AudioSpectralConfig(name="test", f_min=1000.0, f_max=500.0)


# =============================================================================
# TextEmbeddingConfig Tests
# =============================================================================


class TestTextEmbeddingConfig:
    """Tests for TextEmbeddingConfig class."""

    def test_create_with_defaults(self):
        """Test creating TextEmbeddingConfig with default values."""
        config = TextEmbeddingConfig(name="text")

        assert config.vocab_size == 30000
        assert config.embedding_dim == 256
        assert config.max_sequence_length == 512
        assert config.padding_idx == 0
        assert config.use_positional_encoding is True

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="vocab_size must be at least 1"):
            TextEmbeddingConfig(name="test", vocab_size=0)

        with pytest.raises(ValueError, match="embedding_dim must be at least 1"):
            TextEmbeddingConfig(name="test", embedding_dim=0)

        with pytest.raises(ValueError, match="max_sequence_length must be at least 1"):
            TextEmbeddingConfig(name="test", max_sequence_length=0)

        with pytest.raises(ValueError, match="padding_idx must be non-negative"):
            TextEmbeddingConfig(name="test", padding_idx=-1)


# =============================================================================
# ClassifierFreeGuidanceConfig Tests
# =============================================================================


class TestClassifierFreeGuidanceConfig:
    """Tests for ClassifierFreeGuidanceConfig class."""

    def test_create_with_defaults(self):
        """Test creating ClassifierFreeGuidanceConfig with default values."""
        config = ClassifierFreeGuidanceConfig(name="cfg")

        assert config.guidance_scale == 1.0
        assert config.unconditional_conditioning is True

    def test_inheritance(self):
        """Test that ClassifierFreeGuidanceConfig inherits from SamplingExtensionConfig."""
        config = ClassifierFreeGuidanceConfig(name="test")
        assert isinstance(config, SamplingExtensionConfig)


# =============================================================================
# ConstrainedSamplingConfig Tests
# =============================================================================


class TestConstrainedSamplingConfig:
    """Tests for ConstrainedSamplingConfig class."""

    def test_create_with_defaults(self):
        """Test creating ConstrainedSamplingConfig with default values."""
        config = ConstrainedSamplingConfig(name="constrained")

        assert config.constraint_weight == 1.0
        assert config.projection_steps == 1
        assert config.use_gradient_guidance is False

    def test_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="constraint_weight must be non-negative"):
            ConstrainedSamplingConfig(name="test", constraint_weight=-1.0)

        with pytest.raises(ValueError, match="projection_steps must be non-negative"):
            ConstrainedSamplingConfig(name="test", projection_steps=-1)


# =============================================================================
# ExtensionPipelineConfig Tests
# =============================================================================


class TestExtensionPipelineConfig:
    """Tests for ExtensionPipelineConfig class."""

    def test_create_with_defaults(self):
        """Test creating ExtensionPipelineConfig with default values."""
        config = ExtensionPipelineConfig(name="pipeline")

        assert config.extensions == ()
        assert config.aggregate_losses is True
        assert config.loss_reduction == "weighted_sum"

    def test_create_with_extensions(self):
        """Test creating pipeline with multiple extensions."""
        ext1 = ExtensionConfig(name="ext1", weight=1.0)
        ext2 = ConstraintExtensionConfig(name="ext2", weight=0.5)

        config = ExtensionPipelineConfig(
            name="pipeline",
            extensions=(ext1, ext2),
        )

        assert len(config.extensions) == 2
        assert config.extensions[0].name == "ext1"
        assert config.extensions[1].name == "ext2"

    def test_loss_reduction_options(self):
        """Test valid loss reduction options."""
        for reduction in ["sum", "mean", "weighted_sum"]:
            config = ExtensionPipelineConfig(name="test", loss_reduction=reduction)
            assert config.loss_reduction == reduction

    def test_invalid_loss_reduction_raises_error(self):
        """Test that invalid loss_reduction raises ValueError."""
        with pytest.raises(ValueError, match="loss_reduction must be one of"):
            ExtensionPipelineConfig(name="test", loss_reduction="invalid")


# =============================================================================
# Serialization Tests
# =============================================================================


class TestConfigSerialization:
    """Tests for config serialization (to_dict/from_dict)."""

    def test_round_trip_extension_config(self):
        """Test round-trip serialization of ExtensionConfig."""
        original = ExtensionConfig(
            name="test",
            weight=0.5,
            enabled=False,
            tags=("a", "b"),
        )

        config_dict = original.to_dict()
        restored = ExtensionConfig.from_dict(config_dict)

        assert restored.name == original.name
        assert restored.weight == original.weight
        assert restored.enabled == original.enabled
        assert restored.tags == original.tags

    def test_round_trip_protein_config(self):
        """Test round-trip serialization of ProteinExtensionConfig."""
        original = ProteinExtensionConfig(
            name="protein",
            bond_length_weight=2.0,
            backbone_atoms=("N", "CA", "C"),
            ideal_bond_lengths={"N-CA": 1.458},
        )

        config_dict = original.to_dict()
        restored = ProteinExtensionConfig.from_dict(config_dict)

        assert restored.name == original.name
        assert restored.bond_length_weight == original.bond_length_weight
        assert restored.backbone_atoms == original.backbone_atoms
        assert restored.ideal_bond_lengths == original.ideal_bond_lengths

    def test_list_to_tuple_conversion(self):
        """Test that lists are converted to tuples when loading from dict."""
        config = ExtensionConfig.from_dict(
            {
                "name": "test",
                "tags": ["a", "b", "c"],  # List in dict
            }
        )

        assert config.tags == ("a", "b", "c")  # Tuple in config
        assert isinstance(config.tags, tuple)
