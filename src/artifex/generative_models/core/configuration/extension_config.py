"""Frozen dataclass configuration for extensions.

This module provides configuration classes for the extensions system using
frozen dataclasses, following the same pattern as base_dataclass.py.

Key Design Decisions:
1. All configs are frozen dataclasses (immutable)
2. All sequence fields use tuples (not lists)
3. Validation happens in __post_init__ (fail-fast)
4. dacite handles dict → dataclass conversion with type checking
"""

import dataclasses
from typing import Literal

from .base_dataclass import BaseConfig
from .validation import validate_non_negative_float, validate_probability


# =============================================================================
# Base Extension Configurations
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ExtensionConfig(BaseConfig):
    """Base configuration for all extensions.

    This replaces the Pydantic-based ExtensionConfig with a frozen dataclass.

    Attributes:
        weight: Weight for the extension's contribution to loss (default: 1.0).
        enabled: Whether the extension is enabled (default: True).
    """

    weight: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate extension configuration."""
        super().__post_init__()
        validate_non_negative_float(self.weight, "weight")


@dataclasses.dataclass(frozen=True)
class ConstraintExtensionConfig(ExtensionConfig):
    """Configuration for constraint extensions.

    Constraint extensions add physical or domain-specific constraints
    to model outputs through loss terms or projections.

    Attributes:
        tolerance: Tolerance for constraint violations (default: 0.01).
        projection_enabled: Whether to project outputs to satisfy constraints.
    """

    tolerance: float = 0.01
    projection_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate constraint extension configuration."""
        super().__post_init__()
        validate_non_negative_float(self.tolerance, "tolerance")


@dataclasses.dataclass(frozen=True)
class AugmentationExtensionConfig(ExtensionConfig):
    """Configuration for augmentation extensions.

    Augmentation extensions provide data transformation for training.

    Attributes:
        probability: Probability of applying augmentation (default: 1.0).
        deterministic: Whether to use deterministic augmentation (default: False).
    """

    probability: float = 1.0
    deterministic: bool = False

    def __post_init__(self) -> None:
        """Validate augmentation extension configuration."""
        super().__post_init__()
        validate_probability(self.probability, "probability")


@dataclasses.dataclass(frozen=True)
class SamplingExtensionConfig(ExtensionConfig):
    """Configuration for sampling extensions.

    Sampling extensions modify the generation/sampling process.

    Attributes:
        guidance_scale: Scale factor for guidance (default: 1.0).
        temperature: Sampling temperature (default: 1.0).
    """

    guidance_scale: float = 1.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        """Validate sampling extension configuration."""
        super().__post_init__()
        validate_non_negative_float(self.guidance_scale, "guidance_scale")
        validate_non_negative_float(self.temperature, "temperature")


@dataclasses.dataclass(frozen=True)
class LossExtensionConfig(ExtensionConfig):
    """Configuration for loss extensions.

    Loss extensions provide modular loss components.

    Attributes:
        weight_schedule: Schedule for weight adjustment during training.
        warmup_steps: Number of warmup steps before full weight.
    """

    weight_schedule: Literal["constant", "linear", "cosine", "exponential"] = "constant"
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        """Validate loss extension configuration."""
        super().__post_init__()
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")

        valid_schedules = {"constant", "linear", "cosine", "exponential"}
        if self.weight_schedule not in valid_schedules:
            raise ValueError(
                f"weight_schedule must be one of {valid_schedules}, got {self.weight_schedule}"
            )


@dataclasses.dataclass(frozen=True)
class EvaluationExtensionConfig(ExtensionConfig):
    """Configuration for evaluation extensions.

    Evaluation extensions provide domain-specific evaluation metrics.

    Attributes:
        compute_on_train: Whether to compute metrics during training.
        compute_on_eval: Whether to compute metrics during evaluation.
    """

    compute_on_train: bool = False
    compute_on_eval: bool = True

    def __post_init__(self) -> None:
        """Validate evaluation extension configuration."""
        super().__post_init__()


@dataclasses.dataclass(frozen=True)
class CallbackExtensionConfig(ExtensionConfig):
    """Configuration for callback extensions.

    Callback extensions hook into training lifecycle events.

    Attributes:
        frequency: How often to run the callback (in steps).
        on_train: Whether to run during training.
        on_eval: Whether to run during evaluation.
    """

    frequency: int = 1
    on_train: bool = True
    on_eval: bool = True

    def __post_init__(self) -> None:
        """Validate callback extension configuration."""
        super().__post_init__()
        if self.frequency < 1:
            raise ValueError(f"frequency must be at least 1, got {self.frequency}")


@dataclasses.dataclass(frozen=True)
class ModalityExtensionConfig(ExtensionConfig):
    """Configuration for modality extensions.

    Modality extensions provide modality-specific preprocessing and encoding.

    Attributes:
        input_key: Key for input data in batch dict.
        output_key: Key for output data in batch dict.
    """

    input_key: str = "input"
    output_key: str = "output"

    def __post_init__(self) -> None:
        """Validate modality extension configuration."""
        super().__post_init__()


@dataclasses.dataclass(frozen=True)
class ArchitectureExtensionConfig(ExtensionConfig):
    """Configuration for architecture extensions.

    Architecture extensions modify model structure (LoRA, adapters, etc.).

    Attributes:
        rank: Rank for low-rank adaptations (default: 4).
        alpha: Alpha scaling factor (default: 1.0).
        target_modules: Names of modules to adapt.
        dropout: Dropout rate for adaptation layers.
    """

    rank: int = 4
    alpha: float = 1.0
    target_modules: tuple[str, ...] = ()
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate architecture extension configuration."""
        super().__post_init__()
        if self.rank < 1:
            raise ValueError(f"rank must be at least 1, got {self.rank}")
        validate_non_negative_float(self.alpha, "alpha")
        validate_probability(self.dropout, "dropout")


# =============================================================================
# Domain-Specific Extension Configurations
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ProteinExtensionConfig(ConstraintExtensionConfig):
    """Protein-specific extension configuration.

    Configuration for protein structure generation constraints.

    Attributes:
        backbone_atoms: Names of backbone atoms to consider.
        bond_length_weight: Weight for bond length constraints.
        bond_angle_weight: Weight for bond angle constraints.
        dihedral_weight: Weight for dihedral angle constraints.
        ideal_bond_lengths: Dictionary of ideal bond lengths (Angstroms).
        ideal_bond_angles: Dictionary of ideal bond angles (radians).
    """

    backbone_atoms: tuple[str, ...] = ("N", "CA", "C", "O")
    bond_length_weight: float = 1.0
    bond_angle_weight: float = 0.5
    dihedral_weight: float = 0.3

    # These use dict because they're complex mappings
    ideal_bond_lengths: dict[str, float] = dataclasses.field(default_factory=dict)
    ideal_bond_angles: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate protein extension configuration."""
        super().__post_init__()
        validate_non_negative_float(self.bond_length_weight, "bond_length_weight")
        validate_non_negative_float(self.bond_angle_weight, "bond_angle_weight")
        validate_non_negative_float(self.dihedral_weight, "dihedral_weight")


@dataclasses.dataclass(frozen=True)
class ProteinDihedralConfig(ConstraintExtensionConfig):
    """Configuration for protein dihedral angle constraints.

    Attributes:
        target_secondary_structure: Target secondary structure type.
        phi_weight: Weight for phi angle constraints.
        psi_weight: Weight for psi angle constraints.
        omega_weight: Weight for omega angle constraints.
        ideal_phi: Ideal phi angle (radians).
        ideal_psi: Ideal psi angle (radians).
        ideal_omega: Ideal omega angle (radians, default: pi for trans).
    """

    target_secondary_structure: Literal["alpha_helix", "beta_sheet", "coil"] = "alpha_helix"
    phi_weight: float = 0.5
    psi_weight: float = 0.5
    omega_weight: float = 1.0
    ideal_phi: float | None = None  # Will use defaults based on secondary structure
    ideal_psi: float | None = None
    ideal_omega: float = 3.141592653589793  # pi

    def __post_init__(self) -> None:
        """Validate protein dihedral configuration."""
        super().__post_init__()
        validate_non_negative_float(self.phi_weight, "phi_weight")
        validate_non_negative_float(self.psi_weight, "psi_weight")
        validate_non_negative_float(self.omega_weight, "omega_weight")


@dataclasses.dataclass(frozen=True)
class ProteinMixinConfig(ExtensionConfig):
    """Configuration for protein mixin extensions.

    Provides amino acid encoding and protein-specific features.

    Attributes:
        embedding_dim: Dimension for amino acid type embeddings.
        use_one_hot: Whether to use one-hot encoding (vs learned embeddings).
        num_aa_types: Number of amino acid types (20 standard + 1 unknown/padding).
    """

    embedding_dim: int = 16
    use_one_hot: bool = True
    num_aa_types: int = 21

    def __post_init__(self) -> None:
        """Validate protein mixin configuration."""
        super().__post_init__()
        if self.embedding_dim < 1:
            raise ValueError(f"embedding_dim must be at least 1, got {self.embedding_dim}")
        if self.num_aa_types < 1:
            raise ValueError(f"num_aa_types must be at least 1, got {self.num_aa_types}")


@dataclasses.dataclass(frozen=True)
class ChemicalConstraintConfig(ConstraintExtensionConfig):
    """Configuration for chemical/molecular constraints.

    Attributes:
        enforce_valence: Whether to enforce valence constraints.
        enforce_bond_lengths: Whether to enforce bond length constraints.
        enforce_ring_closure: Whether to enforce ring closure constraints.
        max_ring_size: Maximum allowed ring size.
    """

    enforce_valence: bool = True
    enforce_bond_lengths: bool = True
    enforce_ring_closure: bool = True
    max_ring_size: int = 8

    def __post_init__(self) -> None:
        """Validate chemical constraint configuration."""
        super().__post_init__()
        if self.max_ring_size < 3:
            raise ValueError(f"max_ring_size must be at least 3, got {self.max_ring_size}")


@dataclasses.dataclass(frozen=True)
class ImageAugmentationConfig(AugmentationExtensionConfig):
    """Configuration for image augmentation extensions.

    Attributes:
        random_flip_horizontal: Whether to randomly flip horizontally.
        random_flip_vertical: Whether to randomly flip vertically.
        random_rotation: Maximum random rotation in degrees.
        color_jitter: Whether to apply color jitter.
        brightness_range: Range for brightness adjustment.
        contrast_range: Range for contrast adjustment.
    """

    random_flip_horizontal: bool = True
    random_flip_vertical: bool = False
    random_rotation: float = 0.0
    color_jitter: bool = False
    brightness_range: tuple[float, float] = (0.9, 1.1)
    contrast_range: tuple[float, float] = (0.9, 1.1)

    def __post_init__(self) -> None:
        """Validate image augmentation configuration."""
        super().__post_init__()
        validate_non_negative_float(self.random_rotation, "random_rotation")

        if len(self.brightness_range) != 2:
            raise ValueError("brightness_range must have exactly 2 elements")
        if self.brightness_range[0] > self.brightness_range[1]:
            raise ValueError("brightness_range[0] must be <= brightness_range[1]")

        if len(self.contrast_range) != 2:
            raise ValueError("contrast_range must have exactly 2 elements")
        if self.contrast_range[0] > self.contrast_range[1]:
            raise ValueError("contrast_range[0] must be <= contrast_range[1]")


@dataclasses.dataclass(frozen=True)
class AudioSpectralConfig(ExtensionConfig):
    """Configuration for audio spectral analysis extensions.

    Attributes:
        n_fft: FFT size for spectral analysis.
        hop_length: Hop length for STFT.
        n_mels: Number of mel bands.
        sample_rate: Audio sample rate.
        f_min: Minimum frequency for mel scale.
        f_max: Maximum frequency for mel scale (None for Nyquist).
    """

    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    sample_rate: int = 22050
    f_min: float = 0.0
    f_max: float | None = None

    def __post_init__(self) -> None:
        """Validate audio spectral configuration."""
        super().__post_init__()
        if self.n_fft < 1:
            raise ValueError(f"n_fft must be at least 1, got {self.n_fft}")
        if self.hop_length < 1:
            raise ValueError(f"hop_length must be at least 1, got {self.hop_length}")
        if self.n_mels < 1:
            raise ValueError(f"n_mels must be at least 1, got {self.n_mels}")
        if self.sample_rate < 1:
            raise ValueError(f"sample_rate must be at least 1, got {self.sample_rate}")
        validate_non_negative_float(self.f_min, "f_min")
        if self.f_max is not None:
            validate_non_negative_float(self.f_max, "f_max")
            if self.f_max <= self.f_min:
                raise ValueError("f_max must be greater than f_min")


@dataclasses.dataclass(frozen=True)
class TextEmbeddingConfig(ExtensionConfig):
    """Configuration for text embedding extensions.

    Attributes:
        vocab_size: Size of the vocabulary.
        embedding_dim: Dimension of embeddings.
        max_sequence_length: Maximum sequence length.
        padding_idx: Index for padding token.
        use_positional_encoding: Whether to use positional encoding.
    """

    vocab_size: int = 30000
    embedding_dim: int = 256
    max_sequence_length: int = 512
    padding_idx: int = 0
    use_positional_encoding: bool = True

    def __post_init__(self) -> None:
        """Validate text embedding configuration."""
        super().__post_init__()
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be at least 1, got {self.vocab_size}")
        if self.embedding_dim < 1:
            raise ValueError(f"embedding_dim must be at least 1, got {self.embedding_dim}")
        if self.max_sequence_length < 1:
            raise ValueError(
                f"max_sequence_length must be at least 1, got {self.max_sequence_length}"
            )
        if self.padding_idx < 0:
            raise ValueError(f"padding_idx must be non-negative, got {self.padding_idx}")


# =============================================================================
# Sampling Extension Configurations
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ClassifierFreeGuidanceConfig(SamplingExtensionConfig):
    """Configuration for classifier-free guidance.

    Attributes:
        guidance_scale: Scale factor for guidance (typically 1.0-20.0).
        unconditional_conditioning: Whether to use unconditional conditioning.
    """

    unconditional_conditioning: bool = True

    def __post_init__(self) -> None:
        """Validate classifier-free guidance configuration."""
        super().__post_init__()


@dataclasses.dataclass(frozen=True)
class ConstrainedSamplingConfig(SamplingExtensionConfig):
    """Configuration for constrained sampling.

    Attributes:
        constraint_weight: Weight for constraint enforcement during sampling.
        projection_steps: Number of projection steps per sampling step.
        use_gradient_guidance: Whether to use gradient-based guidance.
    """

    constraint_weight: float = 1.0
    projection_steps: int = 1
    use_gradient_guidance: bool = False

    def __post_init__(self) -> None:
        """Validate constrained sampling configuration."""
        super().__post_init__()
        validate_non_negative_float(self.constraint_weight, "constraint_weight")
        if self.projection_steps < 0:
            raise ValueError(f"projection_steps must be non-negative, got {self.projection_steps}")


# =============================================================================
# Composite Extension Configuration
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ExtensionPipelineConfig(BaseConfig):
    """Configuration for a pipeline of extensions.

    Allows composing multiple extensions with individual configs.

    Attributes:
        extensions: Tuple of extension configurations.
        aggregate_losses: Whether to aggregate extension losses.
        loss_reduction: How to reduce multiple extension losses.
    """

    extensions: tuple[ExtensionConfig, ...] = ()
    aggregate_losses: bool = True
    loss_reduction: Literal["sum", "mean", "weighted_sum"] = "weighted_sum"

    def __post_init__(self) -> None:
        """Validate extension pipeline configuration."""
        super().__post_init__()
        valid_reductions = {"sum", "mean", "weighted_sum"}
        if self.loss_reduction not in valid_reductions:
            raise ValueError(
                f"loss_reduction must be one of {valid_reductions}, got {self.loss_reduction}"
            )
