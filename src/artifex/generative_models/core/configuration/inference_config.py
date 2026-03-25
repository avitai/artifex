"""Inference configuration frozen dataclasses.

Design:
- Frozen dataclasses inheriting from BaseConfig
- All validation in __post_init__ using DRY utilities from validation.py
- Three-level hierarchy: InferenceConfig -> DiffusionInferenceConfig
  -> ProteinDiffusionInferenceConfig
- Uses tuples instead of lists for immutable sequences
"""

import dataclasses

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.validation import (
    validate_non_negative_float,
    validate_positive_int,
)


_VALID_DEVICES = frozenset({"cpu", "cuda", "tpu"})


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class InferenceConfig(BaseConfig):
    """Base configuration for model inference.

    Attributes:
        checkpoint_path: Path to model checkpoint (required)
        output_dir: Directory for saving outputs
        batch_size: Inference batch size (must be positive)
        num_samples: Number of samples to generate (must be positive)
        device: Device to run inference on (cpu, cuda, or tpu)
    """

    # Required field — uses None sentinel (dataclass ordering constraint)
    checkpoint_path: str = None  # type: ignore[assignment]  # validated in __post_init__

    # Optional fields with defaults
    output_dir: str = "./outputs"
    batch_size: int = 1
    num_samples: int = 1
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Validate all fields."""
        super(InferenceConfig, self).__post_init__()

        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required and cannot be None")

        validate_positive_int(self.batch_size, "batch_size")
        validate_positive_int(self.num_samples, "num_samples")

        if self.device not in _VALID_DEVICES:
            raise ValueError(f"device must be one of {sorted(_VALID_DEVICES)}, got {self.device!r}")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DiffusionInferenceConfig(InferenceConfig):
    """Configuration for diffusion model inference.

    Extends InferenceConfig with diffusion-specific sampling parameters
    such as sampler type, timesteps, temperature, and classifier guidance.

    Attributes:
        sampler: Sampling algorithm (ddpm, ddim, etc.)
        timesteps: Number of timesteps for sampling (must be positive)
        temperature: Temperature for sampling / noise scale (must be non-negative)
        sample_with_classifier_guidance: Whether to use classifier guidance
        guidance_scale: Scale for classifier-free guidance (must be non-negative)
        save_intermediate_steps: Whether to save intermediate denoising steps
        intermediate_step_interval: Interval for saving intermediate steps (must be positive)
        seed: Random seed for reproducibility (None for no fixed seed)
    """

    sampler: str = "ddpm"
    timesteps: int = 1000
    temperature: float = 1.0
    sample_with_classifier_guidance: bool = False
    guidance_scale: float = 7.5
    save_intermediate_steps: bool = False
    intermediate_step_interval: int = 50
    seed: int | None = 42

    def __post_init__(self) -> None:
        """Validate all fields."""
        super(DiffusionInferenceConfig, self).__post_init__()

        validate_positive_int(self.timesteps, "timesteps")
        validate_non_negative_float(self.temperature, "temperature")
        validate_non_negative_float(self.guidance_scale, "guidance_scale")
        validate_positive_int(self.intermediate_step_interval, "intermediate_step_interval")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProteinDiffusionInferenceConfig(DiffusionInferenceConfig):
    """Configuration for protein diffusion model inference.

    Extends DiffusionInferenceConfig with protein-specific parameters
    for structure generation, metrics calculation, and PDB output.

    Attributes:
        target_seq_length: Target sequence length to generate (must be positive)
        backbone_atom_indices: Indices of backbone atoms to use (immutable tuple)
        calculate_metrics: Whether to calculate quality metrics for generated proteins
        visualize_structures: Whether to visualize generated structures
        save_as_pdb: Whether to save generated structures as PDB files
    """

    target_seq_length: int = 128
    backbone_atom_indices: tuple[int, ...] = (0, 1, 2, 4)
    calculate_metrics: bool = True
    visualize_structures: bool = True
    save_as_pdb: bool = True

    def __post_init__(self) -> None:
        """Validate all fields."""
        super(ProteinDiffusionInferenceConfig, self).__post_init__()

        validate_positive_int(self.target_seq_length, "target_seq_length")
