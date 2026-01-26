"""Inference configuration schemas.

This module defines configuration schemas for model inference.
"""

from pydantic import Field, field_validator

# Import BaseConfig inline to avoid circular imports
from artifex.configs.schema.base import BaseConfig


class InferenceConfig(BaseConfig):
    """Base configuration for model inference."""

    checkpoint_path: str = Field(..., description="Path to model checkpoint")
    output_dir: str = Field("./outputs", description="Directory for saving outputs")
    batch_size: int = Field(1, description="Inference batch size")
    num_samples: int = Field(1, description="Number of samples to generate")
    device: str = Field("cuda", description="Device to run inference on")

    @field_validator("batch_size", "num_samples")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate that value is a positive integer."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device."""
        valid_devices = ["cpu", "cuda", "tpu"]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        return v


class DiffusionInferenceConfig(InferenceConfig):
    """Configuration for diffusion model inference."""

    sampler: str = Field("ddpm", description="Sampling algorithm (ddpm, ddim, etc.)")
    timesteps: int = Field(1000, description="Number of timesteps for sampling")
    temperature: float = Field(1.0, description="Temperature for sampling (noise scale)")
    sample_with_classifier_guidance: bool = Field(
        False, description="Whether to use classifier guidance for sampling"
    )
    guidance_scale: float = Field(7.5, description="Scale for classifier-free guidance")
    save_intermediate_steps: bool = Field(
        False, description="Whether to save intermediate denoising steps"
    )
    intermediate_step_interval: int = Field(
        50, description="Interval for saving intermediate steps"
    )
    seed: int | None = Field(42, description="Random seed for reproducibility")

    @field_validator("temperature", "guidance_scale")
    @classmethod
    def validate_positive_float(cls, v: float) -> float:
        """Validate that value is a positive float."""
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v

    @field_validator("timesteps", "intermediate_step_interval")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate that value is a positive integer."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class ProteinDiffusionInferenceConfig(DiffusionInferenceConfig):
    """Configuration specific to protein diffusion model inference."""

    target_seq_length: int = Field(128, description="Target sequence length to generate")
    backbone_atom_indices: list[int] = Field(
        [0, 1, 2, 4], description="Indices of backbone atoms to use"
    )
    calculate_metrics: bool = Field(
        True,
        description=("Whether to calculate quality metrics for generated proteins"),
    )
    visualize_structures: bool = Field(
        True, description="Whether to visualize generated structures"
    )
    save_as_pdb: bool = Field(True, description="Whether to save generated structures as PDB files")

    @field_validator("target_seq_length")
    @classmethod
    def validate_target_seq_length(cls, v: int) -> int:
        """Validate target sequence length."""
        if v <= 0:
            raise ValueError("Target sequence length must be positive")
        return v
