"""Configuration schema for Diffusion Transformer models.

This should be placed in: src/artifex/configs/schema/models/dit.py
"""

from dataclasses import dataclass
from typing import Any

from artifex.configs.schema.base import BaseConfig
from artifex.generative_models.core.configuration import ModelConfig


@dataclass
class DiTConfig(BaseConfig):
    """Configuration for Diffusion Transformer models.

    DiT replaces the U-Net backbone with a Vision Transformer for better
    scalability and performance in diffusion models.
    """

    # Image parameters
    img_size: int = 32
    patch_size: int = 2
    in_channels: int = 3

    # Transformer architecture
    hidden_size: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0

    # Training parameters
    dropout_rate: float = 0.0
    use_flash_attention: bool = False  # For future optimization

    # Conditional generation
    num_classes: int | None = None
    cfg_scale: float = 1.0  # Classifier-free guidance scale
    class_dropout_prob: float = 0.1  # For classifier-free guidance training

    # Diffusion parameters
    noise_steps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # linear, cosine, quadratic
    learn_sigma: bool = False  # Whether to learn variance

    def __post_init__(self):
        """Validate and set derived parameters."""
        super().__post_init__()

        # Ensure patch size divides image size
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"Image size ({self.img_size}) must be divisible by patch size ({self.patch_size})"
            )

    def to_model_config(self) -> dict[str, Any]:
        """Convert to ModelConfiguration compatible format.

        Returns:
            Dictionary with configuration for ModelConfiguration
        """
        return {
            "name": self.name,
            "model_class": "artifex.generative_models.models.diffusion.dit.DiTModel",
            "input_dim": (self.img_size, self.img_size, self.in_channels),
            "output_dim": (self.img_size, self.img_size, self.in_channels),
            "hidden_dims": [self.hidden_size],
            "parameters": {
                "img_size": self.img_size,
                "patch_size": self.patch_size,
                "hidden_size": self.hidden_size,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "dropout_rate": self.dropout_rate,
                "num_classes": self.num_classes,
                "cfg_scale": self.cfg_scale,
                "class_dropout_prob": self.class_dropout_prob,
                "noise_steps": self.noise_steps,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "beta_schedule": self.beta_schedule,
                "learn_sigma": self.learn_sigma,
                "use_flash_attention": self.use_flash_attention,
            },
        }


@dataclass
class DiTSizePresets:
    """Predefined DiT model sizes following the original paper."""

    @staticmethod
    def DiT_S(img_size: int = 32, patch_size: int = 2, **kwargs) -> DiTConfig:
        """DiT-S: Small model (33M parameters at 256x256)."""
        config = DiTConfig(
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=384,
            depth=12,
            num_heads=6,
            **kwargs,
        )
        config.name = config.name or "dit_s"
        return config

    @staticmethod
    def DiT_B(img_size: int = 32, patch_size: int = 2, **kwargs) -> DiTConfig:
        """DiT-B: Base model (130M parameters at 256x256)."""
        config = DiTConfig(
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=768,
            depth=12,
            num_heads=12,
            **kwargs,
        )
        config.name = config.name or "dit_b"
        return config

    @staticmethod
    def DiT_L(img_size: int = 32, patch_size: int = 2, **kwargs) -> DiTConfig:
        """DiT-L: Large model (458M parameters at 256x256)."""
        config = DiTConfig(
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=1024,
            depth=24,
            num_heads=16,
            **kwargs,
        )
        config.name = config.name or "dit_l"
        return config

    @staticmethod
    def DiT_XL(img_size: int = 32, patch_size: int = 2, **kwargs) -> DiTConfig:
        """DiT-XL: Extra Large model (675M parameters at 256x256)."""
        config = DiTConfig(
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            **kwargs,
        )
        config.name = config.name or "dit_xl"
        return config


# Convenience functions for creating standard DiT configurations
def create_dit_config(
    model_size: str = "B",
    img_size: int = 32,
    patch_size: int = 2,
    num_classes: int | None = None,
    name: str | None = None,
    **kwargs,
) -> DiTConfig:
    """Create a DiT configuration with standard sizing.

    Args:
        model_size: Model size - "S", "B", "L", or "XL"
        img_size: Input image size
        patch_size: Patch size for patchification
        num_classes: Number of classes for conditional generation
        name: Configuration name
        **kwargs: Additional configuration parameters

    Returns:
        DiTConfig instance
    """
    size_map = {
        "S": DiTSizePresets.DiT_S,
        "B": DiTSizePresets.DiT_B,
        "L": DiTSizePresets.DiT_L,
        "XL": DiTSizePresets.DiT_XL,
    }

    if model_size not in size_map:
        raise ValueError(f"Model size must be one of {list(size_map.keys())}")

    config_fn = size_map[model_size]
    config = config_fn(img_size=img_size, patch_size=patch_size, num_classes=num_classes, **kwargs)

    if name:
        config.name = name

    return config


def dit_config_to_model_config(dit_config: DiTConfig) -> ModelConfig:
    """Convert DiTConfig to ModelConfig.

    Args:
        dit_config: DiT configuration

    Returns:
        ModelConfig instance
    """
    config_dict = dit_config.to_model_config()
    return ModelConfig(**config_dict)
