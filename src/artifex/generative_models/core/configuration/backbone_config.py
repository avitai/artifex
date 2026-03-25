"""Typed backbone configuration models for retained diffusion backbones.

The public backbone surface is intentionally limited to backbones with real
builder and runtime support:

- ``UNetBackboneConfig`` for convolutional image diffusion
- ``DiTBackboneConfig`` for diffusion transformers
- ``UNet2DConditionBackboneConfig`` for text-conditioned latent diffusion
- ``UNet1DBackboneConfig`` for sequential diffusion
"""

import dataclasses
from typing import Literal

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.base_network import BaseNetworkConfig
from artifex.generative_models.core.configuration.validation import (
    validate_dropout_rate,
    validate_positive_float,
    validate_positive_int,
    validate_positive_tuple,
)


BackboneTypeLiteral = Literal["unet", "dit", "unet2d_condition", "unet_1d"]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class UNetBackboneConfig(BaseNetworkConfig):
    """Configuration for the retained convolutional U-Net diffusion backbone."""

    backbone_type: Literal["unet"] = "unet"

    in_channels: int = 0
    out_channels: int = 0

    time_embedding_dim: int = 128
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = (16, 8)
    channel_mult: tuple[int, ...] = (1, 2, 4)
    use_attention: bool = True
    resamp_with_conv: bool = True

    def __post_init__(self) -> None:
        """Validate UNet backbone configuration."""
        super(UNetBackboneConfig, self).__post_init__()

        validate_positive_int(self.in_channels, "in_channels")
        validate_positive_int(self.out_channels, "out_channels")
        validate_positive_int(self.time_embedding_dim, "time_embedding_dim")
        validate_positive_int(self.num_res_blocks, "num_res_blocks")

        if not self.channel_mult:
            raise ValueError("channel_mult cannot be empty")
        validate_positive_tuple(self.channel_mult, "channel_mult")

        if self.attention_resolutions:
            validate_positive_tuple(self.attention_resolutions, "attention_resolutions")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DiTBackboneConfig(BaseConfig):
    """Configuration for the retained Diffusion Transformer backbone.

    Unlike convolutional UNets, DiT does not share a generic hidden-layer tuple
    contract. Its runtime shape is driven by patchification, transformer width,
    and explicit image-channel ownership.
    """

    backbone_type: Literal["dit"] = "dit"

    img_size: int = 32
    patch_size: int = 2
    in_channels: int = 3
    hidden_size: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_classes: int | None = None
    learn_sigma: bool = False
    dropout_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate DiT backbone configuration."""
        super(DiTBackboneConfig, self).__post_init__()

        validate_positive_int(self.img_size, "img_size")
        validate_positive_int(self.patch_size, "patch_size")
        validate_positive_int(self.in_channels, "in_channels")
        validate_positive_int(self.hidden_size, "hidden_size")
        validate_positive_int(self.depth, "depth")
        validate_positive_int(self.num_heads, "num_heads")
        validate_positive_float(self.mlp_ratio, "mlp_ratio")
        validate_dropout_rate(self.dropout_rate)

        if self.num_classes is not None:
            validate_positive_int(self.num_classes, "num_classes")

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads, "
                f"got hidden_size={self.hidden_size}, num_heads={self.num_heads}"
            )

        if self.img_size % self.patch_size != 0:
            raise ValueError(
                "img_size must be divisible by patch_size, "
                f"got img_size={self.img_size}, patch_size={self.patch_size}"
            )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class UNet2DConditionBackboneConfig(BaseConfig):
    """Configuration for the retained text-conditioned UNet backbone."""

    backbone_type: Literal["unet2d_condition"] = "unet2d_condition"

    hidden_dims: tuple[int, ...] = ()
    in_channels: int = 4
    out_channels: int = 4
    cross_attention_dim: int = 768
    num_heads: int = 8
    num_res_blocks: int = 2
    attention_levels: tuple[int, ...] = ()
    time_embedding_dim: int = 128

    def __post_init__(self) -> None:
        """Validate conditioned UNet backbone configuration."""
        super(UNet2DConditionBackboneConfig, self).__post_init__()

        if not self.hidden_dims:
            raise ValueError("hidden_dims is required and cannot be empty")

        validate_positive_tuple(self.hidden_dims, "hidden_dims")
        validate_positive_int(self.in_channels, "in_channels")
        validate_positive_int(self.out_channels, "out_channels")
        validate_positive_int(self.cross_attention_dim, "cross_attention_dim")
        validate_positive_int(self.num_heads, "num_heads")
        validate_positive_int(self.num_res_blocks, "num_res_blocks")
        validate_positive_int(self.time_embedding_dim, "time_embedding_dim")

        if not self.attention_levels:
            object.__setattr__(self, "attention_levels", tuple(range(len(self.hidden_dims))))

        if any(level < 0 for level in self.attention_levels):
            raise ValueError("attention_levels must be non-negative")
        if any(level >= len(self.hidden_dims) for level in self.attention_levels):
            raise ValueError("attention_levels must be less than len(hidden_dims)")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class UNet1DBackboneConfig(BaseNetworkConfig):
    """Configuration for the retained 1D UNet diffusion backbone."""

    backbone_type: Literal["unet_1d"] = "unet_1d"

    in_channels: int = 1
    time_embedding_dim: int = 128

    def __post_init__(self) -> None:
        """Validate UNet1D backbone configuration."""
        super(UNet1DBackboneConfig, self).__post_init__()

        validate_positive_int(self.in_channels, "in_channels")
        validate_positive_int(self.time_embedding_dim, "time_embedding_dim")


BackboneConfig = (
    UNetBackboneConfig | DiTBackboneConfig | UNet2DConditionBackboneConfig | UNet1DBackboneConfig
)
