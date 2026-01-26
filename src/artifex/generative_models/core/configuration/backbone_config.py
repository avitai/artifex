"""Backbone configuration for diffusion models.

This module provides a type-safe, extensible backbone configuration system
for diffusion models, following Artifex's protocol-based design pattern.

Configuration Hierarchy:
- BackboneConfig: Union type for all backbone configs (discriminated by backbone_type)
  - UNetBackboneConfig: Standard convolutional U-Net
  - DiTBackboneConfig: Diffusion Transformer
  - UViTBackboneConfig: ViT with U-Net-like skip connections
  - UNet2DConditionBackboneConfig: Text-conditioned UNet

Design Principles:
1. Type Safety: Discriminated unions with backbone_type field
2. Extensibility: Add new backbones by creating new config classes
3. Modularity: Backbone selection is config-driven, not code-driven
4. Follows Principle #4: Methods take configs, NOT individual parameters
"""

import dataclasses
from typing import Literal

from flax import nnx

from artifex.generative_models.core.configuration.base_network import BaseNetworkConfig
from artifex.generative_models.core.configuration.validation import (
    validate_positive_float,
    validate_positive_int,
    validate_positive_tuple,
)


# =============================================================================
# Backbone Type Literals - for discriminated unions
# =============================================================================

BackboneTypeLiteral = Literal["unet", "dit", "uvit", "unet2d_condition", "unet_1d"]


# =============================================================================
# UNet Backbone Config
# =============================================================================


@dataclasses.dataclass(frozen=True)
class UNetBackboneConfig(BaseNetworkConfig):
    """Configuration for UNet backbone in diffusion models.

    This is the default backbone for diffusion models, using a convolutional
    encoder-decoder architecture with skip connections.

    Attributes:
        backbone_type: Discriminator field, always "unet"
        name: Name of the configuration
        hidden_dims: Channel dimensions per UNet level (from BaseNetworkConfig)
        activation: Activation function name (from BaseNetworkConfig)
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_embedding_dim: Dimension of time embeddings
        num_res_blocks: Number of residual blocks per level
        attention_resolutions: Resolutions at which to apply attention
        channel_mult: Channel multipliers for each level
        use_attention: Whether to use attention layers
        resamp_with_conv: Whether to use convolution for resampling
    """

    # Discriminator field for type-safe backbone selection
    backbone_type: Literal["unet"] = "unet"

    # UNet-specific required fields (with dummy defaults for field ordering)
    in_channels: int = 0
    out_channels: int = 0

    # UNet-specific optional fields with defaults
    time_embedding_dim: int = 128
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = (16, 8)
    channel_mult: tuple[int, ...] = (1, 2, 4)
    use_attention: bool = True
    resamp_with_conv: bool = True

    def __post_init__(self) -> None:
        """Validate UNet backbone configuration."""
        super().__post_init__()

        # Validate required fields
        validate_positive_int(self.in_channels, "in_channels")
        validate_positive_int(self.out_channels, "out_channels")
        validate_positive_int(self.time_embedding_dim, "time_embedding_dim")
        validate_positive_int(self.num_res_blocks, "num_res_blocks")

        # Validate channel_mult (cannot be empty)
        if len(self.channel_mult) == 0:
            raise ValueError("channel_mult cannot be empty")
        validate_positive_tuple(self.channel_mult, "channel_mult")

        # attention_resolutions can be empty (no attention)
        if self.attention_resolutions:
            validate_positive_tuple(self.attention_resolutions, "attention_resolutions")


# =============================================================================
# DiT (Diffusion Transformer) Backbone Config
# =============================================================================


@dataclasses.dataclass(frozen=True)
class DiTBackboneConfig(BaseNetworkConfig):
    """Configuration for DiT (Diffusion Transformer) backbone.

    Diffusion Transformer replaces the U-Net with a Vision Transformer
    architecture, offering better scalability for large-scale models.

    Based on "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023).

    Attributes:
        backbone_type: Discriminator field, always "dit"
        name: Name of the configuration
        hidden_dims: Used as (hidden_size,) for compatibility with BaseNetworkConfig
        activation: Activation function (typically "gelu" for transformers)
        img_size: Input image size (square images)
        patch_size: Size of image patches for patchification
        hidden_size: Transformer hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim = hidden_size * mlp_ratio
        num_classes: Number of classes for conditional generation (None for unconditional)
        learn_sigma: Whether to learn variance in addition to noise
        cfg_scale: Classifier-free guidance scale
    """

    # Discriminator field for type-safe backbone selection
    backbone_type: Literal["dit"] = "dit"

    # DiT architecture parameters
    img_size: int = 32
    patch_size: int = 2
    hidden_size: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0

    # Conditional generation parameters
    num_classes: int | None = None
    learn_sigma: bool = False
    cfg_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate DiT backbone configuration."""
        super().__post_init__()

        # Validate DiT architecture parameters
        validate_positive_int(self.img_size, "img_size")
        validate_positive_int(self.patch_size, "patch_size")
        validate_positive_int(self.hidden_size, "hidden_size")
        validate_positive_int(self.depth, "depth")
        validate_positive_int(self.num_heads, "num_heads")
        validate_positive_float(self.mlp_ratio, "mlp_ratio")

        # Validate num_classes (must be positive if specified)
        if self.num_classes is not None:
            validate_positive_int(self.num_classes, "num_classes")

        # Validate hidden_size divisibility by num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads, "
                f"got hidden_size={self.hidden_size}, num_heads={self.num_heads}"
            )

        # Validate patch_size divides img_size
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size must be divisible by patch_size, "
                f"got img_size={self.img_size}, patch_size={self.patch_size}"
            )


# =============================================================================
# U-ViT Backbone Config (Vision Transformer with U-Net-like skip connections)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class UViTBackboneConfig(BaseNetworkConfig):
    """Configuration for U-ViT backbone (ViT with long skip connections).

    U-ViT combines the transformer architecture with U-Net-like skip connections,
    treating all inputs (time, condition, noisy image patches) as tokens.

    Based on "All are Worth Words: A ViT Backbone for Diffusion Models" (CVPR 2023).

    Attributes:
        backbone_type: Discriminator field, always "uvit"
        name: Name of the configuration
        hidden_dims: Hidden dimensions for encoder/decoder layers
        activation: Activation function name
        img_size: Input image size
        patch_size: Size of image patches
        embed_dim: Embedding dimension
        depth: Number of transformer blocks (total, symmetric encoder-decoder)
        num_heads: Number of attention heads
        in_channels: Number of input image channels
        out_channels: Number of output channels
        mlp_ratio: MLP hidden dim = embed_dim * mlp_ratio
        use_skip_connection: Whether to use long skip connections
    """

    # Discriminator field for type-safe backbone selection
    backbone_type: Literal["uvit"] = "uvit"

    # U-ViT architecture parameters
    img_size: int = 32
    patch_size: int = 2
    embed_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    in_channels: int = 3
    out_channels: int = 3
    mlp_ratio: float = 4.0
    use_skip_connection: bool = True

    def __post_init__(self) -> None:
        """Validate U-ViT backbone configuration."""
        super().__post_init__()

        # Validate U-ViT architecture parameters
        validate_positive_int(self.img_size, "img_size")
        validate_positive_int(self.patch_size, "patch_size")
        validate_positive_int(self.embed_dim, "embed_dim")
        validate_positive_int(self.depth, "depth")
        validate_positive_int(self.num_heads, "num_heads")
        validate_positive_int(self.in_channels, "in_channels")
        validate_positive_int(self.out_channels, "out_channels")
        validate_positive_float(self.mlp_ratio, "mlp_ratio")

        # Validate embed_dim divisibility by num_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, "
                f"got embed_dim={self.embed_dim}, num_heads={self.num_heads}"
            )


# =============================================================================
# UNet2DCondition Backbone Config (for text-conditioned models like Stable Diffusion)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class UNet2DConditionBackboneConfig(BaseNetworkConfig):
    """Configuration for UNet2DCondition backbone (text-conditioned UNet).

    This backbone adds cross-attention layers for conditioning on text embeddings,
    as used in Stable Diffusion and similar text-to-image models.

    Attributes:
        backbone_type: Discriminator field, always "unet2d_condition"
        name: Name of the configuration
        hidden_dims: Channel dimensions per level
        activation: Activation function name
        in_channels: Number of input channels (typically 4 for latent space)
        out_channels: Number of output channels
        cross_attention_dim: Dimension of text embeddings (e.g., 768 for CLIP)
        num_heads: Number of attention heads
        num_res_blocks: Number of residual blocks per level
        attention_levels: Which levels to add cross-attention (e.g., [0, 1, 2])
        time_embedding_dim: Dimension of time embeddings
    """

    # Discriminator field for type-safe backbone selection
    backbone_type: Literal["unet2d_condition"] = "unet2d_condition"

    # UNet2DCondition-specific required fields
    in_channels: int = 4
    out_channels: int = 4
    cross_attention_dim: int = 768
    num_heads: int = 8
    num_res_blocks: int = 2
    attention_levels: tuple[int, ...] = (0, 1, 2)
    time_embedding_dim: int = 128

    def __post_init__(self) -> None:
        """Validate UNet2DCondition backbone configuration."""
        super().__post_init__()

        # Validate required fields
        validate_positive_int(self.in_channels, "in_channels")
        validate_positive_int(self.out_channels, "out_channels")
        validate_positive_int(self.cross_attention_dim, "cross_attention_dim")
        validate_positive_int(self.num_heads, "num_heads")
        validate_positive_int(self.num_res_blocks, "num_res_blocks")
        validate_positive_int(self.time_embedding_dim, "time_embedding_dim")


# =============================================================================
# UNet1D Backbone Config (for 1D signals like audio)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class UNet1DBackboneConfig(BaseNetworkConfig):
    """Configuration for 1D UNet backbone for 1D signals like audio.

    This backbone is designed for diffusion models operating on 1D sequences
    (audio waveforms, time series, etc.) using 1D convolutions.

    Attributes:
        backbone_type: Discriminator field, always "unet_1d"
        name: Name of the configuration
        hidden_dims: Channel dimensions per UNet level (from BaseNetworkConfig)
        activation: Activation function name (from BaseNetworkConfig)
        in_channels: Number of input channels (default 1 for mono audio)
        time_embedding_dim: Dimension of time embeddings
    """

    # Discriminator field for type-safe backbone selection
    backbone_type: Literal["unet_1d"] = "unet_1d"

    # UNet1D-specific fields
    in_channels: int = 1
    time_embedding_dim: int = 128

    def __post_init__(self) -> None:
        """Validate UNet1D backbone configuration."""
        super().__post_init__()

        # Validate required fields
        validate_positive_int(self.in_channels, "in_channels")
        validate_positive_int(self.time_embedding_dim, "time_embedding_dim")


# =============================================================================
# Union Type for All Backbone Configs
# =============================================================================

# Type alias for any backbone configuration
# This is a discriminated union based on backbone_type field
BackboneConfig = (
    UNetBackboneConfig
    | DiTBackboneConfig
    | UViTBackboneConfig
    | UNet2DConditionBackboneConfig
    | UNet1DBackboneConfig
)


# =============================================================================
# Backbone Factory Function
# =============================================================================


def create_backbone(config: BackboneConfig, *, rngs: nnx.Rngs) -> nnx.Module:
    """Create a backbone network from configuration.

    This factory function creates the appropriate backbone based on the
    backbone_type discriminator field in the config.

    Args:
        config: Backbone configuration (UNetBackboneConfig, DiTBackboneConfig, etc.)
        rngs: Random number generators for initialization

    Returns:
        Initialized backbone network (UNet, DiffusionTransformer, etc.)

    Raises:
        ValueError: If backbone_type is not supported
        NotImplementedError: If backbone is not yet implemented (e.g., U-ViT)
    """
    match config.backbone_type:
        case "unet":
            from artifex.generative_models.models.backbones.unet import UNet

            # Pass UNetBackboneConfig directly to UNet
            return UNet(config=config, rngs=rngs)

        case "dit":
            from artifex.generative_models.models.backbones.dit import DiffusionTransformer

            return DiffusionTransformer(
                img_size=config.img_size,
                patch_size=config.patch_size,
                in_channels=config.hidden_dims[0] if config.hidden_dims else 3,
                hidden_size=config.hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                num_classes=config.num_classes,
                dropout_rate=config.dropout_rate,
                learn_sigma=config.learn_sigma,
                rngs=rngs,
            )

        case "uvit":
            # U-ViT is not yet implemented in artifex
            raise NotImplementedError(
                "U-ViT backbone is not yet implemented. Use 'unet' or 'dit' backbone types for now."
            )

        case "unet2d_condition":
            from artifex.generative_models.models.backbones.unet_cross_attention import (
                UNet2DCondition,
            )

            return UNet2DCondition(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                hidden_dims=list(config.hidden_dims),
                num_res_blocks=config.num_res_blocks,
                attention_levels=list(config.attention_levels),
                cross_attention_dim=config.cross_attention_dim,
                num_heads=config.num_heads,
                time_embedding_dim=config.time_embedding_dim,
                rngs=rngs,
            )

        case "unet_1d":
            from artifex.generative_models.models.backbones.unet_1d import UNet1D

            return UNet1D(config=config, rngs=rngs)

        case _:
            raise ValueError(
                f"Unknown backbone_type: {config.backbone_type}. "
                f"Supported types: 'unet', 'dit', 'uvit', 'unet2d_condition', 'unet_1d'"
            )


def get_backbone_config_type(backbone_type: BackboneTypeLiteral) -> type[BackboneConfig]:
    """Get the config class for a given backbone type.

    Args:
        backbone_type: One of 'unet', 'dit', 'uvit', 'unet2d_condition'

    Returns:
        The corresponding config class

    Raises:
        ValueError: If backbone_type is not supported
    """
    match backbone_type:
        case "unet":
            return UNetBackboneConfig
        case "dit":
            return DiTBackboneConfig
        case "uvit":
            return UViTBackboneConfig
        case "unet2d_condition":
            return UNet2DConditionBackboneConfig
        case _:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")
