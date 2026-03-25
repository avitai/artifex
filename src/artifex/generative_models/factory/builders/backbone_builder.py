"""Backbone factory for diffusion models.

Creates backbone networks from configuration objects, keeping the
configuration module free of concrete model imports.
"""

from flax import nnx

from artifex.generative_models.core.configuration.backbone_config import (
    BackboneConfig,
    BackboneTypeLiteral,
    DiTBackboneConfig,
    UNet1DBackboneConfig,
    UNet2DConditionBackboneConfig,
    UNetBackboneConfig,
)


def create_backbone(config: BackboneConfig, *, rngs: nnx.Rngs) -> nnx.Module:
    """Create a backbone network from configuration.

    This factory function creates the appropriate backbone based on the
    backbone_type discriminator field in the config.

    Args:
        config: Retained backbone configuration with real runtime support
        rngs: Random number generators for initialization

    Returns:
        Initialized backbone network (UNet, DiffusionTransformer, etc.)

    Raises:
        ValueError: If backbone_type is not supported
    """
    match config.backbone_type:
        case "unet":
            from artifex.generative_models.models.backbones.unet import UNet

            return UNet(config=config, rngs=rngs)

        case "dit":
            from artifex.generative_models.models.backbones.dit import DiffusionTransformer

            return DiffusionTransformer(
                img_size=config.img_size,
                patch_size=config.patch_size,
                in_channels=config.in_channels,
                hidden_size=config.hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                num_classes=config.num_classes,
                dropout_rate=config.dropout_rate,
                learn_sigma=config.learn_sigma,
                rngs=rngs,
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
                "Supported types: 'unet', 'dit', 'unet2d_condition', 'unet_1d'"
            )


def get_backbone_config_type(backbone_type: BackboneTypeLiteral) -> type[BackboneConfig]:
    """Get the config class for a given backbone type.

    Args:
        backbone_type: One of 'unet', 'dit', 'unet2d_condition', 'unet_1d'

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
        case "unet2d_condition":
            return UNet2DConditionBackboneConfig
        case "unet_1d":
            return UNet1DBackboneConfig
        case _:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")
