"""Cases for the denoising backbones in ``artifex.generative_models.models.backbones``."""

from __future__ import annotations

import jax
from substrax.testing.nnx_conformance import Case

from tests.artifex.nnx_conformance.cases.inputs import arrays, BATCH, labels, timesteps


_BACKBONES = "artifex.generative_models.models.backbones"


def dit_inputs(key: jax.Array) -> tuple[jax.Array, ...]:
    """DiT inputs: images, timesteps in [0, 1000) and class labels."""
    k1, k2, k3 = jax.random.split(key, 3)
    return (
        jax.random.normal(k1, (BATCH, 8, 8, 3)),
        jax.random.uniform(k2, (BATCH,), maxval=1000.0),
        labels(k3, 4),
    )


def unet_config() -> object:
    """A small UNet backbone configuration for 8x8 RGB inputs."""
    from artifex.generative_models.core.configuration.backbone_config import (  # noqa: PLC0415
        UNetBackboneConfig,
    )

    return UNetBackboneConfig(
        name="unet",
        hidden_dims=(8, 16),
        activation="gelu",
        in_channels=3,
        out_channels=3,
        time_embedding_dim=16,
        num_res_blocks=1,
        attention_resolutions=(),
        channel_mult=(1, 2),
    )


def unet1d_config() -> object:
    """A small UNet1D backbone configuration for single-channel signals."""
    from artifex.generative_models.core.configuration.backbone_config import (  # noqa: PLC0415
        UNet1DBackboneConfig,
    )

    return UNet1DBackboneConfig(
        name="unet1d", hidden_dims=(8, 16), activation="gelu", in_channels=1, time_embedding_dim=16
    )


def cases() -> dict[str, Case]:
    """Return the backbone cases, by the qualified name of the class each builds."""

    from artifex.generative_models.models.backbones import (  # noqa: PLC0415
        dit,
        unet,
        unet_1d,
        unet_cross_attention,
    )

    cross = unet_cross_attention
    return {
        f"{_BACKBONES}.unet.TimeEmbedding": Case(
            build=lambda r: unet.TimeEmbedding(embedding_dim=16, rngs=r),
            inputs=lambda key: (jax.random.uniform(key, (BATCH,), maxval=1000.0),),
        ),
        f"{_BACKBONES}.unet.ConvBlock": Case(
            build=lambda r: unet.ConvBlock(8, 16, time_embedding_dim=16, rngs=r),
            inputs=arrays((6, 6, 8), (16,)),
        ),
        f"{_BACKBONES}.unet.DownBlock": Case(
            build=lambda r: unet.DownBlock(8, 16, time_embedding_dim=16, rngs=r),
            inputs=arrays((6, 6, 8), (16,)),
        ),
        f"{_BACKBONES}.unet.UpBlock": Case(
            build=lambda r: unet.UpBlock(16, 8, 8, time_embedding_dim=16, rngs=r),
            inputs=arrays((3, 3, 16), (6, 6, 8), (16,)),
        ),
        f"{_BACKBONES}.unet.UNet": Case(
            build=lambda r: unet.UNet(unet_config(), rngs=r),
            inputs=timesteps((8, 8, 3)),
        ),
        f"{_BACKBONES}.dit.TimestepEmbedder": Case(
            build=lambda r: dit.TimestepEmbedder(32, frequency_embedding_size=16, rngs=r),
            inputs=lambda key: (jax.random.uniform(key, (BATCH,), maxval=1000.0),),
        ),
        f"{_BACKBONES}.dit.LabelEmbedder": Case(
            build=lambda r: dit.LabelEmbedder(4, 32, dropout_rate=0.5, rngs=r),
            inputs=lambda key: (labels(key, 4),),
            has_dropout=True,
        ),
        f"{_BACKBONES}.dit.DiTBlock": Case(
            build=lambda r: dit.DiTBlock(32, 2, dropout_rate=0.5, rngs=r),
            inputs=arrays((4, 32), (32,)),
            has_dropout=True,
        ),
        f"{_BACKBONES}.dit.FinalLayer": Case(
            build=lambda r: dit.FinalLayer(32, 2, 3, rngs=r),
            inputs=arrays((4, 32), (32,)),
        ),
        f"{_BACKBONES}.dit.DiffusionTransformer": Case(
            build=lambda r: dit.DiffusionTransformer(
                img_size=8,
                patch_size=2,
                in_channels=3,
                hidden_size=32,
                depth=2,
                num_heads=2,
                num_classes=4,
                dropout_rate=0.5,
                rngs=r,
            ),
            inputs=dit_inputs,
            has_dropout=True,
        ),
        f"{_BACKBONES}.unet_1d.ConvBlock1D": Case(
            build=lambda r: unet_1d.ConvBlock1D(8, 16, time_embedding_dim=16, rngs=r),
            inputs=arrays((12, 8), (16,)),
        ),
        f"{_BACKBONES}.unet_1d.DownBlock1D": Case(
            build=lambda r: unet_1d.DownBlock1D(8, 16, time_embedding_dim=16, rngs=r),
            inputs=arrays((12, 8), (16,)),
        ),
        f"{_BACKBONES}.unet_1d.UpBlock1D": Case(
            build=lambda r: unet_1d.UpBlock1D(16, 8, 8, time_embedding_dim=16, rngs=r),
            inputs=arrays((6, 16), (12, 8), (16,)),
        ),
        f"{_BACKBONES}.unet_1d.UNet1D": Case(
            build=lambda r: unet_1d.UNet1D(unet1d_config(), rngs=r),
            inputs=timesteps((16, 1)),
        ),
        f"{_BACKBONES}.unet_cross_attention.CrossAttentionBlock": Case(
            build=lambda r: cross.CrossAttentionBlock(8, 12, 2, rngs=r),
            inputs=arrays((4, 4, 8), (5, 12)),
        ),
        f"{_BACKBONES}.unet_cross_attention.DownBlockWithCrossAttention": Case(
            build=lambda r: cross.DownBlockWithCrossAttention(8, 16, 16, 12, 2, 1, rngs=r),
            inputs=arrays((6, 6, 8), (16,), (5, 12)),
        ),
        f"{_BACKBONES}.unet_cross_attention.UpBlockWithCrossAttention": Case(
            build=lambda r: cross.UpBlockWithCrossAttention(16, 8, 8, 16, 12, 2, 1, rngs=r),
            inputs=arrays((3, 3, 16), (6, 6, 8), (16,), (5, 12)),
        ),
        f"{_BACKBONES}.unet_cross_attention.UNet2DCondition": Case(
            build=lambda r: cross.UNet2DCondition(
                in_channels=4,
                out_channels=4,
                hidden_dims=[8, 16],
                num_res_blocks=1,
                attention_levels=[1],
                cross_attention_dim=12,
                num_heads=2,
                time_embedding_dim=16,
                rngs=r,
            ),
            inputs=lambda key: (
                *timesteps((8, 8, 4))(key),
                jax.random.normal(jax.random.fold_in(key, 1), (BATCH, 5, 12)),
            ),
            call=lambda module, x, timesteps, text: module(x, timesteps, conditioning=text),
        ),
    }
