"""One conformance case per artifex ``nnx.Module`` class: how to build it and call it.

A case is a :class:`substrax.testing.nnx_conformance.Case`: it builds the module, makes inputs
with a leading batch axis of ``BATCH``, and states the capabilities the module claims (dropout,
batch statistics, rows computed independently). ``test_conformance.py`` runs every substrax
check whose claim a case makes. ``EXEMPT`` names the classes with no case of their own, each
with the reason.

Where a module's activation is configurable, its case uses ``nnx.silu``: a finite difference
taken across a ReLU kink is no derivative (ResNetBlock's float64 check disagrees by 0.64% with
``relu`` and agrees with ``silu``), and the activation is a library function, not this code.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from substrax.testing.nnx_conformance import Case
from substrax.typing import PyTree


BATCH = 4


def _normal(*shape: int) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    return lambda key: (jax.random.normal(key, (BATCH, *shape)),)


def _arrays(*shapes: tuple[int, ...]) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    """Inputs of standard normals, one per shape, each with a leading batch axis of ``BATCH``."""

    def inputs(key: jax.Array) -> tuple[jax.Array, ...]:
        keys = jax.random.split(key, len(shapes))
        return tuple(
            jax.random.normal(k, (BATCH, *shape)) for k, shape in zip(keys, shapes, strict=True)
        )

    return inputs


def _labels(key: jax.Array, num_classes: int) -> jax.Array:
    return jax.random.randint(key, (BATCH,), 0, num_classes)


_CORE = "artifex.generative_models.core"
_LAYERS = f"{_CORE}.layers"


def _egnn_inputs(key: jax.Array) -> tuple[jax.Array, ...]:
    nodes, features, edge = 5, 8, 4
    k1, k2, k3 = jax.random.split(key, 3)
    adjacency = jnp.broadcast_to(jnp.ones((nodes, nodes)) - jnp.eye(nodes), (BATCH, nodes, nodes))
    return (
        jax.random.normal(k1, (BATCH, nodes, features)),
        jax.random.normal(k2, (BATCH, nodes, 3)),
        adjacency,
        jax.random.normal(k3, (BATCH, nodes, nodes, edge)),
    )


def _decoder_inputs(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2 = jax.random.split(key)
    return jax.random.normal(k1, (BATCH, 6, 16)), jax.random.normal(k2, (BATCH, 7, 16))


def _layers() -> dict[str, Case]:
    from artifex.generative_models.core.layers import (  # noqa: PLC0415
        clifford,
        egnn,
        flash_attention,
        positional,
        residual,
        resnet,
        transformers,
    )
    from artifex.generative_models.core.layers.kan import (  # noqa: PLC0415
        basis_layers,
        conv,
        grids,
        spline,
    )

    return {
        f"{_LAYERS}.clifford.linear.CliffordLinear": Case(
            build=lambda r: clifford.CliffordLinear(
                metric=(1, 1), in_channels=3, out_channels=5, rngs=r
            ),
            inputs=_normal(3, 4),
        ),
        f"{_LAYERS}.clifford.conv.CliffordConv1d": Case(
            build=lambda r: clifford.CliffordConv1d(
                metric=(1,), in_channels=3, out_channels=5, kernel_size=3, padding="SAME", rngs=r
            ),
            inputs=_normal(8, 3, 2),
        ),
        f"{_LAYERS}.clifford.conv.CliffordConv2d": Case(
            build=lambda r: clifford.CliffordConv2d(
                metric=(1, 1), in_channels=3, out_channels=5, kernel_size=3, padding="SAME", rngs=r
            ),
            inputs=_normal(6, 6, 3, 4),
        ),
        f"{_LAYERS}.clifford.conv.CliffordConv3d": Case(
            build=lambda r: clifford.CliffordConv3d(
                metric=(1, 1, 1),
                in_channels=2,
                out_channels=3,
                kernel_size=3,
                padding="SAME",
                rngs=r,
            ),
            inputs=_normal(4, 4, 4, 2, 8),
        ),
        f"{_LAYERS}.clifford.spectral.CliffordSpectralConv2d": Case(
            build=lambda r: clifford.CliffordSpectralConv2d(
                metric=(1, 1), in_channels=3, out_channels=5, modes1=2, modes2=2, rngs=r
            ),
            inputs=_normal(8, 8, 3, 4),
        ),
        f"{_LAYERS}.clifford.spectral.CliffordSpectralConv3d": Case(
            build=lambda r: clifford.CliffordSpectralConv3d(
                metric=(1, 1, 1),
                in_channels=2,
                out_channels=3,
                modes1=2,
                modes2=2,
                modes3=2,
                rngs=r,
            ),
            inputs=_normal(4, 4, 4, 2, 8),
        ),
        f"{_LAYERS}.clifford.norm.CliffordBatchNorm": Case(
            build=lambda r: clifford.CliffordBatchNorm(metric=(1, 1), channels=3, rngs=r),
            inputs=_normal(6, 3, 4),
            has_batch_stats=True,
        ),
        f"{_LAYERS}.clifford.norm.CliffordGroupNorm": Case(
            build=lambda r: clifford.CliffordGroupNorm(
                metric=(1, 1), num_groups=2, channels=4, rngs=r
            ),
            inputs=_normal(5, 5, 4, 4),
        ),
        f"{_LAYERS}.clifford.norm.MultiVectorActivation": Case(
            build=lambda r: clifford.MultiVectorActivation(
                channels=3, n_blades=4, input_blades=(0, 1, 2, 3), rngs=r
            ),
            inputs=_normal(3, 4),
        ),
        f"{_LAYERS}.egnn.EGNNLayer": Case(
            build=lambda r: egnn.EGNNLayer(edge_dim=4, hidden_dim=8, dropout_rate=0.5, rngs=r),
            inputs=_egnn_inputs,
            has_dropout=True,
        ),
        f"{_LAYERS}.egnn.EGNNBlock": Case(
            build=lambda r: egnn.EGNNBlock(8, 6, 16, dropout_rate=0.5, rngs=r),
            inputs=_normal(5, 8),
            has_dropout=True,
        ),
        f"{_LAYERS}.flash_attention.FlashMultiHeadAttention": Case(
            build=lambda r: flash_attention.FlashMultiHeadAttention(
                num_heads=2, in_features=16, dropout_rate=0.5, decode=False, rngs=r
            ),
            inputs=_normal(6, 16),
            has_dropout=True,
        ),
        f"{_LAYERS}.positional.SinusoidalPositionalEncoding": Case(
            build=lambda r: positional.SinusoidalPositionalEncoding(
                dim=16, max_len=12, dropout_rate=0.5, rngs=r
            ),
            inputs=_normal(6, 16),
            has_dropout=True,
        ),
        f"{_LAYERS}.positional.LearnedPositionalEncoding": Case(
            build=lambda r: positional.LearnedPositionalEncoding(
                dim=16, max_len=12, dropout_rate=0.5, rngs=r
            ),
            inputs=_normal(6, 16),
            has_dropout=True,
        ),
        f"{_LAYERS}.positional.RotaryPositionalEncoding": Case(
            build=lambda r: positional.RotaryPositionalEncoding(
                dim=16, max_len=12, dropout_rate=0.5, rngs=r
            ),
            inputs=_normal(6, 16),
            has_dropout=True,
        ),
        f"{_LAYERS}.residual.Conv1DResidualBlock": Case(
            build=lambda r: residual.Conv1DResidualBlock(residual_channels=4, dilation=2, rngs=r),
            inputs=_normal(10, 4),
        ),
        f"{_LAYERS}.residual.Conv2DResidualBlock": Case(
            build=lambda r: residual.Conv2DResidualBlock(channels=4, activation=nnx.silu, rngs=r),
            inputs=_normal(6, 6, 4),
        ),
        f"{_LAYERS}.resnet.ResNetBlock": Case(
            build=lambda r: resnet.ResNetBlock(
                in_features=4, features=6, activation=nnx.silu, rngs=r
            ),
            inputs=_normal(6, 6, 4),
            has_batch_stats=True,
            is_rowwise=False,
        ),
        f"{_LAYERS}.resnet.BottleneckBlock": Case(
            build=lambda r: resnet.BottleneckBlock(
                in_features=8, out_features=8, activation=nnx.silu, rngs=r
            ),
            inputs=_normal(6, 6, 8),
            has_batch_stats=True,
            is_rowwise=False,
        ),
        f"{_LAYERS}.transformers.FeedForwardNetwork": Case(
            build=lambda r: transformers.FeedForwardNetwork(
                in_features=16, hidden_features=32, dropout_rate=0.5, rngs=r
            ),
            inputs=_normal(6, 16),
            has_dropout=True,
        ),
        f"{_LAYERS}.transformers.TransformerEncoderBlock": Case(
            build=lambda r: transformers.TransformerEncoderBlock(
                hidden_dim=16, num_heads=2, dropout_rate=0.5, attention_dropout_rate=0.5, rngs=r
            ),
            inputs=_normal(6, 16),
            has_dropout=True,
        ),
        f"{_LAYERS}.transformers.TransformerEncoder": Case(
            build=lambda r: transformers.TransformerEncoder(
                num_layers=2,
                hidden_dim=16,
                num_heads=2,
                dropout_rate=0.5,
                attention_dropout_rate=0.5,
                rngs=r,
            ),
            inputs=_normal(6, 16),
            has_dropout=True,
        ),
        f"{_LAYERS}.transformers.TransformerDecoderBlock": Case(
            build=lambda r: transformers.TransformerDecoderBlock(
                hidden_dim=16, num_heads=2, dropout_rate=0.5, attention_dropout_rate=0.5, rngs=r
            ),
            inputs=_decoder_inputs,
            has_dropout=True,
        ),
        f"{_LAYERS}.transformers.TransformerDecoder": Case(
            build=lambda r: transformers.TransformerDecoder(
                num_layers=2,
                hidden_dim=16,
                num_heads=2,
                dropout_rate=0.5,
                attention_dropout_rate=0.5,
                rngs=r,
            ),
            inputs=_decoder_inputs,
            has_dropout=True,
        ),
        f"{_LAYERS}.kan.spline.DenseKANLayer": Case(
            build=lambda r: spline.DenseKANLayer(n_in=3, n_out=4, rngs=r),
            inputs=_normal(3),
        ),
        f"{_LAYERS}.kan.spline.EfficientKANLayer": Case(
            build=lambda r: spline.EfficientKANLayer(n_in=3, n_out=4, rngs=r),
            inputs=_normal(3),
        ),
        f"{_LAYERS}.kan.basis_layers.ChebyshevKANLayer": Case(
            build=lambda r: basis_layers.ChebyshevKANLayer(n_in=3, n_out=4, rngs=r),
            inputs=_normal(3),
        ),
        f"{_LAYERS}.kan.basis_layers.FourierKANLayer": Case(
            build=lambda r: basis_layers.FourierKANLayer(n_in=3, n_out=4, rngs=r),
            inputs=_normal(3),
        ),
        f"{_LAYERS}.kan.basis_layers.LegendreKANLayer": Case(
            build=lambda r: basis_layers.LegendreKANLayer(n_in=3, n_out=4, rngs=r),
            inputs=_normal(3),
        ),
        f"{_LAYERS}.kan.basis_layers.RBFKANLayer": Case(
            build=lambda r: basis_layers.RBFKANLayer(n_in=3, n_out=4, rngs=r),
            inputs=_normal(3),
        ),
        f"{_LAYERS}.kan.basis_layers.SineKANLayer": Case(
            build=lambda r: basis_layers.SineKANLayer(n_in=3, n_out=4, rngs=r),
            inputs=_normal(3),
        ),
        f"{_LAYERS}.kan.conv.ConvKANLayer": Case(
            build=lambda r: conv.ConvKANLayer(in_channels=2, out_channels=3, rngs=r),
            inputs=_normal(5, 5, 2),
        ),
        f"{_LAYERS}.kan.grids.BSplineBasis": Case(
            build=lambda r: grids.BSplineBasis(k=3),
            inputs=lambda key: (jax.random.uniform(key, (BATCH, 3), minval=-1.0, maxval=1.0),),
            call=lambda module, x: module(
                x, grids.EfficientKANGrid(n_nodes=3, k=3, grid_intervals=3).knots[...]
            ),
            is_differentiable=False,
        ),
    }


def _base() -> dict[str, Case]:
    from artifex.generative_models.core import base  # noqa: PLC0415

    return {
        f"{_CORE}.base.MLP": Case(
            build=lambda r: base.MLP([8, 6], in_features=4, dropout_rate=0.5, rngs=r),
            inputs=_normal(4),
            has_dropout=True,
        ),
        f"{_CORE}.base.CNN": Case(
            build=lambda r: base.CNN([4, 6], in_features=3, activation=nnx.silu, rngs=r),
            inputs=_normal(8, 8, 3),
        ),
    }


_BACKBONES = "artifex.generative_models.models.backbones"


def _dit_inputs(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2, k3 = jax.random.split(key, 3)
    return (
        jax.random.normal(k1, (BATCH, 8, 8, 3)),
        jax.random.uniform(k2, (BATCH,), maxval=1000.0),
        _labels(k3, 4),
    )


def _unet_config() -> object:
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


def _unet1d_config() -> object:
    from artifex.generative_models.core.configuration.backbone_config import (  # noqa: PLC0415
        UNet1DBackboneConfig,
    )

    return UNet1DBackboneConfig(
        name="unet1d", hidden_dims=(8, 16), activation="gelu", in_channels=1, time_embedding_dim=16
    )


def _timesteps(*shapes: tuple[int, ...]) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    """Normal inputs of ``shapes``, then timesteps in [0, 1000)."""

    def inputs(key: jax.Array) -> tuple[jax.Array, ...]:
        k1, k2 = jax.random.split(key)
        return (*_arrays(*shapes)(k1), jax.random.uniform(k2, (BATCH,), maxval=1000.0))

    return inputs


def _backbones() -> dict[str, Case]:
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
            inputs=_arrays((6, 6, 8), (16,)),
        ),
        f"{_BACKBONES}.unet.DownBlock": Case(
            build=lambda r: unet.DownBlock(8, 16, time_embedding_dim=16, rngs=r),
            inputs=_arrays((6, 6, 8), (16,)),
        ),
        f"{_BACKBONES}.unet.UpBlock": Case(
            build=lambda r: unet.UpBlock(16, 8, 8, time_embedding_dim=16, rngs=r),
            inputs=_arrays((3, 3, 16), (6, 6, 8), (16,)),
        ),
        f"{_BACKBONES}.unet.UNet": Case(
            build=lambda r: unet.UNet(_unet_config(), rngs=r),
            inputs=_timesteps((8, 8, 3)),
        ),
        f"{_BACKBONES}.dit.TimestepEmbedder": Case(
            build=lambda r: dit.TimestepEmbedder(32, frequency_embedding_size=16, rngs=r),
            inputs=lambda key: (jax.random.uniform(key, (BATCH,), maxval=1000.0),),
        ),
        f"{_BACKBONES}.dit.LabelEmbedder": Case(
            build=lambda r: dit.LabelEmbedder(4, 32, dropout_rate=0.5, rngs=r),
            inputs=lambda key: (_labels(key, 4),),
            has_dropout=True,
        ),
        f"{_BACKBONES}.dit.DiTBlock": Case(
            build=lambda r: dit.DiTBlock(32, 2, dropout_rate=0.5, rngs=r),
            inputs=_arrays((4, 32), (32,)),
            has_dropout=True,
        ),
        f"{_BACKBONES}.dit.FinalLayer": Case(
            build=lambda r: dit.FinalLayer(32, 2, 3, rngs=r),
            inputs=_arrays((4, 32), (32,)),
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
            inputs=_dit_inputs,
            has_dropout=True,
        ),
        f"{_BACKBONES}.unet_1d.ConvBlock1D": Case(
            build=lambda r: unet_1d.ConvBlock1D(8, 16, time_embedding_dim=16, rngs=r),
            inputs=_arrays((12, 8), (16,)),
        ),
        f"{_BACKBONES}.unet_1d.DownBlock1D": Case(
            build=lambda r: unet_1d.DownBlock1D(8, 16, time_embedding_dim=16, rngs=r),
            inputs=_arrays((12, 8), (16,)),
        ),
        f"{_BACKBONES}.unet_1d.UpBlock1D": Case(
            build=lambda r: unet_1d.UpBlock1D(16, 8, 8, time_embedding_dim=16, rngs=r),
            inputs=_arrays((6, 16), (12, 8), (16,)),
        ),
        f"{_BACKBONES}.unet_1d.UNet1D": Case(
            build=lambda r: unet_1d.UNet1D(_unet1d_config(), rngs=r),
            inputs=_timesteps((16, 1)),
        ),
        f"{_BACKBONES}.unet_cross_attention.CrossAttentionBlock": Case(
            build=lambda r: cross.CrossAttentionBlock(8, 12, 2, rngs=r),
            inputs=_arrays((4, 4, 8), (5, 12)),
        ),
        f"{_BACKBONES}.unet_cross_attention.DownBlockWithCrossAttention": Case(
            build=lambda r: cross.DownBlockWithCrossAttention(8, 16, 16, 12, 2, 1, rngs=r),
            inputs=_arrays((6, 6, 8), (16,), (5, 12)),
        ),
        f"{_BACKBONES}.unet_cross_attention.UpBlockWithCrossAttention": Case(
            build=lambda r: cross.UpBlockWithCrossAttention(16, 8, 8, 16, 12, 2, 1, rngs=r),
            inputs=_arrays((3, 3, 16), (6, 6, 8), (16,), (5, 12)),
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
                *_timesteps((8, 8, 4))(key),
                jax.random.normal(jax.random.fold_in(key, 1), (BATCH, 5, 12)),
            ),
            call=lambda module, x, timesteps, text: module(x, timesteps, conditioning=text),
        ),
    }


_VAE = "artifex.generative_models.models.vae"
_VAE_FEATURES, _VAE_LATENT, _VAE_CLASSES = 5, 3, 3
_IMAGE = (8, 8, 3)


def _encoder_config(input_shape: tuple[int, ...] = (_VAE_FEATURES,), latent: int = _VAE_LATENT):
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        EncoderConfig,
    )

    return EncoderConfig(
        name="encoder",
        hidden_dims=(8, 6),
        activation="silu",
        input_shape=input_shape,
        latent_dim=latent,
    )


def _decoder_config(output_shape: tuple[int, ...] = (_VAE_FEATURES,), latent: int = _VAE_LATENT):
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        DecoderConfig,
    )

    return DecoderConfig(
        name="decoder",
        hidden_dims=(6, 8),
        activation="silu",
        output_shape=output_shape,
        latent_dim=latent,
    )


def _vae_config(config_class: type, **fields: object) -> object:
    return config_class(name="vae", encoder=_encoder_config(), decoder=_decoder_config(), **fields)


def _features_and_labels(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2 = jax.random.split(key)
    return jax.random.normal(k1, (BATCH, _VAE_FEATURES)), _labels(k2, _VAE_CLASSES)


def _latents_and_labels(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2 = jax.random.split(key)
    return jax.random.normal(k1, (BATCH, _VAE_LATENT)), _labels(k2, _VAE_CLASSES)


def _vae() -> dict[str, Case]:
    from artifex.generative_models.core.configuration.vae_config import (  # noqa: PLC0415
        BetaVAEConfig,
        BetaVAEWithCapacityConfig,
        ConditionalVAEConfig,
        VAEConfig,
        VQVAEConfig,
    )
    from artifex.generative_models.models.vae import (  # noqa: PLC0415
        base,
        beta_vae,
        conditional,
        decoders,
        encoders,
        spatial_autoencoder,
        vq_vae,
    )

    spatial = spatial_autoencoder
    conditioned_encoder = (_VAE_FEATURES + _VAE_CLASSES,)
    return {
        f"{_VAE}.encoders.Flatten": Case(
            build=lambda r: encoders.Flatten(rngs=r),
            inputs=_normal(3, 4),
            is_differentiable=False,
        ),
        f"{_VAE}.encoders.MLPEncoder": Case(
            build=lambda r: encoders.MLPEncoder(_encoder_config(), rngs=r),
            inputs=_normal(_VAE_FEATURES),
            has_batch_stats=True,
        ),
        f"{_VAE}.encoders.CNNEncoder": Case(
            build=lambda r: encoders.CNNEncoder(_encoder_config(_IMAGE), rngs=r),
            inputs=_normal(*_IMAGE),
            has_batch_stats=True,
        ),
        f"{_VAE}.encoders.ResNetEncoder": Case(
            build=lambda r: encoders.ResNetEncoder(_encoder_config(_IMAGE), rngs=r),
            inputs=_normal(*_IMAGE),
            has_batch_stats=True,
        ),
        f"{_VAE}.encoders.ConditionalEncoder": Case(
            build=lambda r: encoders.ConditionalEncoder(
                encoders.MLPEncoder(_encoder_config(conditioned_encoder), rngs=r),
                _VAE_CLASSES,
                rngs=r,
            ),
            inputs=_features_and_labels,
            has_batch_stats=True,
        ),
        f"{_VAE}.decoders.MLPDecoder": Case(
            build=lambda r: decoders.MLPDecoder(_decoder_config(), rngs=r),
            inputs=_normal(_VAE_LATENT),
        ),
        f"{_VAE}.decoders.CNNDecoder": Case(
            build=lambda r: decoders.CNNDecoder(_decoder_config(_IMAGE), rngs=r),
            inputs=_normal(_VAE_LATENT),
        ),
        f"{_VAE}.decoders.ResNetDecoder": Case(
            build=lambda r: decoders.ResNetDecoder(_decoder_config(_IMAGE), rngs=r),
            inputs=_normal(_VAE_LATENT),
        ),
        f"{_VAE}.decoders.ConditionalDecoder": Case(
            build=lambda r: decoders.ConditionalDecoder(
                decoders.MLPDecoder(_decoder_config(latent=_VAE_LATENT + _VAE_CLASSES), rngs=r),
                _VAE_CLASSES,
                rngs=r,
            ),
            inputs=_latents_and_labels,
        ),
        f"{_VAE}.spatial_autoencoder.SpatialEncoder": Case(
            build=lambda r: spatial.SpatialEncoder(_encoder_config(_IMAGE, latent=2), rngs=r),
            inputs=_normal(*_IMAGE),
        ),
        f"{_VAE}.spatial_autoencoder.SpatialDecoder": Case(
            build=lambda r: spatial.SpatialDecoder(_decoder_config(_IMAGE, latent=2), rngs=r),
            inputs=_normal(2, 2, 2),
        ),
        f"{_VAE}.base.VAE": Case(
            build=lambda r: base.VAE(_vae_config(VAEConfig), rngs=r),
            inputs=_normal(_VAE_FEATURES),
            has_batch_stats=True,
        ),
        f"{_VAE}.beta_vae.BetaVAE": Case(
            build=lambda r: beta_vae.BetaVAE(_vae_config(BetaVAEConfig, beta_default=2.0), rngs=r),
            inputs=_normal(_VAE_FEATURES),
            has_batch_stats=True,
        ),
        f"{_VAE}.beta_vae.BetaVAEWithCapacity": Case(
            build=lambda r: beta_vae.BetaVAEWithCapacity(
                _vae_config(BetaVAEWithCapacityConfig, use_capacity_control=True), rngs=r
            ),
            inputs=_normal(_VAE_FEATURES),
            has_batch_stats=True,
        ),
        f"{_VAE}.conditional.ConditionalVAE": Case(
            build=lambda r: conditional.ConditionalVAE(
                _vae_config(
                    ConditionalVAEConfig, num_classes=_VAE_CLASSES, condition_type="concat"
                ),
                rngs=r,
            ),
            inputs=_features_and_labels,
            call=lambda module, x, labels: module(x, y=labels),
            has_batch_stats=True,
        ),
        f"{_VAE}.vq_vae.VQVAE": Case(
            build=lambda r: vq_vae.VQVAE(
                _vae_config(VQVAEConfig, num_embeddings=8, embedding_dim=_VAE_LATENT),
                rngs=r,
            ),
            inputs=_normal(_VAE_FEATURES),
            has_batch_stats=True,
            # Its outputs include batch means: the codebook and commitment losses, perplexity.
            is_rowwise=False,
        ),
    }


_FLOW = "artifex.generative_models.models.flow"
_FLOW_DIM, _CONDITION_DIM, _ATOMS = 4, 3, 6
_GLOW_IMAGE = (4, 4, 4)


def _both_ways(module: nnx.Module, x: jax.Array) -> PyTree:
    """A flow's two directions: density (forward) and sampling (inverse), each with its log-det."""
    return module.forward(x), module.inverse(x)


def _conditioned_both_ways(module: nnx.Module, x: jax.Array, condition: jax.Array) -> PyTree:
    return module.forward(x, condition=condition), module.inverse(x, condition=condition)


def _model_both_ways(module: nnx.Module, x: jax.Array) -> PyTree:
    return module(x), module.inverse(x)


def _coupling_network() -> object:
    from artifex.generative_models.core.configuration.flow_config import (  # noqa: PLC0415
        CouplingNetworkConfig,
    )

    return CouplingNetworkConfig(name="coupling", hidden_dims=(8, 8), activation="silu")


def _flow_config(config_class: type, **fields: object) -> object:
    return config_class(
        name="flow",
        coupling_network=_coupling_network(),
        input_dim=_FLOW_DIM,
        latent_dim=_FLOW_DIM,
        **fields,
    )


def _mask() -> jax.Array:
    return jnp.array([1.0, 0.0, 1.0, 0.0])


def _features_and_condition(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2 = jax.random.split(key)
    return (
        jax.random.normal(k1, (BATCH, _FLOW_DIM)),
        jax.random.normal(k2, (BATCH, _CONDITION_DIM)),
    )


def _molecule(key: jax.Array, *, features: int | None = None) -> tuple[jax.Array, ...]:
    """Coordinates, then atom features or atom types, then a mask whose last two atoms are padding."""
    k1, k2 = jax.random.split(key)
    coordinates = jax.random.normal(k1, (BATCH, _ATOMS, 3))
    if features is None:
        second = jax.random.randint(k2, (BATCH, _ATOMS), 0, 5)
    else:
        second = jax.random.normal(k2, (BATCH, _ATOMS, features))
    mask = jnp.broadcast_to(jnp.arange(_ATOMS) < _ATOMS - 2, (BATCH, _ATOMS))
    return coordinates, second, mask


def _spline_inputs(key: jax.Array) -> tuple[jax.Array, ...]:
    bins = 4
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return (
        jax.random.uniform(k1, (BATCH, 2), minval=-2.0, maxval=2.0),
        jax.random.normal(k2, (BATCH, 2, bins)),
        jax.random.normal(k3, (BATCH, 2, bins)),
        jax.random.normal(k4, (BATCH, 2, bins + 1)),
    )


def _flow() -> dict[str, Case]:
    from artifex.generative_models.core.configuration.flow_config import (  # noqa: PLC0415
        ConditionalFlowConfig,
        GlowConfig,
        IAFConfig,
        MAFConfig,
        NeuralSplineConfig,
        RealNVPConfig,
    )
    from artifex.generative_models.models.flow import (  # noqa: PLC0415
        conditional,
        glow,
        iaf,
        made,
        maf,
        neural_spline,
        real_nvp,
        se3_molecular,
    )

    spline = neural_spline
    se3 = se3_molecular
    channels = _GLOW_IMAGE[-1]
    return {
        f"{_FLOW}.real_nvp.CouplingLayer": Case(
            build=lambda r: real_nvp.CouplingLayer(_mask(), [8, 8], rngs=r),
            inputs=_normal(_FLOW_DIM),
            call=_both_ways,
        ),
        f"{_FLOW}.real_nvp.RealNVP": Case(
            build=lambda r: real_nvp.RealNVP(
                _flow_config(RealNVPConfig, num_coupling_layers=2), rngs=r
            ),
            inputs=_normal(_FLOW_DIM),
            call=_model_both_ways,
        ),
        f"{_FLOW}.conditional.ConditionalCouplingLayer": Case(
            build=lambda r: conditional.ConditionalCouplingLayer(
                _mask(), [8, 8], _CONDITION_DIM, rngs=r
            ),
            inputs=_features_and_condition,
            call=_conditioned_both_ways,
        ),
        f"{_FLOW}.conditional.ConditionalNormalizingFlow": Case(
            build=lambda r: conditional.ConditionalNormalizingFlow(
                _flow_config(
                    ConditionalFlowConfig, condition_dim=_CONDITION_DIM, num_coupling_layers=2
                ),
                rngs=r,
            ),
            inputs=_features_and_condition,
            call=_conditioned_both_ways,
        ),
        f"{_FLOW}.conditional.ConditionalRealNVP": Case(
            build=lambda r: conditional.ConditionalRealNVP(
                _flow_config(
                    ConditionalFlowConfig, condition_dim=_CONDITION_DIM, num_coupling_layers=2
                ),
                rngs=r,
            ),
            inputs=_features_and_condition,
            call=_conditioned_both_ways,
        ),
        f"{_FLOW}.glow.ActNormLayer": Case(
            build=lambda r: glow.ActNormLayer(channels, rngs=r),
            inputs=_normal(*_GLOW_IMAGE),
            call=_both_ways,
        ),
        f"{_FLOW}.glow.InvertibleConv1x1": Case(
            build=lambda r: glow.InvertibleConv1x1(channels, rngs=r),
            inputs=_normal(*_GLOW_IMAGE),
            call=_both_ways,
        ),
        f"{_FLOW}.glow.AffineCouplingLayer": Case(
            build=lambda r: glow.AffineCouplingLayer(channels, [8], rngs=r),
            inputs=_normal(*_GLOW_IMAGE),
            call=_both_ways,
        ),
        f"{_FLOW}.glow.GlowBlock": Case(
            build=lambda r: glow.GlowBlock(channels, [8], rngs=r),
            inputs=_normal(*_GLOW_IMAGE),
            call=_both_ways,
        ),
        f"{_FLOW}.glow.Glow": Case(
            build=lambda r: glow.Glow(
                GlowConfig(
                    name="glow",
                    coupling_network=_coupling_network(),
                    input_dim=int(jnp.prod(jnp.array(_GLOW_IMAGE))),
                    latent_dim=int(jnp.prod(jnp.array(_GLOW_IMAGE))),
                    image_shape=_GLOW_IMAGE,
                    blocks_per_scale=2,
                ),
                rngs=r,
            ),
            inputs=_normal(*_GLOW_IMAGE),
            call=_model_both_ways,
        ),
        f"{_FLOW}.made.MADE": Case(
            build=lambda r: made.MADE(_FLOW_DIM, [8, 8], rngs=r, activation="tanh"),
            inputs=_normal(_FLOW_DIM),
        ),
        f"{_FLOW}.maf.MAFLayer": Case(
            build=lambda r: maf.MAFLayer(_FLOW_DIM, [8, 8], rngs=r),
            inputs=_normal(_FLOW_DIM),
            call=_both_ways,
        ),
        f"{_FLOW}.maf.MAF": Case(
            build=lambda r: maf.MAF(_flow_config(MAFConfig, num_layers=2), rngs=r),
            inputs=_normal(_FLOW_DIM),
            call=_model_both_ways,
        ),
        f"{_FLOW}.iaf.IAFLayer": Case(
            build=lambda r: iaf.IAFLayer(_FLOW_DIM, [8, 8], rngs=r),
            inputs=_normal(_FLOW_DIM),
            call=_both_ways,
        ),
        f"{_FLOW}.iaf.IAF": Case(
            build=lambda r: iaf.IAF(_flow_config(IAFConfig, num_layers=2), rngs=r),
            inputs=_normal(_FLOW_DIM),
            call=_model_both_ways,
        ),
        f"{_FLOW}.neural_spline.RationalQuadraticSplineTransform": Case(
            build=lambda r: spline.RationalQuadraticSplineTransform(num_bins=4, rngs=r),
            inputs=_spline_inputs,
            # apply_spline takes constrained parameters, as SplineCouplingLayer passes them.
            call=lambda module, x, widths, heights, derivatives: (
                module.apply_spline(x, *module._constrain_parameters(widths, heights, derivatives)),
                module.apply_spline(
                    x, *module._constrain_parameters(widths, heights, derivatives), inverse=True
                ),
            ),
            is_differentiable=False,
        ),
        f"{_FLOW}.neural_spline.SplineCouplingLayer": Case(
            build=lambda r: spline.SplineCouplingLayer(_mask(), [8, 8], num_bins=4, rngs=r),
            inputs=_normal(_FLOW_DIM),
            call=_both_ways,
        ),
        f"{_FLOW}.neural_spline.NeuralSplineFlow": Case(
            build=lambda r: spline.NeuralSplineFlow(
                _flow_config(NeuralSplineConfig, num_layers=2, num_bins=4), rngs=r
            ),
            inputs=_normal(_FLOW_DIM),
            call=_model_both_ways,
        ),
        f"{_FLOW}.se3_molecular.SE3EquivariantLayer": Case(
            build=lambda r: se3.SE3EquivariantLayer(8, 8, _ATOMS, rngs=r),
            inputs=lambda key: _molecule(key, features=8),
        ),
        f"{_FLOW}.se3_molecular.SE3CouplingLayer": Case(
            build=lambda r: se3.SE3CouplingLayer(8, _ATOMS, 5, rngs=r),
            inputs=_molecule,
            call=lambda module, coordinates, types, mask: (
                module(coordinates, types, mask),
                module(coordinates, types, mask, reverse=True),
            ),
        ),
        f"{_FLOW}.se3_molecular.SE3MolecularFlow": Case(
            build=lambda r: se3.SE3MolecularFlow(
                hidden_dim=8, num_layers=1, num_coupling_layers=2, max_atoms=_ATOMS, rngs=r
            ),
            inputs=_molecule,
            call=lambda module, coordinates, types, mask: module.log_prob(coordinates, types, mask),
        ),
    }


_GAN = "artifex.generative_models.models.gan"
_GAN_LATENT, _GAN_CLASSES, _GAN_DROPOUT = 3, 3, 0.5
_GAN_FLAT = (1, 4, 4)
_GAN_IMAGE = (3, 8, 8)
_CRITIC_IMAGE = (3, 16, 16)
_MULTISCALE_IMAGE = (3, 16, 16)
_CYCLE_IMAGE = (8, 8, 3)
_STYLE_DIM, _STYLE_RESOLUTION = 8, 8


def _scores_and_samples(module: nnx.Module, x: jax.Array, z: jax.Array) -> PyTree:
    """Run both networks of a GAN: the discriminator on data and the generator on latents."""
    return module.discriminator(x), module.generator(z)


def _gan_forward_and_samples(module: nnx.Module, x: jax.Array, z: jax.Array) -> PyTree:
    """The GAN's own forward pass, which only discriminates, and its generator on latents."""
    return module(x), module.generator(z)


def _conditioned_scores_and_samples(
    module: nnx.Module, x: jax.Array, z: jax.Array, labels: jax.Array
) -> PyTree:
    return module.discriminator(x, labels), module.generator(z, labels)


def _translations(module: nnx.Module, a: jax.Array, b: jax.Array) -> PyTree:
    """Both CycleGAN translations and both discriminators."""
    return (
        module.generator_a_to_b(a),
        module.generator_b_to_a(b),
        module.discriminator_a(a),
        module.discriminator_b(b),
    )


def _data_and_latents(image: tuple[int, ...]) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    return _arrays(image, (_GAN_LATENT,))


def _labelled_data_and_latents(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2 = jax.random.split(key)
    x, z = _data_and_latents(_GAN_IMAGE)(k1)
    return x, z, jax.nn.one_hot(_labels(k2, _GAN_CLASSES), _GAN_CLASSES)


def _dense_generator() -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        GeneratorConfig,
    )

    return GeneratorConfig(
        name="generator",
        latent_dim=_GAN_LATENT,
        output_shape=_GAN_FLAT,
        hidden_dims=(8, 6),
        activation="silu",
        batch_norm=True,
        dropout_rate=_GAN_DROPOUT,
    )


def _dense_discriminator() -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        DiscriminatorConfig,
    )

    return DiscriminatorConfig(
        name="discriminator",
        input_shape=_GAN_FLAT,
        hidden_dims=(8, 6),
        activation="silu",
        batch_norm=True,
        dropout_rate=_GAN_DROPOUT,
    )


def _conv_generator(
    output_shape: tuple[int, ...] = _GAN_IMAGE, hidden_dims: tuple[int, ...] = (8, 4)
) -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        ConvGeneratorConfig,
    )

    return ConvGeneratorConfig(
        name="generator",
        latent_dim=_GAN_LATENT,
        output_shape=output_shape,
        hidden_dims=hidden_dims,
        activation="silu",
        batch_norm=True,
    )


def _conv_discriminator(input_shape: tuple[int, ...] = _GAN_IMAGE, **fields: object) -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        ConvDiscriminatorConfig,
    )

    return ConvDiscriminatorConfig(
        name="discriminator",
        input_shape=input_shape,
        hidden_dims=(4, 8),
        activation="silu",
        **{"batch_norm": True, **fields},
    )


def _conditional_generator() -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        ConditionalGeneratorConfig,
        ConditionalParams,
    )

    return ConditionalGeneratorConfig(
        name="generator",
        latent_dim=_GAN_LATENT,
        output_shape=_GAN_IMAGE,
        hidden_dims=(8, 4),
        activation="silu",
        batch_norm=True,
        conditional=ConditionalParams(num_classes=_GAN_CLASSES, embedding_dim=4),
    )


def _conditional_discriminator() -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        ConditionalDiscriminatorConfig,
        ConditionalParams,
    )

    return ConditionalDiscriminatorConfig(
        name="discriminator",
        input_shape=_GAN_IMAGE,
        hidden_dims=(4, 8),
        activation="silu",
        dropout_rate=_GAN_DROPOUT,
        conditional=ConditionalParams(num_classes=_GAN_CLASSES, embedding_dim=4),
        kernel_size=(3, 3),
    )


def _cycle_generator() -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        CycleGANGeneratorConfig,
    )

    # No dropout: any dropout_rate > 0 raises at construction (R453), as it does for the
    # CycleGAN discriminator.
    return CycleGANGeneratorConfig(
        name="generator",
        input_shape=_CYCLE_IMAGE,
        output_shape=_CYCLE_IMAGE,
        hidden_dims=(4, 8),
        n_residual_blocks=1,
        activation="silu",
        batch_norm=True,
    )


def _patch_discriminator(
    input_shape: tuple[int, ...], dropout_rate: float = _GAN_DROPOUT
) -> object:
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        PatchGANDiscriminatorConfig,
    )

    return PatchGANDiscriminatorConfig(
        name="discriminator",
        input_shape=input_shape,
        hidden_dims=(4, 8),
        activation="silu",
        batch_norm=True,
        dropout_rate=dropout_rate,
        num_filters=4,
        num_layers=2,
    )


def _gan_config(config_class: type, generator: object, discriminator: object) -> object:
    return config_class(name="gan", generator=generator, discriminator=discriminator)


def _gan() -> dict[str, Case]:
    from artifex.generative_models.core.configuration.gan_config import (  # noqa: PLC0415
        ConditionalGANConfig,
        CycleGANConfig,
        DCGANConfig,
        GANConfig,
        LSGANConfig,
        WGANConfig,
    )
    from artifex.generative_models.core.configuration.network_configs import (  # noqa: PLC0415
        MultiScalePatchGANConfig,
        StyleGAN3DiscriminatorConfig,
        StyleGAN3GeneratorConfig,
    )
    from artifex.generative_models.models.gan import (  # noqa: PLC0415
        base,
        conditional,
        cyclegan,
        dcgan,
        lsgan,
        patchgan,
        stylegan3,
        wgan,
    )

    style_channels = 4
    cycle_config = CycleGANConfig(
        name="cyclegan",
        generator={"a_to_b": _cycle_generator(), "b_to_a": _cycle_generator()},
        discriminator={
            "disc_a": _patch_discriminator(_CYCLE_IMAGE, dropout_rate=0.0),
            "disc_b": _patch_discriminator(_CYCLE_IMAGE, dropout_rate=0.0),
        },
        input_shape_a=_CYCLE_IMAGE,
        input_shape_b=_CYCLE_IMAGE,
    )
    style_image = (_STYLE_RESOLUTION, _STYLE_RESOLUTION, 3)
    # The base configs' validation requires hidden_dims, activation and a shape, which the
    # StyleGAN3 modules never read (their widths follow the resolution).
    unread = {"hidden_dims": (8,), "activation": "leaky_relu"}
    style_generator = StyleGAN3GeneratorConfig(
        name="generator",
        latent_dim=_GAN_LATENT,
        output_shape=style_image,
        style_dim=_STYLE_DIM,
        mapping_layers=2,
        img_resolution=_STYLE_RESOLUTION,
        img_channels=3,
        **unread,
    )
    style_discriminator = StyleGAN3DiscriminatorConfig(
        name="discriminator",
        input_shape=style_image,
        img_resolution=_STYLE_RESOLUTION,
        img_channels=3,
        base_channels=4,
        max_channels=8,
        **unread,
    )
    return {
        f"{_GAN}.base.Generator": Case(
            build=lambda r: base.Generator(_dense_generator(), rngs=r),
            inputs=_normal(_GAN_LATENT),
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.base.Discriminator": Case(
            build=lambda r: base.Discriminator(_dense_discriminator(), rngs=r),
            inputs=_normal(*_GAN_FLAT[1:]),
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.base.GAN": Case(
            build=lambda r: base.GAN(
                _gan_config(GANConfig, _dense_generator(), _dense_discriminator()), rngs=r
            ),
            inputs=_data_and_latents(_GAN_FLAT[1:]),
            call=_gan_forward_and_samples,
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.dcgan.DCGANGenerator": Case(
            build=lambda r: dcgan.DCGANGenerator(_conv_generator(hidden_dims=(8, 6, 4)), rngs=r),
            inputs=_normal(_GAN_LATENT),
            has_batch_stats=True,
        ),
        f"{_GAN}.dcgan.DCGANDiscriminator": Case(
            build=lambda r: dcgan.DCGANDiscriminator(_conv_discriminator(), rngs=r),
            inputs=_normal(*_GAN_IMAGE),
            has_batch_stats=True,
        ),
        f"{_GAN}.dcgan.DCGAN": Case(
            build=lambda r: dcgan.DCGAN(
                _gan_config(
                    DCGANConfig,
                    _conv_generator(hidden_dims=(8, 6, 4)),
                    _conv_discriminator(),
                ),
                rngs=r,
            ),
            inputs=_data_and_latents(_GAN_IMAGE),
            call=_gan_forward_and_samples,
            has_batch_stats=True,
        ),
        f"{_GAN}.wgan.WGANGenerator": Case(
            build=lambda r: wgan.WGANGenerator(_conv_generator(_CRITIC_IMAGE), rngs=r),
            inputs=_normal(_GAN_LATENT),
            has_batch_stats=True,
        ),
        f"{_GAN}.wgan.WGANDiscriminator": Case(
            build=lambda r: wgan.WGANDiscriminator(
                _conv_discriminator(_CRITIC_IMAGE, batch_norm=False, use_instance_norm=True),
                rngs=r,
            ),
            inputs=_normal(*_CRITIC_IMAGE),
        ),
        f"{_GAN}.wgan.WGAN": Case(
            build=lambda r: wgan.WGAN(
                _gan_config(
                    WGANConfig,
                    _conv_generator(_CRITIC_IMAGE),
                    _conv_discriminator(_CRITIC_IMAGE, batch_norm=False, use_instance_norm=True),
                ),
                rngs=r,
            ),
            inputs=_data_and_latents(_CRITIC_IMAGE),
            call=_scores_and_samples,
            has_batch_stats=True,
        ),
        f"{_GAN}.lsgan.LSGANGenerator": Case(
            build=lambda r: lsgan.LSGANGenerator(_conv_generator(), rngs=r),
            inputs=_normal(_GAN_LATENT),
            has_batch_stats=True,
        ),
        f"{_GAN}.lsgan.LSGANDiscriminator": Case(
            build=lambda r: lsgan.LSGANDiscriminator(
                _conv_discriminator(dropout_rate=_GAN_DROPOUT), rngs=r
            ),
            inputs=_normal(*_GAN_IMAGE),
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.lsgan.LSGAN": Case(
            build=lambda r: lsgan.LSGAN(
                _gan_config(
                    LSGANConfig,
                    _conv_generator(),
                    _conv_discriminator(dropout_rate=_GAN_DROPOUT),
                ),
                rngs=r,
            ),
            inputs=_data_and_latents(_GAN_IMAGE),
            call=_scores_and_samples,
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.conditional.ConditionalGenerator": Case(
            build=lambda r: conditional.ConditionalGenerator(_conditional_generator(), rngs=r),
            inputs=lambda key: _labelled_data_and_latents(key)[1:],
            has_batch_stats=True,
        ),
        f"{_GAN}.conditional.ConditionalDiscriminator": Case(
            build=lambda r: conditional.ConditionalDiscriminator(
                _conditional_discriminator(), rngs=r
            ),
            inputs=lambda key: _labelled_data_and_latents(key)[::2],
            has_dropout=True,
        ),
        f"{_GAN}.conditional.ConditionalGAN": Case(
            build=lambda r: conditional.ConditionalGAN(
                _gan_config(
                    ConditionalGANConfig, _conditional_generator(), _conditional_discriminator()
                ),
                rngs=r,
            ),
            inputs=_labelled_data_and_latents,
            call=_conditioned_scores_and_samples,
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.cyclegan.CycleGANGenerator": Case(
            build=lambda r: cyclegan.CycleGANGenerator(_cycle_generator(), rngs=r),
            inputs=_normal(*_CYCLE_IMAGE),
            has_batch_stats=True,
        ),
        f"{_GAN}.cyclegan.CycleGANDiscriminator": Case(
            build=lambda r: cyclegan.CycleGANDiscriminator(
                _patch_discriminator(_CYCLE_IMAGE, dropout_rate=0.0), rngs=r
            ),
            inputs=_normal(*_CYCLE_IMAGE),
            has_batch_stats=True,
        ),
        f"{_GAN}.cyclegan.CycleGAN": Case(
            build=lambda r: cyclegan.CycleGAN(cycle_config, rngs=r),
            inputs=_arrays(_CYCLE_IMAGE, _CYCLE_IMAGE),
            call=_translations,
            has_batch_stats=True,
        ),
        f"{_GAN}.patchgan.PatchGANDiscriminator": Case(
            build=lambda r: patchgan.PatchGANDiscriminator(
                _patch_discriminator(_GAN_IMAGE), rngs=r
            ),
            inputs=_normal(*_GAN_IMAGE),
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.patchgan.MultiScalePatchGANDiscriminator": Case(
            build=lambda r: patchgan.MultiScalePatchGANDiscriminator(
                MultiScalePatchGANConfig(
                    discriminator=_patch_discriminator(_MULTISCALE_IMAGE), num_discriminators=2
                ),
                rngs=r,
            ),
            inputs=_normal(*_MULTISCALE_IMAGE),
            has_dropout=True,
            has_batch_stats=True,
        ),
        f"{_GAN}.stylegan3.MappingNetwork": Case(
            build=lambda r: stylegan3.MappingNetwork(
                _GAN_LATENT, _STYLE_DIM, num_layers=2, num_ws=2, rngs=r
            ),
            inputs=_normal(_GAN_LATENT),
        ),
        f"{_GAN}.stylegan3.StyleModulatedConv": Case(
            build=lambda r: stylegan3.StyleModulatedConv(
                style_channels, style_channels, 3, _STYLE_DIM, rngs=r
            ),
            inputs=_arrays((4, 4, style_channels), (_STYLE_DIM,)),
        ),
        f"{_GAN}.stylegan3.SynthesisBlock": Case(
            build=lambda r: stylegan3.SynthesisBlock(
                style_channels, style_channels, _STYLE_DIM, upsample=True, rngs=r
            ),
            inputs=_arrays((4, 4, style_channels), (_STYLE_DIM,), (_STYLE_DIM,)),
        ),
        f"{_GAN}.stylegan3.SynthesisNetwork": Case(
            build=lambda r: stylegan3.SynthesisNetwork(_STYLE_DIM, _STYLE_RESOLUTION, 3, rngs=r),
            inputs=_normal(2, _STYLE_DIM),
        ),
        f"{_GAN}.stylegan3.StyleGAN3Generator": Case(
            build=lambda r: stylegan3.StyleGAN3Generator(style_generator, rngs=r),
            inputs=_normal(_GAN_LATENT),
        ),
        f"{_GAN}.stylegan3.StyleGAN3Discriminator": Case(
            build=lambda r: stylegan3.StyleGAN3Discriminator(style_discriminator, rngs=r),
            inputs=_normal(*style_image),
        ),
    }


_DIFFUSION = "artifex.generative_models.models.diffusion"
_STEPS, _SIGNAL = 100, (16, 1)
_TOKENS, _VOCAB, _TEXT_DIM = 5, 20, 12
_SD_IMAGE = (16, 16, 3)


def _noise_schedule() -> object:
    from artifex.generative_models.core.configuration.diffusion_config import (  # noqa: PLC0415
        NoiseScheduleConfig,
    )

    return NoiseScheduleConfig(name="schedule", num_timesteps=_STEPS)


def _diffusion_config(config_class: type, **fields: object) -> object:
    return config_class(
        name="diffusion",
        backbone=_unet1d_config(),
        noise_schedule=_noise_schedule(),
        input_shape=_SIGNAL,
        **fields,
    )


def _signal_and_steps(key: jax.Array) -> tuple[jax.Array, ...]:
    """Noisy signals and integer timesteps that index the schedule."""
    k1, k2 = jax.random.split(key)
    return jax.random.normal(k1, (BATCH, *_SIGNAL)), jax.random.randint(k2, (BATCH,), 0, _STEPS)


def _denoise_and_posterior(module: nnx.Module, x: jax.Array, t: jax.Array) -> PyTree:
    """The denoiser's prediction, then the x0 estimate and posterior the reverse step uses."""
    output = module(x, t)
    x_start = module.predict_start_from_noise(x, t, output["predicted_noise"])
    return output, module.q_posterior_mean_variance(x_start, x, t)


def _tokens_and_mask(key: jax.Array) -> tuple[jax.Array, ...]:
    """Token ids with the last position of every row padded."""
    ids = jax.random.randint(key, (BATCH, _TOKENS), 0, _VOCAB)
    return ids, jnp.ones((BATCH, _TOKENS)).at[:, -1].set(0.0)


def _stable_diffusion_config() -> object:
    from artifex.generative_models.core.configuration.backbone_config import (  # noqa: PLC0415
        UNet2DConditionBackboneConfig,
    )
    from artifex.generative_models.core.configuration.diffusion_config import (  # noqa: PLC0415
        StableDiffusionConfig,
    )

    latent_channels = 2
    return StableDiffusionConfig(
        name="stable_diffusion",
        backbone=UNet2DConditionBackboneConfig(
            name="unet",
            hidden_dims=(8, 16),
            in_channels=latent_channels,
            out_channels=latent_channels,
            cross_attention_dim=_TEXT_DIM,
            num_heads=2,
            num_res_blocks=1,
            attention_levels=(1,),
            time_embedding_dim=16,
        ),
        noise_schedule=_noise_schedule(),
        encoder=_encoder_config(_SD_IMAGE, latent=latent_channels),
        decoder=_decoder_config(_SD_IMAGE, latent=latent_channels),
        input_shape=_SD_IMAGE,
        text_embedding_dim=_TEXT_DIM,
        text_max_length=_TOKENS,
        vocab_size=_VOCAB,
    )


def _image_latents_and_steps(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2 = jax.random.split(key)
    return jax.random.normal(k1, (BATCH, *_SD_IMAGE)), jax.random.randint(k2, (BATCH,), 0, _STEPS)


def _pipeline_round_trip(module: nnx.Module, images: jax.Array, t: jax.Array) -> PyTree:
    """Encode images to latents, predict their noise, and decode them."""
    latents = module.encode_image(images)
    return latents, module.model(latents, t), module.decode_latents(latents)


def _diffusion() -> dict[str, Case]:
    from artifex.generative_models.core.configuration.diffusion_config import (  # noqa: PLC0415
        DDIMConfig,
        DDPMConfig,
        DiffusionConfig,
        DiTConfig,
        LatentDiffusionConfig,
        ScoreDiffusionConfig,
    )
    from artifex.generative_models.modalities.audio.base import (  # noqa: PLC0415
        AudioModalityConfig,
    )
    from artifex.generative_models.models.audio import diffusion as audio  # noqa: PLC0415
    from artifex.generative_models.models.diffusion import (  # noqa: PLC0415
        base,
        clip_text_encoder,
        ddim,
        ddpm,
        dit,
        guidance,
        latent,
        score,
        stable_diffusion,
        stable_diffusion_pipeline,
    )

    latent_features, latent_dim = 5, 16
    ldm_config = LatentDiffusionConfig(
        name="ldm",
        backbone=_unet1d_config(),
        noise_schedule=_noise_schedule(),
        encoder=_encoder_config((latent_features,), latent=latent_dim),
        decoder=_decoder_config((latent_features,), latent=latent_dim),
        input_shape=(latent_features,),
    )
    dit_config = DiTConfig(
        name="dit",
        noise_schedule=_noise_schedule(),
        input_shape=(3, 8, 8),
        patch_size=2,
        hidden_size=32,
        depth=2,
        num_heads=2,
        num_classes=4,
    )
    audio_config = audio.create_audio_diffusion_config(
        modality_config=AudioModalityConfig(sample_rate=16, duration=1.0),
        num_timesteps=_STEPS,
        unet_channels=16,
    )
    simple_image = (4, 4, 1)
    return {
        f"{_DIFFUSION}.base.DiffusionModel": Case(
            build=lambda r: base.DiffusionModel(_diffusion_config(DiffusionConfig), rngs=r),
            inputs=_signal_and_steps,
            call=_denoise_and_posterior,
        ),
        f"{_DIFFUSION}.ddpm.DDPMModel": Case(
            build=lambda r: ddpm.DDPMModel(_diffusion_config(DDPMConfig), rngs=r),
            inputs=_signal_and_steps,
            call=_denoise_and_posterior,
        ),
        f"{_DIFFUSION}.ddim.DDIMModel": Case(
            build=lambda r: ddim.DDIMModel(_diffusion_config(DDIMConfig), rngs=r),
            inputs=_signal_and_steps,
            call=_denoise_and_posterior,
        ),
        f"{_DIFFUSION}.guidance.GuidedDiffusionModel": Case(
            build=lambda r: guidance.GuidedDiffusionModel(
                _diffusion_config(DiffusionConfig),
                rngs=r,
                guidance_method="classifier_free",
                guidance_scale=3.0,
            ),
            inputs=_signal_and_steps,
            call=lambda module, x, t: module.guided_sample_step(x, t),
        ),
        f"{_DIFFUSION}.score.ScoreDiffusionModel": Case(
            build=lambda r: score.ScoreDiffusionModel(
                _diffusion_config(ScoreDiffusionConfig), rngs=r
            ),
            inputs=lambda key: (
                jax.random.normal(key, (BATCH, *_SIGNAL)),
                jax.random.uniform(jax.random.fold_in(key, 1), (BATCH,), minval=0.1),
            ),
            call=lambda module, x, t: module.score(x, t),
        ),
        f"{_DIFFUSION}.latent.SimpleEncoder": Case(
            build=lambda r: latent.SimpleEncoder(simple_image, 3, [8, 6], rngs=r),
            inputs=_normal(*simple_image),
        ),
        f"{_DIFFUSION}.latent.SimpleDecoder": Case(
            build=lambda r: latent.SimpleDecoder(3, simple_image, [6, 8], rngs=r),
            inputs=_normal(3),
        ),
        f"{_DIFFUSION}.latent.LDMModel": Case(
            build=lambda r: latent.LDMModel(ldm_config, rngs=r),
            inputs=lambda key: (
                jax.random.normal(key, (BATCH, latent_features)),
                jax.random.randint(jax.random.fold_in(key, 1), (BATCH,), 0, _STEPS),
            ),
            has_batch_stats=True,
        ),
        f"{_DIFFUSION}.dit.DiTModel": Case(
            build=lambda r: dit.DiTModel(dit_config, rngs=r),
            inputs=_dit_inputs,
        ),
        f"{_DIFFUSION}.clip_text_encoder.CLIPTextEncoder": Case(
            build=lambda r: clip_text_encoder.CLIPTextEncoder(
                _VOCAB, _TOKENS, 16, 2, 2, dropout_rate=0.5, rngs=r
            ),
            inputs=_tokens_and_mask,
            call=lambda module, ids, mask: module(ids, attention_mask=mask),
            has_dropout=True,
        ),
        f"{_DIFFUSION}.stable_diffusion.StableDiffusionModel": Case(
            build=lambda r: stable_diffusion.StableDiffusionModel(
                _stable_diffusion_config(), rngs=r
            ),
            inputs=lambda key: (
                jax.random.normal(key, (BATCH, 4, 4, 2)),
                jax.random.randint(jax.random.fold_in(key, 1), (BATCH,), 0, _STEPS),
            ),
        ),
        f"{_DIFFUSION}.stable_diffusion_pipeline.StableDiffusionPipeline": Case(
            build=lambda r: stable_diffusion_pipeline.StableDiffusionPipeline(
                _stable_diffusion_config(), rngs=r
            ),
            inputs=_image_latents_and_steps,
            call=_pipeline_round_trip,
        ),
        "artifex.generative_models.models.audio.diffusion.AudioDiffusionModel": Case(
            build=lambda r: audio.AudioDiffusionModel(audio_config, rngs=r),
            inputs=_signal_and_steps,
            call=_denoise_and_posterior,
        ),
    }


def all_cases() -> dict[str, Case]:
    """Return every case, by the qualified name of the class it builds."""
    return {**_layers(), **_base(), **_backbones(), **_vae(), **_flow(), **_gan(), **_diffusion()}


EXEMPT: dict[str, str] = {
    f"{_CORE}.base.GenerativeModule": "base class; each subclass's case runs its behaviour",
    f"{_CORE}.base.GenerativeModel": "base class; each subclass's case runs its behaviour",
    f"{_CORE}.base._DepthwiseSeparableConv": "private part of CNN, run through CNN's case",
    f"{_LAYERS}.clifford.conv._CliffordConvNd": "private base of the 1-D, 2-D and 3-D convolutions",
    f"{_LAYERS}.positional.PositionalEncoding": "base class whose __call__ raises NotImplementedError",
    f"{_LAYERS}.kan.grids.DenseKANGrid": "holds a KAN layer's knots; run through DenseKANLayer",
    f"{_LAYERS}.kan.grids.EfficientKANGrid": "holds a KAN layer's knots; run through EfficientKANLayer",
    f"{_LAYERS}.kan.grids.RBFKANGrid": "holds a KAN layer's centres; run through RBFKANLayer",
    f"{_FLOW}.base.FlowLayer": "base class whose forward and inverse raise NotImplementedError",
    f"{_FLOW}.base.NormalizingFlow": (
        "base class holding an empty layer list; each flow model's case runs its forward,"
        " inverse and log-density"
    ),
}


NOT_APPLICABLE: dict[tuple[str, str], str] = {
    (f"{_VAE}.vq_vae.VQVAE", "gradient_matches_finite_differences"): (
        "the straight-through estimator (van den Oord et al. 2017, sec. 3.2; vq_vae.py:106) and"
        " the stop-gradient codebook and commitment terms (vq_vae.py:88-89) are surrogate"
        " gradients by design, not derivatives of the forward pass: in float64 the decoder's"
        " gradient matches finite differences to 9.6e-11, while the encoder's and codebook's,"
        " which pass through the quantizer, differ by 1.75 and 0.155"
    ),
}
"""Checks a case's module does not owe, keyed like the known failures, each with the reason.

A check belongs here when the method the module implements makes the property false by design.
A defect belongs in ``known_failures.py``, which a fix empties.
"""
