"""Cases for the autoregressive family, ``artifex.generative_models.models.autoregressive``."""

from __future__ import annotations

import jax
from substrax.testing.nnx_conformance import Case

from tests.artifex.nnx_conformance.cases.inputs import BATCH, normal


_AUTOREGRESSIVE = "artifex.generative_models.models.autoregressive"
_VOCAB, _LENGTH, _CHANNELS = 11, 8, 4
_IMAGE = (4, 4, 2)


def _tokens(key: jax.Array) -> tuple[jax.Array, ...]:
    return (jax.random.randint(key, (BATCH, _LENGTH), 0, _VOCAB),)


def _pixels(key: jax.Array) -> tuple[jax.Array, ...]:
    """8-bit pixel values, the integers PixelCNN models."""
    return (jax.random.randint(key, (BATCH, *_IMAGE), 0, 256),)


def cases() -> dict[str, Case]:
    """Return the autoregressive cases, by the qualified name of the class each builds."""

    from artifex.generative_models.core.configuration.autoregressive_config import (  # noqa: PLC0415
        PixelCNNConfig,
        TransformerConfig,
        TransformerNetworkConfig,
        WaveNetConfig,
    )
    from artifex.generative_models.models.autoregressive import (  # noqa: PLC0415
        pixel_cnn,
        transformer,
        wavenet,
    )

    transformer_config = TransformerConfig(
        name="transformer",
        vocab_size=_VOCAB,
        sequence_length=_LENGTH,
        dropout_rate=0.5,
        num_layers=2,
        network=TransformerNetworkConfig(
            name="network", hidden_dims=(16,), activation="silu", embed_dim=16, num_heads=2
        ),
    )
    wavenet_config = WaveNetConfig(
        name="wavenet",
        vocab_size=_VOCAB,
        sequence_length=_LENGTH,
        residual_channels=_CHANNELS,
        skip_channels=6,
        num_blocks=1,
        layers_per_block=2,
    )
    pixel_config = PixelCNNConfig(
        name="pixel_cnn",
        image_shape=_IMAGE,
        hidden_channels=_CHANNELS,
        num_layers=2,
        num_residual_blocks=1,
    )
    image = (4, 4, _CHANNELS)
    return {
        f"{_AUTOREGRESSIVE}.pixel_cnn.MaskedConv2D": Case(
            build=lambda r: pixel_cnn.MaskedConv2D(_CHANNELS, 3, mask_type="A", rngs=r),
            inputs=normal(*image),
        ),
        f"{_AUTOREGRESSIVE}.pixel_cnn.PixelCNNResidualBlock": Case(
            build=lambda r: pixel_cnn.PixelCNNResidualBlock(_CHANNELS, rngs=r),
            inputs=normal(*image),
        ),
        f"{_AUTOREGRESSIVE}.pixel_cnn.PixelCNN": Case(
            build=lambda r: pixel_cnn.PixelCNN(pixel_config, rngs=r),
            inputs=_pixels,
        ),
        f"{_AUTOREGRESSIVE}.transformer.TransformerAutoregressiveModel": Case(
            build=lambda r: transformer.TransformerAutoregressiveModel(transformer_config, rngs=r),
            inputs=_tokens,
            has_dropout=True,
            rng_streams=("dropout",),
        ),
        f"{_AUTOREGRESSIVE}.wavenet.CausalConv1D": Case(
            build=lambda r: wavenet.CausalConv1D(_CHANNELS, 3, kernel_size=2, dilation=2, rngs=r),
            inputs=normal(_LENGTH, _CHANNELS),
        ),
        f"{_AUTOREGRESSIVE}.wavenet.GatedActivationUnit": Case(
            build=lambda r: wavenet.GatedActivationUnit(_CHANNELS, 2, dilation=2, rngs=r),
            inputs=normal(_LENGTH, _CHANNELS),
        ),
        f"{_AUTOREGRESSIVE}.wavenet.WaveNet": Case(
            build=lambda r: wavenet.WaveNet(wavenet_config, rngs=r),
            inputs=_tokens,
        ),
    }


EXEMPT: dict[str, str] = {
    f"{_AUTOREGRESSIVE}.base.AutoregressiveModel": (
        "base class whose __call__ raises NotImplementedError; each subclass's case runs it"
    ),
}
