"""Cases for the VAE family, ``artifex.generative_models.models.vae``."""

from __future__ import annotations

import jax
from substrax.testing.nnx_conformance import Case

from tests.artifex.nnx_conformance.cases.inputs import BATCH, labels, normal


_VAE = "artifex.generative_models.models.vae"
_VAE_FEATURES, _VAE_LATENT, _VAE_CLASSES = 5, 3, 3
IMAGE = (8, 8, 3)


def encoder_config(
    input_shape: tuple[int, ...] = (_VAE_FEATURES,), latent: int = _VAE_LATENT
) -> object:
    """An MLP encoder configuration (hidden widths 8 and 6, silu) for ``input_shape``."""
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


def decoder_config(
    output_shape: tuple[int, ...] = (_VAE_FEATURES,), latent: int = _VAE_LATENT
) -> object:
    """An MLP decoder configuration (hidden widths 6 and 8, silu) for ``output_shape``."""
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
    return config_class(name="vae", encoder=encoder_config(), decoder=decoder_config(), **fields)


def _features_and_labels(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2 = jax.random.split(key)
    return jax.random.normal(k1, (BATCH, _VAE_FEATURES)), labels(k2, _VAE_CLASSES)


def _latents_and_labels(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2 = jax.random.split(key)
    return jax.random.normal(k1, (BATCH, _VAE_LATENT)), labels(k2, _VAE_CLASSES)


def cases() -> dict[str, Case]:
    """Return the VAE cases, by the qualified name of the class each builds."""

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
            inputs=normal(3, 4),
            is_differentiable=False,
        ),
        f"{_VAE}.encoders.MLPEncoder": Case(
            build=lambda r: encoders.MLPEncoder(encoder_config(), rngs=r),
            inputs=normal(_VAE_FEATURES),
            has_batch_stats=True,
        ),
        f"{_VAE}.encoders.CNNEncoder": Case(
            build=lambda r: encoders.CNNEncoder(encoder_config(IMAGE), rngs=r),
            inputs=normal(*IMAGE),
            has_batch_stats=True,
        ),
        f"{_VAE}.encoders.ResNetEncoder": Case(
            build=lambda r: encoders.ResNetEncoder(encoder_config(IMAGE), rngs=r),
            inputs=normal(*IMAGE),
            has_batch_stats=True,
        ),
        f"{_VAE}.encoders.ConditionalEncoder": Case(
            build=lambda r: encoders.ConditionalEncoder(
                encoders.MLPEncoder(encoder_config(conditioned_encoder), rngs=r),
                _VAE_CLASSES,
                rngs=r,
            ),
            inputs=_features_and_labels,
            has_batch_stats=True,
        ),
        f"{_VAE}.decoders.MLPDecoder": Case(
            build=lambda r: decoders.MLPDecoder(decoder_config(), rngs=r),
            inputs=normal(_VAE_LATENT),
        ),
        f"{_VAE}.decoders.CNNDecoder": Case(
            build=lambda r: decoders.CNNDecoder(decoder_config(IMAGE), rngs=r),
            inputs=normal(_VAE_LATENT),
        ),
        f"{_VAE}.decoders.ResNetDecoder": Case(
            build=lambda r: decoders.ResNetDecoder(decoder_config(IMAGE), rngs=r),
            inputs=normal(_VAE_LATENT),
        ),
        f"{_VAE}.decoders.ConditionalDecoder": Case(
            build=lambda r: decoders.ConditionalDecoder(
                decoders.MLPDecoder(decoder_config(latent=_VAE_LATENT + _VAE_CLASSES), rngs=r),
                _VAE_CLASSES,
                rngs=r,
            ),
            inputs=_latents_and_labels,
        ),
        f"{_VAE}.spatial_autoencoder.SpatialEncoder": Case(
            build=lambda r: spatial.SpatialEncoder(encoder_config(IMAGE, latent=2), rngs=r),
            inputs=normal(*IMAGE),
        ),
        f"{_VAE}.spatial_autoencoder.SpatialDecoder": Case(
            build=lambda r: spatial.SpatialDecoder(decoder_config(IMAGE, latent=2), rngs=r),
            inputs=normal(2, 2, 2),
        ),
        f"{_VAE}.base.VAE": Case(
            build=lambda r: base.VAE(_vae_config(VAEConfig), rngs=r),
            inputs=normal(_VAE_FEATURES),
            has_batch_stats=True,
        ),
        f"{_VAE}.beta_vae.BetaVAE": Case(
            build=lambda r: beta_vae.BetaVAE(_vae_config(BetaVAEConfig, beta_default=2.0), rngs=r),
            inputs=normal(_VAE_FEATURES),
            has_batch_stats=True,
        ),
        f"{_VAE}.beta_vae.BetaVAEWithCapacity": Case(
            build=lambda r: beta_vae.BetaVAEWithCapacity(
                _vae_config(BetaVAEWithCapacityConfig, use_capacity_control=True), rngs=r
            ),
            inputs=normal(_VAE_FEATURES),
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
            inputs=normal(_VAE_FEATURES),
            has_batch_stats=True,
            # Its outputs include batch means: the codebook and commitment losses, perplexity.
            is_rowwise=False,
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
