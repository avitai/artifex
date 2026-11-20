"""Cases for the diffusion family, ``artifex.generative_models.models.diffusion``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from substrax.testing.nnx_conformance import Case
from substrax.typing import PyTree

from tests.artifex.nnx_conformance.cases.backbones import dit_inputs, unet1d_config
from tests.artifex.nnx_conformance.cases.inputs import BATCH, normal
from tests.artifex.nnx_conformance.cases.vae import decoder_config, encoder_config


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
        backbone=unet1d_config(),
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
        encoder=encoder_config(_SD_IMAGE, latent=latent_channels),
        decoder=decoder_config(_SD_IMAGE, latent=latent_channels),
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


def cases() -> dict[str, Case]:
    """Return the diffusion cases, by the qualified name of the class each builds."""

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
        backbone=unet1d_config(),
        noise_schedule=_noise_schedule(),
        encoder=encoder_config((latent_features,), latent=latent_dim),
        decoder=decoder_config((latent_features,), latent=latent_dim),
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
            inputs=normal(*simple_image),
        ),
        f"{_DIFFUSION}.latent.SimpleDecoder": Case(
            build=lambda r: latent.SimpleDecoder(3, simple_image, [6, 8], rngs=r),
            inputs=normal(3),
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
            inputs=dit_inputs,
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
