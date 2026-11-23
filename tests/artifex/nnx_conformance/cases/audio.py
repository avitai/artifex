"""Cases for the audio models, ``artifex.generative_models.models.audio``.

``AudioDiffusionModel`` is a diffusion model, so its case is in ``diffusion.py``.
"""

from __future__ import annotations

import jax
from substrax.testing.nnx_conformance import Case

from tests.artifex.nnx_conformance.cases.inputs import BATCH, normal


_AUDIO = "artifex.generative_models.models.audio"
_CHANNELS, _STEPS, _LEVELS = 4, 8, 16


def _quantized(key: jax.Array) -> tuple[jax.Array, ...]:
    """mu-law quantized audio: integer levels, the input WaveNet embeds."""
    return (jax.random.randint(key, (BATCH, _STEPS), 0, _LEVELS),)


def cases() -> dict[str, Case]:
    """Return the audio cases, by the qualified name of the class each builds."""

    from artifex.generative_models.modalities.audio.base import (  # noqa: PLC0415
        AudioModalityConfig,
    )
    from artifex.generative_models.models.audio import wavenet  # noqa: PLC0415

    config = wavenet.WaveNetConfig(
        modality_config=AudioModalityConfig(sample_rate=16, duration=1.0),
        n_dilated_blocks=2,
        n_residual_channels=_CHANNELS,
        n_skip_channels=6,
        n_end_channels=8,
        quantization_levels=_LEVELS,
    )
    return {
        f"{_AUDIO}.wavenet.DilatedConv1D": Case(
            build=lambda r: wavenet.DilatedConv1D(_CHANNELS, 3, 2, 2, rngs=r),
            inputs=normal(_STEPS, _CHANNELS),
        ),
        f"{_AUDIO}.wavenet.ResidualBlock": Case(
            build=lambda r: wavenet.ResidualBlock(_CHANNELS, 6, 2, 2, rngs=r),
            inputs=normal(_STEPS, _CHANNELS),
        ),
        f"{_AUDIO}.wavenet.WaveNetAudioModel": Case(
            build=lambda r: wavenet.WaveNetAudioModel(config, rngs=r),
            inputs=_quantized,
        ),
    }


EXEMPT: dict[str, str] = {
    f"{_AUDIO}.base.BaseAudioModel": (
        "base class whose __call__ is GenerativeModel's, which raises NotImplementedError"
        " (core/base.py:202); WaveNetAudioModel's case runs it"
    ),
}
