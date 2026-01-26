"""Audio generation models for generative framework.

This module provides audio-specific model implementations including:
- WaveNet for autoregressive audio generation
- Audio diffusion models for parallel generation
- Shared utilities for audio model development

Example:
    >>> from artifex.generative_models.models.audio import (
    ...     WaveNetAudioModel,
    ...     AudioDiffusionModel,
    ...     create_audio_diffusion_config,
    ... )
    >>> wavenet = WaveNetAudioModel(config=config, rngs=rngs)
    >>> config = create_audio_diffusion_config(modality_config, unet_channels=32)
    >>> diffusion = AudioDiffusionModel(config=config, rngs=rngs)
"""

from .base import (
    AudioModelConfig,
    BaseAudioModel,
)
from .diffusion import (
    AudioDiffusionConfig,
    AudioDiffusionModel,
    create_audio_diffusion_config,
)
from .wavenet import (
    WaveNetAudioModel,
    WaveNetConfig,
)


__all__ = [
    # Base classes
    "AudioModelConfig",
    "BaseAudioModel",
    # WaveNet
    "WaveNetConfig",
    "WaveNetAudioModel",
    # Diffusion
    "AudioDiffusionConfig",
    "AudioDiffusionModel",
    "create_audio_diffusion_config",
]
