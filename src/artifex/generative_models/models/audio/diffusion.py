"""Audio diffusion model for parallel audio generation.

Uses DiffusionConfig with UNet1DBackboneConfig for 1D audio signals.
Follows the (config, *, rngs) signature pattern.
"""

import dataclasses
from typing import Any

import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import (
    DiffusionConfig,
    NoiseScheduleConfig,
)
from artifex.generative_models.core.configuration.backbone_config import (
    UNet1DBackboneConfig,
)
from artifex.generative_models.modalities.audio.base import AudioModalityConfig
from artifex.generative_models.models.diffusion.base import DiffusionModel


@dataclasses.dataclass(frozen=True)
class AudioDiffusionConfig(DiffusionConfig):
    """Configuration for audio diffusion model.

    Extends DiffusionConfig with audio-specific parameters.
    Uses UNet1DBackboneConfig for the backbone.

    Attributes:
        name: Name of the configuration
        backbone: UNet1DBackboneConfig for 1D audio
        noise_schedule: NoiseScheduleConfig for the diffusion process
        input_shape: Shape of input audio (sequence_length,)
        modality_config: Audio modality configuration
    """

    # Audio-specific nested config
    modality_config: AudioModalityConfig = dataclasses.field(default=None)  # type: ignore

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        super().__post_init__()

        if self.modality_config is None:
            raise ValueError("modality_config is required")
        if not isinstance(self.modality_config, AudioModalityConfig):
            raise TypeError(
                f"modality_config must be AudioModalityConfig, "
                f"got {type(self.modality_config).__name__}"
            )


def create_audio_diffusion_config(
    modality_config: AudioModalityConfig,
    *,
    num_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    unet_channels: int = 128,
    name: str = "audio_diffusion",
) -> AudioDiffusionConfig:
    """Create AudioDiffusionConfig with proper nested configs.

    This factory function creates the complete nested config structure
    for audio diffusion models.

    Args:
        modality_config: Audio modality configuration
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting beta value for noise schedule
        beta_end: Ending beta value for noise schedule
        unet_channels: Base number of UNet channels (multiplied for layers)
        name: Name of the configuration

    Returns:
        AudioDiffusionConfig with proper backbone and noise schedule
    """
    # Calculate audio sequence length from modality config
    sequence_length = int(modality_config.sample_rate * modality_config.duration)

    # Create backbone config - scale channels based on unet_channels
    base_channels = unet_channels // 4
    hidden_dims = (
        base_channels,
        base_channels * 2,
        base_channels * 4,
        base_channels * 8,
    )

    backbone = UNet1DBackboneConfig(
        name=f"{name}_backbone",
        hidden_dims=hidden_dims,
        activation="gelu",
        in_channels=1,  # Mono audio
        time_embedding_dim=unet_channels,
    )

    # Create noise schedule config
    noise_schedule = NoiseScheduleConfig(
        name=f"{name}_schedule",
        schedule_type="linear",
        num_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
    )

    return AudioDiffusionConfig(
        name=name,
        backbone=backbone,
        noise_schedule=noise_schedule,
        input_shape=(sequence_length,),
        modality_config=modality_config,
    )


class AudioDiffusionModel(DiffusionModel):
    """Audio diffusion model for parallel generation.

    Uses UNet1D backbone via DiffusionConfig's polymorphic backbone system.
    Follows the (config, *, rngs) signature pattern.

    Example:
        config = create_audio_diffusion_config(
            modality_config=audio_modality_config,
            num_timesteps=100,
            unet_channels=32,
        )
        model = AudioDiffusionModel(config, rngs=rngs)
    """

    def __init__(self, config: AudioDiffusionConfig, *, rngs: nnx.Rngs):
        """Initialize audio diffusion model.

        Args:
            config: AudioDiffusionConfig with UNet1D backbone and audio modality config
            rngs: Random number generators
        """
        # DiffusionModel.__init__ handles backbone creation via create_backbone factory
        super().__init__(config, rngs=rngs)

        # Store audio-specific references
        self.audio_config = config
        self.modality_config = config.modality_config
        self.sample_rate = config.modality_config.sample_rate

    def __call__(
        self,
        x: jnp.ndarray,
        timesteps: jnp.ndarray,
        *,
        conditioning: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Forward pass through the audio diffusion model.

        Args:
            x: Input audio of shape (batch, length) or (batch, length, channels)
            timesteps: Diffusion timesteps
            conditioning: Optional conditioning information (ignored for basic audio)
            **kwargs: Additional arguments passed to the backbone

        Returns:
            Dictionary containing model outputs including predicted_noise
        """
        # Parent's __call__ handles backbone invocation
        return super().__call__(x, timesteps, conditioning=conditioning, **kwargs)

    def generate(
        self,
        n_samples: int = 1,
        duration: float | None = None,
        *,
        clip_denoised: bool = True,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Generate audio samples.

        Args:
            n_samples: Number of samples to generate
            duration: Duration in seconds (uses modality config default if None)
            clip_denoised: Whether to clip the denoised signal to [-1, 1]
            **kwargs: Additional arguments

        Returns:
            Generated audio waveforms of shape (n_samples, sequence_length)
        """
        # Determine audio shape
        if duration is None:
            duration = self.modality_config.duration

        sequence_length = int(duration * self.sample_rate)
        shape = (sequence_length,)  # 1D audio

        # Generate using parent class
        audio = super().generate(n_samples, shape=shape, clip_denoised=clip_denoised)

        return self.postprocess_audio(audio)

    def preprocess_audio(self, audio: jnp.ndarray) -> jnp.ndarray:
        """Preprocess audio for model input.

        Args:
            audio: Raw audio tensor

        Returns:
            Preprocessed audio normalized to [-1, 1]
        """
        # Normalize to [-1, 1] range
        if self.modality_config.normalize:
            audio_max = jnp.max(jnp.abs(audio))
            audio = jnp.where(audio_max > 0, audio / audio_max, audio)

        return audio

    def postprocess_audio(self, audio: jnp.ndarray) -> jnp.ndarray:
        """Postprocess audio from model output.

        Args:
            audio: Model output audio

        Returns:
            Postprocessed audio clipped to valid range
        """
        return jnp.clip(audio, -1.0, 1.0)
