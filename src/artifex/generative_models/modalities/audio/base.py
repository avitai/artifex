"""Base classes and protocols for audio modality.

This module defines the core interfaces and base classes for audio generation,
following the established modality patterns in the framework.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel


class AudioRepresentation(Enum):
    """Audio representation formats supported by the modality."""

    RAW_WAVEFORM = "raw_waveform"
    MEL_SPECTROGRAM = "mel_spectrogram"
    STFT = "stft"


@dataclass
class AudioModalityConfig:
    """Configuration for audio modality processing.

    Args:
        representation: Audio representation format to use
        sample_rate: Audio sample rate in Hz
        n_mel_channels: Number of mel-spectrogram channels (for MEL_SPECTROGRAM)
        hop_length: Hop length for STFT/mel-spectrogram
        n_fft: FFT size for spectral representations
        duration: Default audio duration in seconds
        normalize: Whether to normalize audio values
    """

    representation: AudioRepresentation = AudioRepresentation.RAW_WAVEFORM
    sample_rate: int = 16000
    n_mel_channels: int = 80
    hop_length: int = 256
    n_fft: int = 1024
    duration: float = 2.0
    normalize: bool = True


class AudioGenerationProtocol(Protocol):
    """Protocol for audio generation models."""

    def generate_audio(
        self,
        n_samples: int = 1,
        duration: float = 2.0,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> jnp.ndarray:
        """Generate audio samples.

        Args:
            n_samples: Number of audio samples to generate
            duration: Duration of each audio sample in seconds
            rngs: Random number generators
            **kwargs: Additional generation parameters

        Returns:
            Generated audio array of shape (n_samples, n_time_steps) or
            (n_samples, n_mel_channels, n_time_frames) for spectrograms
        """
        ...

    def compute_likelihood(self, audio: jnp.ndarray) -> jax.Array:
        """Compute likelihood of audio samples.

        Args:
            audio: Audio data to evaluate

        Returns:
            Log-likelihood value
        """
        ...


class AudioModality(GenerativeModel):
    """Base audio modality class providing unified interface for audio generation.

    This class provides a unified interface for different audio generation approaches
    while supporting multiple representation formats.
    """

    def __init__(
        self,
        config: AudioModalityConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize audio modality.

        Args:
            config: Audio modality configuration
            rngs: Random number generators
        """
        super().__init__(rngs=rngs)
        self.config = config or AudioModalityConfig()

        # Calculate derived parameters
        self._n_time_steps = int(self.config.sample_rate * self.config.duration)
        self._n_time_frames = self._n_time_steps // self.config.hop_length

    @property
    def n_time_steps(self) -> int:
        """Number of time steps for raw waveform."""
        return self._n_time_steps

    @property
    def n_time_frames(self) -> int:
        """Number of time frames for spectral representations."""
        return self._n_time_frames

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Output shape for generated audio."""
        if self.config.representation == AudioRepresentation.RAW_WAVEFORM:
            return (self.n_time_steps,)
        elif self.config.representation == AudioRepresentation.MEL_SPECTROGRAM:
            return (self.config.n_mel_channels, self.n_time_frames)
        else:  # STFT
            return (self.config.n_fft // 2 + 1, self.n_time_frames)

    def generate(
        self,
        n_samples: int = 1,
        duration: float | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> jnp.ndarray:
        """Generate audio samples using the configured model.

        Args:
            n_samples: Number of audio samples to generate
            duration: Duration override (uses config default if None)
            rngs: Random number generators
            **kwargs: Additional generation parameters

        Returns:
            Generated audio array
        """
        # Default implementation - subclasses should override
        duration = duration or self.config.duration
        n_steps = int(self.config.sample_rate * duration)

        # Generate simple synthetic audio for base implementation
        if rngs is None:
            raise ValueError("rngs must be provided for sample generation")

        key = rngs.sample()

        if self.config.representation == AudioRepresentation.RAW_WAVEFORM:
            # Generate simple sine wave + noise
            t = jnp.linspace(0, duration, n_steps)
            freq = 440.0  # A4 note
            audio = 0.3 * jnp.sin(2 * jnp.pi * freq * t)
            noise = 0.1 * jax.random.normal(key, (n_samples, n_steps))
            return audio[None, :] + noise
        else:
            # Generate random spectral data for spectrograms
            shape = (n_samples, *self.output_shape)
            return jax.random.normal(key, shape)

    def loss_fn(
        self, batch: dict[str, jnp.ndarray], model_outputs: dict[str, jnp.ndarray], **kwargs
    ) -> jax.Array:
        """Compute loss for audio generation training.

        Args:
            batch: Training batch containing 'audio' key
            model_outputs: Model predictions
            **kwargs: Additional loss parameters

        Returns:
            Loss value
        """
        # Default MSE loss - subclasses should override for specific losses
        target_audio = batch["audio"]
        predicted_audio = model_outputs.get("audio", model_outputs.get("predictions"))

        if predicted_audio is None:
            raise ValueError("Model outputs must contain 'audio' or 'predictions' key")

        return jnp.mean((target_audio - predicted_audio) ** 2)


def create_audio_modality(
    representation: AudioRepresentation | str = AudioRepresentation.RAW_WAVEFORM,
    sample_rate: int = 16000,
    duration: float = 2.0,
    *,
    rngs: nnx.Rngs,
    **kwargs,
) -> AudioModality:
    """Factory function to create audio modality with common configurations.

    Args:
        representation: Audio representation format
        sample_rate: Audio sample rate in Hz
        duration: Audio duration in seconds
        rngs: Random number generators
        **kwargs: Additional configuration parameters

    Returns:
        Configured AudioModality instance
    """
    if isinstance(representation, str):
        representation = AudioRepresentation(representation)

    config = AudioModalityConfig(
        representation=representation, sample_rate=sample_rate, duration=duration, **kwargs
    )

    return AudioModality(config=config, rngs=rngs)
