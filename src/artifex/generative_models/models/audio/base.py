"""Base classes for audio generation models."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel
from artifex.generative_models.modalities.audio.base import (
    AudioModalityConfig,
    AudioRepresentation,
)


@dataclass
class AudioModelConfig:
    """Base configuration for audio models.

    Args:
        modality_config: Audio modality configuration
        hidden_dims: Hidden layer dimensions
        num_layers: Number of model layers
        dropout_rate: Dropout rate for regularization
        activation: Activation function name
    """

    modality_config: AudioModalityConfig
    hidden_dims: int = 512
    num_layers: int = 12
    dropout_rate: float = 0.1
    activation: str = "gelu"


class BaseAudioModel(GenerativeModel):
    """Base class for audio generation models.

    Provides common functionality for audio models including:
    - Audio modality configuration handling
    - Standard generation interface
    - Loss computation utilities
    """

    def __init__(
        self,
        config: AudioModelConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize base audio model.

        Args:
            config: Audio model configuration
            rngs: Random number generators
        """
        super().__init__(rngs=rngs)
        self.config = config
        self.modality_config = config.modality_config

        # Calculate audio dimensions
        self.sample_rate = self.modality_config.sample_rate
        self.n_time_steps = int(self.modality_config.sample_rate * self.modality_config.duration)

        # Set output dimension based on representation
        if self.modality_config.representation == AudioRepresentation.RAW_WAVEFORM:
            self.output_dim = 1  # Single channel waveform
        elif self.modality_config.representation == AudioRepresentation.MEL_SPECTROGRAM:
            self.output_dim = self.modality_config.n_mel_channels
        else:  # STFT
            self.output_dim = self.modality_config.n_fft // 2 + 1

    def get_output_shape(self, batch_size: int = 1) -> tuple[int, ...]:
        """Get output shape for generated audio.

        Args:
            batch_size: Batch size

        Returns:
            Output shape tuple
        """
        if self.modality_config.representation == AudioRepresentation.RAW_WAVEFORM:
            return (batch_size, self.n_time_steps)
        elif self.modality_config.representation == AudioRepresentation.MEL_SPECTROGRAM:
            n_frames = self.n_time_steps // self.modality_config.hop_length
            return (batch_size, self.modality_config.n_mel_channels, n_frames)
        else:  # STFT
            n_frames = self.n_time_steps // self.modality_config.hop_length
            return (batch_size, self.modality_config.n_fft // 2 + 1, n_frames)

    def preprocess_audio(self, audio: jax.Array) -> jax.Array:
        """Preprocess audio for model input.

        Args:
            audio: Raw audio input

        Returns:
            Preprocessed audio
        """
        if self.modality_config.normalize:
            # Normalize to [-1, 1] range
            max_val = jnp.max(jnp.abs(audio))
            audio = jnp.where(max_val > 0, audio / max_val, audio)

        return audio

    def postprocess_audio(self, audio: jax.Array) -> jax.Array:
        """Postprocess generated audio.

        Args:
            audio: Generated audio

        Returns:
            Postprocessed audio
        """
        # Clip to valid range
        audio = jnp.clip(audio, -1.0, 1.0)

        return audio

    def compute_reconstruction_loss(
        self, target: jax.Array, predicted: jax.Array, loss_type: str = "mse"
    ) -> jax.Array:
        """Compute reconstruction loss between target and predicted audio.

        Args:
            target: Target audio
            predicted: Predicted audio
            loss_type: Type of loss ("mse", "l1", "spectral")

        Returns:
            Loss value
        """
        if loss_type == "mse":
            return jnp.mean((target - predicted) ** 2)
        elif loss_type == "l1":
            return jnp.mean(jnp.abs(target - predicted))
        elif loss_type == "spectral":
            # Spectral loss using FFT
            target_fft = jnp.fft.fft(target)
            predicted_fft = jnp.fft.fft(predicted)

            target_mag = jnp.abs(target_fft)
            predicted_mag = jnp.abs(predicted_fft)

            return jnp.mean((target_mag - predicted_mag) ** 2)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def generate(
        self,
        n_samples: int = 1,
        duration: float | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> jax.Array:
        """Generate audio samples.

        Args:
            n_samples: Number of samples to generate
            duration: Duration override
            rngs: Random number generators
            **kwargs: Additional generation parameters

        Returns:
            Generated audio samples
        """
        # Default implementation - subclasses should override
        duration = duration or self.modality_config.duration
        output_shape = self.get_output_shape(n_samples)

        key = (rngs or self.rngs).sample()

        # Generate simple noise as placeholder
        generated = jax.random.normal(key, output_shape)
        return self.postprocess_audio(generated)

    def loss_fn(
        self,
        batch: dict[str, jax.Array],
        model_outputs: dict[str, jax.Array],
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Compute training loss.

        Args:
            batch: Training batch
            model_outputs: Model predictions
            **kwargs: Additional loss parameters

        Returns:
            Loss value
        """
        target_audio = batch["audio"]
        predicted_audio = model_outputs.get("audio", model_outputs.get("predictions"))

        if predicted_audio is None:
            raise ValueError("Model outputs must contain 'audio' or 'predictions' key")

        # Preprocess both target and predicted
        target_audio = self.preprocess_audio(target_audio)
        predicted_audio = self.preprocess_audio(predicted_audio)

        # Compute reconstruction loss
        loss_type = kwargs.get("loss_type", "mse")
        loss = self.compute_reconstruction_loss(target_audio, predicted_audio, loss_type)
        return {
            "loss": float(loss),
            f"{loss_type}_loss": float(loss),
        }
