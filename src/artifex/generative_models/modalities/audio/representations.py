"""Audio representation processing for different formats.

This module handles conversion between different audio representations including
raw waveforms, mel-spectrograms, and STFT representations.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .base import AudioModalityConfig, AudioRepresentation


class AudioProcessor(nnx.Module):
    """Base class for audio representation processing."""

    def __init__(
        self,
        config: AudioModalityConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize audio processor.

        Args:
            config: Audio modality configuration
            rngs: Random number generators (unused for processing)
        """
        super().__init__()
        self.config = config

    def to_representation(self, audio: jnp.ndarray) -> jnp.ndarray:
        """Convert raw waveform to target representation.

        Args:
            audio: Raw waveform of shape (..., n_samples)

        Returns:
            Audio in target representation format
        """
        raise NotImplementedError("Subclasses must implement to_representation")

    def from_representation(self, features: jnp.ndarray) -> jnp.ndarray:
        """Convert from representation back to raw waveform.

        Args:
            features: Audio features in representation format

        Returns:
            Raw waveform audio
        """
        raise NotImplementedError("Subclasses must implement from_representation")


class WaveformProcessor(AudioProcessor):
    """Processor for raw waveform representation (identity transform)."""

    def to_representation(self, audio: jnp.ndarray) -> jnp.ndarray:
        """Identity transform for raw waveforms.

        Args:
            audio: Raw waveform

        Returns:
            Same waveform, optionally normalized
        """
        if self.config.normalize:
            # Normalize to [-1, 1] range
            max_val = jnp.max(jnp.abs(audio))
            return jnp.where(max_val > 0, audio / max_val, audio)
        return audio

    def from_representation(self, features: jnp.ndarray) -> jnp.ndarray:
        """Identity transform for raw waveforms.

        Args:
            features: Raw waveform features

        Returns:
            Same waveform
        """
        return features


class SpectrogramProcessor(AudioProcessor):
    """Processor for spectrogram representations (STFT and Mel-spectrogram)."""

    def __init__(
        self,
        config: AudioModalityConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize spectrogram processor.

        Args:
            config: Audio modality configuration
            rngs: Random number generators
        """
        super().__init__(config, rngs=rngs)

        # Precompute mel filter bank if needed
        if config.representation == AudioRepresentation.MEL_SPECTROGRAM:
            self.mel_filters = self._create_mel_filters()
        else:
            self.mel_filters = None

    def _create_mel_filters(self) -> jnp.ndarray:
        """Create mel-scale filter bank.

        Returns:
            Mel filter bank of shape (n_mel_channels, n_fft // 2 + 1)
        """
        # Simple triangular mel filter bank implementation
        n_fft = self.config.n_fft
        n_mels = self.config.n_mel_channels
        sample_rate = self.config.sample_rate

        # Frequency bins
        freqs = jnp.linspace(0, sample_rate // 2, n_fft // 2 + 1)

        # Mel scale conversion (simplified)
        def hz_to_mel(f):
            return 2595 * jnp.log10(1 + f / 700)

        def mel_to_hz(m):
            return 700 * (10 ** (m / 2595) - 1)

        # Create mel points
        mel_min = hz_to_mel(0)
        mel_max = hz_to_mel(sample_rate // 2)
        mel_points = jnp.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Create triangular filters
        filters = jnp.zeros((n_mels, n_fft // 2 + 1))

        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]

            # Triangular filter
            left_slope = (freqs - left) / (center - left)
            right_slope = (right - freqs) / (right - center)

            filter_vals = jnp.maximum(0, jnp.minimum(left_slope, right_slope))
            filters = filters.at[i].set(filter_vals)

        return filters

    def _stft(self, audio: jnp.ndarray) -> jnp.ndarray:
        """Compute Short-Time Fourier Transform.

        Args:
            audio: Raw waveform of shape (..., n_samples)

        Returns:
            STFT magnitude of shape (..., n_fft // 2 + 1, n_frames)
        """
        # Simple STFT implementation using JAX
        n_fft = self.config.n_fft
        hop_length = self.config.hop_length

        # Pad audio
        pad_amount = n_fft // 2
        audio_padded = jnp.pad(audio, ((0, 0), (pad_amount, pad_amount)), mode="reflect")

        # Create windowed frames using vectorized dynamic_slice
        n_frames = (audio_padded.shape[-1] - n_fft) // hop_length + 1
        window = jnp.hanning(n_fft)

        # Compute start indices for all frames
        frame_starts = jnp.arange(n_frames) * hop_length

        def extract_frame(start: jax.Array) -> jax.Array:
            """Extract and window a single frame from the last axis."""
            frame = jax.lax.dynamic_slice_in_dim(audio_padded, start, n_fft, axis=-1)
            return frame * window

        # vmap over frame indices to extract all frames at once
        frames = jax.vmap(extract_frame)(frame_starts)  # (n_frames, ..., n_fft)

        # Rearrange to (..., n_fft, n_frames)
        # frames is (n_frames, batch, n_fft) -> move n_frames to last axis
        frames = jnp.moveaxis(frames, 0, -1)  # (..., n_fft, n_frames)

        # Apply FFT
        fft_result = jnp.fft.rfft(frames, axis=-2)  # (..., n_fft // 2 + 1, n_frames)
        magnitude = jnp.abs(fft_result)

        return magnitude

    def _istft(self, stft_magnitude: jnp.ndarray) -> jnp.ndarray:
        """Inverse Short-Time Fourier Transform using Griffin-Lim algorithm.

        Args:
            stft_magnitude: STFT magnitude of shape (..., n_fft // 2 + 1, n_frames)

        Returns:
            Reconstructed waveform
        """
        # Simple Griffin-Lim reconstruction
        n_iter = 10  # Number of Griffin-Lim iterations

        # Initialize random phase
        phase = jax.random.uniform(jax.random.key(0), stft_magnitude.shape) * 2 * jnp.pi

        for _ in range(n_iter):
            # Combine magnitude and phase
            complex_stft = stft_magnitude * jnp.exp(1j * phase)

            # ISTFT
            waveform = self._istft_once(complex_stft)

            # Forward STFT to get new phase
            new_stft = self._stft_complex(waveform)
            phase = jnp.angle(new_stft)

        # Final reconstruction
        complex_stft = stft_magnitude * jnp.exp(1j * phase)
        return self._istft_once(complex_stft)

    def _stft_complex(self, audio: jnp.ndarray) -> jnp.ndarray:
        """Compute complex STFT."""
        n_fft = self.config.n_fft
        hop_length = self.config.hop_length

        # Similar to _stft but returns complex values
        pad_amount = n_fft // 2
        audio_padded = jnp.pad(audio, ((0, 0), (pad_amount, pad_amount)), mode="reflect")

        n_frames = (audio_padded.shape[-1] - n_fft) // hop_length + 1
        window = jnp.hanning(n_fft)

        # Vectorized frame extraction
        frame_starts = jnp.arange(n_frames) * hop_length

        def extract_frame(start: jax.Array) -> jax.Array:
            frame = jax.lax.dynamic_slice_in_dim(audio_padded, start, n_fft, axis=-1)
            return frame * window

        frames = jax.vmap(extract_frame)(frame_starts)
        frames = jnp.moveaxis(frames, 0, -1)  # (..., n_fft, n_frames)

        return jnp.fft.rfft(frames, axis=-2)

    def _istft_once(self, complex_stft: jnp.ndarray) -> jnp.ndarray:
        """Single ISTFT operation."""
        n_fft = self.config.n_fft
        hop_length = self.config.hop_length

        # IFFT
        frames = jnp.fft.irfft(complex_stft, n=n_fft, axis=-2)

        # Overlap-add reconstruction using lax.scan
        n_frames = frames.shape[-1]
        audio_length = (n_frames - 1) * hop_length + n_fft
        window = jnp.hanning(n_fft)

        # Scan over frame indices for overlap-add
        def add_frame(
            audio: jax.Array,
            frame_idx: jax.Array,
        ) -> tuple[jax.Array, None]:
            start = frame_idx * hop_length
            windowed_frame = frames[..., frame_idx] * window
            audio = audio.at[..., start : start + n_fft].add(windowed_frame)
            return audio, None

        audio_init = jnp.zeros((*frames.shape[:-2], audio_length))
        audio, _ = jax.lax.scan(add_frame, audio_init, jnp.arange(n_frames))

        # Remove padding
        pad_amount = n_fft // 2
        return audio[..., pad_amount:-pad_amount]

    def to_representation(self, audio: jnp.ndarray) -> jnp.ndarray:
        """Convert waveform to spectrogram representation.

        Args:
            audio: Raw waveform of shape (..., n_samples)

        Returns:
            Spectrogram representation
        """
        # Compute STFT magnitude
        stft_magnitude = self._stft(audio)

        if self.config.representation == AudioRepresentation.MEL_SPECTROGRAM:
            # Apply mel filters
            mel_spec = jnp.einsum("...ft,mf->...mt", stft_magnitude, self.mel_filters)

            # Convert to log scale
            log_mel = jnp.log(mel_spec + 1e-8)
            return log_mel
        else:
            # Return log STFT magnitude
            return jnp.log(stft_magnitude + 1e-8)

    def from_representation(self, features: jnp.ndarray) -> jnp.ndarray:
        """Convert spectrogram back to waveform.

        Args:
            features: Spectrogram features

        Returns:
            Reconstructed waveform
        """
        if self.config.representation == AudioRepresentation.MEL_SPECTROGRAM:
            # Convert from log mel to linear mel
            mel_spec = jnp.exp(features)

            # Pseudo-inverse mel filter (simplified)
            mel_filters_pinv = jnp.linalg.pinv(self.mel_filters + 1e-8)
            stft_magnitude = jnp.einsum("...mt,fm->...ft", mel_spec, mel_filters_pinv)
        else:
            # Convert from log STFT to linear STFT
            stft_magnitude = jnp.exp(features)

        # Reconstruct waveform using Griffin-Lim
        return self._istft(stft_magnitude)


def create_audio_processor(
    config: AudioModalityConfig,
    *,
    rngs: nnx.Rngs | None = None,
) -> AudioProcessor:
    """Factory function to create appropriate audio processor.

    Args:
        config: Audio modality configuration
        rngs: Random number generators

    Returns:
        Appropriate audio processor instance
    """
    if config.representation == AudioRepresentation.RAW_WAVEFORM:
        return WaveformProcessor(config, rngs=rngs)
    else:
        return SpectrogramProcessor(config, rngs=rngs)
