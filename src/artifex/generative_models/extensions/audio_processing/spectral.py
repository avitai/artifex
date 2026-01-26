"""Spectral analysis and feature extraction for audio processing.

This module provides JAX-compatible spectral analysis tools for audio generation
and processing tasks. Uses jax.scipy.signal.stft for JIT-compatible STFT computation.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.scipy.signal import stft as jax_stft

from artifex.generative_models.core.configuration import ExtensionConfig
from artifex.generative_models.extensions.base import ModelExtension


class SpectralAnalysis(ModelExtension):
    """Spectral analysis and feature extraction for audio."""

    def __init__(
        self,
        config: ExtensionConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize spectral analysis module.

        Args:
            config: Extension configuration with spectral parameters:
                - weight: Weight for the extension (default: 1.0)
                - enabled: Whether the extension is enabled (default: True)
                - extensions.spectral.sample_rate: Audio sample rate in Hz
                - extensions.spectral.n_fft: FFT window size
                - extensions.spectral.hop_length: Hop length for STFT
                - extensions.spectral.window_type: Window function type
                - extensions.spectral.n_mels: Number of mel filter banks
            rngs: Random number generator keys
        """
        # Handle configuration
        if not isinstance(config, ExtensionConfig):
            raise TypeError(f"config must be ExtensionConfig, got {type(config).__name__}")

        super().__init__(config, rngs=rngs)

        # Get spectral parameters from extensions field
        spectral_params = getattr(config, "extensions", {}).get("spectral", {})

        self.sample_rate = spectral_params.get("sample_rate", 22050)
        self.n_fft = spectral_params.get("n_fft", 2048)
        self.hop_length = spectral_params.get("hop_length", self.n_fft // 4)
        self.window_type = spectral_params.get("window_type", "hann")
        self.n_mels = spectral_params.get("n_mels", 128)
        self.rngs = rngs

        # Initialize mel filter bank
        self.mel_filters = self._create_mel_filters()

        # Initialize window function
        self.window = self._create_window()

    def _create_window(self) -> jax.Array:
        """Create window function for STFT."""
        if self.window_type == "hann":
            n = jnp.arange(self.n_fft)
            window = 0.5 * (1 - jnp.cos(2 * jnp.pi * n / (self.n_fft - 1)))
        elif self.window_type == "hamming":
            n = jnp.arange(self.n_fft)
            window = 0.54 - 0.46 * jnp.cos(2 * jnp.pi * n / (self.n_fft - 1))
        elif self.window_type == "blackman":
            n = jnp.arange(self.n_fft)
            window = (
                0.42
                - 0.5 * jnp.cos(2 * jnp.pi * n / (self.n_fft - 1))
                + 0.08 * jnp.cos(4 * jnp.pi * n / (self.n_fft - 1))
            )
        else:
            # Default to rectangular window
            window = jnp.ones(self.n_fft)

        return window

    def _create_mel_filters(self) -> jax.Array:
        """Create mel-scale filter bank."""

        # Mel scale conversion functions
        def hz_to_mel(hz: float) -> float:
            return 2595 * jnp.log10(1 + hz / 700)

        def mel_to_hz(mel: float) -> float:
            return 700 * (10 ** (mel / 2595) - 1)

        # Create mel-spaced frequency points
        low_freq_mel = hz_to_mel(0)
        high_freq_mel = hz_to_mel(self.sample_rate / 2)

        mel_points = jnp.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Convert to FFT bin numbers
        bin_points = jnp.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(jnp.int32)

        # Create filter bank
        filters = jnp.zeros((self.n_mels, self.n_fft // 2 + 1))

        for i in range(self.n_mels):
            left_bin = bin_points[i]
            center_bin = bin_points[i + 1]
            right_bin = bin_points[i + 2]

            # Create triangular filter
            for j in range(left_bin, center_bin):
                if center_bin > left_bin:
                    filters = filters.at[i, j].set((j - left_bin) / (center_bin - left_bin))

            for j in range(center_bin, right_bin):
                if right_bin > center_bin:
                    filters = filters.at[i, j].set((right_bin - j) / (right_bin - center_bin))

        return filters

    def compute_stft(self, audio: jax.Array) -> jax.Array:
        """Compute Short-Time Fourier Transform using jax.scipy.signal.stft.

        This implementation is JIT-compatible and uses JAX's built-in STFT
        for efficient computation on GPU/TPU.

        Args:
            audio: Audio signal [length] or [batch, length]

        Returns:
            STFT magnitude spectrogram [freq_bins, time_frames] or [batch, freq_bins, time_frames]
        """
        if audio.ndim == 1:
            # Single audio signal
            return self._stft_single(audio)
        else:
            # Batch of audio signals
            return jax.vmap(self._stft_single)(audio)

    def _stft_single(self, audio: jax.Array) -> jax.Array:
        """Compute STFT for a single audio signal using jax.scipy.signal.stft.

        This is JIT-compatible and avoids Python loops.
        """
        # Compute overlap from hop_length
        noverlap = self.n_fft - self.hop_length

        # Use JAX's built-in STFT (returns freqs, times, Zxx)
        _, _, stft_result = jax_stft(
            audio,
            fs=self.sample_rate,
            window=self.window_type,
            nperseg=self.n_fft,
            noverlap=noverlap,
            padded=True,
            boundary="zeros",
        )

        # Take magnitude (stft_result is complex)
        magnitude = jnp.abs(stft_result)

        return magnitude

    def compute_spectrogram(self, audio: jax.Array) -> jax.Array:
        """Compute power spectrogram.

        Args:
            audio: Audio signal [length] or [batch, length]

        Returns:
            Power spectrogram [freq_bins, time_frames] or [batch, freq_bins, time_frames]
        """
        magnitude = self.compute_stft(audio)
        power = magnitude**2
        return power

    def compute_mel_spectrogram(self, audio: jax.Array) -> jax.Array:
        """Compute mel-scaled spectrogram.

        Args:
            audio: Audio signal [length] or [batch, length]

        Returns:
            Mel spectrogram [n_mels, time_frames] or [batch, n_mels, time_frames]
        """
        # Compute power spectrogram
        power_spec = self.compute_spectrogram(audio)

        if power_spec.ndim == 2:
            # Single spectrogram
            mel_spec = jnp.dot(self.mel_filters, power_spec)
        else:
            # Batch of spectrograms
            mel_spec = jnp.einsum("mf,bft->bmt", self.mel_filters, power_spec)

        return mel_spec

    def compute_log_mel_spectrogram(self, audio: jax.Array, eps: float = 1e-8) -> jax.Array:
        """Compute log mel spectrogram.

        Args:
            audio: Audio signal [length] or [batch, length]
            eps: Small value to avoid log(0)

        Returns:
            Log mel spectrogram [n_mels, time_frames] or [batch, n_mels, time_frames]
        """
        mel_spec = self.compute_mel_spectrogram(audio)
        log_mel_spec = jnp.log(mel_spec + eps)
        return log_mel_spec

    def compute_mfcc(self, audio: jax.Array, n_mfcc: int = 13) -> jax.Array:
        """Compute Mel-Frequency Cepstral Coefficients.

        Args:
            audio: Audio signal [length] or [batch, length]
            n_mfcc: Number of MFCC coefficients

        Returns:
            MFCC features [n_mfcc, time_frames] or [batch, n_mfcc, time_frames]
        """
        # Compute log mel spectrogram
        log_mel_spec = self.compute_log_mel_spectrogram(audio)

        # Apply DCT to get cepstral coefficients
        if log_mel_spec.ndim == 2:
            # Single spectrogram
            mfcc = self._dct_type_ii(log_mel_spec)[:n_mfcc]
        else:
            # Batch of spectrograms — use nnx.vmap for NNX module compatibility
            def dct_and_truncate(x: jax.Array) -> jax.Array:
                return self._dct_type_ii(x)[:n_mfcc]

            mfcc = nnx.vmap(dct_and_truncate, in_axes=0)(log_mel_spec)

        return mfcc

    def _dct_type_ii(self, x: jax.Array) -> jax.Array:
        """Compute Type-II Discrete Cosine Transform along the first axis.

        For MFCC computation, this applies DCT to the mel frequency axis.
        Input: [n_mels, time_frames]
        Output: [n_mels, time_frames] (DCT coefficients along mel axis)
        """
        n = x.shape[0]  # Number of mel bands
        k = jnp.arange(n)[:, None]
        m = jnp.arange(n)[None, :]  # Use n for both dimensions (DCT matrix is [n, n])

        # DCT-II transformation matrix [n, n]
        dct_matrix = jnp.cos(jnp.pi * k * (2 * m + 1) / (2 * n))

        # Apply normalization
        dct_matrix = dct_matrix.at[0].multiply(1 / jnp.sqrt(2))
        dct_matrix = dct_matrix * jnp.sqrt(2 / n)

        # Apply DCT: [n, n] @ [n, time_frames] -> [n, time_frames]
        dct_result = jnp.dot(dct_matrix, x)

        return dct_result

    def compute_spectral_centroid(self, audio: jax.Array) -> jax.Array:
        """Compute spectral centroid (brightness measure).

        Args:
            audio: Audio signal [length] or [batch, length]

        Returns:
            Spectral centroid [time_frames] or [batch, time_frames]
        """
        magnitude = self.compute_stft(audio)

        # Frequency bins
        freqs = jnp.linspace(0, self.sample_rate / 2, magnitude.shape[-2])

        if magnitude.ndim == 2:
            # Single spectrogram
            centroid = jnp.sum(magnitude * freqs[:, None], axis=0) / (
                jnp.sum(magnitude, axis=0) + 1e-8
            )
        else:
            # Batch of spectrograms
            centroid = jnp.sum(magnitude * freqs[None, :, None], axis=1) / (
                jnp.sum(magnitude, axis=1) + 1e-8
            )

        return centroid

    def compute_spectral_bandwidth(self, audio: jax.Array) -> jax.Array:
        """Compute spectral bandwidth.

        Args:
            audio: Audio signal [length] or [batch, length]

        Returns:
            Spectral bandwidth [time_frames] or [batch, time_frames]
        """
        magnitude = self.compute_stft(audio)
        centroid = self.compute_spectral_centroid(audio)

        # Frequency bins
        freqs = jnp.linspace(0, self.sample_rate / 2, magnitude.shape[-2])

        if magnitude.ndim == 2:
            # Single spectrogram
            freq_diff = freqs[:, None] - centroid[None, :]
            bandwidth = jnp.sqrt(
                jnp.sum(magnitude * freq_diff**2, axis=0) / (jnp.sum(magnitude, axis=0) + 1e-8)
            )
        else:
            # Batch of spectrograms
            freq_diff = freqs[None, :, None] - centroid[:, None, :]
            bandwidth = jnp.sqrt(
                jnp.sum(magnitude * freq_diff**2, axis=1) / (jnp.sum(magnitude, axis=1) + 1e-8)
            )

        return bandwidth

    def compute_spectral_rolloff(
        self, audio: jax.Array, rolloff_percent: float = 0.85
    ) -> jax.Array:
        """Compute spectral rolloff frequency.

        Args:
            audio: Audio signal [length] or [batch, length]
            rolloff_percent: Percentage of energy to consider

        Returns:
            Spectral rolloff [time_frames] or [batch, time_frames]
        """
        magnitude = self.compute_stft(audio)

        # Cumulative energy
        cumsum_magnitude = jnp.cumsum(magnitude, axis=-2)
        total_energy = cumsum_magnitude[-1:, :]

        # Find rolloff frequency
        threshold = rolloff_percent * total_energy
        rolloff_indices = jnp.argmax(cumsum_magnitude >= threshold, axis=-2)

        # Convert to frequency
        freqs = jnp.linspace(0, self.sample_rate / 2, magnitude.shape[-2])
        rolloff_freq = freqs[rolloff_indices]

        return rolloff_freq

    def inverse_mel_spectrogram(self, mel_spec: jax.Array) -> jax.Array:
        """Convert mel spectrogram back to linear spectrogram.

        Args:
            mel_spec: Mel spectrogram [n_mels, time_frames] or [batch, n_mels, time_frames]

        Returns:
            Linear spectrogram [freq_bins, time_frames] or [batch, freq_bins, time_frames]
        """
        # Pseudo-inverse of mel filter bank
        mel_filters_inv = jnp.linalg.pinv(self.mel_filters)

        if mel_spec.ndim == 2:
            # Single spectrogram
            linear_spec = jnp.dot(mel_filters_inv, mel_spec)
        else:
            # Batch of spectrograms
            linear_spec = jnp.einsum("fm,bmt->bft", mel_filters_inv, mel_spec)

        return linear_spec

    def extract_spectral_features(self, audio: jax.Array) -> dict[str, jax.Array]:
        """Extract comprehensive spectral features.

        Args:
            audio: Audio signal [length] or [batch, length]

        Returns:
            Dictionary of spectral features
        """
        features = {}

        # Basic spectrograms
        features["spectrogram"] = self.compute_spectrogram(audio)
        features["mel_spectrogram"] = self.compute_mel_spectrogram(audio)
        features["log_mel_spectrogram"] = self.compute_log_mel_spectrogram(audio)
        features["mfcc"] = self.compute_mfcc(audio)

        # Spectral shape features
        features["spectral_centroid"] = self.compute_spectral_centroid(audio)
        features["spectral_bandwidth"] = self.compute_spectral_bandwidth(audio)
        features["spectral_rolloff"] = self.compute_spectral_rolloff(audio)

        return features

    def __call__(self, inputs: Any, model_outputs: Any, **kwargs: Any) -> dict[str, Any]:
        """Process model inputs/outputs.

        Args:
            inputs: Original inputs to the model.
            model_outputs: Outputs from the model.
            **kwargs: Additional keyword arguments.

        Returns:
            dictionary of extension outputs including spectral features.
        """
        if not self.enabled:
            return {"extension_type": "spectral_analysis"}

        # Extract audio data from model outputs
        if isinstance(model_outputs, dict):
            audio = model_outputs.get(
                "audio", model_outputs.get("waveform", model_outputs.get("generated_audio"))
            )
        else:
            # Assume model_outputs is audio directly
            audio = model_outputs

        if audio is None:
            return {
                "extension_type": "spectral_analysis",
                "error": "No audio data found in model outputs",
            }

        # Extract comprehensive spectral features
        features = self.extract_spectral_features(audio)

        return {
            "spectral_features": features,
            "extension_type": "spectral_analysis",
        }
