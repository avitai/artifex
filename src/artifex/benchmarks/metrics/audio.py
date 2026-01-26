"""Audio-specific metrics for generative model evaluation."""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import EvaluationConfig

from .core import MetricBase


class SpectralMetric(MetricBase):
    """Spectral convergence metric for audio quality assessment."""

    def __init__(self, *, config: EvaluationConfig, rngs: nnx.Rngs):
        """Initialize spectral metric.

        Args:
            config: Evaluation configuration (must be EvaluationConfig)
            rngs: NNX Rngs for stochastic operations
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")

        # Initialize base class with the EvaluationConfig
        super().__init__(config=config, rngs=rngs)
        self.eval_batch_size = config.eval_batch_size

        # Spectral parameters from config
        spectral_params = config.metric_params.get("spectral", {})
        self.n_fft = spectral_params.get("n_fft", 1024)
        self.hop_length = spectral_params.get("hop_length", 256)

    def validate_inputs(self, real_data, generated_data) -> bool:
        """Validate input data for spectral computation."""
        if not isinstance(real_data, jnp.ndarray) or not isinstance(generated_data, jnp.ndarray):
            return False
        if real_data.shape != generated_data.shape:
            return False
        if real_data.ndim != 2:  # (batch, time)
            return False
        return True

    def compute(self, real_data, generated_data, **kwargs) -> dict[str, float]:
        """Compute spectral convergence between real and generated audio."""
        # Compute Short-Time Fourier Transform (STFT) for both signals
        real_stft = self._compute_stft(real_data)
        gen_stft = self._compute_stft(generated_data)

        # Compute magnitude spectra
        real_mag = jnp.abs(real_stft)
        gen_mag = jnp.abs(gen_stft)

        # Compute spectral convergence
        numerator = jnp.linalg.norm(real_mag - gen_mag, ord="fro", axis=(-2, -1))
        denominator = jnp.linalg.norm(real_mag, ord="fro", axis=(-2, -1))

        # Avoid division by zero
        spectral_convergence = jnp.where(
            denominator > 1e-8, numerator / denominator, jnp.ones_like(numerator)
        )

        # Average across batch
        avg_spectral_convergence = float(jnp.mean(spectral_convergence))

        return {"spectral_convergence": avg_spectral_convergence}

    def _compute_stft(self, audio):
        """Compute Short-Time Fourier Transform."""
        # Simple mock STFT computation for testing
        # In real implementation, this would use proper STFT

        batch_size, time_length = audio.shape
        n_frames = (time_length - self.n_fft) // self.hop_length + 1
        n_freq_bins = self.n_fft // 2 + 1

        # Mock STFT by applying FFT to windowed segments
        stft = jnp.zeros((batch_size, n_freq_bins, n_frames), dtype=jnp.complex64)

        for i in range(n_frames):
            start_idx = i * self.hop_length
            end_idx = start_idx + self.n_fft

            if end_idx <= time_length:
                # Apply window and FFT
                windowed = audio[:, start_idx:end_idx]
                # Apply Hann window
                window = 0.5 * (1 - jnp.cos(2 * jnp.pi * jnp.arange(self.n_fft) / (self.n_fft - 1)))
                windowed = windowed * window[None, :]

                # Compute FFT
                fft_result = jnp.fft.fft(windowed, n=self.n_fft)
                stft = stft.at[:, :, i].set(fft_result[:, :n_freq_bins])

        return stft


class MelCepstralMetric(MetricBase):
    """Mel-Cepstral Distortion metric for audio quality assessment."""

    def __init__(self, *, config: EvaluationConfig, rngs: nnx.Rngs):
        """Initialize mel-cepstral metric.

        Args:
            config: Evaluation configuration (must be EvaluationConfig)
            rngs: NNX Rngs for stochastic operations
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")

        # Initialize base class with minimal config to satisfy MetricBase requirements
        # Initialize base class with the EvaluationConfig
        super().__init__(config=config, rngs=rngs)
        self.eval_batch_size = config.eval_batch_size

        # MCD parameters from config
        mcd_params = config.metric_params.get("mcd", {})
        self.n_mels = mcd_params.get("n_mels", 80)
        self.sr = mcd_params.get("sr", 16000)
        self.n_fft = mcd_params.get("n_fft", 1024)

    def validate_inputs(self, real_data, generated_data) -> bool:
        """Validate input data for mel-cepstral computation."""
        if not isinstance(real_data, jnp.ndarray) or not isinstance(generated_data, jnp.ndarray):
            return False
        if real_data.shape != generated_data.shape:
            return False
        if real_data.ndim != 2:  # (batch, time)
            return False
        return True

    def compute(self, real_data, generated_data, **kwargs) -> dict[str, float]:
        """Compute mel-cepstral distortion between real and generated audio."""
        # Compute mel-spectrograms
        real_mel = self._compute_mel_spectrogram(real_data)
        gen_mel = self._compute_mel_spectrogram(generated_data)

        # Convert to mel-cepstral coefficients
        real_mfcc = self._mel_to_mfcc(real_mel)
        gen_mfcc = self._mel_to_mfcc(gen_mel)

        # Compute mel-cepstral distortion
        mcd = self._compute_mcd(real_mfcc, gen_mfcc)

        return {"mel_cepstral_distortion": mcd}

    def _compute_mel_spectrogram(self, audio):
        """Compute mel-spectrogram from audio."""
        # Mock mel-spectrogram computation
        # In real implementation, this would use proper mel filterbank

        # First compute magnitude spectrogram
        stft = self._compute_stft(audio)
        magnitude = jnp.abs(stft)

        # Apply mel filterbank (simplified)
        mel_filters = self._create_mel_filterbank()
        mel_spec = jnp.einsum("bft,mf->bmt", magnitude, mel_filters)

        # Convert to log scale
        log_mel_spec = jnp.log(jnp.maximum(mel_spec, 1e-8))

        return log_mel_spec

    def _compute_stft(self, audio):
        """Compute STFT (simplified version)."""
        batch_size, time_length = audio.shape
        hop_length = 256
        n_frames = (time_length - self.n_fft) // hop_length + 1
        n_freq_bins = self.n_fft // 2 + 1

        # Limit frames for efficiency in testing
        n_frames = min(n_frames, 10)

        # Vectorized frame extraction using dynamic_slice
        frame_starts = jnp.arange(n_frames) * hop_length

        def extract_and_fft(start: jax.Array) -> jax.Array:
            """Extract a frame and compute FFT for all batch elements."""
            frame = jax.lax.dynamic_slice_in_dim(audio, start, self.n_fft, axis=-1)
            fft_result = jnp.fft.fft(frame, n=self.n_fft)
            return fft_result[:, :n_freq_bins]

        # vmap over frame indices: (n_frames, batch, n_freq_bins)
        stft_frames = jax.vmap(extract_and_fft)(frame_starts)
        # Transpose to (batch, n_freq_bins, n_frames)
        return jnp.transpose(stft_frames, (1, 2, 0))

    def _create_mel_filterbank(self):
        """Create mel filterbank (simplified)."""
        n_freq_bins = self.n_fft // 2 + 1

        # Simple triangular filters approximation
        mel_filters = jnp.zeros((self.n_mels, n_freq_bins))

        for m in range(self.n_mels):
            # Create triangular filter centered at mel frequency
            center = (m + 1) * n_freq_bins // (self.n_mels + 1)
            width = n_freq_bins // (self.n_mels + 1)

            start = max(0, center - width // 2)
            end = min(n_freq_bins, center + width // 2)

            # Triangular shape
            for f in range(start, end):
                if f <= center:
                    mel_filters = mel_filters.at[m, f].set((f - start) / (center - start + 1e-8))
                else:
                    mel_filters = mel_filters.at[m, f].set((end - f) / (end - center + 1e-8))

        return mel_filters

    def _mel_to_mfcc(self, mel_spec):
        """Convert mel-spectrogram to MFCC."""
        # Apply Discrete Cosine Transform (DCT)
        # Simplified DCT implementation
        batch_size, n_mels, n_frames = mel_spec.shape

        # Create DCT matrix
        dct_matrix = jnp.zeros((n_mels, n_mels))
        for k in range(n_mels):
            for n in range(n_mels):
                if k == 0:
                    dct_matrix = dct_matrix.at[k, n].set(jnp.sqrt(1.0 / n_mels))
                else:
                    dct_matrix = dct_matrix.at[k, n].set(
                        jnp.sqrt(2.0 / n_mels) * jnp.cos(jnp.pi * k * (n + 0.5) / n_mels)
                    )

        # Apply DCT
        mfcc = jnp.einsum("km,bmf->bkf", dct_matrix, mel_spec)

        return mfcc

    def _compute_mcd(self, real_mfcc, gen_mfcc):
        """Compute mel-cepstral distortion."""
        # Use only the first 13 coefficients (standard practice)
        n_coeffs = min(13, real_mfcc.shape[1])
        real_mfcc = real_mfcc[:, :n_coeffs, :]
        gen_mfcc = gen_mfcc[:, :n_coeffs, :]

        # Compute frame-wise euclidean distance
        diff = real_mfcc - gen_mfcc
        frame_distances = jnp.sqrt(jnp.sum(diff**2, axis=1))

        # Average across frames and batch
        mcd = jnp.mean(frame_distances)

        # Scale by constant factor (common in MCD computation)
        mcd = (10.0 / jnp.log(10.0)) * jnp.sqrt(2.0) * mcd

        return float(mcd)


# Factory functions for convenient metric creation
def create_spectral_metric(
    *,
    rngs: nnx.Rngs,
    n_fft: int = 1024,
    hop_length: int = 256,
    batch_size: int = 32,
    config_name: str = "spectral_metric",
) -> SpectralMetric:
    """Create Spectral metric with typed configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        n_fft: FFT window size
        hop_length: Hop length for STFT
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured SpectralMetric instance
    """
    config = EvaluationConfig(
        name=config_name,
        metrics=["spectral"],
        metric_params={
            "spectral": {
                "n_fft": n_fft,
                "hop_length": hop_length,
                "higher_is_better": True,
            }
        },
        eval_batch_size=batch_size,
    )

    return SpectralMetric(config=config, rngs=rngs)


def create_mcd_metric(
    *,
    rngs: nnx.Rngs,
    n_mels: int = 80,
    sr: int = 16000,
    n_fft: int = 1024,
    batch_size: int = 32,
    config_name: str = "mcd_metric",
) -> MelCepstralMetric:
    """Create MCD metric with typed configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        n_mels: Number of mel bands
        sr: Sample rate
        n_fft: FFT window size
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured MelCepstralMetric instance
    """
    config = EvaluationConfig(
        name=config_name,
        metrics=["mcd"],
        metric_params={
            "mcd": {
                "n_mels": n_mels,
                "sr": sr,
                "n_fft": n_fft,
                "higher_is_better": False,
            }
        },
        eval_batch_size=batch_size,
    )

    return MelCepstralMetric(config=config, rngs=rngs)
