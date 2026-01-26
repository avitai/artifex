"""Audio evaluation metrics and assessment tools."""

import jax
import jax.numpy as jnp

from .base import AudioModalityConfig


class AudioMetrics:
    """Container for audio evaluation metrics."""

    def __init__(self, config: AudioModalityConfig):
        """Initialize audio metrics.

        Args:
            config: Audio modality configuration
        """
        self.config = config

    def spectral_convergence(self, generated: jax.Array, reference: jax.Array) -> float:
        """Compute spectral convergence metric.

        Args:
            generated: Generated audio
            reference: Reference audio

        Returns:
            Spectral convergence score
        """
        # Simple implementation using FFT
        gen_fft = jnp.fft.fft(generated)
        ref_fft = jnp.fft.fft(reference)

        gen_mag = jnp.abs(gen_fft)
        ref_mag = jnp.abs(ref_fft)

        # Compute normalized difference (use 2-norm for vectors)
        numerator = jnp.linalg.norm(ref_mag - gen_mag)
        denominator = jnp.linalg.norm(ref_mag)

        return float(numerator / (denominator + 1e-8))

    def log_spectral_distance(self, generated: jax.Array, reference: jax.Array) -> float:
        """Compute log spectral distance.

        Args:
            generated: Generated audio
            reference: Reference audio

        Returns:
            Log spectral distance
        """
        gen_fft = jnp.fft.fft(generated)
        ref_fft = jnp.fft.fft(reference)

        gen_mag = jnp.abs(gen_fft) + 1e-8
        ref_mag = jnp.abs(ref_fft) + 1e-8

        log_diff = jnp.log(gen_mag) - jnp.log(ref_mag)
        return float(jnp.mean(log_diff**2))

    def temporal_coherence(self, audio: jax.Array) -> float:
        """Compute temporal coherence metric.

        Args:
            audio: Audio signal

        Returns:
            Temporal coherence score
        """
        # Simple autocorrelation-based coherence
        autocorr = jnp.correlate(audio, audio, mode="full")
        autocorr = autocorr[autocorr.size // 2 :]

        # Normalize by zero-lag correlation
        normalized_autocorr = autocorr / (autocorr[0] + 1e-8)

        # Measure decay rate as coherence indicator
        coherence = jnp.mean(jnp.abs(normalized_autocorr[: len(normalized_autocorr) // 4]))
        return float(coherence)

    def harmonic_quality(self, audio: jax.Array) -> float:
        """Compute harmonic quality metric.

        Args:
            audio: Audio signal

        Returns:
            Harmonic quality score
        """
        # Compute power spectrum
        fft_result = jnp.fft.fft(audio)
        power_spectrum = jnp.abs(fft_result) ** 2

        # Find dominant frequency
        freqs = jnp.fft.fftfreq(len(audio), 1.0 / self.config.sample_rate)
        dominant_idx = jnp.argmax(power_spectrum[: len(power_spectrum) // 2])
        fundamental_freq = freqs[dominant_idx]

        if fundamental_freq <= 0:
            return 0.0

        # Check for harmonics
        harmonic_strength = 0.0
        for h in range(2, 6):  # Check first few harmonics
            harmonic_freq = h * fundamental_freq
            if harmonic_freq < self.config.sample_rate // 2:
                # Find closest frequency bin
                harmonic_idx = jnp.argmin(jnp.abs(freqs - harmonic_freq))
                harmonic_power = power_spectrum[harmonic_idx]
                fundamental_power = power_spectrum[dominant_idx]

                harmonic_strength += float(harmonic_power / (fundamental_power + 1e-8))

        return float(harmonic_strength / 4.0)  # Normalize by number of harmonics


class AudioEvaluationSuite:
    """Comprehensive audio evaluation suite."""

    def __init__(self, config: AudioModalityConfig):
        """Initialize evaluation suite.

        Args:
            config: Audio modality configuration
        """
        self.config = config
        self.metrics = AudioMetrics(config)

    def evaluate_generation_quality(
        self, generated_audio: jax.Array, reference_audio: jax.Array | None = None
    ) -> dict[str, float]:
        """Evaluate generation quality with multiple metrics.

        Args:
            generated_audio: Generated audio samples
            reference_audio: Reference audio for comparison (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        results = {}

        # Intrinsic quality metrics (no reference needed)
        if generated_audio.ndim > 1:
            # Process each sample separately and average
            temporal_scores = []
            harmonic_scores = []

            for i in range(generated_audio.shape[0]):
                sample = generated_audio[i]
                temporal_scores.append(self.metrics.temporal_coherence(sample))
                harmonic_scores.append(self.metrics.harmonic_quality(sample))

            results["temporal_coherence"] = float(jnp.mean(jnp.array(temporal_scores)))
            results["harmonic_quality"] = float(jnp.mean(jnp.array(harmonic_scores)))
        else:
            results["temporal_coherence"] = self.metrics.temporal_coherence(generated_audio)
            results["harmonic_quality"] = self.metrics.harmonic_quality(generated_audio)

        # Reference-based metrics
        if reference_audio is not None:
            if generated_audio.shape != reference_audio.shape:
                # Handle shape mismatch by padding/truncating
                min_len = min(generated_audio.shape[-1], reference_audio.shape[-1])
                gen_truncated = generated_audio[..., :min_len]
                ref_truncated = reference_audio[..., :min_len]
            else:
                gen_truncated = generated_audio
                ref_truncated = reference_audio

            if gen_truncated.ndim > 1:
                # Average over samples
                spectral_scores = []
                log_spectral_scores = []

                for i in range(gen_truncated.shape[0]):
                    gen_sample = gen_truncated[i]
                    ref_sample = ref_truncated[i] if ref_truncated.ndim > 1 else ref_truncated

                    spectral_scores.append(
                        self.metrics.spectral_convergence(gen_sample, ref_sample)
                    )
                    log_spectral_scores.append(
                        self.metrics.log_spectral_distance(gen_sample, ref_sample)
                    )

                results["spectral_convergence"] = float(jnp.mean(jnp.array(spectral_scores)))
                results["log_spectral_distance"] = float(jnp.mean(jnp.array(log_spectral_scores)))
            else:
                results["spectral_convergence"] = self.metrics.spectral_convergence(
                    gen_truncated, ref_truncated
                )
                results["log_spectral_distance"] = self.metrics.log_spectral_distance(
                    gen_truncated, ref_truncated
                )

        # Composite quality score
        quality_components = [
            results.get("temporal_coherence", 0.0),
            results.get("harmonic_quality", 0.0),
        ]

        if reference_audio is not None:
            # Invert spectral metrics (lower is better) for composite score
            spectral_conv = results.get("spectral_convergence", 1.0)
            log_spectral = results.get("log_spectral_distance", 1.0)

            quality_components.extend(
                [
                    1.0 / (1.0 + spectral_conv),  # Invert so higher is better
                    1.0 / (1.0 + log_spectral),  # Invert so higher is better
                ]
            )

        results["audio_quality_index"] = float(jnp.mean(jnp.array(quality_components)))

        return results

    def evaluate_diversity(self, generated_samples: jax.Array) -> dict[str, float]:
        """Evaluate diversity of generated samples.

        Args:
            generated_samples: Batch of generated audio samples

        Returns:
            Diversity metrics
        """
        if generated_samples.ndim < 2:
            return {"sample_diversity": 0.0}

        n_samples = generated_samples.shape[0]
        if n_samples < 2:
            return {"sample_diversity": 0.0}

        # Compute pairwise spectral distances
        distances = []

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                sample_i = generated_samples[i]
                sample_j = generated_samples[j]

                # Use spectral convergence as distance measure
                distance = self.metrics.spectral_convergence(sample_i, sample_j)
                distances.append(distance)

        diversity = float(jnp.mean(jnp.array(distances)))

        return {"sample_diversity": diversity}


def compute_audio_metrics(
    generated_audio: jax.Array,
    reference_audio: jax.Array | None = None,
    config: AudioModalityConfig | None = None,
) -> dict[str, float]:
    """Convenience function to compute audio metrics.

    Args:
        generated_audio: Generated audio samples
        reference_audio: Reference audio for comparison
        config: Audio modality configuration

    Returns:
        Dictionary of audio metrics
    """
    if config is None:
        config = AudioModalityConfig()

    evaluator = AudioEvaluationSuite(config)

    # Combine quality and diversity metrics
    quality_metrics = evaluator.evaluate_generation_quality(generated_audio, reference_audio)

    diversity_metrics = evaluator.evaluate_diversity(generated_audio)

    return {**quality_metrics, **diversity_metrics}
