"""Temporal analysis and feature extraction for audio processing.

This module provides temporal pattern analysis for audio generation tasks,
including rhythm, tempo, and time-domain feature extraction.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import ExtensionConfig
from artifex.generative_models.extensions.base import ModelExtension


class TemporalAnalysis(ModelExtension):
    """Temporal pattern analysis for audio generation."""

    def __init__(
        self,
        config: ExtensionConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize temporal analysis module.

        Args:
            config: Extension configuration with temporal parameters:
                - weight: Weight for the extension (default: 1.0)
                - enabled: Whether the extension is enabled (default: True)
                - extensions.temporal.sample_rate: Audio sample rate in Hz
                - extensions.temporal.frame_length: Frame length for analysis
                - extensions.temporal.hop_length: Hop length between frames
            rngs: Random number generator keys
        """
        # Handle configuration
        if not isinstance(config, ExtensionConfig):
            raise TypeError(f"config must be ExtensionConfig, got {type(config).__name__}")

        super().__init__(config, rngs=rngs)

        # Get temporal parameters from extensions field
        temporal_params = getattr(config, "extensions", {}).get("temporal", {})

        self.sample_rate = temporal_params.get("sample_rate", 22050)
        self.frame_length = temporal_params.get("frame_length", 2048)
        self.hop_length = temporal_params.get("hop_length", self.frame_length // 4)
        self.rngs = rngs

    def compute_zero_crossing_rate(self, audio: jax.Array) -> jax.Array:
        """Compute zero crossing rate.

        Args:
            audio: Audio signal [length] or [batch, length]

        Returns:
            Zero crossing rate [frames] or [batch, frames]
        """
        if audio.ndim == 1:
            return self._zcr_single(audio)
        else:
            return jax.vmap(self._zcr_single)(audio)

    def _zcr_single(self, audio: jax.Array) -> jax.Array:
        """Compute ZCR for a single audio signal."""
        # Compute sign changes
        sign_changes = jnp.diff(jnp.sign(audio))
        zero_crossings = jnp.abs(sign_changes) > 0

        # Frame the zero crossings
        n_frames = (len(audio) - self.frame_length) // self.hop_length + 1
        zcr_frames = []

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.frame_length - 1  # -1 because diff reduces length by 1
            if end < len(zero_crossings):
                frame_zcr = jnp.mean(zero_crossings[start:end])
                zcr_frames.append(frame_zcr)

        return jnp.array(zcr_frames)

    def compute_energy(self, audio: jax.Array) -> jax.Array:
        """Compute short-time energy.

        Args:
            audio: Audio signal [length] or [batch, length]

        Returns:
            Energy [frames] or [batch, frames]
        """
        if audio.ndim == 1:
            return self._energy_single(audio)
        else:
            return jax.vmap(self._energy_single)(audio)

    def _energy_single(self, audio: jax.Array) -> jax.Array:
        """Compute energy for a single audio signal."""
        n_frames = (len(audio) - self.frame_length) // self.hop_length + 1
        energy_frames = []

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start : start + self.frame_length]
            frame_energy = jnp.sum(frame**2)
            energy_frames.append(frame_energy)

        return jnp.array(energy_frames)

    def compute_rms(self, audio: jax.Array) -> jax.Array:
        """Compute root mean square energy.

        Args:
            audio: Audio signal [length] or [batch, length]

        Returns:
            RMS energy [frames] or [batch, frames]
        """
        energy = self.compute_energy(audio)
        rms = jnp.sqrt(energy / self.frame_length)
        return rms

    def estimate_tempo(
        self, audio: jax.Array, tempo_range: tuple[float, float] = (60, 200)
    ) -> jax.Array:
        """Estimate tempo using onset detection and autocorrelation.

        Args:
            audio: Audio signal [length] or [batch, length]
            tempo_range: Range of tempos to consider (BPM)

        Returns:
            Estimated tempo in BPM [scalar] or [batch]
        """
        if audio.ndim == 1:
            return self._tempo_single(audio, tempo_range)

        # Use nnx.vmap for NNX module compatibility (self is captured correctly)
        def tempo_fn(x: jax.Array) -> jax.Array:
            return self._tempo_single(x, tempo_range)

        return nnx.vmap(tempo_fn, in_axes=0)(audio)

    def _tempo_single(self, audio: jax.Array, tempo_range: tuple[float, float]) -> jax.Array:
        """Estimate tempo for a single audio signal."""
        # Compute onset strength
        onset_strength = self.compute_onset_strength(audio)

        # Compute autocorrelation
        autocorr = jnp.correlate(onset_strength, onset_strength, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        # Convert tempo range to lag range
        min_tempo, max_tempo = tempo_range
        max_lag = int(60 * self.sample_rate / self.hop_length / min_tempo)
        min_lag = int(60 * self.sample_rate / self.hop_length / max_tempo)

        # Find peak in autocorrelation within tempo range
        if max_lag < len(autocorr) and min_lag < max_lag:
            valid_autocorr = autocorr[min_lag:max_lag]
            best_lag = jnp.argmax(valid_autocorr) + min_lag

            # Convert lag to tempo
            tempo = 60 * self.sample_rate / self.hop_length / best_lag
        else:
            # Default tempo if range is invalid
            tempo = 120.0

        return jnp.array(tempo)

    def compute_onset_strength(self, audio: jax.Array) -> jax.Array:
        """Compute onset strength function.

        Args:
            audio: Audio signal [length] or [batch, length]

        Returns:
            Onset strength [frames] or [batch, frames]
        """
        if audio.ndim == 1:
            return self._onset_strength_single(audio)
        else:
            return jax.vmap(self._onset_strength_single)(audio)

    def _onset_strength_single(self, audio: jax.Array) -> jax.Array:
        """Compute onset strength for a single audio signal."""
        # Compute spectrogram (simplified)
        n_frames = (len(audio) - self.frame_length) // self.hop_length + 1
        spectral_frames = []

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start : start + self.frame_length]

            # Apply window and FFT
            windowed = frame * jnp.hanning(len(frame))
            spectrum = jnp.abs(jnp.fft.fft(windowed))
            magnitude = spectrum[: len(spectrum) // 2]
            spectral_frames.append(magnitude)

        spectral_frames = jnp.stack(spectral_frames)  # [frames, freq_bins]

        # Compute spectral flux (onset strength)
        spectral_diff = jnp.diff(spectral_frames, axis=0)
        onset_strength = jnp.sum(jnp.maximum(spectral_diff, 0), axis=1)

        # Pad to match original frame count
        onset_strength = jnp.concatenate([jnp.array([0.0]), onset_strength])

        return onset_strength

    def detect_beats(self, audio: jax.Array, tempo: jax.Array | None = None) -> jax.Array:
        """Detect beat positions.

        Args:
            audio: Audio signal [length] or [batch, length]
            tempo: Known tempo (optional) [scalar] or [batch]

        Returns:
            Beat positions in samples [n_beats] or [batch, n_beats]
        """
        if audio.ndim == 1:
            estimated_tempo = tempo if tempo is not None else self.estimate_tempo(audio)
            return self._detect_beats_single(audio, estimated_tempo)
        else:
            if tempo is None:
                tempo = self.estimate_tempo(audio)
            return jax.vmap(self._detect_beats_single)(audio, tempo)

    def _detect_beats_single(self, audio: jax.Array, tempo: jax.Array) -> jax.Array:
        """Detect beats for a single audio signal."""
        # Compute onset strength
        onset_strength = self.compute_onset_strength(audio)

        # Expected beat period in frames
        beat_period = 60 * self.sample_rate / self.hop_length / tempo

        # Simple peak picking with expected period
        peaks = []
        last_beat = -beat_period

        for i, strength in enumerate(onset_strength):
            if (i - last_beat) >= beat_period * 0.5:  # Minimum gap between beats
                # Check if this is a local maximum
                start_idx = max(0, i - 2)
                end_idx = min(len(onset_strength), i + 3)
                local_region = onset_strength[start_idx:end_idx]

                if strength == jnp.max(local_region) and strength > jnp.mean(onset_strength):
                    peaks.append(i * self.hop_length)  # Convert to samples
                    last_beat = i

        return jnp.array(peaks)

    def compute_rhythm_features(self, audio: jax.Array) -> dict[str, jax.Array]:
        """Compute comprehensive rhythm features.

        Args:
            audio: Audio signal [length] or [batch, length]

        Returns:
            Dictionary of rhythm features
        """
        features = {}

        # Tempo estimation
        features["tempo"] = self.estimate_tempo(audio)

        # Onset detection
        features["onset_strength"] = self.compute_onset_strength(audio)

        # Beat detection
        features["beats"] = self.detect_beats(audio, features["tempo"])

        return features

    def compute_temporal_features(self, audio: jax.Array) -> dict[str, jax.Array]:
        """Extract comprehensive temporal features.

        Args:
            audio: Audio signal [length] or [batch, length]

        Returns:
            Dictionary of temporal features
        """
        features = {}

        # Basic temporal features
        features["zero_crossing_rate"] = self.compute_zero_crossing_rate(audio)
        features["energy"] = self.compute_energy(audio)
        features["rms"] = self.compute_rms(audio)

        # Rhythm features
        rhythm_features = self.compute_rhythm_features(audio)
        features.update(rhythm_features)

        return features

    def compute_pulse_clarity(self, audio: jax.Array) -> jax.Array:
        """Compute pulse clarity measure.

        Args:
            audio: Audio signal [length] or [batch, length]

        Returns:
            Pulse clarity [scalar] or [batch]
        """
        onset_strength = self.compute_onset_strength(audio)

        # Compute autocorrelation
        autocorr = jnp.correlate(onset_strength, onset_strength, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        # Pulse clarity is the ratio of the maximum autocorrelation
        # to the mean autocorrelation (excluding the zero lag)
        if len(autocorr) > 1:
            max_corr = jnp.max(autocorr[1:])  # Exclude zero lag
            mean_corr = jnp.mean(autocorr[1:])
            pulse_clarity = max_corr / (mean_corr + 1e-8)
        else:
            pulse_clarity = 0.0

        return jnp.array(pulse_clarity)

    def segment_by_energy(self, audio: jax.Array, threshold: float = 0.1) -> list[tuple[int, int]]:
        """Segment audio by energy thresholding.

        Args:
            audio: Audio signal [length]
            threshold: Energy threshold (relative to max energy)

        Returns:
            List of (start, end) sample indices for segments
        """
        energy = self.compute_energy(audio)

        # Normalize energy
        max_energy = jnp.max(energy)
        normalized_energy = energy / (max_energy + 1e-8)

        # Find segments above threshold
        above_threshold = normalized_energy > threshold

        # Find start and end points
        segments = []
        in_segment = False
        start_frame = 0

        for i, above in enumerate(above_threshold):
            if above and not in_segment:
                start_frame = i
                in_segment = True
            elif not above and in_segment:
                end_frame = i
                # Convert frame indices to sample indices
                start_sample = start_frame * self.hop_length
                end_sample = end_frame * self.hop_length
                segments.append((start_sample, end_sample))
                in_segment = False

        # Handle case where audio ends while in a segment
        if in_segment:
            end_sample = len(audio)
            start_sample = start_frame * self.hop_length
            segments.append((start_sample, end_sample))

        return segments

    def __call__(self, inputs: Any, model_outputs: Any, **kwargs: Any) -> dict[str, Any]:
        """Process model inputs/outputs.

        Args:
            inputs: Original inputs to the model.
            model_outputs: Outputs from the model.
            **kwargs: Additional keyword arguments.

        Returns:
            dictionary of extension outputs including temporal features.
        """
        if not self.enabled:
            return {"extension_type": "temporal_analysis"}

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
                "extension_type": "temporal_analysis",
                "error": "No audio data found in model outputs",
            }

        # Extract comprehensive temporal features
        features = self.compute_temporal_features(audio)

        # Add additional temporal analysis results
        features["pulse_clarity"] = self.compute_pulse_clarity(audio)

        # Segment audio by energy
        if audio.ndim == 1:  # Only for single audio, not batch
            segments = self.segment_by_energy(audio)
            features["energy_segments"] = segments

        return {
            "temporal_features": features,
            "extension_type": "temporal_analysis",
        }
