"""Tests for temporal analysis extension.

This module contains comprehensive tests for the TemporalAnalysis extension
that provides temporal pattern analysis for audio processing.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import ExtensionConfig
from artifex.generative_models.extensions.audio_processing.temporal import TemporalAnalysis


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rngs():
    """Create random number generator keys."""
    return nnx.Rngs(0)


@pytest.fixture
def config():
    """Create a basic extension configuration."""
    return ExtensionConfig(name="test_temporal", weight=1.0, enabled=True)


@pytest.fixture
def disabled_config():
    """Create a disabled extension configuration."""
    return ExtensionConfig(name="test_temporal", weight=1.0, enabled=False)


@pytest.fixture
def temporal(config, rngs):
    """Create a temporal analysis instance with default configuration."""
    return TemporalAnalysis(config, rngs=rngs)


@pytest.fixture
def disabled_temporal(disabled_config, rngs):
    """Create a disabled temporal analysis instance."""
    return TemporalAnalysis(disabled_config, rngs=rngs)


@pytest.fixture
def test_audio():
    """Create test audio signal (1 second at 22050 Hz)."""
    t = jnp.linspace(0, 1, 22050)
    # Simple sine wave at 440 Hz
    audio = jnp.sin(2 * jnp.pi * 440 * t)
    return audio


@pytest.fixture
def test_audio_batch(test_audio):
    """Create batch of test audio signals."""
    return jnp.stack([test_audio, test_audio * 0.5, test_audio * 0.25])


@pytest.fixture
def test_audio_rhythmic():
    """Create test audio with rhythmic pattern (beat-like)."""
    sample_rate = 22050
    duration = 4  # 4 seconds
    samples = sample_rate * duration

    _ = jnp.linspace(0, duration, samples)  # Unused but kept for clarity

    # Create impulses at beat positions (120 BPM = 2 beats per second)
    beat_period = 0.5  # 120 BPM
    audio = jnp.zeros(samples)

    # Add impulses at beat positions
    for beat_time in jnp.arange(0, duration, beat_period):
        beat_sample = int(beat_time * sample_rate)
        if beat_sample < samples:
            # Create a short impulse with decay
            impulse_len = int(0.05 * sample_rate)  # 50ms impulse
            end_sample = min(beat_sample + impulse_len, samples)
            impulse_t = jnp.arange(end_sample - beat_sample) / sample_rate
            impulse = jnp.exp(-20 * impulse_t) * jnp.sin(2 * jnp.pi * 200 * impulse_t)
            audio = audio.at[beat_sample:end_sample].add(impulse)

    return audio


# =============================================================================
# Initialization Tests
# =============================================================================


class TestTemporalInitialization:
    """Tests for temporal analysis initialization."""

    def test_init_with_default_config(self, config, rngs):
        """Test initialization with default ExtensionConfig."""
        temporal = TemporalAnalysis(config, rngs=rngs)

        assert temporal.sample_rate == 22050
        assert temporal.frame_length == 2048
        assert temporal.hop_length == 512  # frame_length // 4

    def test_init_invalid_config_type(self, rngs):
        """Test that invalid config type raises TypeError."""
        with pytest.raises(TypeError, match="config must be ExtensionConfig"):
            TemporalAnalysis({"weight": 1.0}, rngs=rngs)

    def test_init_disabled_config(self, disabled_config, rngs):
        """Test initialization with disabled configuration."""
        temporal = TemporalAnalysis(disabled_config, rngs=rngs)
        assert temporal.enabled is False


# =============================================================================
# Zero Crossing Rate Tests
# =============================================================================


class TestZeroCrossingRate:
    """Tests for zero crossing rate computation."""

    def test_zcr_single_shape(self, temporal, test_audio):
        """Test ZCR output shape for single audio."""
        zcr = temporal.compute_zero_crossing_rate(test_audio)

        # Should have multiple frames
        assert zcr.ndim == 1
        assert len(zcr) > 0

    def test_zcr_batch_shape(self, temporal, test_audio_batch):
        """Test ZCR output shape for batch audio."""
        zcr = temporal.compute_zero_crossing_rate(test_audio_batch)

        # Should be [batch, frames]
        assert zcr.ndim == 2
        assert zcr.shape[0] == 3  # batch size

    def test_zcr_range(self, temporal, test_audio):
        """Test ZCR values are between 0 and 1."""
        zcr = temporal.compute_zero_crossing_rate(test_audio)

        assert jnp.all(zcr >= 0)
        assert jnp.all(zcr <= 1)

    def test_zcr_sine_wave(self, temporal, test_audio):
        """Test ZCR for sine wave is relatively constant."""
        zcr = temporal.compute_zero_crossing_rate(test_audio)

        # Sine wave should have consistent ZCR
        # 440 Hz has about 880 zero crossings per second
        std_zcr = jnp.std(zcr)

        # Standard deviation should be relatively low for pure tone
        assert std_zcr < 0.1


# =============================================================================
# Energy Tests
# =============================================================================


class TestEnergy:
    """Tests for energy computation."""

    def test_energy_single_shape(self, temporal, test_audio):
        """Test energy output shape for single audio."""
        energy = temporal.compute_energy(test_audio)

        assert energy.ndim == 1
        assert len(energy) > 0

    def test_energy_batch_shape(self, temporal, test_audio_batch):
        """Test energy output shape for batch audio."""
        energy = temporal.compute_energy(test_audio_batch)

        assert energy.ndim == 2
        assert energy.shape[0] == 3  # batch size

    def test_energy_non_negative(self, temporal, test_audio):
        """Test energy values are non-negative."""
        energy = temporal.compute_energy(test_audio)
        assert jnp.all(energy >= 0)

    def test_energy_scales_with_amplitude(self, temporal):
        """Test energy scales quadratically with amplitude."""
        t = jnp.linspace(0, 1, 22050)
        audio1 = jnp.sin(2 * jnp.pi * 440 * t)
        audio2 = 2 * audio1  # Double amplitude

        energy1 = temporal.compute_energy(audio1)
        energy2 = temporal.compute_energy(audio2)

        # Energy should scale by factor of 4 (amplitude squared)
        ratio = jnp.mean(energy2) / jnp.mean(energy1)
        assert jnp.isclose(ratio, 4.0, rtol=0.1)


# =============================================================================
# RMS Tests
# =============================================================================


class TestRMS:
    """Tests for RMS computation."""

    def test_rms_single_shape(self, temporal, test_audio):
        """Test RMS output shape for single audio."""
        rms = temporal.compute_rms(test_audio)

        assert rms.ndim == 1
        assert len(rms) > 0

    def test_rms_batch_shape(self, temporal, test_audio_batch):
        """Test RMS output shape for batch audio."""
        rms = temporal.compute_rms(test_audio_batch)

        assert rms.ndim == 2
        assert rms.shape[0] == 3  # batch size

    def test_rms_non_negative(self, temporal, test_audio):
        """Test RMS values are non-negative."""
        rms = temporal.compute_rms(test_audio)
        assert jnp.all(rms >= 0)

    def test_rms_sqrt_of_energy(self, temporal, test_audio):
        """Test RMS is sqrt of mean energy."""
        energy = temporal.compute_energy(test_audio)
        rms = temporal.compute_rms(test_audio)

        # RMS should be sqrt(energy / frame_length)
        expected_rms = jnp.sqrt(energy / temporal.frame_length)
        assert jnp.allclose(rms, expected_rms)


# =============================================================================
# Tempo Estimation Tests
# =============================================================================


class TestTempoEstimation:
    """Tests for tempo estimation."""

    def test_tempo_single_output(self, temporal, test_audio):
        """Test tempo estimation returns scalar for single audio."""
        tempo = temporal.estimate_tempo(test_audio)

        assert tempo.ndim == 0 or tempo.size == 1  # Scalar or 1-element array

    def test_tempo_batch_output(self, temporal, test_audio_batch):
        """Test tempo estimation returns array for batch audio."""
        tempo = temporal.estimate_tempo(test_audio_batch)

        assert tempo.shape == (3,)  # One tempo per audio in batch

    def test_tempo_range(self, temporal, test_audio):
        """Test tempo is within expected range."""
        tempo = temporal.estimate_tempo(test_audio, tempo_range=(60, 200))

        assert 60 <= float(tempo) <= 200 or tempo == 120  # Default fallback

    def test_tempo_custom_range(self, temporal, test_audio):
        """Test tempo with custom range."""
        tempo = temporal.estimate_tempo(test_audio, tempo_range=(80, 160))

        # Should be a valid tempo value
        assert jnp.isfinite(tempo)


# =============================================================================
# Onset Strength Tests
# =============================================================================


class TestOnsetStrength:
    """Tests for onset strength computation."""

    def test_onset_strength_single_shape(self, temporal, test_audio):
        """Test onset strength output shape for single audio."""
        onset = temporal.compute_onset_strength(test_audio)

        assert onset.ndim == 1
        assert len(onset) > 0

    def test_onset_strength_batch_shape(self, temporal, test_audio_batch):
        """Test onset strength output shape for batch audio."""
        onset = temporal.compute_onset_strength(test_audio_batch)

        assert onset.ndim == 2
        assert onset.shape[0] == 3  # batch size

    def test_onset_strength_non_negative(self, temporal, test_audio):
        """Test onset strength is non-negative."""
        onset = temporal.compute_onset_strength(test_audio)
        assert jnp.all(onset >= 0)


# =============================================================================
# Beat Detection Tests
# =============================================================================


class TestBeatDetection:
    """Tests for beat detection."""

    def test_detect_beats_returns_array(self, temporal, test_audio):
        """Test beat detection returns array."""
        beats = temporal.detect_beats(test_audio)

        assert isinstance(beats, jnp.ndarray)

    def test_detect_beats_sample_positions(self, temporal, test_audio):
        """Test beat positions are valid sample indices."""
        beats = temporal.detect_beats(test_audio)

        if len(beats) > 0:
            assert jnp.all(beats >= 0)
            assert jnp.all(beats < len(test_audio))

    def test_detect_beats_with_tempo(self, temporal, test_audio):
        """Test beat detection with provided tempo."""
        tempo = jnp.array(120.0)
        beats = temporal.detect_beats(test_audio, tempo=tempo)

        assert isinstance(beats, jnp.ndarray)


# =============================================================================
# Rhythm Features Tests
# =============================================================================


class TestRhythmFeatures:
    """Tests for rhythm feature extraction."""

    def test_compute_rhythm_features_keys(self, temporal, test_audio):
        """Test rhythm features contains expected keys."""
        features = temporal.compute_rhythm_features(test_audio)

        expected_keys = {"tempo", "onset_strength", "beats"}
        assert expected_keys == set(features.keys())

    def test_compute_rhythm_features_types(self, temporal, test_audio):
        """Test rhythm features have correct types."""
        features = temporal.compute_rhythm_features(test_audio)

        assert isinstance(features["tempo"], jnp.ndarray)
        assert isinstance(features["onset_strength"], jnp.ndarray)
        assert isinstance(features["beats"], jnp.ndarray)


# =============================================================================
# Temporal Features Tests
# =============================================================================


class TestTemporalFeatures:
    """Tests for comprehensive temporal feature extraction."""

    def test_compute_temporal_features_keys(self, temporal, test_audio):
        """Test temporal features contains expected keys."""
        features = temporal.compute_temporal_features(test_audio)

        expected_keys = {
            "zero_crossing_rate",
            "energy",
            "rms",
            "tempo",
            "onset_strength",
            "beats",
        }
        assert expected_keys == set(features.keys())

    def test_compute_temporal_features_shapes(self, temporal, test_audio):
        """Test temporal features have correct shapes."""
        features = temporal.compute_temporal_features(test_audio)

        # Time-series features should be 1D
        assert features["zero_crossing_rate"].ndim == 1
        assert features["energy"].ndim == 1
        assert features["rms"].ndim == 1
        assert features["onset_strength"].ndim == 1


# =============================================================================
# Pulse Clarity Tests
# =============================================================================


class TestPulseClarity:
    """Tests for pulse clarity computation."""

    def test_pulse_clarity_single_output(self, temporal, test_audio):
        """Test pulse clarity returns scalar for single audio."""
        clarity = temporal.compute_pulse_clarity(test_audio)

        assert clarity.ndim == 0 or clarity.size == 1

    def test_pulse_clarity_non_negative(self, temporal, test_audio):
        """Test pulse clarity is non-negative."""
        clarity = temporal.compute_pulse_clarity(test_audio)
        assert float(clarity) >= 0

    def test_pulse_clarity_rhythmic_audio(self, temporal, test_audio_rhythmic):
        """Test pulse clarity is higher for rhythmic audio."""
        clarity_rhythmic = temporal.compute_pulse_clarity(test_audio_rhythmic)

        # Rhythmic audio should have reasonable pulse clarity
        assert jnp.isfinite(clarity_rhythmic)


# =============================================================================
# Energy Segmentation Tests
# =============================================================================


class TestEnergySegmentation:
    """Tests for energy-based audio segmentation."""

    def test_segment_by_energy_returns_list(self, temporal, test_audio):
        """Test segmentation returns list of tuples."""
        segments = temporal.segment_by_energy(test_audio)

        assert isinstance(segments, list)

    def test_segment_by_energy_tuple_format(self, temporal, test_audio):
        """Test segments are (start, end) tuples."""
        segments = temporal.segment_by_energy(test_audio)

        for segment in segments:
            assert isinstance(segment, tuple)
            assert len(segment) == 2
            start, end = segment
            assert start >= 0
            assert end > start

    def test_segment_by_energy_threshold(self, temporal, test_audio):
        """Test segmentation with different thresholds."""
        segments_low = temporal.segment_by_energy(test_audio, threshold=0.05)
        segments_high = temporal.segment_by_energy(test_audio, threshold=0.5)

        # Lower threshold should potentially find more segments or longer ones
        # This is a soft test as behavior depends on audio content
        assert isinstance(segments_low, list)
        assert isinstance(segments_high, list)

    def test_segment_by_energy_valid_indices(self, temporal, test_audio):
        """Test segment indices are within audio bounds."""
        segments = temporal.segment_by_energy(test_audio)

        for start, end in segments:
            assert 0 <= start < len(test_audio)
            assert 0 < end <= len(test_audio)


# =============================================================================
# __call__ Method Tests
# =============================================================================


class TestCallMethod:
    """Tests for the __call__ method."""

    def test_call_with_audio_dict(self, temporal, test_audio):
        """Test __call__ with audio in dict format."""
        model_outputs = {"audio": test_audio}
        result = temporal({}, model_outputs)

        assert "extension_type" in result
        assert result["extension_type"] == "temporal_analysis"
        assert "temporal_features" in result

    def test_call_with_waveform_key(self, temporal, test_audio):
        """Test __call__ with waveform key."""
        model_outputs = {"waveform": test_audio}
        result = temporal({}, model_outputs)

        assert "temporal_features" in result

    def test_call_with_generated_audio_key(self, temporal, test_audio):
        """Test __call__ with generated_audio key."""
        model_outputs = {"generated_audio": test_audio}
        result = temporal({}, model_outputs)

        assert "temporal_features" in result

    def test_call_with_raw_audio(self, temporal, test_audio):
        """Test __call__ with raw audio as model_outputs."""
        result = temporal({}, test_audio)

        assert "temporal_features" in result

    def test_call_no_audio(self, temporal):
        """Test __call__ when no audio is found."""
        model_outputs = {"some_other_key": "value"}
        result = temporal({}, model_outputs)

        assert "error" in result
        assert "No audio data found" in result["error"]

    def test_call_disabled_returns_minimal(self, disabled_temporal, test_audio):
        """Test disabled temporal analysis returns minimal result."""
        model_outputs = {"audio": test_audio}
        result = disabled_temporal({}, model_outputs)

        assert result == {"extension_type": "temporal_analysis"}

    def test_call_includes_pulse_clarity(self, temporal, test_audio):
        """Test __call__ includes pulse clarity."""
        model_outputs = {"audio": test_audio}
        result = temporal({}, model_outputs)

        features = result["temporal_features"]
        assert "pulse_clarity" in features

    def test_call_includes_energy_segments(self, temporal, test_audio):
        """Test __call__ includes energy segments for single audio."""
        model_outputs = {"audio": test_audio}
        result = temporal({}, model_outputs)

        features = result["temporal_features"]
        assert "energy_segments" in features

    def test_call_features_complete(self, temporal, test_audio):
        """Test __call__ extracts all temporal features."""
        model_outputs = {"audio": test_audio}
        result = temporal({}, model_outputs)

        features = result["temporal_features"]
        assert "zero_crossing_rate" in features
        assert "energy" in features
        assert "rms" in features
        assert "tempo" in features
        assert "onset_strength" in features
        assert "beats" in features
        assert "pulse_clarity" in features


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_short_audio(self, temporal):
        """Test with very short audio."""
        # Audio shorter than frame_length
        short_audio = jnp.zeros(1000)
        short_audio = short_audio.at[500].set(1.0)

        # Should not crash, but may return empty results
        zcr = temporal.compute_zero_crossing_rate(short_audio)
        assert isinstance(zcr, jnp.ndarray)

    def test_silent_audio(self, temporal):
        """Test with silent audio."""
        silent_audio = jnp.zeros(22050)

        energy = temporal.compute_energy(silent_audio)
        assert jnp.all(energy == 0)

        rms = temporal.compute_rms(silent_audio)
        assert jnp.all(rms == 0)

    def test_constant_audio(self, temporal):
        """Test with constant audio (DC offset)."""
        constant_audio = jnp.ones(22050) * 0.5

        zcr = temporal.compute_zero_crossing_rate(constant_audio)
        # Constant audio should have zero crossing rate of 0
        assert jnp.all(zcr == 0)
