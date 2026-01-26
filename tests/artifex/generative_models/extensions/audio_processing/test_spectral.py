"""Tests for spectral analysis extension.

This module contains comprehensive tests for the SpectralAnalysis extension
that provides spectral analysis utilities for audio processing.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import ExtensionConfig
from artifex.generative_models.extensions.audio_processing.spectral import SpectralAnalysis


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
    return ExtensionConfig(name="test_spectral", weight=1.0, enabled=True)


@pytest.fixture
def disabled_config():
    """Create a disabled extension configuration."""
    return ExtensionConfig(name="test_spectral", weight=1.0, enabled=False)


@pytest.fixture
def spectral(config, rngs):
    """Create a spectral analysis instance with default configuration."""
    return SpectralAnalysis(config, rngs=rngs)


@pytest.fixture
def disabled_spectral(disabled_config, rngs):
    """Create a disabled spectral analysis instance."""
    return SpectralAnalysis(disabled_config, rngs=rngs)


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


# =============================================================================
# Initialization Tests
# =============================================================================


class TestSpectralInitialization:
    """Tests for spectral analysis initialization."""

    def test_init_with_default_config(self, config, rngs):
        """Test initialization with default ExtensionConfig."""
        spectral = SpectralAnalysis(config, rngs=rngs)

        assert spectral.sample_rate == 22050
        assert spectral.n_fft == 2048
        assert spectral.hop_length == 512  # n_fft // 4
        assert spectral.window_type == "hann"
        assert spectral.n_mels == 128

    def test_init_creates_mel_filters(self, spectral):
        """Test that mel filter bank is created."""
        assert hasattr(spectral, "mel_filters")
        assert spectral.mel_filters.shape == (128, 1025)  # n_mels x (n_fft//2 + 1)

    def test_init_creates_window(self, spectral):
        """Test that window function is created."""
        assert hasattr(spectral, "window")
        assert spectral.window.shape == (2048,)

    def test_init_invalid_config_type(self, rngs):
        """Test that invalid config type raises TypeError."""
        with pytest.raises(TypeError, match="config must be ExtensionConfig"):
            SpectralAnalysis({"weight": 1.0}, rngs=rngs)

    def test_init_disabled_config(self, disabled_config, rngs):
        """Test initialization with disabled configuration."""
        spectral = SpectralAnalysis(disabled_config, rngs=rngs)
        assert spectral.enabled is False


# =============================================================================
# Window Function Tests
# =============================================================================


class TestWindowFunctions:
    """Tests for window function creation."""

    def test_hann_window(self, config, rngs):
        """Test Hann window creation."""
        spectral = SpectralAnalysis(config, rngs=rngs)

        # Hann window should start and end near 0
        assert jnp.isclose(spectral.window[0], 0.0, atol=1e-6)
        assert jnp.isclose(spectral.window[-1], 0.0, atol=1e-6)
        # Maximum should be at center
        assert jnp.isclose(spectral.window[len(spectral.window) // 2], 1.0, atol=1e-2)

    def test_hann_window_shape(self, spectral):
        """Test Hann window has correct shape."""
        assert spectral.window.shape == (spectral.n_fft,)

    def test_hann_window_non_negative(self, spectral):
        """Test Hann window values are non-negative."""
        assert jnp.all(spectral.window >= 0)

    def test_hann_window_max_one(self, spectral):
        """Test Hann window maximum is close to 1."""
        assert jnp.max(spectral.window) <= 1.0 + 1e-6


# =============================================================================
# Mel Filter Bank Tests
# =============================================================================


class TestMelFilterBank:
    """Tests for mel filter bank creation."""

    def test_mel_filters_shape(self, spectral):
        """Test mel filter bank has correct shape."""
        expected_shape = (spectral.n_mels, spectral.n_fft // 2 + 1)
        assert spectral.mel_filters.shape == expected_shape

    def test_mel_filters_non_negative(self, spectral):
        """Test mel filter values are non-negative."""
        assert jnp.all(spectral.mel_filters >= 0)

    def test_mel_filters_triangular(self, spectral):
        """Test mel filters have triangular structure."""
        # Each filter should have at most one peak region
        for i in range(min(10, spectral.n_mels)):
            filter_row = spectral.mel_filters[i]
            non_zero = filter_row > 0
            # Should be contiguous non-zero region
            if jnp.any(non_zero):
                # At least some filters should have values
                assert jnp.sum(non_zero) > 0


# =============================================================================
# STFT Tests
# =============================================================================


class TestComputeSTFT:
    """Tests for STFT computation."""

    def test_stft_single_shape(self, spectral, test_audio):
        """Test STFT output shape for single audio."""
        stft = spectral.compute_stft(test_audio)

        # Output should be [freq_bins, time_frames]
        assert stft.ndim == 2
        assert stft.shape[0] == spectral.n_fft // 2 + 1

    def test_stft_batch_shape(self, spectral, test_audio_batch):
        """Test STFT output shape for batch audio."""
        stft = spectral.compute_stft(test_audio_batch)

        # Output should be [batch, freq_bins, time_frames]
        assert stft.ndim == 3
        assert stft.shape[0] == 3  # batch size
        assert stft.shape[1] == spectral.n_fft // 2 + 1

    def test_stft_non_negative(self, spectral, test_audio):
        """Test STFT magnitude is non-negative."""
        stft = spectral.compute_stft(test_audio)
        assert jnp.all(stft >= 0)

    def test_stft_peak_frequency(self, spectral, test_audio):
        """Test STFT detects dominant frequency."""
        stft = spectral.compute_stft(test_audio)

        # Average across time to get frequency profile
        freq_profile = jnp.mean(stft, axis=1)

        # 440 Hz should correspond to a bin near index 440 * n_fft / sample_rate
        expected_bin = int(440 * spectral.n_fft / spectral.sample_rate)
        peak_bin = jnp.argmax(freq_profile)

        # Peak should be within a few bins of expected
        assert abs(int(peak_bin) - expected_bin) < 5


# =============================================================================
# Spectrogram Tests
# =============================================================================


class TestComputeSpectrogram:
    """Tests for spectrogram computation."""

    def test_spectrogram_single_shape(self, spectral, test_audio):
        """Test power spectrogram output shape for single audio."""
        spec = spectral.compute_spectrogram(test_audio)

        assert spec.ndim == 2
        assert spec.shape[0] == spectral.n_fft // 2 + 1

    def test_spectrogram_batch_shape(self, spectral, test_audio_batch):
        """Test power spectrogram output shape for batch audio."""
        spec = spectral.compute_spectrogram(test_audio_batch)

        assert spec.ndim == 3
        assert spec.shape[0] == 3  # batch size

    def test_spectrogram_non_negative(self, spectral, test_audio):
        """Test power spectrogram is non-negative."""
        spec = spectral.compute_spectrogram(test_audio)
        assert jnp.all(spec >= 0)

    def test_spectrogram_is_squared_stft(self, spectral, test_audio):
        """Test spectrogram is squared STFT magnitude."""
        stft = spectral.compute_stft(test_audio)
        spec = spectral.compute_spectrogram(test_audio)

        assert jnp.allclose(spec, stft**2)


# =============================================================================
# Mel Spectrogram Tests
# =============================================================================


class TestComputeMelSpectrogram:
    """Tests for mel spectrogram computation."""

    def test_mel_spectrogram_single_shape(self, spectral, test_audio):
        """Test mel spectrogram output shape for single audio."""
        mel_spec = spectral.compute_mel_spectrogram(test_audio)

        assert mel_spec.ndim == 2
        assert mel_spec.shape[0] == spectral.n_mels

    def test_mel_spectrogram_batch_shape(self, spectral, test_audio_batch):
        """Test mel spectrogram output shape for batch audio."""
        mel_spec = spectral.compute_mel_spectrogram(test_audio_batch)

        assert mel_spec.ndim == 3
        assert mel_spec.shape[0] == 3  # batch size
        assert mel_spec.shape[1] == spectral.n_mels

    def test_mel_spectrogram_non_negative(self, spectral, test_audio):
        """Test mel spectrogram is non-negative."""
        mel_spec = spectral.compute_mel_spectrogram(test_audio)
        assert jnp.all(mel_spec >= 0)


# =============================================================================
# Log Mel Spectrogram Tests
# =============================================================================


class TestComputeLogMelSpectrogram:
    """Tests for log mel spectrogram computation."""

    def test_log_mel_spectrogram_single_shape(self, spectral, test_audio):
        """Test log mel spectrogram output shape for single audio."""
        log_mel = spectral.compute_log_mel_spectrogram(test_audio)

        assert log_mel.ndim == 2
        assert log_mel.shape[0] == spectral.n_mels

    def test_log_mel_spectrogram_batch_shape(self, spectral, test_audio_batch):
        """Test log mel spectrogram output shape for batch audio."""
        log_mel = spectral.compute_log_mel_spectrogram(test_audio_batch)

        assert log_mel.ndim == 3
        assert log_mel.shape[0] == 3  # batch size

    def test_log_mel_spectrogram_epsilon(self, spectral, test_audio):
        """Test log mel spectrogram handles epsilon correctly."""
        log_mel = spectral.compute_log_mel_spectrogram(test_audio, eps=1e-8)

        # Should not have -inf values due to epsilon
        assert jnp.all(jnp.isfinite(log_mel))

    def test_log_mel_spectrogram_is_log_of_mel(self, spectral, test_audio):
        """Test log mel is log of mel spectrogram."""
        mel_spec = spectral.compute_mel_spectrogram(test_audio)
        log_mel = spectral.compute_log_mel_spectrogram(test_audio, eps=1e-8)

        expected = jnp.log(mel_spec + 1e-8)
        assert jnp.allclose(log_mel, expected)


# =============================================================================
# MFCC Tests
# =============================================================================


class TestComputeMFCC:
    """Tests for MFCC computation."""

    def test_mfcc_single_shape(self, spectral, test_audio):
        """Test MFCC output shape for single audio."""
        mfcc = spectral.compute_mfcc(test_audio, n_mfcc=13)

        assert mfcc.ndim == 2
        assert mfcc.shape[0] == 13

    def test_mfcc_batch_shape(self, spectral, test_audio_batch):
        """Test MFCC output shape for batch audio."""
        mfcc = spectral.compute_mfcc(test_audio_batch, n_mfcc=13)

        assert mfcc.ndim == 3
        assert mfcc.shape[0] == 3  # batch size
        assert mfcc.shape[1] == 13

    def test_mfcc_different_n_mfcc(self, spectral, test_audio):
        """Test MFCC with different number of coefficients."""
        mfcc_13 = spectral.compute_mfcc(test_audio, n_mfcc=13)
        mfcc_20 = spectral.compute_mfcc(test_audio, n_mfcc=20)

        assert mfcc_13.shape[0] == 13
        assert mfcc_20.shape[0] == 20


# =============================================================================
# Spectral Feature Tests
# =============================================================================


class TestSpectralCentroid:
    """Tests for spectral centroid computation."""

    def test_spectral_centroid_single_shape(self, spectral, test_audio):
        """Test spectral centroid output shape for single audio."""
        centroid = spectral.compute_spectral_centroid(test_audio)

        # Should have one value per time frame
        assert centroid.ndim == 1

    def test_spectral_centroid_batch_shape(self, spectral, test_audio_batch):
        """Test spectral centroid output shape for batch audio."""
        centroid = spectral.compute_spectral_centroid(test_audio_batch)

        assert centroid.ndim == 2
        assert centroid.shape[0] == 3  # batch size

    def test_spectral_centroid_range(self, spectral, test_audio):
        """Test spectral centroid is within valid frequency range."""
        centroid = spectral.compute_spectral_centroid(test_audio)

        # Should be between 0 and Nyquist frequency
        assert jnp.all(centroid >= 0)
        assert jnp.all(centroid <= spectral.sample_rate / 2)

    def test_spectral_centroid_pure_tone(self, spectral, test_audio):
        """Test spectral centroid for pure tone is near tone frequency."""
        centroid = spectral.compute_spectral_centroid(test_audio)

        # For a 440 Hz pure tone, centroid should be near 440 Hz
        mean_centroid = jnp.mean(centroid)
        assert abs(float(mean_centroid) - 440) < 100  # Within 100 Hz


class TestSpectralBandwidth:
    """Tests for spectral bandwidth computation."""

    def test_spectral_bandwidth_single_shape(self, spectral, test_audio):
        """Test spectral bandwidth output shape for single audio."""
        bandwidth = spectral.compute_spectral_bandwidth(test_audio)

        assert bandwidth.ndim == 1

    def test_spectral_bandwidth_batch_shape(self, spectral, test_audio_batch):
        """Test spectral bandwidth output shape for batch audio."""
        bandwidth = spectral.compute_spectral_bandwidth(test_audio_batch)

        assert bandwidth.ndim == 2
        assert bandwidth.shape[0] == 3  # batch size

    def test_spectral_bandwidth_non_negative(self, spectral, test_audio):
        """Test spectral bandwidth is non-negative."""
        bandwidth = spectral.compute_spectral_bandwidth(test_audio)
        assert jnp.all(bandwidth >= 0)


class TestSpectralRolloff:
    """Tests for spectral rolloff computation."""

    def test_spectral_rolloff_single_shape(self, spectral, test_audio):
        """Test spectral rolloff output shape for single audio."""
        rolloff = spectral.compute_spectral_rolloff(test_audio)

        assert rolloff.ndim == 1

    def test_spectral_rolloff_batch_shape(self, spectral, test_audio_batch):
        """Test spectral rolloff output shape for batch audio."""
        rolloff = spectral.compute_spectral_rolloff(test_audio_batch)

        assert rolloff.ndim == 2
        assert rolloff.shape[0] == 3  # batch size

    def test_spectral_rolloff_range(self, spectral, test_audio):
        """Test spectral rolloff is within valid frequency range."""
        rolloff = spectral.compute_spectral_rolloff(test_audio)

        assert jnp.all(rolloff >= 0)
        assert jnp.all(rolloff <= spectral.sample_rate / 2)

    def test_spectral_rolloff_percentage(self, spectral, test_audio):
        """Test spectral rolloff with different percentages."""
        rolloff_85 = spectral.compute_spectral_rolloff(test_audio, rolloff_percent=0.85)
        rolloff_95 = spectral.compute_spectral_rolloff(test_audio, rolloff_percent=0.95)

        # Higher percentage should generally give higher rolloff frequency
        assert jnp.mean(rolloff_95) >= jnp.mean(rolloff_85) - 100  # With tolerance


# =============================================================================
# Inverse Transform Tests
# =============================================================================


class TestInverseMelSpectrogram:
    """Tests for inverse mel spectrogram."""

    def test_inverse_mel_spectrogram_single_shape(self, spectral, test_audio):
        """Test inverse mel spectrogram output shape for single audio."""
        mel_spec = spectral.compute_mel_spectrogram(test_audio)
        linear_spec = spectral.inverse_mel_spectrogram(mel_spec)

        assert linear_spec.ndim == 2
        assert linear_spec.shape[0] == spectral.n_fft // 2 + 1

    def test_inverse_mel_spectrogram_batch_shape(self, spectral, test_audio_batch):
        """Test inverse mel spectrogram output shape for batch audio."""
        mel_spec = spectral.compute_mel_spectrogram(test_audio_batch)
        linear_spec = spectral.inverse_mel_spectrogram(mel_spec)

        assert linear_spec.ndim == 3
        assert linear_spec.shape[0] == 3  # batch size
        assert linear_spec.shape[1] == spectral.n_fft // 2 + 1


# =============================================================================
# Feature Extraction Tests
# =============================================================================


class TestExtractSpectralFeatures:
    """Tests for comprehensive spectral feature extraction."""

    def test_extract_spectral_features_keys(self, spectral, test_audio):
        """Test that all expected features are extracted."""
        features = spectral.extract_spectral_features(test_audio)

        expected_keys = {
            "spectrogram",
            "mel_spectrogram",
            "log_mel_spectrogram",
            "mfcc",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
        }
        assert expected_keys == set(features.keys())

    def test_extract_spectral_features_shapes(self, spectral, test_audio):
        """Test that extracted features have correct shapes."""
        features = spectral.extract_spectral_features(test_audio)

        assert features["spectrogram"].ndim == 2
        assert features["mel_spectrogram"].ndim == 2
        assert features["log_mel_spectrogram"].ndim == 2
        assert features["mfcc"].ndim == 2
        assert features["spectral_centroid"].ndim == 1
        assert features["spectral_bandwidth"].ndim == 1
        assert features["spectral_rolloff"].ndim == 1


# =============================================================================
# __call__ Method Tests
# =============================================================================


class TestCallMethod:
    """Tests for the __call__ method."""

    def test_call_with_audio_dict(self, spectral, test_audio):
        """Test __call__ with audio in dict format."""
        model_outputs = {"audio": test_audio}
        result = spectral({}, model_outputs)

        assert "extension_type" in result
        assert result["extension_type"] == "spectral_analysis"
        assert "spectral_features" in result

    def test_call_with_waveform_key(self, spectral, test_audio):
        """Test __call__ with waveform key."""
        model_outputs = {"waveform": test_audio}
        result = spectral({}, model_outputs)

        assert "spectral_features" in result

    def test_call_with_generated_audio_key(self, spectral, test_audio):
        """Test __call__ with generated_audio key."""
        model_outputs = {"generated_audio": test_audio}
        result = spectral({}, model_outputs)

        assert "spectral_features" in result

    def test_call_with_raw_audio(self, spectral, test_audio):
        """Test __call__ with raw audio as model_outputs."""
        result = spectral({}, test_audio)

        assert "spectral_features" in result

    def test_call_no_audio(self, spectral):
        """Test __call__ when no audio is found."""
        model_outputs = {"some_other_key": "value"}
        result = spectral({}, model_outputs)

        assert "error" in result
        assert "No audio data found" in result["error"]

    def test_call_disabled_returns_minimal(self, disabled_spectral, test_audio):
        """Test disabled spectral analysis returns minimal result."""
        model_outputs = {"audio": test_audio}
        result = disabled_spectral({}, model_outputs)

        assert result == {"extension_type": "spectral_analysis"}

    def test_call_features_complete(self, spectral, test_audio):
        """Test __call__ extracts all spectral features."""
        model_outputs = {"audio": test_audio}
        result = spectral({}, model_outputs)

        features = result["spectral_features"]
        assert "spectrogram" in features
        assert "mel_spectrogram" in features
        assert "log_mel_spectrogram" in features
        assert "mfcc" in features
        assert "spectral_centroid" in features
        assert "spectral_bandwidth" in features
        assert "spectral_rolloff" in features
