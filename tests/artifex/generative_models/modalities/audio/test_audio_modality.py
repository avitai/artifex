"""Tests for audio modality implementation."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.modalities.audio import (
    AudioModality,
    AudioModalityConfig,
    AudioRepresentation,
    compute_audio_metrics,
    create_audio_modality,
    SyntheticAudioDataset,
)


@pytest.fixture
def rngs():
    """Random number generators for testing."""
    return nnx.Rngs(42)


@pytest.fixture
def audio_config():
    """Default audio configuration for testing."""
    return AudioModalityConfig(
        representation=AudioRepresentation.RAW_WAVEFORM,
        sample_rate=16000,
        duration=1.0,  # Short duration for tests
    )


class TestAudioModalityConfig:
    """Test audio modality configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AudioModalityConfig()

        assert config.representation == AudioRepresentation.RAW_WAVEFORM
        assert config.sample_rate == 16000
        assert config.n_mel_channels == 80
        assert config.hop_length == 256
        assert config.n_fft == 1024
        assert config.duration == 2.0
        assert config.normalize is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AudioModalityConfig(
            representation=AudioRepresentation.MEL_SPECTROGRAM,
            sample_rate=22050,
            duration=3.0,
            normalize=False,
        )

        assert config.representation == AudioRepresentation.MEL_SPECTROGRAM
        assert config.sample_rate == 22050
        assert config.duration == 3.0
        assert config.normalize is False


class TestAudioModality:
    """Test audio modality base class."""

    def test_initialization(self, audio_config, rngs):
        """Test audio modality initialization."""
        modality = AudioModality(config=audio_config, rngs=rngs)

        assert modality.config == audio_config
        assert modality.n_time_steps == 16000  # 1 second at 16kHz
        assert modality.n_time_frames == 16000 // 256  # hop_length = 256

    def test_output_shape_waveform(self, rngs):
        """Test output shape for waveform representation."""
        config = AudioModalityConfig(
            representation=AudioRepresentation.RAW_WAVEFORM,
            duration=1.0,
        )
        modality = AudioModality(config=config, rngs=rngs)

        expected_shape = (16000,)  # 1 second at 16kHz
        assert modality.output_shape == expected_shape

    def test_output_shape_mel_spectrogram(self, rngs):
        """Test output shape for mel-spectrogram representation."""
        config = AudioModalityConfig(
            representation=AudioRepresentation.MEL_SPECTROGRAM,
            duration=1.0,
            n_mel_channels=80,
            hop_length=256,
        )
        modality = AudioModality(config=config, rngs=rngs)

        n_frames = 16000 // 256  # duration * sample_rate / hop_length
        expected_shape = (80, n_frames)
        assert modality.output_shape == expected_shape

    def test_generate_waveform(self, audio_config, rngs):
        """Test generation of waveform audio."""
        modality = AudioModality(config=audio_config, rngs=rngs)

        generated = modality.generate(n_samples=2, rngs=rngs)

        assert generated.shape == (2, 16000)  # 2 samples, 1 second each
        assert jnp.isfinite(generated).all()
        assert jnp.abs(generated).max() <= 1.0  # Should be normalized

    def test_generate_mel_spectrogram(self, rngs):
        """Test generation of mel-spectrogram audio."""
        config = AudioModalityConfig(
            representation=AudioRepresentation.MEL_SPECTROGRAM,
            duration=1.0,
        )
        modality = AudioModality(config=config, rngs=rngs)

        generated = modality.generate(n_samples=2, rngs=rngs)

        n_frames = 16000 // 256
        assert generated.shape == (2, 80, n_frames)
        assert jnp.isfinite(generated).all()

    def test_generate_custom_duration(self, audio_config, rngs):
        """Test generation with custom duration."""
        modality = AudioModality(config=audio_config, rngs=rngs)

        generated = modality.generate(n_samples=1, duration=0.5, rngs=rngs)

        assert generated.shape == (1, 8000)  # 0.5 seconds at 16kHz

    def test_loss_fn(self, audio_config, rngs):
        """Test loss function computation."""
        modality = AudioModality(config=audio_config, rngs=rngs)

        # Create mock batch and model outputs
        batch_size = 4
        audio_length = 16000

        batch = {"audio": jax.random.normal(rngs.sample(), (batch_size, audio_length))}

        model_outputs = {"audio": jax.random.normal(rngs.sample(), (batch_size, audio_length))}

        loss = modality.loss_fn(batch, model_outputs)

        assert isinstance(loss, float) or jnp.isscalar(loss)
        assert jnp.isfinite(loss)
        assert loss >= 0.0  # MSE loss should be non-negative


class TestCreateAudioModality:
    """Test audio modality factory function."""

    def test_create_with_defaults(self, rngs):
        """Test creating audio modality with default parameters."""
        modality = create_audio_modality(rngs=rngs)

        assert isinstance(modality, AudioModality)
        assert modality.config.representation == AudioRepresentation.RAW_WAVEFORM
        assert modality.config.sample_rate == 16000
        assert modality.config.duration == 2.0

    def test_create_with_string_representation(self, rngs):
        """Test creating with string representation."""
        modality = create_audio_modality(representation="mel_spectrogram", rngs=rngs)

        assert modality.config.representation == AudioRepresentation.MEL_SPECTROGRAM

    def test_create_with_custom_params(self, rngs):
        """Test creating with custom parameters."""
        modality = create_audio_modality(
            representation=AudioRepresentation.STFT,
            sample_rate=22050,
            duration=3.0,
            normalize=False,
            rngs=rngs,
        )

        assert modality.config.representation == AudioRepresentation.STFT
        assert modality.config.sample_rate == 22050
        assert modality.config.duration == 3.0
        assert modality.config.normalize is False


class TestSyntheticAudioDataset:
    """Test synthetic audio dataset."""

    def test_dataset_creation(self, audio_config):
        """Test synthetic dataset creation."""
        dataset = SyntheticAudioDataset(
            config=audio_config, n_samples=10, audio_types=["sine", "noise"]
        )

        assert len(dataset) == 10
        assert dataset.name == "SyntheticAudioDataset"

    def test_dataset_item_access(self, audio_config):
        """Test accessing dataset items."""
        dataset = SyntheticAudioDataset(
            config=audio_config,
            n_samples=5,
        )

        item = dataset[0]

        assert isinstance(item, dict)
        assert "audio" in item
        assert "audio_type" in item
        assert "sample_rate" in item
        assert "duration" in item

        assert item["audio"].shape == (16000,)  # 1 second at 16kHz
        assert item["sample_rate"] == 16000
        assert item["duration"] == 1.0

    def test_dataset_collate(self, audio_config):
        """Test dataset collation for batching."""
        dataset = SyntheticAudioDataset(
            config=audio_config,
            n_samples=5,
        )

        batch = [dataset[i] for i in range(3)]
        collated = dataset.collate_fn(batch)

        assert "audio" in collated
        assert collated["audio"].shape == (3, 16000)  # Batch of 3 samples
        assert jnp.isfinite(collated["audio"]).all()


class TestAudioMetrics:
    """Test audio evaluation metrics."""

    def test_compute_audio_metrics(self, audio_config):
        """Test audio metrics computation."""
        # Generate test audio
        key = jax.random.key(42)
        generated = jax.random.normal(key, (2, 16000))

        metrics = compute_audio_metrics(generated_audio=generated, config=audio_config)

        assert isinstance(metrics, dict)
        assert "temporal_coherence" in metrics
        assert "harmonic_quality" in metrics
        assert "sample_diversity" in metrics
        assert "audio_quality_index" in metrics

        # Check that all values are finite
        for key, value in metrics.items():
            assert jnp.isfinite(value), f"Metric {key} is not finite: {value}"

    def test_metrics_with_reference(self, audio_config):
        """Test metrics computation with reference audio."""
        key = jax.random.key(42)
        generated = jax.random.normal(key, (2, 16000))
        reference = jax.random.normal(jax.random.split(key)[1], (2, 16000))

        metrics = compute_audio_metrics(
            generated_audio=generated, reference_audio=reference, config=audio_config
        )

        # Should include reference-based metrics
        assert "spectral_convergence" in metrics
        assert "log_spectral_distance" in metrics

        for key, value in metrics.items():
            assert jnp.isfinite(value), f"Metric {key} is not finite: {value}"


if __name__ == "__main__":
    pytest.main([__file__])
