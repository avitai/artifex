"""Test audio metrics with unified configuration system.

Following TDD principles - write tests first, then implement.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import EvaluationConfig


class TestAudioMetricsUnifiedConfig:
    """Test audio metrics with new unified configuration system."""

    @pytest.fixture
    def rngs(self):
        """Create test RNGs."""
        return nnx.Rngs(42)

    @pytest.fixture
    def test_audio(self):
        """Create test audio data."""
        # Simulated audio data (batch_size=3, time_length=8000)
        real_audio = jnp.sin(2 * jnp.pi * 440 * jnp.arange(8000) / 16000)  # 440 Hz sine wave
        real_audio = jnp.stack([real_audio] * 3)  # Batch of 3

        # Slightly different frequency for generated audio
        generated_audio = jnp.sin(2 * jnp.pi * 445 * jnp.arange(8000) / 16000)  # 445 Hz
        generated_audio = jnp.stack([generated_audio] * 3)

        return real_audio, generated_audio

    def test_spectral_metric_requires_evaluation_config(self, rngs):
        """Test that Spectral metric requires EvaluationConfig."""
        from artifex.benchmarks.metrics.audio import SpectralMetric

        # Should only accept EvaluationConfig
        config = EvaluationConfig(
            name="spectral_metric",
            metrics=["spectral"],
            metric_params={
                "spectral": {
                    "n_fft": 1024,
                    "hop_length": 256,
                    "higher_is_better": True,
                }
            },
            eval_batch_size=32,
        )

        # This should work
        metric = SpectralMetric(rngs=rngs, config=config)
        assert metric.config == config
        assert metric.n_fft == 1024
        assert metric.hop_length == 256

        # This should NOT work - no backward compatibility
        with pytest.raises(TypeError, match="config must be an EvaluationConfig"):
            SpectralMetric(rngs=rngs, config={"name": "spectral"})

    def test_spectral_computation(self, rngs, test_audio):
        """Test Spectral metric computation with typed config."""
        from artifex.benchmarks.metrics.audio import SpectralMetric

        config = EvaluationConfig(
            name="spectral_test",
            metrics=["spectral"],
            metric_params={
                "spectral": {
                    "n_fft": 512,
                    "hop_length": 128,
                }
            },
            eval_batch_size=16,
        )

        metric = SpectralMetric(rngs=rngs, config=config)
        real_audio, generated_audio = test_audio

        result = metric.compute(real_audio, generated_audio)

        assert "spectral_convergence" in result
        assert isinstance(result["spectral_convergence"], float)
        assert result["spectral_convergence"] >= 0

    def test_mcd_metric_requires_evaluation_config(self, rngs):
        """Test that MCD metric requires EvaluationConfig."""
        from artifex.benchmarks.metrics.audio import MelCepstralMetric

        config = EvaluationConfig(
            name="mcd_metric",
            metrics=["mcd"],
            metric_params={
                "mcd": {
                    "n_mels": 80,
                    "sr": 16000,
                    "n_fft": 1024,
                    "higher_is_better": False,
                }
            },
            eval_batch_size=32,
        )

        # This should work
        metric = MelCepstralMetric(rngs=rngs, config=config)
        assert metric.config == config
        assert metric.n_mels == 80
        assert metric.sr == 16000

        # No dict allowed
        with pytest.raises(TypeError):
            MelCepstralMetric(rngs=rngs, config={"name": "mcd"})

    def test_audio_metric_factory_functions(self, rngs):
        """Test audio metric factory functions."""
        from artifex.benchmarks.metrics.audio import create_mcd_metric, create_spectral_metric

        # Spectral factory
        spectral = create_spectral_metric(rngs=rngs, n_fft=2048, hop_length=512, batch_size=64)
        assert isinstance(spectral.config, EvaluationConfig)
        assert spectral.config.eval_batch_size == 64
        assert spectral.n_fft == 2048

        # MCD factory
        mcd = create_mcd_metric(rngs=rngs, n_mels=128, sr=22050, batch_size=32)
        assert mcd.config.eval_batch_size == 32
        assert mcd.n_mels == 128
        assert mcd.sr == 22050

    def test_audio_metrics_inherit_from_base(self, rngs):
        """Test that all audio metrics inherit from MetricBase."""
        from artifex.benchmarks.metrics.audio import SpectralMetric
        from artifex.benchmarks.metrics.core import MetricBase

        config = EvaluationConfig(
            name="test_inheritance", metrics=["spectral"], metric_params={"spectral": {}}
        )

        spectral = SpectralMetric(rngs=rngs, config=config)
        assert isinstance(spectral, MetricBase)

        # All should have required methods
        assert hasattr(spectral, "compute")
        assert hasattr(spectral, "validate_inputs")
        assert hasattr(spectral, "rngs")

    def test_validation_inputs_for_audio_metrics(self, rngs, test_audio):
        """Test input validation for audio metrics."""
        from artifex.benchmarks.metrics.audio import SpectralMetric

        config = EvaluationConfig(
            name="validation_test", metrics=["spectral"], metric_params={"spectral": {}}
        )

        metric = SpectralMetric(rngs=rngs, config=config)
        real_audio, generated_audio = test_audio

        # Valid inputs
        assert metric.validate_inputs(real_audio, generated_audio)

        # Invalid inputs - not arrays
        assert not metric.validate_inputs([1, 2, 3], generated_audio)

        # Invalid inputs - shape mismatch
        assert not metric.validate_inputs(real_audio, generated_audio[:, :1000])

        # Invalid inputs - wrong dimensions
        assert not metric.validate_inputs(real_audio[0], generated_audio[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
