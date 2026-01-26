"""Test metrics with unified configuration system.

Following TDD principles - write tests first, then implement.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import EvaluationConfig


class TestImageMetricsUnifiedConfig:
    """Test image metrics with new unified configuration system."""

    @pytest.fixture
    def rngs(self):
        """Create test RNGs."""
        return nnx.Rngs(42)

    @pytest.fixture
    def test_images(self):
        """Create test image data."""
        batch_size = 4
        real_images = jnp.ones((batch_size, 32, 32, 3))
        generated_images = jnp.ones((batch_size, 32, 32, 3)) * 0.9
        return real_images, generated_images

    def test_fid_metric_requires_evaluation_config(self, rngs):
        """Test that FID metric requires EvaluationConfig."""
        from artifex.benchmarks.metrics.image import FIDMetric

        # Should only accept EvaluationConfig, not dict
        config = EvaluationConfig(
            name="fid_metric",
            metrics=["fid"],
            metric_params={
                "fid": {
                    "mock_inception": True,
                    "higher_is_better": False,
                }
            },
            eval_batch_size=64,
        )

        # This should work
        metric = FIDMetric(rngs=rngs, config=config)
        assert metric.config == config
        assert metric.eval_batch_size == 64

        # This should NOT work anymore - no backward compatibility
        with pytest.raises(TypeError, match="config must be an EvaluationConfig"):
            FIDMetric(rngs=rngs, config={"name": "fid"})

    def test_fid_metric_computation(self, rngs, test_images):
        """Test FID metric computation with typed config."""
        from artifex.benchmarks.metrics.image import FIDMetric

        config = EvaluationConfig(
            name="fid_test",
            metrics=["fid"],
            metric_params={
                "fid": {
                    "mock_inception": True,
                    "higher_is_better": False,
                }
            },
            eval_batch_size=32,
        )

        metric = FIDMetric(rngs=rngs, config=config)
        real_images, generated_images = test_images

        result = metric.compute(real_images, generated_images)

        assert "fid_score" in result
        assert isinstance(result["fid_score"], float)
        assert result["fid_score"] >= 0

    def test_lpips_metric_requires_evaluation_config(self, rngs):
        """Test that LPIPS metric requires EvaluationConfig."""
        from artifex.benchmarks.metrics.image import LPIPSMetric

        config = EvaluationConfig(
            name="lpips_metric",
            metrics=["lpips"],
            metric_params={
                "lpips": {
                    "mock_implementation": True,
                    "higher_is_better": False,
                }
            },
            eval_batch_size=16,
        )

        # This should work
        metric = LPIPSMetric(rngs=rngs, config=config)
        assert metric.config == config

        # No dict configs allowed
        with pytest.raises(TypeError):
            LPIPSMetric(rngs=rngs, config={"name": "lpips"})

    def test_metric_factory_functions(self, rngs, test_images):
        """Test metric factory functions."""
        from artifex.benchmarks.metrics.image import create_fid_metric, create_lpips_metric

        # Factory functions should return properly configured metrics
        fid_metric = create_fid_metric(rngs=rngs, mock_inception=True, batch_size=128)

        assert isinstance(fid_metric.config, EvaluationConfig)
        assert fid_metric.config.eval_batch_size == 128
        assert "fid" in fid_metric.config.metrics

        lpips_metric = create_lpips_metric(rngs=rngs, mock_implementation=True, batch_size=64)

        assert isinstance(lpips_metric.config, EvaluationConfig)
        assert lpips_metric.config.eval_batch_size == 64
        assert "lpips" in lpips_metric.config.metrics

    def test_ssim_metric_with_config(self, rngs, test_images):
        """Test SSIM metric with evaluation config."""
        from artifex.benchmarks.metrics.image import SSIMMetric

        config = EvaluationConfig(
            name="ssim_metric",
            metrics=["ssim"],
            metric_params={
                "ssim": {
                    "higher_is_better": True,
                    "window_size": 11,
                }
            },
            eval_batch_size=32,
        )

        metric = SSIMMetric(rngs=rngs, config=config)
        real_images, generated_images = test_images

        result = metric.compute(real_images, generated_images)

        assert "ssim_score" in result
        assert 0 <= result["ssim_score"] <= 1

    def test_is_metric_with_config(self, rngs, test_images):
        """Test Inception Score metric with evaluation config."""
        from artifex.benchmarks.metrics.image import ISMetric

        config = EvaluationConfig(
            name="is_metric",
            metrics=["inception_score"],
            metric_params={
                "inception_score": {
                    "mock_inception": True,
                    "splits": 10,
                    "higher_is_better": True,
                }
            },
            eval_batch_size=32,
        )

        metric = ISMetric(rngs=rngs, config=config)
        real_images, generated_images = test_images

        result = metric.compute(real_images, generated_images)

        assert "inception_score" in result
        assert result["inception_score"] > 0

    def test_config_factory_creates_named_config(self, rngs):
        """Test that metric factory creates properly named config."""
        from artifex.benchmarks.metrics.image import create_fid_metric

        # Create a metric with a unique name
        metric = create_fid_metric(rngs=rngs, config_name="test_fid_config")

        # Verify the config was created with the correct name
        assert metric.config.name == "test_fid_config"
        assert "fid" in metric.config.metrics

    def test_metric_base_integration(self, rngs):
        """Test that metrics properly integrate with MetricBase."""
        from artifex.benchmarks.metrics.core import MetricBase
        from artifex.benchmarks.metrics.image import FIDMetric

        config = EvaluationConfig(
            name="fid_base_test", metrics=["fid"], metric_params={"fid": {"mock_inception": True}}
        )

        metric = FIDMetric(rngs=rngs, config=config)

        # Should be instance of MetricBase
        assert isinstance(metric, MetricBase)

        # Should have required attributes from base
        assert hasattr(metric, "compute")
        assert hasattr(metric, "validate_inputs")
        assert hasattr(metric, "rngs")


class TestTextMetricsUnifiedConfig:
    """Test text metrics with unified configuration."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def test_texts(self):
        """Create test text data."""
        real_texts = ["This is a test.", "Another test sentence."]
        generated_texts = ["This is test.", "Another sentence test."]
        return real_texts, generated_texts

    def test_bleu_metric_requires_config(self, rngs):
        """Test BLEU metric with unified config."""
        from artifex.benchmarks.metrics.text import BLEUMetric

        config = EvaluationConfig(
            name="bleu_metric",
            metrics=["bleu"],
            metric_params={
                "bleu": {
                    "max_n": 4,
                    "smooth": True,
                }
            },
        )

        metric = BLEUMetric(rngs=rngs, config=config)
        assert metric.config == config

        # No dict allowed
        with pytest.raises(TypeError):
            BLEUMetric(rngs=rngs, config={"name": "bleu"})


class TestAudioMetricsUnifiedConfig:
    """Test audio metrics with unified configuration."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_spectral_metric_requires_config(self, rngs):
        """Test spectral metric with unified config."""
        from artifex.benchmarks.metrics.audio import SpectralMetric

        config = EvaluationConfig(
            name="spectral_metric",
            metrics=["spectral_distance"],
            metric_params={
                "spectral_distance": {
                    "sample_rate": 16000,
                    "n_fft": 2048,
                }
            },
        )

        metric = SpectralMetric(rngs=rngs, config=config)
        assert isinstance(metric.config, EvaluationConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
