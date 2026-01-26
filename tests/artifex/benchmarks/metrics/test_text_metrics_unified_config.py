"""Test text metrics with unified configuration system.

Following TDD principles - write tests first, then implement.
"""

import pytest
from flax import nnx

from artifex.generative_models.core.configuration import EvaluationConfig


class TestTextMetricsUnifiedConfig:
    """Test text metrics with new unified configuration system."""

    @pytest.fixture
    def rngs(self):
        """Create test RNGs."""
        return nnx.Rngs(42)

    @pytest.fixture
    def test_texts(self):
        """Create test text data."""
        real_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
        ]
        generated_texts = [
            "The fast brown fox leaps over the sleepy dog.",
            "A journey of thousand miles starts with one step.",
            "To be or not be, that is question.",
        ]
        return real_texts, generated_texts

    def test_bleu_metric_requires_evaluation_config(self, rngs):
        """Test that BLEU metric requires EvaluationConfig."""
        from artifex.benchmarks.metrics.text import BLEUMetric

        # Should only accept EvaluationConfig
        config = EvaluationConfig(
            name="bleu_metric",
            metrics=["bleu"],
            metric_params={
                "bleu": {
                    "max_n": 4,
                    "smooth": True,
                    "weights": [0.25, 0.25, 0.25, 0.25],
                }
            },
            eval_batch_size=32,
        )

        # This should work
        metric = BLEUMetric(rngs=rngs, config=config)
        assert metric.config == config
        assert metric.max_n == 4
        assert metric.smooth

        # This should NOT work - no backward compatibility
        with pytest.raises(TypeError, match="config must be an EvaluationConfig"):
            BLEUMetric(rngs=rngs, config={"name": "bleu"})

    def test_bleu_computation(self, rngs, test_texts):
        """Test BLEU metric computation with typed config."""
        from artifex.benchmarks.metrics.text import BLEUMetric

        config = EvaluationConfig(
            name="bleu_test",
            metrics=["bleu"],
            metric_params={
                "bleu": {
                    "max_n": 4,
                    "smooth": True,
                }
            },
            eval_batch_size=16,
        )

        metric = BLEUMetric(rngs=rngs, config=config)
        real_texts, generated_texts = test_texts

        result = metric.compute(real_texts, generated_texts)

        assert "bleu_score" in result
        assert 0 <= result["bleu_score"] <= 1
        assert isinstance(result["bleu_score"], float)

    def test_rouge_metric_requires_evaluation_config(self, rngs):
        """Test that ROUGE metric requires EvaluationConfig."""
        from artifex.benchmarks.metrics.text import ROUGEMetric

        config = EvaluationConfig(
            name="rouge_metric",
            metrics=["rouge"],
            metric_params={
                "rouge": {
                    "rouge_types": ["rouge1", "rouge2", "rougeL"],
                    "use_stemmer": True,
                }
            },
            eval_batch_size=32,
        )

        # This should work
        metric = ROUGEMetric(rngs=rngs, config=config)
        assert metric.config == config
        assert "rouge1" in metric.rouge_types

        # No dict allowed
        with pytest.raises(TypeError):
            ROUGEMetric(rngs=rngs, config={"name": "rouge"})

    def test_perplexity_metric_requires_evaluation_config(self, rngs):
        """Test that Perplexity metric requires EvaluationConfig."""
        from artifex.benchmarks.metrics.text import PerplexityMetric

        config = EvaluationConfig(
            name="perplexity_metric",
            metrics=["perplexity"],
            metric_params={
                "perplexity": {
                    "model_name": "gpt2",
                    "use_mock": True,  # For testing
                }
            },
            eval_batch_size=8,
        )

        metric = PerplexityMetric(rngs=rngs, config=config)
        assert metric.config == config
        assert metric.use_mock

    def test_diversity_metric_with_config(self, rngs, test_texts):
        """Test Diversity metric with evaluation config."""
        from artifex.benchmarks.metrics.text import DiversityMetric

        config = EvaluationConfig(
            name="diversity_metric",
            metrics=["diversity"],
            metric_params={
                "diversity": {
                    "n_gram_sizes": [1, 2, 3],
                    "measure_self_bleu": True,
                }
            },
            eval_batch_size=32,
        )

        metric = DiversityMetric(rngs=rngs, config=config)
        _, generated_texts = test_texts

        result = metric.compute(generated_texts, generated_texts)

        assert "diversity_score" in result
        assert 0 <= result["diversity_score"] <= 1

    def test_text_metric_factory_functions(self, rngs):
        """Test text metric factory functions."""
        from artifex.benchmarks.metrics.text import (
            create_bleu_metric,
            create_diversity_metric,
            create_perplexity_metric,
            create_rouge_metric,
        )

        # BLEU factory
        bleu = create_bleu_metric(rngs=rngs, max_n=4, smooth=True, batch_size=64)
        assert isinstance(bleu.config, EvaluationConfig)
        assert bleu.config.eval_batch_size == 64
        assert bleu.max_n == 4

        # ROUGE factory
        rouge = create_rouge_metric(rngs=rngs, rouge_types=["rouge1", "rougeL"], batch_size=32)
        assert rouge.config.eval_batch_size == 32
        assert "rouge1" in rouge.rouge_types

        # Perplexity factory
        perplexity = create_perplexity_metric(
            rngs=rngs, model_name="gpt2", use_mock=True, batch_size=16
        )
        assert perplexity.config.eval_batch_size == 16
        assert perplexity.model_name == "gpt2"

        # Diversity factory
        diversity = create_diversity_metric(rngs=rngs, n_gram_sizes=[1, 2], batch_size=32)
        assert diversity.config.eval_batch_size == 32
        assert diversity.n_gram_sizes == [1, 2]

    def test_text_metrics_inherit_from_base(self, rngs):
        """Test that all text metrics inherit from MetricBase."""
        from artifex.benchmarks.metrics.core import MetricBase
        from artifex.benchmarks.metrics.text import BLEUMetric

        config = EvaluationConfig(
            name="test_inheritance", metrics=["bleu"], metric_params={"bleu": {}}
        )

        bleu = BLEUMetric(rngs=rngs, config=config)
        assert isinstance(bleu, MetricBase)

        # All should have required methods
        assert hasattr(bleu, "compute")
        assert hasattr(bleu, "validate_inputs")
        assert hasattr(bleu, "rngs")

    def test_config_factory_creates_named_config(self, rngs):
        """Test that text metric factory creates properly named config."""
        from artifex.benchmarks.metrics.text import create_bleu_metric

        # Create with unique name
        metric = create_bleu_metric(rngs=rngs, config_name="test_bleu_registration")

        # Verify the config was created with the correct name
        assert metric.config.name == "test_bleu_registration"
        assert "bleu" in metric.config.metrics

    def test_validation_inputs_for_text_metrics(self, rngs, test_texts):
        """Test input validation for text metrics."""
        from artifex.benchmarks.metrics.text import BLEUMetric

        config = EvaluationConfig(
            name="validation_test", metrics=["bleu"], metric_params={"bleu": {}}
        )

        metric = BLEUMetric(rngs=rngs, config=config)
        real_texts, generated_texts = test_texts

        # Valid inputs
        assert metric.validate_inputs(real_texts, generated_texts)

        # Invalid inputs - empty lists
        assert not metric.validate_inputs([], [])

        # Invalid inputs - mismatched lengths
        assert not metric.validate_inputs(real_texts, generated_texts[:1])

        # Invalid inputs - non-string data
        assert not metric.validate_inputs([1, 2, 3], ["text"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
