"""Tests for evaluation system with unified configuration."""

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from artifex.benchmarks.core import BenchmarkBase, BenchmarkRunner, PerformanceTracker
from artifex.generative_models.core.configuration import EvaluationConfig
from artifex.generative_models.core.evaluation.metrics.pipeline import EvaluationPipeline


def _test_feature_extractor(images):
    means = jnp.mean(images, axis=(1, 2, 3))
    stds = jnp.std(images, axis=(1, 2, 3))
    return jnp.stack([means, stds], axis=1)


def _test_classifier(images):
    means = jnp.mean(images, axis=(1, 2, 3), keepdims=True)
    return jnp.tile(means, (1, 10))


def _test_language_model(inputs):
    return jnp.full(inputs.shape, -0.5)


class TestEvaluationPipeline:
    """Test EvaluationPipeline with unified configuration."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def evaluation_config(self):
        return EvaluationConfig(
            name="test_eval",
            metrics=["image:fid", "image:is", "text:perplexity"],
            metric_params={
                "fid": {"feature_extractor": _test_feature_extractor},
                "is": {"classifier": _test_classifier},
                "perplexity": {"model": _test_language_model},
            },
            eval_batch_size=16,
            save_predictions=True,
        )

    def test_init_with_typed_config(self, evaluation_config, rngs):
        pipeline = EvaluationPipeline(evaluation_config, rngs=rngs)
        assert pipeline.config == evaluation_config
        assert "image" in pipeline.metrics
        assert "text" in pipeline.metrics
        assert len(pipeline.metrics["image"]) == 2
        assert len(pipeline.metrics["text"]) == 1

    def test_init_rejects_non_config(self, rngs):
        with pytest.raises(TypeError, match="config must be EvaluationConfig"):
            EvaluationPipeline("invalid", rngs=rngs)

    def test_init_rejects_none(self, rngs):
        with pytest.raises(TypeError, match="config must be EvaluationConfig"):
            EvaluationPipeline(None, rngs=rngs)

    def test_dict_config_is_rejected(self, rngs):
        dict_config = {
            "modalities": ["image", "text"],
            "image_metrics": ["fid", "is"],
            "text_metrics": ["perplexity"],
        }
        with pytest.raises(TypeError, match="config must be EvaluationConfig"):
            EvaluationPipeline(dict_config, rngs=rngs)

    def test_evaluate_with_typed_config(self, evaluation_config, rngs):
        pipeline = EvaluationPipeline(evaluation_config, rngs=rngs)
        results = pipeline.evaluate(
            {
                "image": {
                    "real": jnp.ones((10, 16, 16, 3)),
                    "generated": jnp.ones((10, 16, 16, 3)) * 0.9,
                },
                "text": {"inputs": jnp.ones((10, 16), dtype=jnp.int32)},
            }
        )
        assert "image" in results
        assert "text" in results
        assert "fid" in results["image"]
        assert "is_mean" in results["image"]
        assert "perplexity" in results["text"]

    def test_pipeline_rejects_missing_dependencies(self, rngs):
        with pytest.raises(ValueError, match="feature_extractor"):
            EvaluationPipeline(EvaluationConfig(name="bad_fid", metrics=["image:fid"]), rngs=rngs)
        with pytest.raises(ValueError, match="classifier"):
            EvaluationPipeline(EvaluationConfig(name="bad_is", metrics=["image:is"]), rngs=rngs)
        with pytest.raises(ValueError, match="model"):
            EvaluationPipeline(
                EvaluationConfig(name="bad_ppl", metrics=["text:perplexity"]), rngs=rngs
            )

    def test_pipeline_requires_prefixed_supported_metrics(self, rngs):
        with pytest.raises(ValueError, match="modality:metric"):
            EvaluationPipeline(
                EvaluationConfig(
                    name="bad_spec",
                    metrics=["fid"],
                    metric_params={"fid": {"feature_extractor": _test_feature_extractor}},
                ),
                rngs=rngs,
            )
        with pytest.raises(ValueError, match="Unsupported evaluation metric spec"):
            EvaluationPipeline(
                EvaluationConfig(
                    name="unsupported",
                    metrics=["text:bleu"],
                    metric_params={"bleu": {"model": _test_language_model}},
                ),
                rngs=rngs,
            )

    def test_metric_params_from_typed_config(self, rngs):
        config = EvaluationConfig(
            name="test_with_params",
            metrics=["image:fid"],
            metric_params={"fid": {"feature_extractor": _test_feature_extractor, "batch_size": 64}},
        )
        pipeline = EvaluationPipeline(config, rngs=rngs)
        assert callable(pipeline.config.metric_params["fid"]["feature_extractor"])
        assert pipeline.config.metric_params["fid"]["batch_size"] == 64


class TestPerformanceTracker:
    """Test PerformanceTracker with unified configuration."""

    def test_init_with_evaluation_config(self):
        config = EvaluationConfig(
            name="test_tracker",
            metrics=["accuracy", "latency_ms"],
            metric_params={"target_metrics": {"accuracy": 0.95, "latency_ms": 100}},
        )
        tracker = PerformanceTracker(config)
        assert tracker.config == config

    def test_init_rejects_non_evaluation_config(self):
        config = {"target_metrics": {"accuracy": 0.95, "latency_ms": 100}}
        with pytest.raises(TypeError, match="config must be EvaluationConfig"):
            PerformanceTracker(config)

    def test_track_and_check_targets(self):
        config = EvaluationConfig(
            name="test_track",
            metrics=["accuracy", "latency_ms"],
            metric_params={"target_metrics": {"accuracy": 0.90, "latency_ms": 50}},
        )
        tracker = PerformanceTracker(config)
        tracker.track_metrics({"accuracy": 0.92, "latency_ms": 45}, step=1)
        assert tracker.check_target_achievement() is True
        tracker.track_metrics({"accuracy": 0.85, "latency_ms": 60}, step=2)
        assert tracker.check_target_achievement() is False


class MockBenchmark(BenchmarkBase):
    """Mock benchmark for testing."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def _setup_benchmark_components(self):
        pass

    def run_training(self):
        return {"training_loss": 0.1, "training_time": 100}

    def run_evaluation(self):
        return {"accuracy": 0.95, "latency_ms": 50}

    def get_performance_targets(self):
        return {"accuracy": 0.90, "latency_ms": 100}


class TestBenchmarkRunner:
    """Test BenchmarkRunner with unified configuration."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def mock_benchmark(self, rngs):
        config = EvaluationConfig(
            name="test_benchmark",
            metrics=["accuracy"],
            metric_params={"target_metrics": {"accuracy": 0.90}},
        )
        return MockBenchmark(config, rngs=rngs)

    def test_runner_with_benchmark(self, mock_benchmark):
        runner = BenchmarkRunner(mock_benchmark)
        assert runner.benchmark == mock_benchmark
        assert runner.results_history == []

    def test_run_full_benchmark(self, mock_benchmark):
        runner = BenchmarkRunner(mock_benchmark)
        results = runner.run_full_benchmark()
        assert "training_results" in results
        assert "evaluation_results" in results
        assert "targets_achieved" in results
        assert results["targets_achieved"] is True
        assert runner.get_run_count() == 1

    def test_compare_performance(self, mock_benchmark):
        runner = BenchmarkRunner(mock_benchmark)
        runner.run_full_benchmark()
        runner.run_full_benchmark()
        comparison = runner.compare_performance()
        assert comparison["num_runs"] == 2
        assert "runs_summary" in comparison
        assert len(comparison["runs_summary"]) == 2
        assert "metrics_summary" in comparison

    def test_clear_history(self, mock_benchmark):
        runner = BenchmarkRunner(mock_benchmark)
        runner.run_full_benchmark()
        assert runner.get_run_count() == 1
        runner.clear_history()
        assert runner.get_run_count() == 0
        assert runner.get_latest_results() is None
