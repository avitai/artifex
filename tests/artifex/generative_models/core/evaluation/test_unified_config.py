"""Tests for the benchmark runner and tracker with unified configuration."""

import flax.nnx as nnx
import pytest

from artifex.benchmarks.core import BenchmarkBase, BenchmarkRunner, PerformanceTracker
from artifex.generative_models.core.configuration import EvaluationConfig


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
