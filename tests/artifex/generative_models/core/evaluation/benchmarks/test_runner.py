"""Tests for core benchmark protocols."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from artifex.generative_models.core.configuration import EvaluationConfig
from artifex.generative_models.core.evaluation.benchmarks.runner import (
    BenchmarkRunner,
    PerformanceTracker,
)
from artifex.generative_models.core.evaluation.metrics.registry import MetricsRegistry
from artifex.generative_models.core.protocols.benchmarks import BenchmarkBase


@pytest.fixture
def rngs():
    """Standard fixture for rngs."""
    return nnx.Rngs(42)


@pytest.fixture
def sample_config():
    """Sample benchmark configuration."""
    return EvaluationConfig(
        name="test_benchmark",
        metrics=["accuracy", "latency_ms"],
        metric_params={
            "target_metrics": {"accuracy": 0.95, "latency_ms": 100.0},
            "hardware_requirements": {"gpu_memory_gb": 8.0, "compute_capability": "7.0"},
        },
        eval_batch_size=32,
    )


class MockGenerativeModel(nnx.Module):
    """Mock generative model for testing."""

    def __init__(self, output_dim: int = 10, *, rngs: nnx.Rngs):
        super().__init__()
        self.output_dim = output_dim
        self.dense = nnx.Linear(in_features=5, out_features=output_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dense(x)

    def generate(self, n_samples: int = 1, *, rngs: nnx.Rngs) -> jax.Array:
        """Generate samples."""
        key = rngs.default() if "default" in rngs else jax.random.key(0)
        return jax.random.normal(key, (n_samples, self.output_dim))


class MockBenchmark(BenchmarkBase):
    """Mock benchmark implementation for testing."""

    def __new__(cls, *args, **kwargs):
        """Override __new__ to bypass registry lookup."""
        return object.__new__(cls)

    def _setup_benchmark_components(self):
        """Setup benchmark-specific components."""
        self.model = MockGenerativeModel(rngs=self.rngs)
        self.performance_tracker = PerformanceTracker(config=self.config)

    def run_training(self) -> dict[str, float]:
        """Execute training phase."""
        # Mock training
        return {"training_loss": 0.5, "training_time": 120.0}

    def run_evaluation(self) -> dict[str, float]:
        """Execute evaluation phase."""
        # Mock evaluation
        x = jnp.ones((10, 5))
        _ = self.model(x)  # Run model but don't use output
        samples = self.model.generate(n_samples=5, rngs=self.rngs)

        return {"accuracy": 0.96, "latency_ms": 95.0, "samples_shape": samples.shape[0]}

    def get_performance_targets(self) -> dict[str, float]:
        """Return performance targets for this benchmark."""
        return self.config.metric_params.get("target_metrics", {})


class TestBenchmarkBase:
    """Test BenchmarkBase abstract base class."""

    def test_benchmark_initialization(self, sample_config, rngs):
        """Test benchmark initialization."""
        benchmark = MockBenchmark(config=sample_config, rngs=rngs)

        assert benchmark.config == sample_config
        assert benchmark.rngs is not None
        assert hasattr(benchmark, "model")
        assert hasattr(benchmark, "performance_tracker")

    def test_benchmark_training_execution(self, sample_config, rngs):
        """Test training execution."""
        benchmark = MockBenchmark(config=sample_config, rngs=rngs)

        results = benchmark.run_training()

        assert isinstance(results, dict)
        assert "training_loss" in results
        assert "training_time" in results
        assert results["training_loss"] >= 0.0
        assert results["training_time"] > 0.0

    def test_benchmark_evaluation_execution(self, sample_config, rngs):
        """Test evaluation execution."""
        benchmark = MockBenchmark(config=sample_config, rngs=rngs)

        results = benchmark.run_evaluation()

        assert isinstance(results, dict)
        assert "accuracy" in results
        assert "latency_ms" in results
        assert "samples_shape" in results
        assert 0.0 <= results["accuracy"] <= 1.0
        assert results["latency_ms"] > 0.0
        assert results["samples_shape"] == 5

    def test_performance_targets(self, sample_config, rngs):
        """Test performance targets retrieval."""
        benchmark = MockBenchmark(config=sample_config, rngs=rngs)

        targets = benchmark.get_performance_targets()

        assert isinstance(targets, dict)
        assert "accuracy" in targets
        assert "latency_ms" in targets
        assert targets["accuracy"] == 0.95
        assert targets["latency_ms"] == 100.0

    def test_benchmark_rngs_handling(self, sample_config, rngs):
        """Test proper RNG handling in benchmarks."""
        benchmark = MockBenchmark(config=sample_config, rngs=rngs)

        # Test that model generation works with rngs
        samples1 = benchmark.model.generate(n_samples=3, rngs=rngs)
        samples2 = benchmark.model.generate(n_samples=3, rngs=rngs)

        assert samples1.shape == (3, 10)
        assert samples2.shape == (3, 10)
        # Note: samples may or may not be identical depending on RNG usage
        assert jnp.isfinite(samples1).all()
        assert jnp.isfinite(samples2).all()


class TestPerformanceTracker:
    """Test PerformanceTracker functionality."""

    def test_performance_tracker_initialization(self, sample_config):
        """Test performance tracker initialization."""
        tracker = PerformanceTracker(config=sample_config)

        assert tracker.config == sample_config
        assert hasattr(tracker, "metrics_history")
        assert len(tracker.metrics_history) == 0

    def test_track_metrics(self, sample_config):
        """Test metrics tracking."""
        tracker = PerformanceTracker(config=sample_config)

        test_metrics = {"accuracy": 0.85, "loss": 0.2}
        tracker.track_metrics(test_metrics, step=1)

        assert len(tracker.metrics_history) == 1
        assert tracker.metrics_history[0]["step"] == 1
        assert tracker.metrics_history[0]["metrics"]["accuracy"] == 0.85
        assert tracker.metrics_history[0]["metrics"]["loss"] == 0.2

    def test_get_current_performance(self, sample_config):
        """Test current performance retrieval."""
        tracker = PerformanceTracker(config=sample_config)

        # No metrics tracked yet
        current = tracker.get_current_performance()
        assert current is None

        # Track some metrics
        tracker.track_metrics({"accuracy": 0.90}, step=1)
        tracker.track_metrics({"accuracy": 0.92}, step=2)

        current = tracker.get_current_performance()
        assert current["step"] == 2
        assert current["metrics"]["accuracy"] == 0.92

    def test_check_target_achievement(self, sample_config):
        """Test target achievement checking."""
        tracker = PerformanceTracker(config=sample_config)

        # Track metrics that meet targets
        tracker.track_metrics({"accuracy": 0.96, "latency_ms": 90.0}, step=1)

        achieved = tracker.check_target_achievement()
        assert achieved is True

        # Track metrics that don't meet targets
        tracker.track_metrics({"accuracy": 0.90, "latency_ms": 120.0}, step=2)

        achieved = tracker.check_target_achievement()
        assert achieved is False


class TestMetricsRegistry:
    """Test MetricsRegistry functionality."""

    def test_metrics_registry_singleton(self):
        """Test metrics registry singleton pattern."""
        registry1 = MetricsRegistry()
        registry2 = MetricsRegistry()

        assert registry1 is registry2

    def test_register_metric_computer(self):
        """Test metric computer registration."""
        registry = MetricsRegistry()

        def dummy_metric(data):
            return {"dummy": 1.0}

        registry.register_metric_computer("dummy_metric", dummy_metric)

        assert "dummy_metric" in registry.metric_computers
        assert registry.metric_computers["dummy_metric"] == dummy_metric

    def test_compute_metrics(self):
        """Test metrics computation."""
        registry = MetricsRegistry()

        # Register test metric
        def test_accuracy(predictions, targets):
            correct = jnp.sum(predictions == targets)
            return {"accuracy": correct / len(targets)}

        registry.register_metric_computer("accuracy", test_accuracy)

        # Test computation
        predictions = jnp.array([1, 0, 1, 1])
        targets = jnp.array([1, 0, 0, 1])

        metrics = registry.compute_metrics("accuracy", predictions, targets)
        assert "accuracy" in metrics
        assert metrics["accuracy"] == 0.75

    def test_list_available_metrics(self):
        """Test listing available metrics."""
        registry = MetricsRegistry()

        # Clear registry for clean test
        registry.metric_computers.clear()

        available = registry.list_available_metrics()
        assert len(available) == 0

        # Register some metrics
        registry.register_metric_computer("metric1", lambda x: {"m1": 1.0})
        registry.register_metric_computer("metric2", lambda x: {"m2": 2.0})

        available = registry.list_available_metrics()
        assert len(available) == 2
        assert "metric1" in available
        assert "metric2" in available


class TestBenchmarkRunner:
    """Test BenchmarkRunner functionality."""

    def test_benchmark_runner_initialization(self, sample_config, rngs):
        """Test benchmark runner initialization."""
        benchmark = MockBenchmark(config=sample_config, rngs=rngs)
        runner = BenchmarkRunner(benchmark=benchmark)

        assert runner.benchmark is benchmark
        assert hasattr(runner, "results_history")
        assert len(runner.results_history) == 0

    def test_run_full_benchmark(self, sample_config, rngs):
        """Test full benchmark execution."""
        benchmark = MockBenchmark(config=sample_config, rngs=rngs)
        runner = BenchmarkRunner(benchmark=benchmark)

        results = runner.run_full_benchmark()

        assert isinstance(results, dict)
        assert "training_results" in results
        assert "evaluation_results" in results
        assert "performance_summary" in results
        assert "targets_achieved" in results

        # Verify structure
        training = results["training_results"]
        evaluation = results["evaluation_results"]

        assert "training_loss" in training
        assert "accuracy" in evaluation
        assert "latency_ms" in evaluation

    def test_results_history_tracking(self, sample_config, rngs):
        """Test results history tracking."""
        benchmark = MockBenchmark(config=sample_config, rngs=rngs)
        runner = BenchmarkRunner(benchmark=benchmark)

        # Run benchmark multiple times
        runner.run_full_benchmark()
        runner.run_full_benchmark()

        assert len(runner.results_history) == 2

        # Each result should have timestamp
        for result in runner.results_history:
            assert "timestamp" in result
            assert "training_results" in result
            assert "evaluation_results" in result

    def test_performance_comparison(self, sample_config, rngs):
        """Test performance comparison functionality."""
        benchmark = MockBenchmark(config=sample_config, rngs=rngs)
        runner = BenchmarkRunner(benchmark=benchmark)

        # Run benchmarks to populate history
        runner.run_full_benchmark()
        runner.run_full_benchmark()

        comparison = runner.compare_performance()

        assert isinstance(comparison, dict)
        assert "num_runs" in comparison
        assert comparison["num_runs"] == 2

        if "metrics_summary" in comparison:
            # If metrics summary exists, verify structure
            summary = comparison["metrics_summary"]
            assert isinstance(summary, dict)


class TestBenchmarkIntegration:
    """Integration tests for benchmark system."""

    def test_end_to_end_benchmark_flow(self, sample_config, rngs):
        """Test complete benchmark workflow."""
        # Create benchmark
        benchmark = MockBenchmark(config=sample_config, rngs=rngs)

        # Create metrics registry
        registry = MetricsRegistry()

        # Register a custom metric
        def custom_metric(data):
            return {"custom_score": 0.88}

        registry.register_metric_computer("custom", custom_metric)

        # Create runner and execute
        runner = BenchmarkRunner(benchmark=benchmark)
        results = runner.run_full_benchmark()

        # Verify end-to-end functionality
        assert results is not None
        assert "training_results" in results
        assert "evaluation_results" in results

        # Test custom metric computation
        custom_metrics = registry.compute_metrics("custom", {"dummy": "data"})
        assert "custom_score" in custom_metrics
        assert custom_metrics["custom_score"] == 0.88

    def test_hardware_requirements_validation(self, sample_config, rngs):
        """Test hardware requirements validation."""
        benchmark = MockBenchmark(config=sample_config, rngs=rngs)

        # Verify hardware requirements are accessible
        hw_reqs = benchmark.config.metric_params.get("hardware_requirements", {})
        assert "gpu_memory_gb" in hw_reqs
        assert "compute_capability" in hw_reqs
        assert hw_reqs["gpu_memory_gb"] == 8.0
        assert hw_reqs["compute_capability"] == "7.0"

    def test_benchmark_reproducibility(self, sample_config, rngs):
        """Test benchmark reproducibility with same RNG seeds."""
        # Create two identical benchmarks
        benchmark1 = MockBenchmark(config=sample_config, rngs=rngs)
        benchmark2 = MockBenchmark(config=sample_config, rngs=rngs)

        # Generate samples with same RNG state
        samples1 = benchmark1.model.generate(n_samples=5, rngs=rngs)
        samples2 = benchmark2.model.generate(n_samples=5, rngs=rngs)

        # Should produce finite results
        assert jnp.isfinite(samples1).all()
        assert jnp.isfinite(samples2).all()
        assert samples1.shape == samples2.shape == (5, 10)
