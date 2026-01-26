"""Tests for latency benchmarks."""

import time

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.benchmarks.base import BenchmarkResult
from artifex.benchmarks.performance.latency import (
    LatencyBenchmark,
    measure_inference_latency,
)


class MockModel:
    """Mock model for testing latency benchmarks."""

    def __init__(self, sleep_time=0.001, model_name="mock_model"):
        """Initialize with a predetermined sleep time for consistent testing."""
        self.model_name = model_name
        self.sleep_time = sleep_time
        self.sample_called = False
        self.predict_called = False

    def sample(self, *, batch_size=1, rngs=None):
        """Mock sample method with a sleep."""
        self.sample_called = True
        time.sleep(self.sleep_time)
        return jnp.ones((batch_size, 10))

    def predict(self, x, *, rngs=None):
        """Mock predict method with a sleep."""
        self.predict_called = True
        time.sleep(self.sleep_time)
        return jnp.ones_like(x)


class TestLatencyMeasurement:
    """Tests for latency measurement functions."""

    def test_measure_inference_latency(self):
        """Test the latency measurement function."""
        model = MockModel(sleep_time=0.001)
        rngs = nnx.Rngs(sample=jax.random.key(42))

        # Test with sample method
        latency, std_dev = measure_inference_latency(
            model=model,
            method="sample",
            num_runs=10,
            warmup_runs=2,
            rngs=rngs,
        )

        assert model.sample_called
        assert latency > 0
        assert std_dev >= 0

        # Reset model
        model = MockModel(sleep_time=0.001)

        # Test with predict method
        inputs = jnp.ones((1, 10))
        latency, std_dev = measure_inference_latency(
            model=model,
            method="predict",
            num_runs=10,
            warmup_runs=2,
            inputs=inputs,
        )

        assert model.predict_called
        assert latency > 0
        assert std_dev >= 0

    def test_measure_latency_with_longer_sleep(self):
        """Test with a longer sleep time to ensure accuracy."""
        model = MockModel(sleep_time=0.01)
        rngs = nnx.Rngs(sample=jax.random.key(42))

        latency, _ = measure_inference_latency(
            model=model,
            method="sample",
            num_runs=5,
            warmup_runs=1,
            rngs=rngs,
        )

        # Should be close to the sleep time
        assert 0.005 < latency < 0.02


class TestLatencyBenchmark:
    """Tests for the LatencyBenchmark class."""

    def test_init(self):
        """Test initialization of LatencyBenchmark."""
        benchmark = LatencyBenchmark()

        assert benchmark.config.name == "latency"
        assert "inference_latency_ms" in benchmark.config.metric_names
        assert "latency_std_dev_ms" in benchmark.config.metric_names

        benchmark = LatencyBenchmark(
            method="predict",
            batch_size=4,
            num_runs=20,
            warmup_runs=5,
        )

        assert benchmark.method == "predict"
        assert benchmark.batch_size == 4
        assert benchmark.num_runs == 20
        assert benchmark.warmup_runs == 5

    def test_run_sample_method(self):
        """Test running benchmark with sample method."""
        model = MockModel(sleep_time=0.001)
        benchmark = LatencyBenchmark(method="sample", num_runs=5)

        result = benchmark.run(model=model)

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "latency"
        assert result.model_name == "mock_model"
        assert "inference_latency_ms" in result.metrics
        assert "latency_std_dev_ms" in result.metrics
        assert "samples_per_second" in result.metrics

        # Latency should be positive
        assert result.metrics["inference_latency_ms"] > 0
        assert result.metrics["samples_per_second"] > 0

    def test_run_predict_method(self):
        """Test running benchmark with predict method."""
        model = MockModel(sleep_time=0.001)
        benchmark = LatencyBenchmark(method="predict", num_runs=5)

        # Need to provide dataset for predict method
        dataset = jnp.ones((10, 10))
        result = benchmark.run(model=model, dataset=dataset)

        assert isinstance(result, BenchmarkResult)
        assert "inference_latency_ms" in result.metrics
        assert result.metrics["inference_latency_ms"] > 0

    def test_run_with_different_batch_sizes(self):
        """Test running benchmark with different batch sizes."""
        model = MockModel(sleep_time=0.001)

        # Small batch
        benchmark_small = LatencyBenchmark(batch_size=1, num_runs=5)
        result_small = benchmark_small.run(model=model)

        # Larger batch
        benchmark_large = LatencyBenchmark(batch_size=10, num_runs=5)
        result_large = benchmark_large.run(model=model)

        # Latency per sample should be lower for larger batch
        samples_per_second_small = result_small.metrics["samples_per_second"]
        samples_per_second_large = result_large.metrics["samples_per_second"]

        # Larger batch should have higher throughput
        assert samples_per_second_large > samples_per_second_small
