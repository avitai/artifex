"""Tests for latency benchmark with NNX models."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from tests.utils.test_models import SimpleNNXModel

from artifex.benchmarks.performance.latency import (
    LatencyBenchmark,
    measure_inference_latency,
)


class TestLatencyBenchmarkWithNNX:
    """Tests for latency benchmark with NNX models."""

    def setup_method(self):
        """Set up for each test."""
        key = jax.random.PRNGKey(0)
        # Use the default feature size from centralized SimpleNNXModel (10 in_features)
        self.feature_size = 10
        rngs = nnx.Rngs(params=key)
        self.model = SimpleNNXModel(features=5, in_features=self.feature_size, rngs=rngs)

        # Create a simple dataset for prediction tests with matching feature dimensions
        self.dataset = np.ones((10, self.feature_size))

    def test_measure_inference_latency_sample(self):
        """Test measuring inference latency for sampling."""
        key = jax.random.PRNGKey(1)
        nnx_rngs = nnx.Rngs(sample=key)
        # Convert to dictionary for measure_inference_latency
        rngs = {"sample": nnx_rngs.sample}

        # Measure latency
        avg_latency, std_dev = measure_inference_latency(
            model=self.model,
            method="sample",
            num_runs=3,  # Small number for testing
            warmup_runs=1,
            batch_size=2,
            rngs=rngs,
        )

        # Verify results
        assert avg_latency > 0
        assert std_dev >= 0

    def test_measure_inference_latency_predict(self):
        """Test measuring inference latency for prediction."""
        key = jax.random.PRNGKey(1)
        nnx_rngs = nnx.Rngs(dropout=key)
        # Convert to dictionary for measure_inference_latency
        rngs = {"dropout": nnx_rngs.dropout}

        # Measure latency
        avg_latency, std_dev = measure_inference_latency(
            model=self.model,
            method="predict",
            num_runs=3,  # Small number for testing
            warmup_runs=1,
            batch_size=2,
            rngs=rngs,
            inputs=jnp.ones((2, self.feature_size)),
        )

        # Verify results
        assert avg_latency > 0
        assert std_dev >= 0

    def test_latency_benchmark_sample(self):
        """Test the latency benchmark with sample method."""
        # Create benchmark
        benchmark = LatencyBenchmark(
            method="sample",
            batch_size=2,
            num_runs=3,  # Small number for testing
            warmup_runs=1,
            random_seed=42,
        )

        # Run benchmark
        result = benchmark.run(self.model)

        # Verify results
        assert "inference_latency_ms" in result.metrics
        assert "latency_std_dev_ms" in result.metrics
        assert "samples_per_second" in result.metrics
        assert result.model_name == "SimpleNNXModel"

    def test_latency_benchmark_predict(self):
        """Test the latency benchmark with predict method."""
        # Create benchmark
        benchmark = LatencyBenchmark(
            method="predict",
            batch_size=2,
            num_runs=3,  # Small number for testing
            warmup_runs=1,
            random_seed=42,
        )

        # Run benchmark
        result = benchmark.run(self.model, dataset=self.dataset)

        # Verify results
        assert "inference_latency_ms" in result.metrics
        assert "latency_std_dev_ms" in result.metrics
        assert "samples_per_second" in result.metrics
        assert result.model_name == "SimpleNNXModel"
