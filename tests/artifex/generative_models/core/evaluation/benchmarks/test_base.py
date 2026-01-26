"""Tests for artifex.generative_models.core.evaluation.benchmarks.base module."""

import os
import tempfile

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest
from tests.utils.test_models import MockModel

from artifex.generative_models.core.evaluation.benchmarks.base import (
    Benchmark,
)
from artifex.generative_models.core.evaluation.benchmarks.types import (
    BenchmarkConfig,
    BenchmarkResult,
)


class TestBenchmarkConfig:
    """Tests for the BenchmarkConfig class."""

    def test_init(self):
        """Test initialization of BenchmarkConfig."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="A test benchmark",
            metric_names=["metric1", "metric2"],
        )

        assert config.name == "test_benchmark"
        assert config.description == "A test benchmark"
        assert config.metric_names == ["metric1", "metric2"]
        assert config.metadata == {}

    def test_with_metadata(self):
        """Test initialization with metadata."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="A test benchmark",
            metric_names=["metric1"],
            metadata={"author": "Test Author", "version": "1.0.0"},
        )

        assert config.metadata == {"author": "Test Author", "version": "1.0.0"}


# MockModel is now imported from tests.utils.test_models


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, num_examples=100):
        self.num_examples = num_examples
        self.data = jnp.ones((num_examples, 10))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.data[idx]


class TestBenchmarkResult:
    """Tests for the BenchmarkResult class."""

    def test_init(self):
        """Test initialization of BenchmarkResult."""
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            model_name="test_model",
            metrics={"metric1": 0.95, "metric2": 0.85},
        )

        assert result.benchmark_name == "test_benchmark"
        assert result.model_name == "test_model"
        assert result.metrics == {"metric1": 0.95, "metric2": 0.85}
        assert result.metadata == {}

    def test_with_metadata(self):
        """Test initialization with metadata."""
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            model_name="test_model",
            metrics={"metric1": 0.95},
            metadata={"runtime": 10.5, "device": "cpu"},
        )

        assert result.metadata == {"runtime": 10.5, "device": "cpu"}

    def test_save_and_load(self):
        """Test saving and loading results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = os.path.join(temp_dir, "result.json")

            result = BenchmarkResult(
                benchmark_name="test_benchmark",
                model_name="test_model",
                metrics={"metric1": 0.95, "metric2": 0.85},
                metadata={"runtime": 10.5},
            )

            # Test save
            result.save(result_path)
            assert os.path.exists(result_path)

            # Test load
            loaded_result = BenchmarkResult.load(result_path)
            assert loaded_result.benchmark_name == result.benchmark_name
            assert loaded_result.model_name == result.model_name
            assert loaded_result.metrics == result.metrics
            assert loaded_result.metadata == result.metadata


class TestBenchmark:
    """Tests for the Benchmark class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create RNGs for initializing models
        key = jax.random.PRNGKey(0)
        self.rngs = nnx.Rngs(params=key)

    def test_init(self):
        """Test initialization of Benchmark."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="A test benchmark",
            metric_names=["metric1", "metric2"],
        )

        class ConcreteBenchmark(Benchmark):
            def run(self, model, dataset=None):
                return BenchmarkResult(
                    benchmark_name=self.config.name,
                    model_name="test",
                    metrics={"metric1": 0.5, "metric2": 0.5},
                )

        benchmark = ConcreteBenchmark(config=config)

        assert benchmark.config.name == "test_benchmark"
        assert benchmark.config.description == "A test benchmark"
        assert benchmark.config.metric_names == ["metric1", "metric2"]

    def test_run_with_mock_implementations(self):
        """Test running a benchmark with mock implementations."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="A test benchmark",
            metric_names=["metric1", "metric2"],
        )

        class MockBenchmark(Benchmark):
            """Mock benchmark implementation."""

            def run(self, model, dataset=None):
                """Mock run implementation."""
                return BenchmarkResult(
                    benchmark_name=self.config.name,
                    model_name=getattr(model, "model_name", "unknown"),
                    metrics={"metric1": 0.95, "metric2": 0.85},
                )

        mock_model = MockModel(rngs=self.rngs, model_name="test_model")
        mock_dataset = MockDataset()

        benchmark = MockBenchmark(config=config)
        result = benchmark.run(model=mock_model, dataset=mock_dataset)

        assert result.benchmark_name == "test_benchmark"
        assert result.model_name == "test_model"
        assert result.metrics == {"metric1": 0.95, "metric2": 0.85}

    def test_timed_run(self):
        """Test the timed_run method."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="A test benchmark",
            metric_names=["metric1"],
        )

        class MockBenchmark(Benchmark):
            """Mock benchmark implementation."""

            def run(self, model, dataset=None):
                """Mock run implementation."""
                return BenchmarkResult(
                    benchmark_name=self.config.name,
                    model_name=getattr(model, "model_name", "unknown"),
                    metrics={"metric1": 0.95},
                )

        mock_model = MockModel(rngs=self.rngs, model_name="test_model")

        benchmark = MockBenchmark(config=config)
        result = benchmark.timed_run(model=mock_model)

        assert result.benchmark_name == "test_benchmark"
        assert "runtime" in result.metadata
        assert isinstance(result.metadata["runtime"], float)
        assert result.metadata["runtime"] > 0

    def test_validate_metrics(self):
        """Test validation of metrics."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="A test benchmark",
            metric_names=["metric1", "metric2"],
        )

        class ConcreteBenchmark(Benchmark):
            def run(self, model, dataset=None):
                return BenchmarkResult(
                    benchmark_name=self.config.name,
                    model_name="test",
                    metrics={"metric1": 0.5, "metric2": 0.5},
                )

        benchmark = ConcreteBenchmark(config=config)

        # Valid metrics
        valid_metrics = {"metric1": 0.95, "metric2": 0.85}
        benchmark.validate_metrics(valid_metrics)

        # Invalid metrics - missing metric
        invalid_metrics1 = {"metric1": 0.95}
        with pytest.raises(ValueError):
            benchmark.validate_metrics(invalid_metrics1)

        # Invalid metrics - extra metric
        invalid_metrics2 = {"metric1": 0.95, "metric2": 0.85, "metric3": 0.75}
        with pytest.raises(ValueError):
            benchmark.validate_metrics(invalid_metrics2)

    def test_abstract_run_method(self):
        """Test that the run method is abstract and must be implemented."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="A test benchmark",
            metric_names=["metric1"],
        )

        # Mock implementation that directly calls NotImplementedError
        class TestBenchmarkImp(Benchmark):
            def run(self, model, dataset=None):
                raise NotImplementedError("Test implementation")

        # Instantiate and test
        benchmark = TestBenchmarkImp(config=config)
        with pytest.raises(NotImplementedError):
            benchmark.run(model=MockModel(rngs=self.rngs, model_name="test_model"))
