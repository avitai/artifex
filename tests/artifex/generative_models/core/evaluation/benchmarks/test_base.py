"""Tests for the benchmark foundation surface under artifex.benchmarks.core."""

import tempfile
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest
from calibrax.core import BenchmarkResult as CalibraxBenchmarkResult, Metric
from tests.utils.test_models import MockModel

from artifex.benchmarks.core import (
    Benchmark,
    benchmark_result,
    BenchmarkConfig,
    BenchmarkResult,
    metric_values,
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
    """Artifex results are calibrax's BenchmarkResult, built by the benchmark."""

    def test_result_is_calibrax_type(self):
        """Artifex re-exports calibrax's result type; there is no second one."""
        assert BenchmarkResult is CalibraxBenchmarkResult

    def test_benchmark_result_records_name_model_metrics_and_config(self):
        """Benchmark.result names the result after the config and tags the model."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="A test benchmark",
            metric_names=["metric1", "metric2"],
        )
        benchmark = _ConstantBenchmark(config=config)

        result = benchmark.result("test_model", {"metric1": 0.95, "metric2": jnp.float32(0.85)})

        assert isinstance(result, CalibraxBenchmarkResult)
        assert result.name == "test_benchmark"
        assert result.tags == {"model_name": "test_model"}
        assert result.metrics["metric1"] == Metric(value=0.95)
        assert result.metrics["metric2"].value == pytest.approx(0.85)
        assert isinstance(result.metrics["metric2"].value, float)
        assert result.config == {
            "name": "test_benchmark",
            "description": "A test benchmark",
            "metric_names": ["metric1", "metric2"],
            "metadata": {},
        }
        assert result.metadata == {}

    def test_metadata_is_sanitised_at_build_time(self):
        """JAX scalars in metadata become Python numbers."""
        benchmark = _ConstantBenchmark(config=_config())

        result = benchmark.result("m", {"metric1": 1.0}, metadata={"runtime": jnp.float32(10.5)})

        assert result.metadata == {"runtime": pytest.approx(10.5)}
        assert isinstance(result.metadata["runtime"], float)

    def test_benchmark_result_function_takes_any_name(self):
        """Suites that are not Benchmark subclasses build results the same way."""
        result = benchmark_result("stylegan3", "generator", {"fid_score": 12.0}, metadata={"a": 1})

        assert result.name == "stylegan3"
        assert result.tags["model_name"] == "generator"
        assert metric_values(result) == {"fid_score": 12.0}
        assert result.metadata == {"a": 1}
        assert result.config == {}

    def test_metric_values_reads_plain_floats(self):
        result = benchmark_result("b", "m", {"x": 0.5, "y": True})

        assert metric_values(result) == {"x": 0.5, "y": 1.0}

    def test_save_and_load(self):
        """Saving goes through calibrax's JSON layout and loads back whole."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            result = _ConstantBenchmark(config=_config()).result(
                "test_model", {"metric1": 0.95, "metric2": 0.85}, metadata={"runtime": 10.5}
            )

            result.save(result_path)
            loaded_result = BenchmarkResult.load(result_path)

            assert loaded_result.name == result.name
            assert loaded_result.tags == result.tags
            assert metric_values(loaded_result) == metric_values(result)
            assert loaded_result.metadata == result.metadata
            assert loaded_result.config == result.config


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="test_benchmark", description="A test benchmark", metric_names=["metric1"]
    )


class _ConstantBenchmark(Benchmark):
    """Benchmark returning fixed metrics through Benchmark.result."""

    def run(self, model, dataset=None):
        return self.result(getattr(model, "model_name", "unknown"), {"metric1": 0.95})


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
                return self.result("test", {"metric1": 0.5, "metric2": 0.5})

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
                return self.result(
                    getattr(model, "model_name", "unknown"), {"metric1": 0.95, "metric2": 0.85}
                )

        mock_model = MockModel(rngs=self.rngs, model_name="test_model")
        mock_dataset = MockDataset()

        benchmark = MockBenchmark(config=config)
        result = benchmark.run(model=mock_model, dataset=mock_dataset)

        assert result.name == "test_benchmark"
        assert result.tags["model_name"] == "test_model"
        assert metric_values(result) == {"metric1": 0.95, "metric2": 0.85}

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
                return self.result(getattr(model, "model_name", "unknown"), {"metric1": 0.95})

        mock_model = MockModel(rngs=self.rngs, model_name="test_model")

        benchmark = MockBenchmark(config=config)
        result = benchmark.timed_run(model=mock_model)

        assert result.name == "test_benchmark"
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
                return self.result("test", {"metric1": 0.5, "metric2": 0.5})

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
