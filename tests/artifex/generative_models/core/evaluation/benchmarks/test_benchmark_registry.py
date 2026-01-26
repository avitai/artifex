"""Tests for artifex.benchmarks.registry module."""

import pytest

from artifex.benchmarks.base import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
)
from artifex.benchmarks.registry import (
    BenchmarkRegistry,
    get_benchmark,
    list_benchmarks,
    register_benchmark,
)


class TestBenchmarkRegistry:
    """Tests for the BenchmarkRegistry class."""

    def setup_method(self):
        """Set up the test environment."""
        # Clear the registry before each test
        BenchmarkRegistry.reset()

    def test_registry_singleton(self):
        """Test that the registry is a singleton."""
        registry1 = BenchmarkRegistry()
        registry2 = BenchmarkRegistry()

        assert registry1 is registry2

    def test_register_function(self):
        """Test registering a benchmark using the register_benchmark function."""

        class TestBenchmark(Benchmark):
            """Test benchmark implementation."""

            def run(self, model, dataset=None):
                """Run the benchmark."""
                return BenchmarkResult(
                    benchmark_name=self.config.name,
                    model_name="test_model",
                    metrics={"metric1": 0.95},
                )

        config = BenchmarkConfig(
            name="test_benchmark",
            description="A test benchmark",
            metric_names=["metric1"],
        )

        # Create and register the benchmark
        benchmark = TestBenchmark(config=config)
        register_benchmark("test_benchmark", benchmark)

        # Check that it's in the registry
        registry = BenchmarkRegistry()
        assert "test_benchmark" in registry.benchmarks
        assert registry.benchmarks["test_benchmark"] is benchmark

    def test_register_decorator(self):
        """Test registering a benchmark using the register_benchmark decorator."""

        @register_benchmark("decorated_benchmark")
        class DecoratedBenchmark(Benchmark):
            """Test benchmark with decorator."""

            def __init__(self, config=None):
                """Initialize the benchmark."""
                if config is None:
                    config = BenchmarkConfig(
                        name="decorated_benchmark",
                        description="A decorated benchmark",
                        metric_names=["metric1"],
                    )
                super().__init__(config=config)

            def run(self, model, dataset=None):
                """Run the benchmark."""
                return BenchmarkResult(
                    benchmark_name=self.config.name,
                    model_name="test_model",
                    metrics={"metric1": 0.95},
                )

        # Check that it's in the registry
        registry = BenchmarkRegistry()
        assert "decorated_benchmark" in registry.benchmarks
        assert isinstance(registry.benchmarks["decorated_benchmark"], DecoratedBenchmark)

    def test_get_benchmark(self):
        """Test getting a benchmark from the registry."""

        class TestBenchmark(Benchmark):
            """Test benchmark implementation."""

            def run(self, model, dataset=None):
                """Run the benchmark."""
                return BenchmarkResult(
                    benchmark_name=self.config.name,
                    model_name="test_model",
                    metrics={"metric1": 0.95},
                )

        config = BenchmarkConfig(
            name="test_benchmark",
            description="A test benchmark",
            metric_names=["metric1"],
        )

        # Create and register the benchmark
        benchmark = TestBenchmark(config=config)
        register_benchmark("test_benchmark", benchmark)

        # Get the benchmark
        retrieved_benchmark = get_benchmark("test_benchmark")
        assert retrieved_benchmark is benchmark

    def test_get_nonexistent_benchmark(self):
        """Test getting a benchmark that doesn't exist."""
        with pytest.raises(KeyError):
            get_benchmark("nonexistent_benchmark")

    def test_list_benchmarks(self):
        """Test listing all benchmarks."""

        class TestBenchmark1(Benchmark):
            """Test benchmark 1."""

            def run(self, model, dataset=None):
                """Run the benchmark."""
                return BenchmarkResult(
                    benchmark_name=self.config.name,
                    model_name="test_model",
                    metrics={"metric1": 0.95},
                )

        class TestBenchmark2(Benchmark):
            """Test benchmark 2."""

            def run(self, model, dataset=None):
                """Run the benchmark."""
                return BenchmarkResult(
                    benchmark_name=self.config.name,
                    model_name="test_model",
                    metrics={"metric2": 0.85},
                )

        config1 = BenchmarkConfig(
            name="test_benchmark_1",
            description="Test benchmark 1",
            metric_names=["metric1"],
        )

        config2 = BenchmarkConfig(
            name="test_benchmark_2",
            description="Test benchmark 2",
            metric_names=["metric2"],
        )

        # Create and register the benchmarks
        benchmark1 = TestBenchmark1(config=config1)
        benchmark2 = TestBenchmark2(config=config2)

        register_benchmark("test_benchmark_1", benchmark1)
        register_benchmark("test_benchmark_2", benchmark2)

        # List the benchmarks
        benchmarks = list_benchmarks()
        assert len(benchmarks) == 2
        assert "test_benchmark_1" in benchmarks
        assert "test_benchmark_2" in benchmarks
