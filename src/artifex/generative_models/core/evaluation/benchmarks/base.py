"""Base benchmark implementations for artifex.generative_models.core.evaluation."""

import time
from abc import ABC, abstractmethod
from typing import Any

from artifex.generative_models.core.evaluation.benchmarks.types import (
    BenchmarkConfig,
    BenchmarkResult,
)
from artifex.generative_models.core.protocols.evaluation import (
    BatchableDatasetProtocol,
    DatasetProtocol,
    ModelProtocol,
)


class Benchmark(ABC):
    """Base class for benchmarks.

    This class defines the interface for all benchmarks in the system.
    All benchmarks must exclusively support NNX models.

    Attributes:
        config: Configuration for the benchmark.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize a benchmark.

        Args:
            config: Configuration for the benchmark.
        """
        self.config = config

    @abstractmethod
    def run(
        self,
        model: ModelProtocol,
        dataset: DatasetProtocol | BatchableDatasetProtocol | None = None,
    ) -> BenchmarkResult:
        """Run the benchmark.

        Args:
            model: NNX model to benchmark.
            dataset: Dataset to use for benchmarking.

        Returns:
            Result of the benchmark.
        """
        raise NotImplementedError("Subclasses must implement the run method.")

    def timed_run(
        self,
        model: ModelProtocol,
        dataset: DatasetProtocol | BatchableDatasetProtocol | None = None,
    ) -> BenchmarkResult:
        """Run the benchmark and measure the runtime.

        Args:
            model: NNX model to benchmark.
            dataset: Dataset to use for benchmarking.

        Returns:
            Result of the benchmark with runtime metadata.
        """
        start_time = time.time()
        result = self.run(model, dataset)
        end_time = time.time()

        # Add runtime to metadata
        result.metadata["runtime"] = end_time - start_time

        return result

    def validate_metrics(self, metrics: dict[str, float]) -> None:
        """Validate that the metrics match the expected metric names.

        Args:
            metrics: Metrics to validate.

        Raises:
            ValueError: If the metrics don't match the expected metrics.
        """
        # Check that all expected metrics are present
        for name in self.config.metric_names:
            if name not in metrics:
                raise ValueError(f"Expected metric '{name}' not found in results.")

        # Check that no unexpected metrics are present
        for name in metrics:
            if name not in self.config.metric_names:
                raise ValueError(f"Unexpected metric '{name}' found in results.")


class BenchmarkSuite(ABC):
    """Base class for benchmark suites.

    A benchmark suite manages multiple related benchmarks and provides
    orchestration for running them together.
    """

    def __init__(self, name: str, description: str = ""):
        """Initialize the benchmark suite.

        Args:
            name: Name of the benchmark suite
            description: Description of the suite
        """
        self.name = name
        self.description = description
        self.benchmarks: list[Benchmark] = []

    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add a benchmark to the suite.

        Args:
            benchmark: Benchmark to add
        """
        self.benchmarks.append(benchmark)

    def run_all(self, model, **kwargs) -> dict[str, BenchmarkResult]:
        """Run all benchmarks in the suite.

        Args:
            model: Model to benchmark
            **kwargs: Additional parameters for benchmarks

        Returns:
            dictionary mapping benchmark names to results
        """
        results = {}

        for benchmark in self.benchmarks:
            print(f"Running benchmark: {benchmark.config.name}")
            result = benchmark.run(model, **kwargs)
            results[benchmark.config.name] = result

        return results

    def get_summary(self, results: dict[str, BenchmarkResult]) -> dict[str, Any]:
        """Get a summary of benchmark results.

        Args:
            results: Results from run_all

        Returns:
            Summary statistics
        """
        summary: dict[str, Any] = {
            "suite_name": self.name,
            "num_benchmarks": len(results),
            "benchmark_names": list(results.keys()),
            "all_metrics": {},
        }

        # Aggregate all metrics
        for benchmark_name, result in results.items():
            for metric_name, value in result.metrics.items():
                full_metric_name = f"{benchmark_name}_{metric_name}"
                summary["all_metrics"][full_metric_name] = value

        return summary
