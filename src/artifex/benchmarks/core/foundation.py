"""Retained benchmark foundation for Artifex.

This module owns the framework-local benchmark config and suite abstractions
that remain on top of the Calibrax registry and protocol layer. Results are
calibrax's ``BenchmarkResult``; ``benchmark_result`` and ``Benchmark.result``
build them from a metric dictionary.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from calibrax.core import BenchmarkResult, Metric

from artifex.generative_models.core.protocols.evaluation import (
    BatchableDatasetProtocol,
    BenchmarkModelProtocol,
    DatasetProtocol,
)


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""

    name: str
    description: str
    metric_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def _to_python(value: Any) -> Any:
    """Convert JAX and numpy scalars (nested in dicts, lists and tuples) to Python values."""
    if isinstance(value, Mapping):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return value.item()
    return value


def benchmark_result(
    name: str,
    model_name: str,
    metrics: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> BenchmarkResult:
    """Build a calibrax result from a benchmark's metric dictionary.

    Args:
        name: Benchmark name.
        model_name: Name of the evaluated model, stored under ``tags["model_name"]``.
        metrics: Metric values; each becomes a ``Metric`` with a float value.
        metadata: Extra information; JAX and numpy scalars become Python values.
        config: Benchmark configuration as a dictionary.

    Returns:
        The result with its metrics wrapped and its metadata sanitised.
    """
    return BenchmarkResult(
        name=name,
        tags={"model_name": model_name},
        metrics={key: Metric(value=float(value)) for key, value in metrics.items()},
        metadata=_to_python(dict(metadata or {})),
        config=_to_python(dict(config or {})),
    )


def metric_values(result: BenchmarkResult) -> dict[str, float]:
    """Plain ``name -> value`` view of a result's metrics."""
    return {name: metric.value for name, metric in result.metrics.items()}


class Benchmark(ABC):
    """Abstract benchmark base class used by the retained benchmark surface."""

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize the benchmark with its configuration."""
        self.config = config

    def result(
        self,
        model_name: str,
        metrics: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Build this benchmark's result, named after its config and carrying it.

        Args:
            model_name: Name of the evaluated model.
            metrics: Metric values keyed by metric name.
            metadata: Extra information about the run.

        Returns:
            A calibrax ``BenchmarkResult``.
        """
        return benchmark_result(
            self.config.name,
            model_name,
            metrics,
            metadata=metadata,
            config=asdict(self.config),
        )

    def setup(self) -> None:
        """Set up benchmark resources before execution."""

    def run_training(self) -> dict[str, float]:
        """Execute the training phase."""
        return {}

    def run_evaluation(self) -> dict[str, float]:
        """Execute the evaluation phase."""
        return {}

    def teardown(self) -> None:
        """Release benchmark resources after execution."""

    def get_performance_targets(self) -> dict[str, float]:
        """Return expected performance targets."""
        return {}

    @abstractmethod
    def run(
        self,
        model: BenchmarkModelProtocol,
        dataset: DatasetProtocol | BatchableDatasetProtocol | None = None,
    ) -> BenchmarkResult:
        """Run the benchmark."""
        raise NotImplementedError("Subclasses must implement the run method.")

    def timed_run(
        self,
        model: BenchmarkModelProtocol,
        dataset: DatasetProtocol | BatchableDatasetProtocol | None = None,
    ) -> BenchmarkResult:
        """Run the benchmark and measure runtime."""
        start_time = time.time()
        result = self.run(model, dataset)
        runtime = time.time() - start_time
        return replace(result, metadata={**result.metadata, "runtime": runtime})

    def validate_metrics(self, metrics: dict[str, float]) -> None:
        """Validate that returned metrics match the configured metric names."""
        for name in self.config.metric_names:
            if name not in metrics:
                raise ValueError(f"Expected metric '{name}' not found in results.")

        for name in metrics:
            if name not in self.config.metric_names:
                raise ValueError(f"Unexpected metric '{name}' found in results.")


class BenchmarkSuite(ABC):
    """Base class for benchmark suites."""

    def __init__(self, name: str, description: str = ""):
        """Initialize the benchmark suite metadata."""
        self.name = name
        self.description = description
        self.benchmarks: list[Benchmark] = []

    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)

    def run_all(self, model, **kwargs) -> dict[str, BenchmarkResult]:
        """Run all benchmarks in the suite."""
        results: dict[str, BenchmarkResult] = {}
        for benchmark in self.benchmarks:
            logger.info("Running benchmark: %s", benchmark.config.name)
            result = benchmark.run(model, **kwargs)
            results[benchmark.config.name] = result
        return results

    def get_summary(self, results: dict[str, BenchmarkResult]) -> dict[str, Any]:
        """Get a summary of benchmark results."""
        summary: dict[str, Any] = {
            "suite_name": self.name,
            "num_benchmarks": len(results),
            "benchmark_names": list(results.keys()),
            "all_metrics": {},
        }
        for benchmark_name, result in results.items():
            for metric_name, value in metric_values(result).items():
                summary["all_metrics"][f"{benchmark_name}_{metric_name}"] = value
        return summary


__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkSuite",
    "benchmark_result",
    "metric_values",
]
