"""Retained benchmark foundation for Artifex.

This module owns the framework-local benchmark config, result, and suite
abstractions that remain on top of the CalibraX registry and protocol
layer.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifex.generative_models.core.protocols.evaluation import (
    BatchableDatasetProtocol,
    BenchmarkModelProtocol,
    DatasetProtocol,
)
from artifex.utils.file_utils import ensure_valid_output_path


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""

    name: str
    description: str
    metric_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of running a benchmark."""

    benchmark_name: str
    model_name: str
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Save the result to a file."""
        valid_path = ensure_valid_output_path(path, base_dir="benchmark_results")
        Path(valid_path).parent.mkdir(parents=True, exist_ok=True)

        import jax
        import jax.numpy as jnp

        def is_jax_array(obj: Any) -> bool:
            return isinstance(obj, (jnp.ndarray, jax.Array))

        serializable_dict: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if is_jax_array(value):
                serializable_dict[key] = float(value) if value.size == 1 else value.tolist()
            elif isinstance(value, dict):
                serializable_dict[key] = {
                    nested_key: float(nested_value)
                    if is_jax_array(nested_value) and nested_value.size == 1
                    else nested_value.tolist()
                    if is_jax_array(nested_value)
                    else nested_value
                    for nested_key, nested_value in value.items()
                }
            else:
                serializable_dict[key] = value

        with open(valid_path, "w", encoding="utf-8") as handle:
            json.dump(serializable_dict, handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "BenchmarkResult":
        """Load a result from a file."""
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(**data)


class Benchmark(ABC):
    """Abstract benchmark base class used by the retained benchmark surface."""

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize the benchmark with its configuration."""
        self.config = config

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
        end_time = time.time()
        result.metadata["runtime"] = end_time - start_time
        return result

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
            for metric_name, value in result.metrics.items():
                summary["all_metrics"][f"{benchmark_name}_{metric_name}"] = value
        return summary


__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkSuite",
]
