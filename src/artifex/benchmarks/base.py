"""Base classes for benchmarks.

This module re-exports the base classes from artifex.generative_models.core.evaluation.benchmarks
to maintain backward compatibility.
"""

from artifex.benchmarks.datasets.base import DatasetProtocol
from artifex.generative_models.core.evaluation.benchmarks.base import (
    Benchmark,
    BenchmarkSuite,
)
from artifex.generative_models.core.evaluation.benchmarks.types import (
    BenchmarkConfig,
    BenchmarkResult,
)
from artifex.generative_models.core.protocols.evaluation import ModelProtocol


__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkSuite",
    "DatasetProtocol",
    "ModelProtocol",
]
