"""Benchmark protocols and interfaces."""

from artifex.generative_models.core.evaluation.metrics.registry import MetricsRegistry

from .core import BenchmarkConfig, BenchmarkResult


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "MetricsRegistry",
]
