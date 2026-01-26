"""Evaluation package for artifex.generative_models.core.

This package provides evaluation capabilities including benchmarks and metrics
for assessing generative model performance across different modalities.

Modules:
    benchmarks: Benchmark implementations and runners
    metrics: Evaluation metrics for different modalities
"""

# Import main components for convenience
from artifex.generative_models.core.evaluation import benchmarks, metrics


__all__ = [
    "benchmarks",
    "metrics",
]
