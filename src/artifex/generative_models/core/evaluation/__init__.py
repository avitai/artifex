"""Evaluation package for artifex.generative_models.core.

The retained core evaluation surface is metrics-only. Benchmark ownership
lives under `artifex.benchmarks.core`.
"""

from artifex.generative_models.core.evaluation import metrics


__all__ = [
    "metrics",
]
