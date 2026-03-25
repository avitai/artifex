"""Benchmark suites for generative models.

This module provides benchmark suites that combine multiple benchmarks for
complete evaluation.
"""

from artifex.benchmarks.suites.protein_benchmarks import (
    ProteinBenchmarkSuite,
    ProteinStructureBenchmark,
)


__all__ = [
    "ProteinBenchmarkSuite",
    "ProteinStructureBenchmark",
]
