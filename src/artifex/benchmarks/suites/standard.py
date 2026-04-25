"""Standard benchmark suites for generative models.

This module provides predefined benchmark suites for evaluating different
aspects of generative models.
"""

from artifex.benchmarks.core import Benchmark
from artifex.benchmarks.metrics.precision_recall import PrecisionRecallBenchmark
from artifex.benchmarks.performance.latency import LatencyBenchmark
from artifex.benchmarks.suites.registry import register_suite


def get_quality_suite() -> list[Benchmark]:
    """Get a suite of benchmarks for evaluating the quality of generated samples.

    Returns:
        list of benchmarks in the suite.
    """
    return [
        PrecisionRecallBenchmark(
            num_clusters=10,
            num_samples=1000,
            random_seed=42,
        ),
    ]


def get_performance_suite() -> list[Benchmark]:
    """Get a suite of benchmarks for evaluating model performance.

    Returns:
        list of benchmarks in the suite.
    """
    return [
        LatencyBenchmark(
            method="sample",
            batch_size=1,
            num_runs=100,
            warmup_runs=10,
        ),
        LatencyBenchmark(
            method="sample",
            batch_size=16,
            num_runs=100,
            warmup_runs=10,
        ),
        LatencyBenchmark(
            method="predict",
            batch_size=1,
            num_runs=100,
            warmup_runs=10,
        ),
        LatencyBenchmark(
            method="predict",
            batch_size=16,
            num_runs=100,
            warmup_runs=10,
        ),
    ]


def get_standard_suite() -> list[Benchmark]:
    """Get a complete suite of benchmarks for evaluating generative models.

    This suite includes both quality and performance benchmarks.

    Returns:
        list of benchmarks in the suite.
    """
    return get_quality_suite() + get_performance_suite()


# Register the standard suites
register_suite("quality", get_quality_suite)
register_suite("performance", get_performance_suite)
register_suite("standard", get_standard_suite)
