from __future__ import annotations

from artifex.benchmarks.metrics.precision_recall import PrecisionRecallBenchmark
from artifex.benchmarks.performance.latency import LatencyBenchmark
from artifex.benchmarks.suites import standard
from artifex.benchmarks.suites.registry import get_suite


def test_quality_suite_contains_precision_recall_benchmark() -> None:
    """The standard quality suite should expose the retained quality benchmark."""
    suite = standard.get_quality_suite()

    assert len(suite) == 1
    assert isinstance(suite[0], PrecisionRecallBenchmark)


def test_performance_suite_covers_sample_and_predict_batch_sizes() -> None:
    """The performance suite should retain sample and predict latency checks."""
    suite = standard.get_performance_suite()

    assert len(suite) == 4
    assert all(isinstance(benchmark, LatencyBenchmark) for benchmark in suite)
    assert {(benchmark.method, benchmark.batch_size) for benchmark in suite} == {
        ("sample", 1),
        ("sample", 16),
        ("predict", 1),
        ("predict", 16),
    }


def test_standard_suite_combines_quality_and_performance_suites() -> None:
    """The complete standard suite should concatenate the retained component suites."""
    standard_suite = standard.get_standard_suite()

    assert len(standard_suite) == len(standard.get_quality_suite()) + len(
        standard.get_performance_suite()
    )
    assert isinstance(standard_suite[0], PrecisionRecallBenchmark)
    assert all(isinstance(benchmark, LatencyBenchmark) for benchmark in standard_suite[1:])


def test_standard_suites_are_registered() -> None:
    """Importing the module should register standard suite factories."""
    quality_suite = get_suite("quality")
    performance_suite = get_suite("performance")
    standard_suite = get_suite("standard")

    assert len(quality_suite) == len(standard.get_quality_suite())
    assert len(performance_suite) == len(standard.get_performance_suite())
    assert len(standard_suite) == len(standard.get_standard_suite())
    assert isinstance(quality_suite[0], PrecisionRecallBenchmark)
    assert all(isinstance(benchmark, LatencyBenchmark) for benchmark in performance_suite)
