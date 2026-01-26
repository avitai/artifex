"""Performance benchmarks for generative models.

This package provides benchmarks for evaluating the computational performance
of generative models, including:

- Latency: Measures the time it takes to generate samples or make predictions.
- Throughput: Measures the number of samples that can be generated per second.
- Memory: Measures the memory usage during inference.
- Scaling: Measures how performance scales with model and batch size.
- Optimization: Measures training performance and convergence rates.
"""

from artifex.benchmarks.performance.latency import (
    LatencyBenchmark,
    measure_inference_latency,
)
from artifex.benchmarks.performance.optimization import (
    OptimizationBenchmark,
    OptimizationMetricsConfig,
    OptimizerComparisonBenchmark,
    TrainerProtocol,
    TrainingConvergenceBenchmark,
    TrainingCurvePoint,
)


__all__ = [
    # Latency benchmarks
    "LatencyBenchmark",
    "measure_inference_latency",
    # Optimization benchmarks
    "OptimizationBenchmark",
    "TrainingConvergenceBenchmark",
    "OptimizerComparisonBenchmark",
    "OptimizationMetricsConfig",
    "TrainerProtocol",
    "TrainingCurvePoint",
]
