"""Benchmarks for generative models.

This package provides benchmark systems for evaluating generative models
across different modalities and tasks.
"""

import flax.nnx as nnx

from artifex.benchmarks.core import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    PerformanceTracker,
)
from artifex.benchmarks.model_adapters import (
    adapt_model,
    NNXGenerativeModelAdapter,
    register_adapter,
)
from artifex.benchmarks.registry import (
    BenchmarkRegistry,
    get_benchmark,
    list_benchmarks,
    register_benchmark,
)
from artifex.benchmarks.suites.geometric_suite import (
    GeometricBenchmarkSuite,
    PointCloudGenerationBenchmark,
)
from artifex.benchmarks.suites.multi_beta_vae_suite import (
    MultiBetaVAEBenchmark,
    MultiBetaVAEBenchmarkSuite,
)
from artifex.benchmarks.suites.protein_ligand_suite import (
    ProteinLigandBenchmarkSuite,
    ProteinLigandCoDesignBenchmark,
)
from artifex.generative_models.core.protocols.evaluation import (
    BatchableDatasetProtocol,
    BenchmarkModelProtocol,
    DatasetProtocol,
)


__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkModelProtocol",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "PerformanceTracker",
    "BatchableDatasetProtocol",
    "DatasetProtocol",
    "BenchmarkRegistry",
    "NNXGenerativeModelAdapter",
    "adapt_model",
    "register_adapter",
    "get_benchmark",
    "list_benchmarks",
    "register_benchmark",
    "GeometricBenchmarkSuite",
    "PointCloudGenerationBenchmark",
    "ProteinLigandBenchmarkSuite",
    "ProteinLigandCoDesignBenchmark",
    "MultiBetaVAEBenchmarkSuite",
    "MultiBetaVAEBenchmark",
    "create_benchmark_suite",
]


def create_benchmark_suite(
    suite_type: str,
    config: dict | None = None,
    *,
    rngs: nnx.Rngs,
) -> GeometricBenchmarkSuite | ProteinLigandBenchmarkSuite | MultiBetaVAEBenchmarkSuite:
    """Create a benchmark suite of the specified type."""
    if config is None:
        config = {}

    if suite_type == "geometric":
        return GeometricBenchmarkSuite(config=config, rngs=rngs)
    if suite_type == "protein_ligand":
        return ProteinLigandBenchmarkSuite(
            dataset_config=config.get("dataset_config"),
            benchmark_config=config.get("benchmark_config"),
            rngs=rngs,
        )
    if suite_type == "multi_beta_vae":
        return MultiBetaVAEBenchmarkSuite(
            dataset_config=config.get("dataset_config"),
            benchmark_config=config.get("benchmark_config"),
            rngs=rngs,
        )
    raise ValueError(f"Unknown benchmark suite type: {suite_type}")
