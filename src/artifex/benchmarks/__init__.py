"""Benchmarks for generative models.

This package provides a comprehensive benchmark system for evaluating
generative models across different modalities and tasks.
"""

import flax.nnx as nnx

from artifex.benchmarks.base import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
)
from artifex.benchmarks.model_adapters import (
    adapt_model,
    BenchmarkModelAdapter,
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
    DatasetProtocol,
    ModelProtocol,
)


__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkSuite",
    "DatasetProtocol",
    "ModelProtocol",
    "BenchmarkRegistry",
    "BenchmarkModelAdapter",
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
    """Create a benchmark suite of the specified type.

    Args:
        suite_type: Type of benchmark suite to create
        config: Configuration for the benchmark suite
        rngs: Random number generator keys

    Returns:
        Benchmark suite instance

    Raises:
        ValueError: If the specified suite type is not recognized
    """
    if config is None:
        config = {}

    if suite_type == "geometric":
        return GeometricBenchmarkSuite(config=config, rngs=rngs)
    elif suite_type == "protein_ligand":
        return ProteinLigandBenchmarkSuite(
            dataset_config=config.get("dataset_config"),
            benchmark_config=config.get("benchmark_config"),
            rngs=rngs,
        )
    elif suite_type == "multi_beta_vae":
        return MultiBetaVAEBenchmarkSuite(
            dataset_config=config.get("dataset_config"),
            benchmark_config=config.get("benchmark_config"),
            rngs=rngs,
        )
    else:
        raise ValueError(f"Unknown benchmark suite type: {suite_type}")


# Register built-in benchmarks
def _register_builtin_benchmarks():
    """Register built-in benchmarks with the registry."""
    _ = BenchmarkRegistry()

    # Register any built-in benchmarks here
    # This is just a placeholder for now
    pass


# Initialize registry with built-in benchmarks
_register_builtin_benchmarks()
