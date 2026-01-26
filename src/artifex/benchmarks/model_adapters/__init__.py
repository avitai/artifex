"""Model adapters for benchmarks.

This module provides adapters for different model types to be used with the
benchmark system.
"""

from artifex.benchmarks.model_adapters.base import (
    adapt_model,
    BenchmarkModelAdapter,
    NNXModelAdapter,
    register_adapter,
)
from artifex.benchmarks.model_adapters.protein_adapters import (
    ProteinPointCloudAdapter,
)


__all__ = [
    "adapt_model",
    "register_adapter",
    "BenchmarkModelAdapter",
    "NNXModelAdapter",
    "ProteinPointCloudAdapter",
]
