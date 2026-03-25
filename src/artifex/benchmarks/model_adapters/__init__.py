"""Model adapters for benchmarks.

Provides NNXGenerativeModelAdapter extending calibrax NNXBenchmarkAdapter
for adapting NNX generative models to the benchmark system.
"""

from artifex.benchmarks.model_adapters.generative import (
    adapt_model,
    NNXGenerativeModelAdapter,
    register_adapter,
)
from artifex.benchmarks.model_adapters.protein_adapters import (
    ProteinPointCloudAdapter,
)


__all__ = [
    "NNXGenerativeModelAdapter",
    "adapt_model",
    "register_adapter",
    "ProteinPointCloudAdapter",
]
