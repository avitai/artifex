"""Core benchmark protocols and infrastructure."""

from dataclasses import dataclass
from typing import Any

from artifex.generative_models.core.configuration import EvaluationConfig


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.

    This class provides a standard configuration interface for all benchmarks
    in the generative models benchmark system.
    """

    name: str
    batch_size: int = 32
    num_samples: int = 1000
    metrics: list | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        """Initialize default metrics if not provided."""
        if self.metrics is None:
            self.metrics = []


@dataclass
class BenchmarkResult:
    """Standard result format for benchmark execution.

    This class provides a consistent result format across all benchmarks
    in the generative models benchmark system.
    """

    model_name: str
    dataset_name: str
    metrics: dict[str, float]
    config: EvaluationConfig | BenchmarkConfig
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Initialize default metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
