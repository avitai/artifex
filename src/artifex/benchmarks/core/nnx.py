"""Artifex-specific NNX benchmark base classes.

The CalibraX protocol layer owns the generic benchmark contract. This
module keeps only the NNX-specific abstract base classes that Artifex
still needs for local benchmark implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

import flax.nnx as nnx

from artifex.generative_models.core.configuration import EvaluationConfig


class BenchmarkBase(nnx.Module, ABC):
    """NNX benchmark base class for Artifex-specific implementations."""

    def __init__(self, config: EvaluationConfig, *, rngs: nnx.Rngs):
        super().__init__()
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be EvaluationConfig, got {type(config).__name__}")
        self.config = config
        self.rngs = rngs
        self._setup_benchmark_components()

    def setup(self) -> None:
        """Set up benchmark resources before execution."""

    def teardown(self) -> None:
        """Release benchmark resources after execution."""

    @abstractmethod
    def _setup_benchmark_components(self) -> None:
        """Set up benchmark-specific components."""

    @abstractmethod
    def run_training(self) -> dict[str, float]:
        """Execute the training phase of the benchmark."""

    @abstractmethod
    def run_evaluation(self) -> dict[str, float]:
        """Execute the evaluation phase of the benchmark."""

    @abstractmethod
    def get_performance_targets(self) -> dict[str, float]:
        """Return performance targets for this benchmark."""

    def validate_targets_achieved(self, metrics: dict[str, float]) -> bool:
        """Check if metrics meet performance targets."""
        targets = self.get_performance_targets()
        for metric_name, target_value in targets.items():
            if metric_name not in metrics:
                return False
            current_value = metrics[metric_name]
            if "latency" in metric_name.lower() or "time" in metric_name.lower():
                if current_value > target_value:
                    return False
            elif current_value < target_value:
                return False
        return True


class BenchmarkWithValidation(Protocol):
    """Protocol for benchmarks that support performance validation."""

    def validate_performance(self, results: dict[str, float | int]) -> dict[str, bool]:
        """Validate performance against targets."""
        ...

    def get_benchmark_info(self) -> dict[str, Any]:
        """Get benchmark metadata and configuration."""
        ...


__all__ = [
    "BenchmarkBase",
    "BenchmarkWithValidation",
]
