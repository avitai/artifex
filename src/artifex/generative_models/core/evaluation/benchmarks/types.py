"""Type definitions for artifex.generative_models.core.evaluation.benchmarks."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifex.utils.file_utils import ensure_valid_output_path


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark.

    Attributes:
        name: Name of the benchmark.
        description: Description of the benchmark.
        metric_names: Names of the metrics computed by the benchmark.
        metadata: Additional metadata for the benchmark.
    """

    name: str
    description: str
    metric_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of running a benchmark.

    Attributes:
        benchmark_name: Name of the benchmark.
        model_name: Name of the model.
        metrics: dictionary of metric values.
        metadata: Additional metadata for the result.
    """

    benchmark_name: str
    model_name: str
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Save the result to a file.

        Args:
            path: Path to save the result to.
        """
        # Ensure the path is in the benchmark_results directory
        valid_path = ensure_valid_output_path(path, base_dir="benchmark_results")

        # Create parent directories if needed
        Path(valid_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert JAX arrays to Python types for JSON serialization
        import jax
        import jax.numpy as jnp

        def is_jax_array(obj):
            """Check if object is a JAX array (compatible with all JAX versions)."""
            return isinstance(obj, (jnp.ndarray, jax.Array))

        serializable_dict = {}
        for key, value in self.__dict__.items():
            if is_jax_array(value):
                serializable_dict[key] = float(value) if value.size == 1 else value.tolist()
            elif isinstance(value, dict):
                # Recursively handle nested dicts
                serializable_dict[key] = {
                    k: float(v)
                    if is_jax_array(v) and v.size == 1
                    else v.tolist()
                    if is_jax_array(v)
                    else v
                    for k, v in value.items()
                }
            else:
                serializable_dict[key] = value

        with open(valid_path, "w") as f:
            json.dump(serializable_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BenchmarkResult":
        """Load a result from a file.

        Args:
            path: Path to load the result from.

        Returns:
            The loaded result.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
