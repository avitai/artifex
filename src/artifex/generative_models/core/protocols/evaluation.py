"""Protocols for evaluation of generative models.

BenchmarkModelProtocol is artifex-specific (requires rngs for NNX models).
DatasetProtocol and BatchableDatasetProtocol are re-exported from calibrax.
"""

from typing import Protocol, runtime_checkable

import jax
from calibrax.core import BatchableDatasetProtocol, DatasetProtocol
from flax import nnx


@runtime_checkable
class BenchmarkModelProtocol(Protocol):
    """Protocol for NNX models that can be benchmarked.

    All models must be built with Flax NNX and must follow the proper
    RNG handling patterns. The rngs parameter is required for all operations.
    """

    @property
    def model_name(self) -> str:
        """Get the name of the model."""
        ...

    def predict(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        """Make predictions from input data.

        Args:
            x: Input data.
            rngs: NNX Rngs objects for stochastic operations.

        Returns:
            Model predictions.
        """
        ...

    def sample(self, *, batch_size: int = 1, rngs: nnx.Rngs) -> jax.Array:
        """Generate samples from the model.

        Args:
            batch_size: Number of samples to generate.
            rngs: NNX Rngs objects for stochastic operations.

        Returns:
            Generated samples.
        """
        ...


__all__ = [
    "BatchableDatasetProtocol",
    "BenchmarkModelProtocol",
    "DatasetProtocol",
]
