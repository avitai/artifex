"""Protocols for evaluation of generative models."""

from typing import Any, Protocol, runtime_checkable

import jax
from flax import nnx


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol defining the interface for NNX models that can be benchmarked.

    All models must be built with Flax NNX and must follow the proper
    RNG handling patterns described in the critical technical guidelines.
    The rngs parameter is required for all operations.
    """

    @property
    def model_name(self) -> str:
        """Get the name of the model.

        Returns:
            The model name.
        """
        ...

    def predict(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        """Make predictions from input data.

        Args:
            x: Input data.
            rngs: dictionary of NNX Rngs objects for stochastic operations.

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


class DatasetProtocol(Protocol):
    """Protocol defining the interface for datasets that can be used in benchmarks."""

    def __len__(self) -> int:
        """Get the number of examples in the dataset.

        Returns:
            Number of examples.
        """
        ...

    def __getitem__(self, idx: int) -> Any:
        """Get an example from the dataset.

        Args:
            idx: Index of the example.

        Returns:
            The example.
        """
        ...


class BatchableDatasetProtocol(Protocol):
    """Protocol for datasets that support batch retrieval."""

    def __len__(self) -> int:
        """Get the number of examples in the dataset.

        Returns:
            Number of examples.
        """
        ...

    def __getitem__(self, idx: int) -> Any:
        """Get an example from the dataset.

        Args:
            idx: Index of the example.

        Returns:
            The example.
        """
        ...

    def get_batch(self, batch_size: int, start_idx: int) -> dict[str, Any]:
        """Get a batch of data.

        Args:
            batch_size: Size of the batch
            start_idx: Starting index

        Returns:
            Batch data dictionary
        """
        ...
