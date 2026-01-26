"""Base utilities for generative models.

This module provides type aliases and common utilities for generative models.
All generative models should inherit from GenerativeModel in core.nn module.
"""

from typing import Any, Protocol, runtime_checkable

import jax
from flax import nnx


@runtime_checkable
class GenerativeModelProtocol(Protocol):
    """Protocol defining the interface for all generative models.

    This protocol specifies the essential methods that all generative models
    must implement:

    1. __call__: Forward pass through the model
    2. generate: Sample generation from the model
    3. loss_fn: Loss computation for model training
    4. sample: Alias for generate (backward compatibility)

    All methods must accept an `rngs` parameter for proper NNX compatibility.

    Using a protocol allows for both static type checking (at development time) and
    runtime verification using isinstance(), without requiring inheritance from a base class.
    This approach enables greater flexibility in implementation while maintaining a
    consistent interface.
    """

    def __call__(self, x: Any, *, rngs: nnx.Rngs | None = None, **kwargs) -> dict[str, Any]:
        """Forward pass through the model.

        This method represents the standard forward pass for inference.

        Args:
            x: Input data for the model.
            rngs: Optional random number generator for stochastic operations.
            **kwargs: Additional keyword arguments for the model.

        Returns:
            A dictionary containing the model outputs, which typically includes
            predictions, intermediate activations, or any other relevant information.
        """
        ...

    def generate(self, n_samples: int = 1, *, rngs: nnx.Rngs | None = None, **kwargs) -> jax.Array:
        """Generate samples from the model.

        This method is used for sample generation during inference.

        Args:
            n_samples: Number of samples to generate.
            rngs: Optional random number generator for stochastic operations.
            **kwargs: Additional keyword arguments for controlling the generation process.

        Returns:
            JAX arrays containing generated samples. The exact shape and structure
            depends on the specific model implementation.
        """
        ...

    def loss_fn(
        self,
        batch: Any,
        model_outputs: dict[str, Any],
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Compute loss function for model training.

        This method calculates the loss used for model optimization.

        Args:
            batch: Batch of data used for computing the loss.
            model_outputs: Dictionary of model outputs from the forward pass.
            rngs: Optional random number generator for stochastic operations.
            **kwargs: Additional keyword arguments for loss computation.

        Returns:
            A dictionary containing the loss value and any auxiliary losses or metrics.
            Typically includes a 'loss' key with the primary loss value.
        """
        ...

    def sample(self, num_samples: int, **kwargs: Any) -> jax.Array:
        """Generate samples from the model (alias for generate).

        Args:
            num_samples: Number of samples to generate.
            **kwargs: Additional keyword arguments.

        Returns:
            Generated samples as JAX arrays.
        """
        ...


# Type alias for any object that implements the GenerativeModel interface
# All models should inherit from artifex.generative_models.core.base.GenerativeModel
GenerativeModelType = Any

__all__ = ["GenerativeModelType", "GenerativeModelProtocol"]
