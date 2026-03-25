"""Shared protocol types for generative models."""

from typing import Any, Protocol, runtime_checkable

import jax
from flax import nnx


@runtime_checkable
class GenerativeModelProtocol(Protocol):
    """Shared inference/generation protocol for generative models.

    This base protocol captures only the capabilities that Artifex treats as
    broadly shared across model families:

    1. `__call__`: forward/inference pass
    2. `generate`: ergonomic user-facing generation entrypoint

    Family-native mathematical capabilities such as `sample`, `log_prob`,
    `encode`, `decode`, or explicit objective helpers stay on the concrete
    families that meaningfully support them.
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


@runtime_checkable
class TrainableGenerativeModelProtocol(GenerativeModelProtocol, Protocol):
    """Protocol for single-objective model surfaces that expose `loss_fn`.

    Multi-objective families such as adversarial models may instead expose
    explicit family-local objective helpers or use trainer-owned objectives.
    """

    def loss_fn(
        self,
        batch: Any,
        model_outputs: dict[str, Any],
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Compute a model-owned training objective for single-objective families."""
        ...


# Type alias for any object that implements the GenerativeModel interface
# All models should inherit from artifex.generative_models.core.base.GenerativeModel
GenerativeModelType = Any

__all__ = [
    "GenerativeModelType",
    "GenerativeModelProtocol",
    "TrainableGenerativeModelProtocol",
]
