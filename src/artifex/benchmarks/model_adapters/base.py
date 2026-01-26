"""Base model adapters for benchmarks.

This module provides the base adapter classes for different model types to be
used with the benchmark system.
"""

from abc import ABC, abstractmethod
from typing import Any, Type

import flax.nnx as nnx
import jax

from artifex.generative_models.core.protocols.evaluation import ModelProtocol


class BenchmarkModelAdapter(ModelProtocol, ABC):
    """Base class for model adapters.

    This class provides a common interface for NNX model types to be used
    with the benchmark system. All models must support the rngs parameter.
    """

    def __init__(self, model: Any) -> None:
        """Initialize the adapter.

        Args:
            model: The model to adapt.
        """
        self.model = model
        # Get the model name - handle nnx.Variable case
        model_name_attr = getattr(model, "model_name", None)
        if model_name_attr is not None and hasattr(model_name_attr, "value"):  # Handle nnx.Variable
            self._model_name = str(model_name_attr.value)
        else:
            self._model_name = model_name_attr if model_name_attr is not None else "unknown"

    @property
    def model_name(self) -> str:
        """Get the model name.

        Returns:
            The name of the model.
        """
        return self._model_name

    @abstractmethod
    def predict(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        """Make predictions from input data.

        Args:
            x: Input data.
            rngs: NNX Rngs objects for stochastic operations.

        Returns:
            Model predictions.
        """
        raise NotImplementedError("Subclasses must implement the predict method.")

    @abstractmethod
    def sample(self, *, batch_size: int = 1, rngs: nnx.Rngs) -> jax.Array:
        """Generate samples from the model.

        Args:
            batch_size: Number of samples to generate.
            rngs: NNX Rngs objects for stochastic operations.

        Returns:
            Generated samples.
        """
        raise NotImplementedError("Subclasses must implement the sample method.")

    @classmethod
    def can_adapt(cls, model: Any) -> bool:
        """Check if this adapter can adapt the given model.

        This method should be overridden by subclasses to determine if they
        can adapt a specific model type.

        Args:
            model: The model to check.

        Returns:
            True if this adapter can adapt the model, False otherwise.
        """
        return False


class NNXModelAdapter(BenchmarkModelAdapter):
    """Adapter for Flax NNX models.

    This adapter is specifically designed for models built with Flax NNX,
    following the proper state access patterns and RNG handling described
    in the critical technical guidelines.
    """

    @classmethod
    def can_adapt(cls, model: Any) -> bool:
        """Check if this adapter can adapt the given model.

        Args:
            model: The model to check.

        Returns:
            True if this adapter can adapt the model, False otherwise.
        """
        # Check if the model is an nnx.Module
        return isinstance(model, nnx.Module)

    def predict(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        """Make predictions using the NNX model.

        Args:
            x: Input data.
            rngs: NNX Rngs objects for stochastic operations.

        Returns:
            Model predictions.
        """
        if hasattr(self.model, "__call__") and callable(self.model.__call__):
            result = self.model(x, rngs=rngs)
        elif hasattr(self.model, "predict") and callable(self.model.predict):
            result = self.model.predict(x, rngs=rngs)
        else:
            model_type = type(self.model).__name__
            raise ValueError(f"NNX model {model_type} has no predict or __call__ method")

        # Ensure result is a JAX array
        return jax.numpy.asarray(result) if not isinstance(result, jax.Array) else result

    def sample(self, *, batch_size: int = 1, rngs: nnx.Rngs) -> jax.Array:
        """Generate samples from the NNX model.

        Args:
            batch_size: Number of samples to generate.
            rngs: NNX Rngs objects for stochastic operations.

        Returns:
            Generated samples.
        """
        # Use the provided rngs directly - no need for rng_key handling

        # Use the model's sample method if available
        if hasattr(self.model, "sample") and callable(self.model.sample):
            result = self.model.sample(batch_size=batch_size, rngs=rngs)
        elif hasattr(self.model, "generate") and callable(self.model.generate):
            result = self.model.generate(batch_size=batch_size, rngs=rngs)
        else:
            # No sampling method available
            model_type = type(self.model).__name__
            raise ValueError(f"NNX model {model_type} has no sample or generate method")

        # Ensure result is a JAX array
        return jax.numpy.asarray(result) if not isinstance(result, jax.Array) else result


# Registry of adapters - only include NNX adapters
_adapters: list[Type[BenchmarkModelAdapter]] = [
    NNXModelAdapter,  # Only adapter - specifically for NNX modules
]


def register_adapter(adapter_cls: Type[BenchmarkModelAdapter]) -> None:
    """Register a model adapter.

    Args:
        adapter_cls: The adapter class to register.
    """
    # Add to beginning of list for higher priority
    _adapters.insert(0, adapter_cls)


def adapt_model(model: Any) -> ModelProtocol:
    """Adapt a model to the ModelProtocol interface.

    This function tries to find a suitable adapter for the given model.
    Only NNX models are supported.

    Args:
        model: The model to adapt.

    Returns:
        An adapter for the model.

    Raises:
        ValueError: If no adapter can be found for the model.
    """
    # Try to find a suitable adapter by calling can_adapt
    for adapter_cls in _adapters:
        if adapter_cls.can_adapt(model):
            return adapter_cls(model)

    # No adapter found
    raise ValueError(
        f"No adapter found for model of type {type(model).__name__}. "
        "Only Flax NNX models are supported."
    )
