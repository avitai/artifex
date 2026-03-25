"""Generative model adapter extending calibrax NNXBenchmarkAdapter.

Provides NNXGenerativeModelAdapter with domain-specific predict() and
sample() methods that handle RNG state for stochastic generative models.
"""

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from calibrax.core import AdapterRegistry, NNXBenchmarkAdapter


class NNXGenerativeModelAdapter(NNXBenchmarkAdapter):
    """Adapter for NNX generative models.

    Extends calibrax NNXBenchmarkAdapter with domain-specific predict()
    and sample() methods that require explicit RNG state for stochastic
    generative model operations.
    """

    def __init__(self, model: nnx.Module) -> None:
        """Initialize the adapter.

        Args:
            model: NNX module to adapt.
        """
        super().__init__(model)
        # Resolve model_name, handling nnx.Variable wrapping
        model_name_attr = getattr(model, "model_name", None)
        if isinstance(model_name_attr, nnx.Variable):
            self._model_name_str = str(model_name_attr.get_value())
        elif isinstance(model_name_attr, str):
            self._model_name_str = model_name_attr
        else:
            self._model_name_str = getattr(model, "name", None) or "unknown"

    @property
    def model_name(self) -> str:
        """Get the model name.

        Returns:
            The name of the adapted model.
        """
        return self._model_name_str

    def predict(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        """Make predictions from input data.

        Resolves the prediction method in order: model.__call__, model.predict.

        Args:
            x: Input data.
            rngs: NNX Rngs for stochastic operations.

        Returns:
            Model predictions.

        Raises:
            ValueError: If model has no predict or __call__ method.
        """
        if hasattr(self.model, "__call__") and callable(self.model.__call__):
            result = self.model(x, rngs=rngs)
        elif hasattr(self.model, "predict") and callable(self.model.predict):
            result = self.model.predict(x, rngs=rngs)
        else:
            model_type = type(self.model).__name__
            raise ValueError(f"NNX model {model_type} has no predict or __call__ method")

        return jnp.asarray(result) if not isinstance(result, jax.Array) else result

    def sample(self, *, batch_size: int = 1, rngs: nnx.Rngs) -> jax.Array:
        """Generate samples from the model.

        Resolves the sampling method in order: model.sample, model.generate.

        Args:
            batch_size: Number of samples to generate.
            rngs: NNX Rngs for stochastic operations.

        Returns:
            Generated samples.

        Raises:
            ValueError: If model has no sample or generate method.
        """
        if hasattr(self.model, "sample") and callable(self.model.sample):
            result = self.model.sample(batch_size=batch_size, rngs=rngs)
        elif hasattr(self.model, "generate") and callable(self.model.generate):
            result = self.model.generate(batch_size=batch_size, rngs=rngs)
        else:
            model_type = type(self.model).__name__
            raise ValueError(f"NNX model {model_type} has no sample or generate method")

        return jnp.asarray(result) if not isinstance(result, jax.Array) else result


# Module-level adapter registry backed by calibrax
_adapter_registry = AdapterRegistry()
_adapter_registry.register(NNXGenerativeModelAdapter)


def register_adapter(adapter_cls: type) -> None:
    """Register a model adapter (highest priority).

    Args:
        adapter_cls: Adapter class with can_adapt() classmethod.
    """
    _adapter_registry.register(adapter_cls)


def adapt_model(model: Any) -> NNXGenerativeModelAdapter:
    """Adapt a model using the adapter registry.

    Only NNX models are supported.

    Args:
        model: The model to adapt.

    Returns:
        An adapter wrapping the model.

    Raises:
        ValueError: If no adapter can handle the model.
    """
    return _adapter_registry.adapt(model)
