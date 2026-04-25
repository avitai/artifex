"""Typed benchmark model surfaces for generative model adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable, TypeAlias

import jax
from flax import nnx


ResultMap: TypeAlias = Mapping[str, Any]


@runtime_checkable
class CallablePredictorProtocol(Protocol):
    """Protocol for models that predict by direct call."""

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        """Return predictions for the input batch."""
        ...


@runtime_checkable
class PredictMethodProtocol(Protocol):
    """Protocol for models exposing a predict method."""

    def predict(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        """Return predictions for the input batch."""
        ...


@runtime_checkable
class SamplerProtocol(Protocol):
    """Protocol for models exposing a sample method."""

    def sample(self, *, batch_size: int = 1, rngs: nnx.Rngs) -> jax.Array:
        """Return generated samples."""
        ...


@runtime_checkable
class GeneratorProtocol(Protocol):
    """Protocol for models exposing a generate method."""

    def generate(self, *, batch_size: int = 1, rngs: nnx.Rngs) -> jax.Array:
        """Return generated samples."""
        ...


@runtime_checkable
class BatchableDatasetSurface(Protocol):
    """Protocol for datasets that return named JAX batches."""

    def get_batch(self, batch_size: int, *, rngs: nnx.Rngs | None = None) -> ResultMap:
        """Return one batch of named arrays or metadata."""
        ...


__all__ = [
    "BatchableDatasetSurface",
    "CallablePredictorProtocol",
    "GeneratorProtocol",
    "PredictMethodProtocol",
    "ResultMap",
    "SamplerProtocol",
]
