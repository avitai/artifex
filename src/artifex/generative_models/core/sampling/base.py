"""Base classes for sampling algorithms."""

import abc
from typing import Any

import jax


class SamplingAlgorithm(abc.ABC):
    """Abstract base class for sampling algorithms."""

    @abc.abstractmethod
    def init(self, x: jax.Array, key: jax.Array) -> Any:
        """Initialize the sampler state.

        Args:
            x: Initial position.
            key: Random key.

        Returns:
            Initial state.
        """
        pass

    @abc.abstractmethod
    def step(self, state: Any) -> tuple[Any, dict]:
        """Perform one sampling step.

        Args:
            state: Current state.

        Returns:
            New state and auxiliary information.
        """
        pass
