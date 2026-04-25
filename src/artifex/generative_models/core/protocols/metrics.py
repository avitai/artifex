"""Metric protocol definitions for artifex.generative_models.core.

MetricBase is the single shared runtime base for Artifex metrics. It owns only
metric metadata and call semantics; config adaptation belongs to the concrete
runtime layers that choose to use configs.
"""

from __future__ import annotations

import flax.nnx as nnx


class MetricBase(nnx.Module):
    """Single shared runtime base class for Artifex metrics."""

    def __init__(
        self,
        *,
        name: str | None = None,
        batch_size: int = 32,
        rngs: nnx.Rngs | None = None,
        modality: str = "unknown",
        higher_is_better: bool = True,
    ) -> None:
        """Initialize metric protocol state."""
        super().__init__()

        if name is None:
            raise TypeError("name must be provided")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(modality, str) or not modality.strip():
            raise ValueError("modality must be a non-empty string")
        if not isinstance(higher_is_better, bool):
            raise TypeError("higher_is_better must be a bool")

        self._name = name
        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.config: object | None = None
        self.rngs = rngs
        self.modality = modality
        self._higher_is_better = higher_is_better

    @property
    def name(self) -> str:
        """Metric name identifier."""
        return self._name

    @property
    def higher_is_better(self) -> bool:
        """Whether higher values indicate better performance."""
        return self._higher_is_better

    def compute(self, *args, **kwargs) -> dict[str, float]:
        """Compute metric values."""
        raise NotImplementedError("Subclasses must implement compute().")

    def validate_inputs(self, *args, **kwargs) -> None:
        """Validate input compatibility for the metric."""

    def __call__(self, *args, **kwargs) -> dict[str, float]:
        """Alias for compute to keep metric usage uniform."""
        return self.compute(*args, **kwargs)
