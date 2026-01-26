"""Metrics registry for artifex.generative_models.core.evaluation."""

from typing import Any, Callable

import jax
import jax.numpy as jnp


class MetricsRegistry:
    """Singleton registry for metric computation functions.

    This registry manages all available metrics and provides
    a centralized way to compute them. It supports:
    - Registration of custom metrics
    - Metric computation with consistent interface
    - Discovery of available metrics
    """

    _instance: "MetricsRegistry" = None  # type: ignore
    _initialized = False

    def __new__(cls) -> "MetricsRegistry":
        """Create a new metrics registry instance.

        Returns:
            An instance of the MetricsRegistry
        """
        if cls._instance is None:
            cls._instance = super(MetricsRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the metrics registry."""
        if not self._initialized:
            self.metric_computers: dict[str, Callable] = {}
            self._initialized = True

    def register_metric_computer(self, name: str, computer: Callable) -> None:
        """Register a metric computation function.

        Args:
            name: Name of the metric
            computer: Function that computes the metric
        """
        self.metric_computers[name] = computer

    def compute_metrics(self, metric_name: str, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Compute specified metric.

        Args:
            metric_name: Name of registered metric
            *args: Arguments to pass to metric computer
            **kwargs: Keyword arguments to pass to metric computer

        Returns:
            dictionary of computed metrics

        Raises:
            KeyError: If metric is not registered
        """
        if metric_name not in self.metric_computers:
            raise KeyError(f"Metric '{metric_name}' not registered")

        computer = self.metric_computers[metric_name]
        return computer(*args, **kwargs)

    def list_available_metrics(self) -> list[str]:
        """Get list of available metrics.

        Returns:
            list of registered metric names
        """
        return list(self.metric_computers.keys())

    def has_metric(self, metric_name: str) -> bool:
        """Check if metric is available.

        Args:
            metric_name: Name of metric to check

        Returns:
            True if metric is registered, False otherwise
        """
        return metric_name in self.metric_computers


# Register standard metrics
def _register_standard_metrics() -> None:
    """Register commonly used metrics."""
    registry = MetricsRegistry()

    def accuracy_metric(predictions: jax.Array, targets: jax.Array) -> dict[str, float]:
        """Compute accuracy metric."""
        correct = jnp.sum(predictions == targets)
        accuracy = correct / len(targets)
        return {"accuracy": float(accuracy)}

    def mse_metric(predictions: jax.Array, targets: jax.Array) -> dict[str, float]:
        """Compute mean squared error."""
        mse = jnp.mean((predictions - targets) ** 2)
        return {"mse": float(mse)}

    def mae_metric(predictions: jax.Array, targets: jax.Array) -> dict[str, float]:
        """Compute mean absolute error."""
        mae = jnp.mean(jnp.abs(predictions - targets))
        return {"mae": float(mae)}

    # Register standard metrics
    registry.register_metric_computer("accuracy", accuracy_metric)
    registry.register_metric_computer("mse", mse_metric)
    registry.register_metric_computer("mae", mae_metric)


# Initialize standard metrics when module is imported
_register_standard_metrics()
