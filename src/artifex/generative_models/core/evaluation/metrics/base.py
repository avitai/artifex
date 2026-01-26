"""Base classes for evaluation metrics using Flax NNX.

This module provides the base classes for all evaluation metrics,
following NNX patterns and ensuring JAX compatibility.
"""

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class MetricModule(nnx.Module):
    """Base class for all metrics following NNX patterns.

    This class provides the foundation for implementing evaluation metrics
    that are compatible with JAX transformations and NNX modules.

    Attributes:
        name: Name of the metric
        batch_size: Batch size for processing data
    """

    def __init__(
        self,
        name: str,
        batch_size: int = 32,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the metric module.

        Args:
            name: Name of the metric
            batch_size: Batch size for processing data
            rngs: Optional RNG state for stochastic operations
        """
        super().__init__()
        self.name = name
        self.batch_size = batch_size

    def compute(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Compute the metric.

        This method should be implemented by all subclasses.

        Returns:
            Dictionary with metric names as keys and values as floats
        """
        raise NotImplementedError("Subclasses must implement the compute method")

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Alias for compute method for consistency."""
        return self.compute(*args, **kwargs)


class FeatureBasedMetric(MetricModule):
    """Base class for metrics that require feature extraction.

    This class extends MetricModule for metrics that need to extract
    features from data before computing the metric (e.g., FID, IS).

    Attributes:
        feature_extractor: Optional feature extraction function
    """

    def __init__(
        self,
        name: str,
        feature_extractor: Any | None = None,
        batch_size: int = 32,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the feature-based metric.

        Args:
            name: Name of the metric
            feature_extractor: Optional function to extract features
            batch_size: Batch size for processing
            rngs: Optional RNG state
        """
        super().__init__(name=name, batch_size=batch_size, rngs=rngs)
        self.feature_extractor = feature_extractor

    def extract_features(self, data: jax.Array, batch_size: int | None = None) -> jax.Array:
        """Extract features from data.

        Args:
            data: Input data
            batch_size: Optional batch size override

        Returns:
            Extracted features
        """
        if self.feature_extractor is None:
            # If no feature extractor, return data as-is
            return data

        batch_size = batch_size or self.batch_size
        n_samples = data.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        features = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch = data[start_idx:end_idx]
            batch_features = self.feature_extractor(batch)
            features.append(batch_features)

        return jnp.concatenate(features, axis=0)


class DistributionMetric(MetricModule):
    """Base class for metrics comparing distributions.

    This class provides common functionality for metrics that
    compare statistical properties of distributions.
    """

    def __init__(
        self,
        name: str,
        batch_size: int = 32,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the distribution metric.

        Args:
            name: Name of the metric
            batch_size: Batch size for processing
            rngs: Optional RNG state
        """
        super().__init__(name=name, batch_size=batch_size, rngs=rngs)

    @staticmethod
    def compute_statistics(features: jax.Array) -> dict[str, jax.Array]:
        """Compute statistical properties of features.

        Args:
            features: Feature array

        Returns:
            Dictionary with statistics (mean, covariance, etc.)
        """
        mu = jnp.mean(features, axis=0)
        sigma = jnp.cov(features, rowvar=False)

        return {
            "mean": mu,
            "covariance": sigma,
            "std": jnp.std(features, axis=0),
        }


class SequenceMetric(MetricModule):
    """Base class for sequence-based metrics.

    This class provides functionality for metrics that operate
    on sequences (e.g., text, time series).
    """

    def __init__(
        self,
        name: str,
        batch_size: int = 32,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the sequence metric.

        Args:
            name: Name of the metric
            batch_size: Batch size for processing
            rngs: Optional RNG state
        """
        super().__init__(name=name, batch_size=batch_size, rngs=rngs)

    def process_sequences(self, sequences: jax.Array, masks: jax.Array | None = None) -> jax.Array:
        """Process sequences with optional masking.

        Args:
            sequences: Input sequences
            masks: Optional mask array

        Returns:
            Processed sequences
        """
        if masks is None:
            return sequences

        # Apply mask to sequences
        return sequences * masks
