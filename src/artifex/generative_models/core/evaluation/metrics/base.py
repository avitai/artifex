"""Internal helper classes for evaluation metrics using Flax NNX.

These helpers stay private to the evaluation implementation layer. The shared
metric protocol base lives in ``artifex.generative_models.core.protocols``.
"""

from __future__ import annotations

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from artifex.generative_models.core.protocols.metrics import MetricBase


class FeatureBasedMetric(MetricBase):
    """Internal helper for metrics that require feature extraction."""

    def __init__(
        self,
        *,
        name: str,
        feature_extractor: Any | None = None,
        batch_size: int = 32,
        rngs: nnx.Rngs | None = None,
        modality: str = "unknown",
        higher_is_better: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            batch_size=batch_size,
            rngs=rngs,
            modality=modality,
            higher_is_better=higher_is_better,
        )
        self.feature_extractor = feature_extractor

    def extract_features(self, data: jax.Array, batch_size: int | None = None) -> jax.Array:
        """Extract features from data."""
        if self.feature_extractor is None:
            return data

        effective_batch_size = self.batch_size if batch_size is None else batch_size
        n_samples = data.shape[0]
        n_batches = (n_samples + effective_batch_size - 1) // effective_batch_size

        features = []
        for i in range(n_batches):
            start_idx = i * effective_batch_size
            end_idx = min((i + 1) * effective_batch_size, n_samples)
            batch = data[start_idx:end_idx]
            batch_features = self.feature_extractor(batch)
            features.append(batch_features)

        return jnp.concatenate(features, axis=0)


class DistributionMetric(MetricBase):
    """Internal helper for metrics that compare distributions."""

    @staticmethod
    def compute_statistics(features: jax.Array) -> dict[str, jax.Array]:
        """Compute statistical properties of features."""
        mu = jnp.mean(features, axis=0)
        sigma = jnp.cov(features, rowvar=False)

        return {
            "mean": mu,
            "covariance": sigma,
            "std": jnp.std(features, axis=0),
        }


class SequenceMetric(MetricBase):
    """Internal helper for metrics that operate on sequences."""

    def process_sequences(self, sequences: jax.Array, masks: jax.Array | None = None) -> jax.Array:
        """Process sequences with optional masking."""
        if masks is None:
            return sequences
        return sequences * masks
