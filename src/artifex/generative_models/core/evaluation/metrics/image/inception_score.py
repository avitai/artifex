"""Inception Score metric with explicit classifier dependencies.

Artifex does not ship a built-in Inception-v3 classifier for Inception
Score. Callers must provide a real classifier callable.
"""

from collections.abc import Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from ..base import FeatureBasedMetric


class InceptionScore(FeatureBasedMetric):
    """Compute the Inception Score with a caller-supplied classifier."""

    def __init__(
        self,
        classifier: Callable[[jax.Array], jax.Array] | None,
        batch_size: int = 32,
        splits: int = 10,
        name: str = "inception_score",
        *,
        rngs: nnx.Rngs | None = None,
    ):
        if classifier is None:
            raise ValueError(
                "InceptionScore requires an explicit callable classifier. "
                "Artifex does not ship a default Inception-v3 classifier."
            )
        if not callable(classifier):
            raise TypeError("classifier must be callable")

        super().__init__(
            name=name,
            feature_extractor=classifier,
            batch_size=batch_size,
            rngs=rngs,
            modality="image",
            higher_is_better=True,
        )
        self.splits = splits

    def get_predictions(self, images: jax.Array) -> jax.Array:
        """Get classification predictions for images."""
        logits = self.extract_features(images)
        return nnx.softmax(logits, axis=-1)

    @staticmethod
    def calculate_kl_divergence(p: jax.Array, q: jax.Array) -> jax.Array:
        """Calculate the KL divergence between distributions p and q."""
        eps = 1e-10
        q_safe = jnp.maximum(q, eps)
        p_safe = jnp.maximum(p, eps)
        return jnp.sum(p_safe * (jnp.log(p_safe) - jnp.log(q_safe)))

    def calculate_score(self, predictions: jax.Array, splits: int) -> tuple[float, float]:
        """Calculate the Inception Score from class predictions."""
        if splits <= 0:
            raise ValueError("splits must be a positive integer")

        n = predictions.shape[0]
        if n < splits:
            raise ValueError("InceptionScore requires at least as many samples as splits.")

        split_size = n // splits
        scores = []
        for i in range(splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size
            split_preds = predictions[start_idx:end_idx]
            marginal = jnp.mean(split_preds, axis=0)
            kl_divs = jnp.array(
                [self.calculate_kl_divergence(pred, marginal) for pred in split_preds]
            )
            scores.append(float(jnp.exp(jnp.mean(kl_divs))))
        scores_array = jnp.array(scores)
        return float(jnp.mean(scores_array)), float(jnp.std(scores_array))

    def compute(
        self,
        images: jax.Array,
        splits: int | None = None,
    ) -> dict[str, float]:
        """Compute the Inception Score for generated images."""
        predictions = self.get_predictions(images)
        effective_splits = self.splits if splits is None else splits
        mean_score, std_score = self.calculate_score(predictions, effective_splits)
        return {f"{self.name}_mean": mean_score, f"{self.name}_std": std_score}
