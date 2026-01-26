"""Inception Score implementation using JAX and NNX.

The Inception Score (IS) is a metric for evaluating generative models,
particularly GANs, based on two criteria:
1. Quality: Generated images should contain clear, recognizable objects
2. Diversity: The model should generate diverse images across classes
"""

from typing import Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from ..base import FeatureBasedMetric


class InceptionScore(FeatureBasedMetric):
    """Computes the Inception Score for a set of generated images.

    The Inception Score uses the Inception model to classify generated images
    and measures both the confidence of classifications (quality) and the
    diversity of generated images.

    Attributes:
        classifier: Function that returns classification logits
        batch_size: Batch size for feature extraction
        splits: Number of splits to use when computing the score
    """

    def __init__(
        self,
        classifier: Callable | None = None,
        batch_size: int = 32,
        splits: int = 10,
        name: str = "inception_score",
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the Inception Score metric.

        Args:
            classifier: Function that returns classification logits for images
                If None, the default Inception-v3 classifier is used
            batch_size: Batch size for feature extraction
            splits: Number of splits to use when computing the score
            name: Name of the metric
            rngs: Optional RNG state
        """
        super().__init__(
            name=name,
            feature_extractor=classifier,  # Reuse feature_extractor for classifier
            batch_size=batch_size,
            rngs=rngs,
        )
        self.splits = splits

        # Set default classifier if none provided
        if self.feature_extractor is None:
            self.feature_extractor = self._get_default_classifier()

    def _get_default_classifier(self) -> Callable:
        """Returns the default classifier (Inception-v3).

        Returns:
            A callable that classifies images
        """

        # In production, this would load the actual Inception-v3 model
        # For now, we're using a placeholder
        def classifier(images):
            # Placeholder - in production, use actual Inception-v3 model
            # This returns random logits of appropriate size for testing
            key = jax.random.PRNGKey(0)
            return jax.random.normal(key, (images.shape[0], 1000))

        return classifier

    def get_predictions(self, images: jax.Array) -> jax.Array:
        """Get classification predictions for images.

        Args:
            images: Images to classify

        Returns:
            Softmax probabilities for each image
        """
        # Extract logits using the classifier
        logits = self.extract_features(images)

        # Convert logits to probabilities with softmax
        probs = nnx.softmax(logits, axis=-1)

        return probs

    @staticmethod
    def calculate_kl_divergence(p: jax.Array, q: jax.Array) -> jax.Array:
        """Calculate the KL divergence between distributions p and q.

        Args:
            p: First probability distribution
            q: Second probability distribution

        Returns:
            KL divergence between p and q
        """
        # Add small epsilon to prevent log(0)
        eps = 1e-10
        q_safe = jnp.maximum(q, eps)
        p_safe = jnp.maximum(p, eps)

        return jnp.sum(p_safe * (jnp.log(p_safe) - jnp.log(q_safe)))

    def calculate_score(self, predictions: jax.Array, splits: int) -> tuple[float, float]:
        """Calculate the Inception Score from predictions.

        Args:
            predictions: Softmax probabilities for each image
            splits: Number of splits to use

        Returns:
            Tuple of (mean_score, std_score)
        """
        n = predictions.shape[0]
        split_size = n // splits

        # Calculate inception scores for each split
        scores = []
        for i in range(splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size

            # Get predictions for this split
            split_preds = predictions[start_idx:end_idx]

            # Calculate marginal distribution (average over samples)
            marginal = jnp.mean(split_preds, axis=0)

            # Calculate KL divergence between individual predictions and marginal
            kl_divs = jnp.array(
                [self.calculate_kl_divergence(pred, marginal) for pred in split_preds]
            )

            # Calculate mean KL divergence and exponentiate
            score = jnp.exp(jnp.mean(kl_divs))
            scores.append(float(score))

        # Return mean and standard deviation
        scores_array = jnp.array(scores)
        return float(jnp.mean(scores_array)), float(jnp.std(scores_array))

    def compute(
        self,
        images: jax.Array,
        splits: int | None = None,
    ) -> dict[str, float]:
        """Compute the Inception Score for generated images.

        Args:
            images: Generated images
            splits: Number of splits (uses self.splits if None)

        Returns:
            Dictionary containing the Inception Score mean and std
        """
        # Get predictions
        predictions = self.get_predictions(images)

        # Calculate score
        splits = splits or self.splits
        mean_score, std_score = self.calculate_score(predictions, splits)

        return {f"{self.name}_mean": mean_score, f"{self.name}_std": std_score}
