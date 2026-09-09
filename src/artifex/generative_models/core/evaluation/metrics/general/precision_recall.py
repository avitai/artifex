"""Precision and Recall metrics for generative models using JAX.

This module implements the improved Precision and Recall metrics for evaluating
generative models as proposed in "Improved Precision and Recall Metric for
Assessing Generative Models" (Kynkäänniemi et al., 2019).

All implementations use JAX for compatibility with NNX modules.
"""

from collections.abc import Callable

import flax.nnx as nnx
import jax
from calibrax.metrics.functional.generative import (
    density_weighted_precision,
    density_weighted_recall,
    manifold_precision,
    manifold_radii,
    manifold_recall,
)

from ..base import FeatureBasedMetric


class PrecisionRecall(FeatureBasedMetric):
    """Computes Precision and Recall metrics for generative models.

    These metrics assess the quality and diversity of generated samples:
    - Precision: Fraction of generated samples that fall within the manifold
      of real samples (measures quality/fidelity)
    - Recall: Fraction of real samples covered by the manifold of generated
      samples (measures diversity)

    Attributes:
        k: Number of nearest neighbors for manifold estimation
    """

    def __init__(
        self,
        feature_extractor: Callable[[jax.Array], jax.Array] | None = None,
        batch_size: int = 32,
        k: int = 3,
        name: str = "precision_recall",
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the Precision and Recall metric.

        Args:
            feature_extractor: Function that extracts features from samples
            batch_size: Batch size for feature extraction
            k: Number of nearest neighbors for manifold estimation
            name: Name of the metric
            rngs: Optional RNG state
        """
        super().__init__(
            name=name, feature_extractor=feature_extractor, batch_size=batch_size, rngs=rngs
        )
        self.k = k

    def compute_manifold_radii(self, features: jax.Array, k: int) -> jax.Array:
        """Compute manifold radii for each feature point.

        The manifold radius is the distance to the k-th nearest neighbor (excluding self).

        Args:
            features: Feature vectors of shape (n_samples, feature_dim)
            k: Number of nearest neighbors to consider (excluding self)

        Returns:
            Manifold radii for each feature point of shape (n_samples,)
        """
        return manifold_radii(features, k=k)

    def compute_precision(self, real_features: jax.Array, gen_features: jax.Array, k: int) -> float:
        """Compute precision: fraction of generated samples within real manifold.

        Args:
            real_features: Features of real samples
            gen_features: Features of generated samples
            k: Number of nearest neighbors for manifold estimation

        Returns:
            Precision value (0 to 1)
        """
        return float(manifold_precision(real_features, gen_features, k=k))

    def compute_recall(self, real_features: jax.Array, gen_features: jax.Array, k: int) -> float:
        """Compute recall: fraction of real samples covered by generated manifold.

        Args:
            real_features: Features of real samples
            gen_features: Features of generated samples
            k: Number of nearest neighbors for manifold estimation

        Returns:
            Recall value (0 to 1)
        """
        return float(manifold_recall(real_features, gen_features, k=k))

    def compute_f1_score(self, precision: float, recall: float) -> float:
        """Compute F1 score from precision and recall.

        Args:
            precision: Precision value
            recall: Recall value

        Returns:
            F1 score
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def compute(
        self,
        real_samples: jax.Array,
        generated_samples: jax.Array,
        k: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        """Compute precision and recall for the given samples.

        Args:
            real_samples: Real samples
            generated_samples: Generated samples
            k: Number of nearest neighbors (uses self.k if None)
            batch_size: Optional batch size for feature extraction

        Returns:
            Dictionary containing precision, recall, and F1 score
        """
        # Extract features
        real_features = self.extract_features(real_samples, batch_size)
        gen_features = self.extract_features(generated_samples, batch_size)

        # Use provided k or default
        k_value = k or self.k

        # Compute metrics
        precision = self.compute_precision(real_features, gen_features, k_value)
        recall = self.compute_recall(real_features, gen_features, k_value)
        f1 = self.compute_f1_score(precision, recall)

        return {
            f"{self.name}_precision": precision,
            f"{self.name}_recall": recall,
            f"{self.name}_f1": f1,
        }


def precision_recall(
    real_samples: jax.Array,
    generated_samples: jax.Array,
    feature_extractor: Callable[[jax.Array], jax.Array] | None = None,
    batch_size: int = 32,
    k: int = 3,
) -> dict[str, float]:
    """Convenience function to compute precision and recall.

    Args:
        real_samples: Real samples
        generated_samples: Generated samples
        feature_extractor: Function that extracts features from samples
        batch_size: Batch size for feature extraction
        k: Number of nearest neighbors for manifold estimation

    Returns:
        Dictionary containing precision, recall, and F1 score
    """
    # Create default rngs for convenience function
    rngs = nnx.Rngs(0)
    pr_metric = PrecisionRecall(
        feature_extractor=feature_extractor, batch_size=batch_size, k=k, rngs=rngs
    )
    return pr_metric.compute(real_samples, generated_samples, k)


def density_precision_recall(
    real_samples: jax.Array,
    generated_samples: jax.Array,
    feature_extractor: Callable[[jax.Array], jax.Array] | None = None,
    batch_size: int = 32,
    k: int = 5,
) -> dict[str, float]:
    """Convenience function to compute improved precision and recall.

    Args:
        real_samples: Real samples
        generated_samples: Generated samples
        feature_extractor: Function that extracts features from samples
        batch_size: Batch size for feature extraction
        k: Number of nearest neighbors for density estimation

    Returns:
        Dictionary containing precision, recall, and F1 score
    """
    # Create default rngs for convenience function
    rngs = nnx.Rngs(0)
    pr_metric = DensityPrecisionRecall(
        feature_extractor=feature_extractor, batch_size=batch_size, k=k, rngs=rngs
    )
    return pr_metric.compute(real_samples, generated_samples, k)


class DensityPrecisionRecall(FeatureBasedMetric):
    """Improved Precision and Recall metric based on density estimation.

    This implementation uses a k-nearest neighbor density estimator to more
    robustly measure the overlap between real and generated distributions.

    Attributes:
        k: Number of nearest neighbors for density estimation
    """

    def __init__(
        self,
        feature_extractor: Callable[[jax.Array], jax.Array] | None = None,
        batch_size: int = 32,
        k: int = 5,
        name: str = "density_precision_recall",
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the Density Precision and Recall metric.

        Args:
            feature_extractor: Function that extracts features from samples
            batch_size: Batch size for feature extraction
            k: Number of nearest neighbors for density estimation
            name: Name of the metric
            rngs: Optional RNG state
        """
        super().__init__(
            name=name, feature_extractor=feature_extractor, batch_size=batch_size, rngs=rngs
        )
        self.k = k

    def estimate_densities(self, features: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
        """Estimate densities for each feature point using k-NN.

        Args:
            features: Feature vectors
            k: Number of nearest neighbors (excluding self)

        Returns:
            Tuple of (radii, densities) for each feature point
        """
        radii = manifold_radii(features, k=k)
        return radii, 1.0 / (radii + 1e-10)

    def compute_improved_precision_recall(
        self, real_features: jax.Array, gen_features: jax.Array, k: int
    ) -> tuple[float, float]:
        """Compute improved precision and recall using density estimation.

        Args:
            real_features: Features of real samples
            gen_features: Features of generated samples
            k: Number of nearest neighbors for density estimation

        Returns:
            Tuple of (precision, recall)
        """
        return (
            float(density_weighted_precision(real_features, gen_features, k=k)),
            float(density_weighted_recall(real_features, gen_features, k=k)),
        )

    def compute(
        self,
        real_samples: jax.Array,
        generated_samples: jax.Array,
        k: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        """Compute improved precision and recall for the given samples.

        Args:
            real_samples: Real samples
            generated_samples: Generated samples
            k: Number of nearest neighbors (uses self.k if None)
            batch_size: Optional batch size for feature extraction

        Returns:
            Dictionary containing precision, recall, and F1 score
        """
        # Extract features
        real_features = self.extract_features(real_samples, batch_size)
        gen_features = self.extract_features(generated_samples, batch_size)

        # Use provided k or default
        k_value = k or self.k

        # Compute improved precision and recall
        precision, recall = self.compute_improved_precision_recall(
            real_features, gen_features, k_value
        )

        # Compute F1 score
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            f"{self.name}_precision": precision,
            f"{self.name}_recall": recall,
            f"{self.name}_f1": f1,
        }
