"""Fréchet Inception Distance (FID) metric with explicit dependencies.

Artifex does not ship a built-in Inception-v3 checkpoint or default feature
extractor for FID. Callers must provide a real feature extractor callable.
"""

from collections.abc import Callable

import flax.nnx as nnx
import jax

from ..base import DistributionMetric, FeatureBasedMetric
from ..metric_ops import frechet_distance_from_statistics


class FrechetInceptionDistance(FeatureBasedMetric, DistributionMetric):
    """Compute FID using a caller-supplied feature extractor."""

    def __init__(
        self,
        feature_extractor: Callable[[jax.Array], jax.Array] | None,
        batch_size: int = 32,
        name: str = "fid",
        *,
        rngs: nnx.Rngs | None = None,
    ):
        if feature_extractor is None:
            raise ValueError(
                "FrechetInceptionDistance requires an explicit callable "
                "feature_extractor. Artifex does not ship a default Inception-v3 "
                "extractor."
            )
        if not callable(feature_extractor):
            raise TypeError("feature_extractor must be callable")

        FeatureBasedMetric.__init__(
            self,
            name=name,
            feature_extractor=feature_extractor,
            batch_size=batch_size,
            rngs=rngs,
            modality="image",
            higher_is_better=False,
        )

    @staticmethod
    def calculate_frechet_distance(
        mu1: jax.Array,
        sigma1: jax.Array,
        mu2: jax.Array,
        sigma2: jax.Array,
    ) -> float:
        """Calculate the Fréchet distance between two multivariate Gaussians."""
        return frechet_distance_from_statistics(mu1, sigma1, mu2, sigma2)

    def compute(
        self,
        real_images: jax.Array,
        generated_images: jax.Array,
    ) -> dict[str, float]:
        """Compute FID between real and generated images."""
        real_features = self.extract_features(real_images)
        gen_features = self.extract_features(generated_images)
        real_stats = self.compute_statistics(real_features)
        gen_stats = self.compute_statistics(gen_features)
        fid = self.calculate_frechet_distance(
            real_stats["mean"],
            real_stats["covariance"],
            gen_stats["mean"],
            gen_stats["covariance"],
        )
        return {self.name: fid}
