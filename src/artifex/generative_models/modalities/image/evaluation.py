"""Image evaluation metrics for the image modality.

This module provides evaluation metrics for image generation,
including MSE, PSNR, SSIM, MS-SSIM, Vendi Score, and perceptual metrics.
Delegates to calibrax for core metric computations.
"""

import logging

import jax
import jax.numpy as jnp
from calibrax.metrics.functional.image import (
    ms_ssim as calibrax_ms_ssim,
    psnr as calibrax_psnr,
    ssim as calibrax_ssim,
    vendi_score as calibrax_vendi_score,
)
from calibrax.metrics.functional.regression import mse as calibrax_mse
from flax import nnx

from .base import ImageModalityConfig


logger = logging.getLogger(__name__)


class ImageMetrics(nnx.Module):
    """Base class for image quality metrics."""

    def __init__(self, config: ImageModalityConfig, *, rngs: nnx.Rngs):
        """Initialize image metrics.

        Args:
            config: Image modality configuration
            rngs: Random number generators
        """
        super().__init__()
        self.config = config
        self.rngs = rngs

    def compute_metric(
        self, generated_images: jnp.ndarray, reference_images: jnp.ndarray | None = None
    ) -> dict[str, float]:
        """Compute the metric.

        Args:
            generated_images: Generated images
            reference_images: Reference images (optional)

        Returns:
            Dictionary with metric values
        """
        raise NotImplementedError


class MSEMetric(ImageMetrics):
    """Mean Squared Error between generated and reference images.

    Delegates to calibrax.metrics.functional.regression.mse.
    """

    def compute_metric(
        self, generated_images: jnp.ndarray, reference_images: jnp.ndarray | None = None
    ) -> dict[str, float]:
        """Compute MSE between generated and reference images.

        Args:
            generated_images: Generated images
            reference_images: Reference images

        Returns:
            Dictionary with MSE value
        """
        if reference_images is None:
            raise ValueError("MSE metric requires reference images")

        mse_value = calibrax_mse(generated_images, reference_images)
        return {"mse": float(mse_value)}


class PSNRMetric(ImageMetrics):
    """Peak Signal-to-Noise Ratio metric.

    Delegates to calibrax.metrics.functional.image.psnr per image,
    then averages across the batch.
    """

    def compute_metric(
        self, generated_images: jnp.ndarray, reference_images: jnp.ndarray | None = None
    ) -> dict[str, float]:
        """Compute PSNR between generated and reference images.

        Args:
            generated_images: Generated images [batch, H, W, C]
            reference_images: Reference images [batch, H, W, C]

        Returns:
            Dictionary with PSNR value
        """
        if reference_images is None:
            raise ValueError("PSNR metric requires reference images")

        psnr_per_image = jax.vmap(calibrax_psnr)(generated_images, reference_images)
        mean_psnr = jnp.mean(psnr_per_image)
        return {"psnr": float(mean_psnr)}


class SSIMMetric(ImageMetrics):
    """Structural Similarity Index Metric.

    Delegates to calibrax.metrics.functional.image.ssim per image,
    then averages across the batch.
    """

    def __init__(
        self,
        config: ImageModalityConfig,
        window_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize SSIM metric.

        Args:
            config: Image modality configuration
            window_size: Window size for SSIM computation
            k1: K1 parameter for SSIM
            k2: K2 parameter for SSIM
            rngs: Random number generators
        """
        super().__init__(config, rngs=rngs)
        self.window_size = window_size
        self.k1 = k1
        self.k2 = k2

    def compute_metric(
        self, generated_images: jnp.ndarray, reference_images: jnp.ndarray | None = None
    ) -> dict[str, float]:
        """Compute SSIM between generated and reference images.

        Args:
            generated_images: Generated images [batch, H, W, C]
            reference_images: Reference images [batch, H, W, C]

        Returns:
            Dictionary with SSIM value
        """
        if reference_images is None:
            raise ValueError("SSIM metric requires reference images")

        def _ssim_single(gen_img: jnp.ndarray, ref_img: jnp.ndarray) -> jnp.ndarray:
            return calibrax_ssim(
                gen_img,
                ref_img,
                filter_size=self.window_size,
                k1=self.k1,
                k2=self.k2,
            )

        ssim_per_image = jax.vmap(_ssim_single)(generated_images, reference_images)
        mean_ssim = jnp.mean(ssim_per_image)
        return {"ssim": float(mean_ssim)}


class MSSSIMMetric(ImageMetrics):
    """Multi-Scale Structural Similarity Index Metric.

    Delegates to calibrax.metrics.functional.image.ms_ssim per image.
    Requires images at least ~160x160 for default 5-scale decomposition.
    """

    def compute_metric(
        self, generated_images: jnp.ndarray, reference_images: jnp.ndarray | None = None
    ) -> dict[str, float]:
        """Compute MS-SSIM between generated and reference images.

        Args:
            generated_images: Generated images [batch, H, W, C]
            reference_images: Reference images [batch, H, W, C]

        Returns:
            Dictionary with MS-SSIM value
        """
        if reference_images is None:
            raise ValueError("MS-SSIM metric requires reference images")

        ms_ssim_per_image = jax.vmap(calibrax_ms_ssim)(generated_images, reference_images)
        mean_ms_ssim = jnp.mean(ms_ssim_per_image)
        return {"ms_ssim": float(mean_ms_ssim)}


class VendiScoreMetric(ImageMetrics):
    """Vendi Score diversity metric for generated images.

    Measures diversity using eigenvalue entropy of a similarity matrix.
    Delegates to calibrax.metrics.functional.image.vendi_score.
    """

    def compute_metric(
        self, generated_images: jnp.ndarray, reference_images: jnp.ndarray | None = None
    ) -> dict[str, float]:
        """Compute Vendi Score for generated images.

        Args:
            generated_images: Generated images [batch, H, W, C]
            reference_images: Not used (diversity is self-referential)

        Returns:
            Dictionary with Vendi Score (>= 1.0, higher means more diverse)
        """
        batch_size = generated_images.shape[0]
        flat_images = generated_images.reshape(batch_size, -1)
        norms = jnp.linalg.norm(flat_images, axis=1, keepdims=True)
        normalized = flat_images / jnp.maximum(norms, 1e-8)
        similarity_matrix = normalized @ normalized.T
        vendi = calibrax_vendi_score(similarity_matrix)
        return {"vendi_score": float(vendi)}


class PerceptualDistanceMetric(ImageMetrics):
    """Simplified perceptual distance metric based on feature differences."""

    def __init__(
        self,
        config: ImageModalityConfig,
        feature_layers: list[str] | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize perceptual distance metric.

        Args:
            config: Image modality configuration
            feature_layers: Feature layers to use (simplified implementation)
            rngs: Random number generators
        """
        super().__init__(config, rngs=rngs)
        self.feature_layers = feature_layers or ["conv_features"]

    def _extract_features(self, images: jnp.ndarray) -> jnp.ndarray:
        """Extract simple features from images.

        This is a simplified implementation. In practice, you would use
        pre-trained networks like VGG or ResNet for feature extraction.
        """
        # Simple feature extraction using gradients and textures
        # Sobel operators for edge detection
        sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=jnp.float32)
        sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=jnp.float32)

        features = []
        for img in images:
            if img.shape[-1] == 1:  # Grayscale
                img_gray = img[..., 0]
            else:  # Convert to grayscale for feature extraction
                weights = jnp.array([0.299, 0.587, 0.114])
                img_gray = jnp.sum(img * weights, axis=-1)

            # Compute gradients
            grad_x = jax.scipy.signal.convolve2d(img_gray, sobel_x, mode="same")
            grad_y = jax.scipy.signal.convolve2d(img_gray, sobel_y, mode="same")
            gradient_magnitude = jnp.sqrt(grad_x**2 + grad_y**2)

            # Pool features
            pooled_gradient = jnp.mean(gradient_magnitude)
            pooled_intensity = jnp.mean(img_gray)
            pooled_variance = jnp.var(img_gray)

            feature_vector = jnp.array([pooled_gradient, pooled_intensity, pooled_variance])
            features.append(feature_vector)

        return jnp.stack(features)

    def compute_metric(
        self, generated_images: jnp.ndarray, reference_images: jnp.ndarray | None = None
    ) -> dict[str, float]:
        """Compute perceptual distance between generated and reference images.

        Args:
            generated_images: Generated images
            reference_images: Reference images

        Returns:
            Dictionary with perceptual distance value
        """
        if reference_images is None:
            raise ValueError("Perceptual distance metric requires reference images")

        # Extract features
        gen_features = self._extract_features(generated_images)
        ref_features = self._extract_features(reference_images)

        # Compute L2 distance in feature space
        feature_distance = jnp.mean(jnp.linalg.norm(gen_features - ref_features, axis=1))

        return {"perceptual_distance": float(feature_distance)}


class ImageEvaluationSuite(nnx.Module):
    """Complete image evaluation suite."""

    def __init__(
        self,
        config: ImageModalityConfig,
        metrics: list[str] | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize image evaluation suite.

        Args:
            config: Image modality configuration
            metrics: List of metrics to compute
            rngs: Random number generators
        """
        super().__init__()
        self.config = config
        self.rngs = rngs

        # Default metrics
        if metrics is None:
            metrics = ["mse", "psnr", "ssim", "perceptual_distance"]

        # Initialize metric instances using nnx.Dict for proper parameter tracking
        metrics_dict: dict[str, nnx.Module] = {}
        for metric_name in metrics:
            if metric_name == "mse":
                metrics_dict[metric_name] = MSEMetric(config, rngs=rngs)
            elif metric_name == "psnr":
                metrics_dict[metric_name] = PSNRMetric(config, rngs=rngs)
            elif metric_name == "ssim":
                metrics_dict[metric_name] = SSIMMetric(config, rngs=rngs)
            elif metric_name == "ms_ssim":
                metrics_dict[metric_name] = MSSSIMMetric(config, rngs=rngs)
            elif metric_name == "vendi_score":
                metrics_dict[metric_name] = VendiScoreMetric(config, rngs=rngs)
            elif metric_name == "perceptual_distance":
                metrics_dict[metric_name] = PerceptualDistanceMetric(config, rngs=rngs)
            else:
                raise ValueError(f"Unknown metric: {metric_name}")
        self.metrics = nnx.Dict(metrics_dict)

    def evaluate(
        self, generated_images: jnp.ndarray, reference_images: jnp.ndarray | None = None
    ) -> dict[str, float]:
        """Evaluate generated images using all configured metrics.

        Args:
            generated_images: Generated images
            reference_images: Reference images (required for some metrics)

        Returns:
            Dictionary with all metric values
        """
        results = {}

        for metric_name, metric_instance in self.metrics.items():
            try:
                metric_results = metric_instance.compute_metric(generated_images, reference_images)
                results.update(metric_results)
            except ValueError as e:
                logger.warning("Could not compute %s: %s", metric_name, e)
                results[metric_name] = float("nan")

        return results

    def compute_diversity_metrics(self, generated_images: jnp.ndarray) -> dict[str, float]:
        """Compute diversity metrics for generated images.

        Args:
            generated_images: Generated images

        Returns:
            Dictionary with diversity metrics
        """
        # Compute pairwise distances
        batch_size = generated_images.shape[0]
        pairwise_distances = []

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                img1 = generated_images[i]
                img2 = generated_images[j]
                distance = jnp.mean((img1 - img2) ** 2)
                pairwise_distances.append(distance)

        if pairwise_distances:
            pairwise_distances = jnp.array(pairwise_distances)
            mean_distance = jnp.mean(pairwise_distances)
            std_distance = jnp.std(pairwise_distances)
        else:
            mean_distance = 0.0
            std_distance = 0.0

        # Compute entropy of pixel intensities
        flattened = generated_images.reshape(-1)
        # Discretize to compute histogram
        bins = 256
        hist, _ = jnp.histogram(flattened, bins=bins, range=(0.0, 1.0))
        hist = hist / jnp.sum(hist)  # Normalize
        # Compute entropy
        entropy = -jnp.sum(hist * jnp.log(hist + 1e-8))

        return {
            "mean_pairwise_distance": float(mean_distance),
            "std_pairwise_distance": float(std_distance),
            "pixel_entropy": float(entropy),
        }


def compute_image_metrics(
    generated_images: jnp.ndarray,
    reference_images: jnp.ndarray | None = None,
    config: ImageModalityConfig | None = None,
    metrics: list[str] | None = None,
    *,
    rngs: nnx.Rngs,
) -> dict[str, float]:
    """Compute image quality metrics.

    Args:
        generated_images: Generated images
        reference_images: Reference images
        config: Image modality configuration
        metrics: List of metrics to compute
        rngs: Random number generators

    Returns:
        Dictionary with metric values
    """
    if config is None:
        # Infer config from image shape
        height, width, channels = generated_images.shape[1:]
        config = ImageModalityConfig(height=height, width=width, channels=channels)

    evaluation_suite = ImageEvaluationSuite(config, metrics=metrics, rngs=rngs)
    return evaluation_suite.evaluate(generated_images, reference_images)
