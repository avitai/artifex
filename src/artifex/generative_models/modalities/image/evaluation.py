"""Image evaluation metrics for the image modality.

This module provides comprehensive evaluation metrics for image generation,
including Frechet Inception Distance (FID), Inception Score (IS), and perceptual metrics.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .base import ImageModalityConfig


class ImageMetrics(nnx.Module):
    """Base class for image quality metrics."""

    def __init__(self, config: ImageModalityConfig, *, rngs: nnx.Rngs):
        """Initialize image metrics.

        Args:
            config: Image modality configuration
            rngs: Random number generators
        """
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
    """Mean Squared Error between generated and reference images."""

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

        mse = jnp.mean((generated_images - reference_images) ** 2)
        return {"mse": float(mse)}


class PSNRMetric(ImageMetrics):
    """Peak Signal-to-Noise Ratio metric."""

    def compute_metric(
        self, generated_images: jnp.ndarray, reference_images: jnp.ndarray | None = None
    ) -> dict[str, float]:
        """Compute PSNR between generated and reference images.

        Args:
            generated_images: Generated images
            reference_images: Reference images

        Returns:
            Dictionary with PSNR value
        """
        if reference_images is None:
            raise ValueError("PSNR metric requires reference images")

        mse = jnp.mean((generated_images - reference_images) ** 2)
        # Assuming images are in [0, 1] range
        max_pixel_value = 1.0
        psnr = 20 * jnp.log10(max_pixel_value / jnp.sqrt(mse + 1e-8))
        return {"psnr": float(psnr)}


class SSIMMetric(ImageMetrics):
    """Structural Similarity Index Metric."""

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

    def _gaussian_kernel(self, size: int, sigma: float = 1.5) -> jnp.ndarray:
        """Create a Gaussian kernel for SSIM computation."""
        coords = jnp.arange(size, dtype=jnp.float32) - (size - 1) / 2
        kernel = jnp.exp(-(coords**2) / (2 * sigma**2))
        kernel = kernel / jnp.sum(kernel)
        return kernel[:, None] * kernel[None, :]

    def _ssim_single_channel(self, img1: jnp.ndarray, img2: jnp.ndarray) -> jnp.ndarray:
        """Compute SSIM for a single channel."""
        # Create Gaussian kernel
        kernel = self._gaussian_kernel(self.window_size)
        kernel = kernel / jnp.sum(kernel)

        # Pad images for convolution
        pad_size = self.window_size // 2
        padding = [(pad_size, pad_size), (pad_size, pad_size)]

        img1_padded = jnp.pad(img1, padding, mode="reflect")
        img2_padded = jnp.pad(img2, padding, mode="reflect")

        # Compute local means
        mu1 = jax.scipy.signal.convolve2d(img1_padded, kernel, mode="valid")
        mu2 = jax.scipy.signal.convolve2d(img2_padded, kernel, mode="valid")

        # Compute local variances and covariance
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = jax.scipy.signal.convolve2d(img1_padded**2, kernel, mode="valid") - mu1_sq
        sigma2_sq = jax.scipy.signal.convolve2d(img2_padded**2, kernel, mode="valid") - mu2_sq
        sigma12 = (
            jax.scipy.signal.convolve2d(img1_padded * img2_padded, kernel, mode="valid") - mu1_mu2
        )

        # SSIM constants
        c1 = (self.k1 * 1.0) ** 2  # Assuming dynamic range of 1.0
        c2 = (self.k2 * 1.0) ** 2

        # Compute SSIM
        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = numerator / denominator

        return jnp.mean(ssim_map)

    def compute_metric(
        self, generated_images: jnp.ndarray, reference_images: jnp.ndarray | None = None
    ) -> dict[str, float]:
        """Compute SSIM between generated and reference images.

        Args:
            generated_images: Generated images
            reference_images: Reference images

        Returns:
            Dictionary with SSIM value
        """
        if reference_images is None:
            raise ValueError("SSIM metric requires reference images")

        # Compute SSIM for each image pair and channel
        ssim_values = []

        for gen_img, ref_img in zip(generated_images, reference_images):
            if gen_img.shape[-1] == 1:  # Grayscale
                ssim_val = self._ssim_single_channel(gen_img[..., 0], ref_img[..., 0])
                ssim_values.append(ssim_val)
            else:  # Multi-channel
                channel_ssims = []
                for c in range(gen_img.shape[-1]):
                    ssim_val = self._ssim_single_channel(gen_img[..., c], ref_img[..., c])
                    channel_ssims.append(ssim_val)
                ssim_values.append(jnp.mean(jnp.array(channel_ssims)))

        mean_ssim = jnp.mean(jnp.array(ssim_values))
        return {"ssim": float(mean_ssim)}


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
    """Comprehensive image evaluation suite."""

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
                print(f"Warning: Could not compute {metric_name}: {e}")
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
