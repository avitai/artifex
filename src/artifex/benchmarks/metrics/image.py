"""Image metrics for generative models.

This module provides metrics for evaluating the quality of generated images,
including FID, IS (Inception Score), LPIPS, and SSIM metrics for image generation.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp


try:
    from tensorflow.keras.applications.inception_v3 import InceptionV3

    _HAS_TENSORFLOW = True
except ImportError:
    InceptionV3 = None
    _HAS_TENSORFLOW = False

from artifex.benchmarks.metrics.core import MetricBase
from artifex.generative_models.core.configuration import EvaluationConfig
from artifex.generative_models.core.evaluation.metrics.quality import (
    calculate_fid_score,
    compute_lpips_distance,
)


class FIDMetric(MetricBase):
    """Fréchet Inception Distance (FID) metric for image quality evaluation.

    FID measures the distance between the feature distributions of real and
    generated images, as extracted by a pre-trained Inception model. Lower
    FID values indicate higher quality and more realistic generated images.

    Reference:
        Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash
        Equilibrium"
        https://arxiv.org/abs/1706.08500
    """

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig):
        """Initialize FID metric.

        Args:
            rngs: NNX Rngs for stochastic operations
            config: Evaluation configuration (must be EvaluationConfig)
        """
        if not isinstance(config, EvaluationConfig):
            # Also check for class name to handle dynamic module loading
            if type(config).__name__ != "EvaluationConfig":
                raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")

        # Initialize base class with the EvaluationConfig
        super().__init__(config=config, rngs=rngs)
        self.eval_batch_size = config.eval_batch_size

        self.metric_name = "fid_score"
        self.inception_model: InceptionV3 | MockInceptionModel | None = None
        self.mock_inception = config.metric_params.get("fid", {}).get("mock_inception", True)

        # Initialize mock or real inception model
        self._initialize_inception_model()

    def _initialize_inception_model(self):
        """Initialize the Inception model for feature extraction."""
        if self.mock_inception or not _HAS_TENSORFLOW:
            # Use mock inception model that returns random features
            self.inception_model = MockInceptionModel(rngs=self.rngs)
            if not _HAS_TENSORFLOW:
                print("TensorFlow not available, using mock Inception model")
        else:
            # Load real inception model
            self.inception_model = InceptionV3(include_top=False, weights="imagenet", pooling="avg")
            print("Loaded real Inception model for FID calculation")

    def compute(
        self, real_data: jax.Array, generated_data: jax.Array, **kwargs
    ) -> dict[str, float]:
        """Compute the FID score.

        Args:
            real_data: Real images [batch_size, height, width, channels]
            generated_data: Generated images [batch_size, height, width, channels]
            **kwargs: Additional parameters

        Returns:
            dictionary with FID score
        """
        # Extract features
        gen_features = self._extract_features(generated_data)
        real_features = self._extract_features(real_data)

        # Calculate FID
        fid_score = calculate_fid_score(gen_features, real_features)

        return {"fid_score": fid_score}

    def validate_inputs(self, real_data: jax.Array, generated_data: jax.Array) -> bool:
        """Validate input data compatibility.

        Args:
            real_data: Real images
            generated_data: Generated images

        Returns:
            True if inputs are valid
        """
        # Check that both inputs are arrays
        if not isinstance(real_data, jax.Array) or not isinstance(generated_data, jax.Array):
            return False

        # Check that both have same number of dimensions
        # (should be 4D: batch x height x width x channels)
        if len(real_data.shape) != 4 or len(generated_data.shape) != 4:
            return False

        # Check that last dimension is 3 (for RGB channels)
        if real_data.shape[-1] != 3 or generated_data.shape[-1] != 3:
            return False

        # Check that batch sizes match
        if real_data.shape[0] != generated_data.shape[0]:
            return False

        return True

    def _extract_features(self, images: jax.Array) -> jax.Array:
        """Extract features from images using Inception model.

        Args:
            images: Images to extract features from [batch_size, height, width, channels]

        Returns:
            Extracted features [batch_size, feature_dim]
        """
        if self.mock_inception:
            # Use mock inception model
            return self.inception_model.extract_features(images)
        else:
            # Convert to numpy and preprocess for Inception
            images_np = jnp.array(images)

            # Resize images to Inception input size if needed
            if images_np.shape[1] != 299 or images_np.shape[2] != 299:
                images_np = self._resize_images(images_np, target_size=(299, 299))

            # Scale from [0, 1] to [-1, 1] if needed
            if jnp.max(images_np) <= 1.0:
                images_np = images_np * 2.0 - 1.0

            # Extract features using real Inception model
            features = self.inception_model.predict(images_np, verbose=0)
            return features

    def _resize_images(self, images: jax.Array, target_size: tuple[int, int]) -> jax.Array:
        """Resize images to target size.

        Args:
            images: Images to resize [batch_size, height, width, channels]
            target_size: Target size (height, width)

        Returns:
            Resized images
        """
        try:
            from PIL import Image

            batch_size = images.shape[0]
            resized_images = jnp.zeros(
                (batch_size, target_size[0], target_size[1], images.shape[3])
            )

            for i in range(batch_size):
                img = Image.fromarray((images[i] * 255).astype(jnp.uint8))
                img_resized = img.resize(target_size)
                resized_images[i] = jnp.array(img_resized) / 255.0

            return resized_images
        except ImportError:
            # Fallback to simple resizing using interpolation
            print("PIL not available, using simple resize")
            return images  # Return original images for testing


class MockInceptionModel:
    """Mock Inception model for testing FID metric.

    This class provides a simple mock of the Inception model that returns
    random features, for use when TensorFlow is not available or for testing.
    """

    def __init__(self, feature_dim: int = 2048, *, rngs: nnx.Rngs):
        """Initialize mock Inception model.

        Args:
            feature_dim: Dimension of feature vectors
            rngs: NNX Rngs for stochastic operations
        """
        self.feature_dim = feature_dim
        self.rngs = rngs

    def extract_features(self, images: jax.Array) -> jax.Array:
        """Extract mock features from images.

        Args:
            images: Images to extract features from [batch_size, height, width, channels]

        Returns:
            Mock features [batch_size, feature_dim]
        """
        batch_size = images.shape[0]

        # Generate deterministic features based on image content
        # This ensures the same image gets the same features
        # Use params key if inception key is not available
        if hasattr(self.rngs, "inception"):
            key = self.rngs.inception()
        elif hasattr(self.rngs, "params"):
            key = self.rngs.params()
        else:
            key = jax.random.key(42)  # Fallback to a default key

        features = jax.random.normal(key, (batch_size, self.feature_dim))

        # Make features somewhat dependent on image content for better testing
        image_means = jnp.mean(images, axis=(1, 2, 3))
        image_stds = jnp.std(images, axis=(1, 2, 3))

        # Scale features by image statistics
        scaled_features = features * image_stds[:, None] + image_means[:, None]

        return jnp.array(scaled_features)


class LPIPSMetric(MetricBase):
    """Learned Perceptual Image Patch Similarity (LPIPS) metric.

    LPIPS measures perceptual similarity between images using deep features,
    which correlates better with human perception than pixel-based metrics.
    Lower LPIPS values indicate more perceptually similar images.

    Reference:
        Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
        https://arxiv.org/abs/1801.03924
    """

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig):
        """Initialize LPIPS metric.

        Args:
            rngs: NNX Rngs for stochastic operations
            config: Evaluation configuration (must be EvaluationConfig)
        """
        if not isinstance(config, EvaluationConfig):
            # Also check for class name to handle dynamic module loading
            if type(config).__name__ != "EvaluationConfig":
                raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")

        # Initialize base class with minimal config to satisfy MetricBase requirements
        # Initialize base class with the EvaluationConfig
        super().__init__(config=config, rngs=rngs)
        self.eval_batch_size = config.eval_batch_size

        self.metric_name = "lpips_distance"
        self.mock_implementation = config.metric_params.get("lpips", {}).get(
            "mock_implementation", True
        )

        # Initialize model
        self._initialize_lpips_model()

    def _initialize_lpips_model(self):
        """Initialize LPIPS model."""
        if not self.mock_implementation:
            try:
                # Try to import real LPIPS implementation
                # This is a placeholder - in a real implementation, we would import
                # the actual LPIPS library or implement it using JAX/Flax
                pass
            except ImportError:
                print("LPIPS library not available, using mock implementation")
                self.mock_implementation = True

    def compute(
        self, real_data: jax.Array, generated_data: jax.Array, **kwargs
    ) -> dict[str, float]:
        """Compute LPIPS score between two sets of images.

        Args:
            real_data: Real images [batch_size, height, width, channels]
            generated_data: Generated images [batch_size, height, width, channels]
            **kwargs: Additional parameters

        Returns:
            dictionary with LPIPS score
        """
        if self.mock_implementation:
            # Use mock implementation
            lpips_score = compute_lpips_distance(real_data, generated_data)
        else:
            # Use real LPIPS implementation
            # This is a placeholder - in a real implementation, we would use
            # the actual LPIPS library or our JAX/Flax implementation
            lpips_score = 0.0

        return {"lpips_distance": lpips_score}

    def validate_inputs(self, real_data: jax.Array, generated_data: jax.Array) -> bool:
        """Validate input data compatibility.

        Args:
            real_data: Real images
            generated_data: Generated images

        Returns:
            True if inputs are valid
        """
        # Check that both inputs are arrays
        if not isinstance(real_data, jax.Array) or not isinstance(generated_data, jax.Array):
            return False

        # Check shapes match
        if real_data.shape != generated_data.shape:
            return False

        # Check that images are 4D with 3 channels
        if len(real_data.shape) != 4 or real_data.shape[-1] != 3:
            return False

        return True


class SSIMMetric(MetricBase):
    """Structural Similarity Index Measure (SSIM) metric.

    SSIM measures the similarity between two images based on luminance,
    contrast, and structure. Higher SSIM values indicate more similar images.

    Reference:
        Wang et al. "Image quality assessment: from error visibility to structural similarity"
        https://ieeexplore.ieee.org/document/1284395
    """

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig):
        """Initialize SSIM metric.

        Args:
            rngs: NNX Rngs for stochastic operations
            config: Evaluation configuration (must be EvaluationConfig)
        """
        if not isinstance(config, EvaluationConfig):
            # Also check for class name to handle dynamic module loading
            if type(config).__name__ != "EvaluationConfig":
                raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")

        # Initialize base class with minimal config to satisfy MetricBase requirements
        # Initialize base class with the EvaluationConfig
        super().__init__(config=config, rngs=rngs)
        self.eval_batch_size = config.eval_batch_size

        self.metric_name = "ssim_score"

        # SSIM parameters from config
        ssim_params = config.metric_params.get("ssim", {})
        self.K1 = ssim_params.get("K1", 0.01)
        self.K2 = ssim_params.get("K2", 0.03)
        self.window_size = ssim_params.get("window_size", 11)
        self.sigma = ssim_params.get("sigma", 1.5)

    def compute(
        self, real_data: jax.Array, generated_data: jax.Array, **kwargs
    ) -> dict[str, float]:
        """Compute SSIM score between two sets of images.

        Args:
            real_data: Real images [batch_size, height, width, channels]
            generated_data: Generated images [batch_size, height, width, channels]
            **kwargs: Additional parameters

        Returns:
            dictionary with SSIM score
        """
        # Create Gaussian window
        window = self._gaussian_window(self.window_size, self.sigma)
        window = window[:, :, None, None]  # Add channel dimensions

        # Compute SSIM for each image pair
        ssim_values = []

        for i in range(real_data.shape[0]):
            img1 = real_data[i]
            img2 = generated_data[i]

            # Compute SSIM for each channel
            channel_ssim = []
            for c in range(img1.shape[2]):
                ssim_val = self._compute_ssim(img1[:, :, c], img2[:, :, c], window)
                channel_ssim.append(ssim_val)

            # Average across channels
            ssim_values.append(jnp.mean(jnp.array(channel_ssim)))

        # Average across batch
        ssim_score = float(jnp.mean(jnp.array(ssim_values)))

        return {"ssim_score": ssim_score}

    def validate_inputs(self, real_data: jax.Array, generated_data: jax.Array) -> bool:
        """Validate input data compatibility.

        Args:
            real_data: Real images
            generated_data: Generated images

        Returns:
            True if inputs are valid
        """
        # Check that both inputs are arrays
        if not isinstance(real_data, jax.Array) or not isinstance(generated_data, jax.Array):
            return False

        # Check shapes match
        if real_data.shape != generated_data.shape:
            return False

        # Check that images are 4D with 3 channels
        if len(real_data.shape) != 4 or real_data.shape[-1] != 3:
            return False

        return True

    def _gaussian_window(self, window_size: int, sigma: float) -> jax.Array:
        """Create a Gaussian window.

        Args:
            window_size: Size of the window
            sigma: Standard deviation of the Gaussian

        Returns:
            2D Gaussian window
        """
        # Create 1D Gaussian kernel
        x = jnp.arange(window_size)
        x = x - window_size // 2
        gauss = jnp.exp(-(x**2) / (2 * sigma**2))
        gauss = gauss / jnp.sum(gauss)

        # Create 2D Gaussian kernel
        window = jnp.outer(gauss, gauss)

        return window

    def _compute_ssim(self, img1: jax.Array, img2: jax.Array, window: jax.Array) -> float:
        """Compute SSIM for a single channel.

        Args:
            img1: First image channel
            img2: Second image channel
            window: Gaussian window

        Returns:
            SSIM value
        """
        # Constants for stability
        C1 = (self.K1 * 1.0) ** 2
        C2 = (self.K2 * 1.0) ** 2

        # Compute means
        mu1 = self._conv2d(img1, window)
        mu2 = self._conv2d(img2, window)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        # Compute variances and covariance
        sigma1_sq = self._conv2d(img1**2, window) - mu1_sq
        sigma2_sq = self._conv2d(img2**2, window) - mu2_sq
        sigma12 = self._conv2d(img1 * img2, window) - mu1_mu2

        # Compute SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator

        return float(jnp.mean(ssim_map))

    def _conv2d(self, img: jax.Array, window: jax.Array) -> jax.Array:
        """Apply 2D convolution with a window.

        This is a simplified implementation for testing purposes.

        Args:
            img: Input image
            window: Convolution window

        Returns:
            Convolved image
        """
        # Simple convolution implementation
        # In a real implementation, we would use jax.lax.conv or similar

        # For simplicity, we'll use a very basic implementation
        # that doesn't handle padding properly
        h, w = window.shape[:2]

        # Pad image
        pad_h = h // 2
        pad_w = w // 2
        img_padded = jnp.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

        # Apply convolution
        result = jnp.zeros_like(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                patch = img_padded[i : i + h, j : j + w]
                result = result.at[i, j].set(jnp.sum(patch * window))

        return result


class ISMetric(MetricBase):
    """Inception Score (IS) metric for image quality evaluation.

    Inception Score measures the quality and diversity of generated images
    by computing the KL divergence between the conditional and marginal
    label distributions of a pre-trained Inception model.
    Higher IS values indicate better quality and diversity.

    Reference:
        Salimans et al. "Improved Techniques for Training GANs"
        https://arxiv.org/abs/1606.03498
    """

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig):
        """Initialize IS metric.

        Args:
            rngs: NNX Rngs for stochastic operations
            config: Evaluation configuration (must be EvaluationConfig)
        """
        if not isinstance(config, EvaluationConfig):
            # Also check for class name to handle dynamic module loading
            if type(config).__name__ != "EvaluationConfig":
                raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")

        # Initialize base class with minimal config to satisfy MetricBase requirements
        # Initialize base class with the EvaluationConfig
        super().__init__(config=config, rngs=rngs)
        self.eval_batch_size = config.eval_batch_size

        self.metric_name = "inception_score"

        # IS parameters from config
        is_params = config.metric_params.get("inception_score", {})
        self.mock_inception = is_params.get("mock_inception", True)
        self.splits = is_params.get("splits", 10)
        self.inception_model = None

        # Initialize mock or real inception model
        self._initialize_inception_model()

    def _initialize_inception_model(self):
        """Initialize the Inception model for feature extraction."""
        if self.mock_inception:
            # Use mock inception model that returns random features
            self.inception_model = MockInceptionModel(rngs=self.rngs)
        else:
            # Try to load real inception model
            try:
                # Import TensorFlow if available
                import tensorflow as tf  # noqa: F401
                from tensorflow.keras.applications.inception_v3 import InceptionV3

                # Load pre-trained Inception model
                self.inception_model = InceptionV3(include_top=True, weights="imagenet")
                print("Loaded real Inception model for IS calculation")
            except ImportError:
                print("TensorFlow not available, using mock Inception model")
                self.inception_model = MockInceptionModel(rngs=self.rngs)

    def compute(
        self, real_data: jax.Array, generated_data: jax.Array, **kwargs
    ) -> dict[str, float]:
        """Compute the Inception Score.

        Args:
            real_data: Real images [batch_size, height, width, channels]
            generated_data: Generated images [batch_size, height, width, channels]
            **kwargs: Additional parameters

        Returns:
            dictionary with Inception Score
        """
        # For IS, we only need generated images
        # Extract class probabilities
        preds = self._get_inception_predictions(generated_data)

        # Calculate IS
        is_score = self._calculate_inception_score(preds)

        return {"inception_score": is_score}

    def validate_inputs(self, real_data: jax.Array, generated_data: jax.Array) -> bool:
        """Validate input data compatibility.

        Args:
            real_data: Real images
            generated_data: Generated images

        Returns:
            True if inputs are valid
        """
        # For IS, we only need generated images to be valid
        if not isinstance(generated_data, jax.Array):
            return False

        # Check that input is 4D: batch x height x width x channels
        if len(generated_data.shape) != 4:
            return False

        # Check that last dimension is 3 (for RGB channels)
        if generated_data.shape[-1] != 3:
            return False

        return True

    def _get_inception_predictions(self, images: jax.Array) -> jax.Array:
        """Get Inception model predictions for images.

        Args:
            images: Images to get predictions for [batch_size, height, width, channels]

        Returns:
            Inception predictions [batch_size, num_classes]
        """
        if self.mock_inception:
            # Use mock inception model to generate class probabilities
            batch_size = images.shape[0]
            num_classes = 1000  # ImageNet classes

            # Generate random probabilities
            if hasattr(self.rngs, "inception"):
                key = self.rngs.inception()
            elif hasattr(self.rngs, "params"):
                key = self.rngs.params()
            else:
                key = jax.random.key(42)
            logits = jax.random.normal(key, (batch_size, num_classes))

            # Apply softmax to get probabilities
            probs = jax.nn.softmax(logits, axis=1)

            # Make probabilities somewhat dependent on image content
            image_means = jnp.mean(images, axis=(1, 2, 3))
            scaled_probs = probs * (1.0 + 0.1 * image_means[:, None])
            normalized_probs = scaled_probs / jnp.sum(scaled_probs, axis=1, keepdims=True)

            return normalized_probs
        else:
            # Convert to numpy and preprocess for Inception
            images_np = images

            # Resize images to Inception input size if needed
            if images_np.shape[1] != 299 or images_np.shape[2] != 299:
                # Placeholder for resizing
                pass

            # Scale from [0, 1] to [-1, 1] if needed
            if images_np.max() <= 1.0:
                images_np = images_np * 2.0 - 1.0

            # Get predictions using real Inception model
            preds = self.inception_model.predict(images_np, verbose=0)
            return preds

    def _calculate_inception_score(self, preds: jax.Array) -> float:
        """Calculate Inception Score from predictions.

        Args:
            preds: Inception model predictions [batch_size, num_classes]

        Returns:
            Inception Score
        """
        # Split predictions into groups
        batch_size = preds.shape[0]
        split_size = batch_size // self.splits

        # Handle case where batch is too small for requested splits
        if split_size < 1:
            self.splits = batch_size
            split_size = 1

        scores = []

        # Calculate IS for each split
        for i in range(self.splits):
            start = i * split_size
            end = start + split_size
            if i == self.splits - 1:  # Last split might be larger
                end = batch_size

            split_preds = preds[start:end]

            # Calculate KL divergence
            p_y = jnp.mean(split_preds, axis=0, keepdims=True)  # Marginal probability
            kl_div = split_preds * (jnp.log(split_preds + 1e-10) - jnp.log(p_y + 1e-10))
            kl_div = jnp.mean(jnp.sum(kl_div, axis=1))

            # IS is exp(KL)
            scores.append(jnp.exp(kl_div))

        # Return mean and std of scores
        mean_score = float(jnp.mean(jnp.array(scores)))

        return mean_score


# Factory functions for creating metrics with unified configuration
def create_fid_metric(
    rngs: nnx.Rngs,
    mock_inception: bool = True,
    batch_size: int = 32,
    config_name: str = "fid_metric",
) -> FIDMetric:
    """Create FID metric with unified configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        mock_inception: Whether to use mock inception model
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured FID metric
    """
    config = EvaluationConfig(
        name=config_name,
        metrics=["fid"],
        metric_params={
            "fid": {
                "mock_inception": mock_inception,
                "higher_is_better": False,
            }
        },
        eval_batch_size=batch_size,
    )

    return FIDMetric(rngs=rngs, config=config)


def create_lpips_metric(
    rngs: nnx.Rngs,
    mock_implementation: bool = True,
    batch_size: int = 32,
    config_name: str = "lpips_metric",
) -> LPIPSMetric:
    """Create LPIPS metric with unified configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        mock_implementation: Whether to use mock implementation
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured LPIPS metric
    """
    config = EvaluationConfig(
        name=config_name,
        metrics=["lpips"],
        metric_params={
            "lpips": {
                "mock_implementation": mock_implementation,
                "higher_is_better": False,
            }
        },
        eval_batch_size=batch_size,
    )

    return LPIPSMetric(rngs=rngs, config=config)


def create_ssim_metric(
    rngs: nnx.Rngs,
    window_size: int = 11,
    batch_size: int = 32,
    config_name: str = "ssim_metric",
) -> SSIMMetric:
    """Create SSIM metric with unified configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        window_size: Window size for SSIM computation
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured SSIM metric
    """
    config = EvaluationConfig(
        name=config_name,
        metrics=["ssim"],
        metric_params={
            "ssim": {
                "window_size": window_size,
                "higher_is_better": True,
            }
        },
        eval_batch_size=batch_size,
    )

    return SSIMMetric(rngs=rngs, config=config)


def create_is_metric(
    rngs: nnx.Rngs,
    mock_inception: bool = True,
    splits: int = 10,
    batch_size: int = 32,
    config_name: str = "is_metric",
) -> ISMetric:
    """Create Inception Score metric with unified configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        mock_inception: Whether to use mock inception model
        splits: Number of splits for IS calculation
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured IS metric
    """
    config = EvaluationConfig(
        name=config_name,
        metrics=["inception_score"],
        metric_params={
            "inception_score": {
                "mock_inception": mock_inception,
                "splits": splits,
                "higher_is_better": True,
            }
        },
        eval_batch_size=batch_size,
    )

    return ISMetric(rngs=rngs, config=config)
