"""Image metrics for generative models.

This module provides metrics for evaluating the quality of generated images,
including FID, IS (Inception Score), LPIPS, and SSIM metrics for image generation.
"""

import logging
from collections.abc import Callable
from typing import cast

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from calibrax.metrics.functional.image import ssim as calibrax_ssim

from artifex.benchmarks.metrics.core import _init_metric_from_config, MetricBase
from artifex.benchmarks.runtime_guards import demo_mode_from_mapping, require_demo_mode
from artifex.generative_models.core.configuration import EvaluationConfig
from artifex.generative_models.core.evaluation.metrics.metric_ops import (
    frechet_distance_from_statistics,
)
from artifex.generative_models.core.evaluation.metrics.quality import (
    calculate_fid_score,
    compute_lpips_distance,
)


logger = logging.getLogger(__name__)


def _resolved_metric_rngs(rngs: nnx.Rngs | None) -> nnx.Rngs:
    """Provide a deterministic fallback RNG container for retained mock paths."""
    if rngs is not None:
        return rngs
    return nnx.Rngs(params=jax.random.PRNGKey(0))


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
        self.metric_name = "fid_score"
        self.feature_extractor: Callable[[jax.Array], jax.Array]
        self.fid_params = _init_metric_from_config(
            self,
            config=config,
            rngs=rngs,
            metric_key="fid",
            modality="image",
            higher_is_better=False,
        )
        self.mock_inception = self.fid_params.get("mock_inception", False)
        self.demo_mode = demo_mode_from_mapping(self.fid_params)

        if self.mock_inception:
            require_demo_mode(
                enabled=self.demo_mode,
                component="FIDMetric",
                detail=(
                    "The retained mock Inception feature extractor is demo-only and not the "
                    "shipped benchmark dependency path."
                ),
            )

        self._initialize_feature_extractor()

    def _initialize_feature_extractor(self) -> None:
        """Initialize the configured feature extractor."""
        if self.mock_inception:
            mock_model = MockInceptionModel(rngs=_resolved_metric_rngs(self.rngs))
            self.feature_extractor = mock_model.extract_features
            logger.info("Using retained mock Inception model in explicit demo mode")
            return

        feature_extractor = self.fid_params.get("feature_extractor")
        if not callable(feature_extractor):
            raise ValueError(
                "FIDMetric requires an explicit callable feature_extractor in supported mode. "
                "Pass mock_inception=True only for the retained demo workflow."
            )
        self.feature_extractor = cast(Callable[[jax.Array], jax.Array], feature_extractor)

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
        fid_score = float(calculate_fid_score(gen_features, real_features))

        return {"fid_score": fid_score}

    def validate_inputs(self, real_data: jax.Array, generated_data: jax.Array) -> None:
        """Validate input data compatibility.

        Args:
            real_data: Real images
            generated_data: Generated images

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(real_data, jax.Array) or not isinstance(generated_data, jax.Array):
            raise ValueError("Both inputs must be jax.Array")
        if len(real_data.shape) != 4 or len(generated_data.shape) != 4:
            raise ValueError("Images must be 4D (batch, height, width, channels)")
        if real_data.shape[-1] != 3 or generated_data.shape[-1] != 3:
            raise ValueError("Images must have 3 channels (RGB)")
        if real_data.shape[0] != generated_data.shape[0]:
            raise ValueError("Batch sizes must match")

    def compute_statistics(self, images: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute mean and covariance of Inception features for a set of images.

        Args:
            images: Images [batch_size, height, width, channels]

        Returns:
            Tuple of (mean, covariance) of extracted features
        """
        features = self._extract_features(images)
        mean = jnp.mean(features, axis=0)
        # Covariance with small regularisation for numerical stability
        centered = features - mean
        cov = (centered.T @ centered) / max(features.shape[0] - 1, 1)
        return mean, cov

    def compute_fid(
        self,
        real_mean: jax.Array,
        real_cov: jax.Array,
        fake_mean: jax.Array,
        fake_cov: jax.Array,
    ) -> float:
        """Compute FID from pre-computed statistics.

        Args:
            real_mean: Mean of real image features
            real_cov: Covariance of real image features
            fake_mean: Mean of generated image features
            fake_cov: Covariance of generated image features

        Returns:
            FID score (lower is better, non-negative)
        """
        return float(frechet_distance_from_statistics(real_mean, real_cov, fake_mean, fake_cov))

    def _extract_features(self, images: jax.Array) -> jax.Array:
        """Extract features from images using Inception model.

        Args:
            images: Images to extract features from [batch_size, height, width, channels]

        Returns:
            Extracted features [batch_size, feature_dim]
        """
        return jnp.asarray(self.feature_extractor(images))

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
                image_uint8 = np.asarray(images[i] * 255, dtype=np.uint8)
                img = Image.fromarray(image_uint8)
                img_resized = img.resize(target_size)
                resized_images[i] = jnp.array(img_resized) / 255.0

            return resized_images
        except ImportError:
            # Fallback to simple resizing using interpolation
            logger.info("PIL not available, using simple resize")
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

    def predict(self, images: jax.Array) -> jax.Array:
        """Return deterministic mock class probabilities for generated images."""
        batch_size = images.shape[0]
        if hasattr(self.rngs, "inception"):
            key = self.rngs.inception()
        elif hasattr(self.rngs, "params"):
            key = self.rngs.params()
        else:
            key = jax.random.key(42)

        logits = jax.random.normal(key, (batch_size, 1000))
        probabilities = nnx.softmax(logits, axis=-1)
        image_means = jnp.mean(images, axis=(1, 2, 3))
        scaled = probabilities * (1.0 + 0.1 * image_means[:, None])
        return scaled / jnp.sum(scaled, axis=-1, keepdims=True)


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
        self.metric_name = "lpips_distance"
        self.lpips_params = _init_metric_from_config(
            self,
            config=config,
            rngs=rngs,
            metric_key="lpips",
            modality="image",
            higher_is_better=False,
        )
        self.mock_implementation = self.lpips_params.get("mock_implementation", False)
        self.demo_mode = demo_mode_from_mapping(self.lpips_params)

        if self.mock_implementation:
            require_demo_mode(
                enabled=self.demo_mode,
                component="LPIPSMetric",
                detail=(
                    "The retained LPIPS path uses a placeholder perceptual distance backend and "
                    "is demo-only."
                ),
            )

        self._initialize_lpips_model()

    def _initialize_lpips_model(self):
        """Initialize LPIPS model."""
        if self.mock_implementation:
            logger.info("Using retained mock LPIPS implementation in explicit demo mode")
            return

        raise RuntimeError(
            "LPIPSMetric does not ship a benchmark-grade perceptual backend. Pass "
            "mock_implementation=True only for the retained demo workflow."
        )

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
            lpips_score = float(compute_lpips_distance(real_data, generated_data))
            return {"lpips_distance": lpips_score}

        raise RuntimeError(
            "LPIPSMetric does not ship a benchmark-grade perceptual backend outside explicit "
            "demo mode."
        )

    def compute_distance(self, images1: jax.Array, images2: jax.Array) -> jax.Array:
        """Compute per-sample LPIPS distance between two image batches.

        Args:
            images1: First set of images [batch_size, H, W, C]
            images2: Second set of images [batch_size, H, W, C]

        Returns:
            Per-sample LPIPS distances [batch_size]
        """
        batch_size = images1.shape[0]
        distances = jnp.zeros(batch_size)
        for i in range(batch_size):
            img1 = images1[i : i + 1]
            img2 = images2[i : i + 1]
            score = compute_lpips_distance(img1, img2)
            distances = distances.at[i].set(score)
        return distances

    def validate_inputs(self, real_data: jax.Array, generated_data: jax.Array) -> None:
        """Validate input data compatibility.

        Args:
            real_data: Real images
            generated_data: Generated images

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(real_data, jax.Array) or not isinstance(generated_data, jax.Array):
            raise ValueError("Both inputs must be jax.Array")
        if real_data.shape != generated_data.shape:
            raise ValueError("Input shapes must match")
        if len(real_data.shape) != 4 or real_data.shape[-1] != 3:
            raise ValueError("Images must be 4D with 3 channels")


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
        _init_metric_from_config(
            self,
            config=config,
            rngs=rngs,
            metric_key="ssim",
            modality="image",
            higher_is_better=True,
        )

        self.metric_name = "ssim_score"

        # SSIM parameters from config
        ssim_params = _init_metric_from_config(
            self,
            config=config,
            rngs=rngs,
            metric_key="ssim",
            modality="image",
            higher_is_better=True,
        )
        self.K1 = ssim_params.get("K1", 0.01)
        self.K2 = ssim_params.get("K2", 0.03)
        self.window_size = ssim_params.get("window_size", 11)
        self.sigma = ssim_params.get("sigma", 1.5)

    def compute(
        self, real_data: jax.Array, generated_data: jax.Array, **kwargs
    ) -> dict[str, float]:
        """Compute SSIM score between two sets of images.

        Delegates per-image SSIM computation to calibrax.

        Args:
            real_data: Real images [batch_size, height, width, channels]
            generated_data: Generated images [batch_size, height, width, channels]
            **kwargs: Additional parameters

        Returns:
            dictionary with SSIM score
        """

        def _ssim_single(img1: jax.Array, img2: jax.Array) -> jax.Array:
            return calibrax_ssim(
                img1,
                img2,
                filter_size=self.window_size,
                k1=self.K1,
                k2=self.K2,
                filter_sigma=self.sigma,
            )

        ssim_per_image = jax.vmap(_ssim_single)(real_data, generated_data)
        ssim_score = float(jnp.mean(ssim_per_image))
        return {"ssim_score": ssim_score}

    def validate_inputs(self, real_data: jax.Array, generated_data: jax.Array) -> None:
        """Validate input data compatibility.

        Args:
            real_data: Real images
            generated_data: Generated images

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(real_data, jax.Array) or not isinstance(generated_data, jax.Array):
            raise ValueError("Both inputs must be jax.Array")
        if real_data.shape != generated_data.shape:
            raise ValueError("Input shapes must match")
        if len(real_data.shape) != 4 or real_data.shape[-1] != 3:
            raise ValueError("Images must be 4D with 3 channels")


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
        self.metric_name = "inception_score"

        # IS parameters from config
        is_params = _init_metric_from_config(
            self,
            config=config,
            rngs=rngs,
            metric_key="inception_score",
            modality="image",
            higher_is_better=True,
        )
        self.mock_inception = is_params.get("mock_inception", False)
        self.demo_mode = demo_mode_from_mapping(is_params)
        self.splits = is_params.get("splits", 10)
        self.classifier: Callable[[jax.Array], jax.Array]

        if self.mock_inception:
            require_demo_mode(
                enabled=self.demo_mode,
                component="ISMetric",
                detail=(
                    "The retained mock Inception classifier is demo-only and not the shipped "
                    "benchmark dependency path."
                ),
            )

        self._initialize_classifier(is_params)

    def _initialize_classifier(self, is_params: dict[str, object]) -> None:
        """Initialize the configured classifier."""
        if self.mock_inception:
            mock_model = MockInceptionModel(rngs=_resolved_metric_rngs(self.rngs))
            self.classifier = mock_model.predict
            logger.info("Using retained mock Inception classifier in explicit demo mode")
            return

        classifier = is_params.get("classifier")
        if not callable(classifier):
            raise ValueError(
                "ISMetric requires an explicit callable classifier in supported mode. "
                "Pass mock_inception=True only for the retained demo workflow."
            )
        self.classifier = cast(Callable[[jax.Array], jax.Array], classifier)

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

    def validate_inputs(self, real_data: jax.Array, generated_data: jax.Array) -> None:
        """Validate input data compatibility.

        Args:
            real_data: Real images
            generated_data: Generated images

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(generated_data, jax.Array):
            raise ValueError("Generated data must be jax.Array")
        if len(generated_data.shape) != 4:
            raise ValueError("Images must be 4D (batch, height, width, channels)")
        if generated_data.shape[-1] != 3:
            raise ValueError("Images must have 3 channels (RGB)")

    def _get_inception_predictions(self, images: jax.Array) -> jax.Array:
        """Get Inception model predictions for images.

        Args:
            images: Images to get predictions for [batch_size, height, width, channels]

        Returns:
            Inception predictions [batch_size, num_classes]
        """
        predictions = jnp.asarray(self.classifier(images))
        if self.mock_inception:
            return predictions
        return nnx.softmax(predictions, axis=-1)

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
    mock_inception: bool = False,
    feature_extractor: Callable[[jax.Array], jax.Array] | None = None,
    batch_size: int = 32,
    config_name: str = "fid_metric",
) -> FIDMetric:
    """Create FID metric with unified configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        mock_inception: Whether to use mock inception model
        feature_extractor: Callable feature extractor for supported mode
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured FID metric
    """
    fid_params: dict[str, object] = {
        "mock_inception": mock_inception,
        "higher_is_better": False,
    }
    if feature_extractor is not None:
        fid_params["feature_extractor"] = feature_extractor

    config = EvaluationConfig(
        name=config_name,
        metrics=["fid"],
        metric_params={"fid": fid_params},
        eval_batch_size=batch_size,
    )

    return FIDMetric(rngs=rngs, config=config)


def create_lpips_metric(
    rngs: nnx.Rngs,
    mock_implementation: bool = False,
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
    mock_inception: bool = False,
    classifier: Callable[[jax.Array], jax.Array] | None = None,
    splits: int = 10,
    batch_size: int = 32,
    config_name: str = "is_metric",
) -> ISMetric:
    """Create Inception Score metric with unified configuration.

    Args:
        rngs: NNX Rngs for stochastic operations
        mock_inception: Whether to use mock inception model
        classifier: Callable classifier for supported mode
        splits: Number of splits for IS calculation
        batch_size: Evaluation batch size
        config_name: Name for the configuration

    Returns:
        Configured IS metric
    """
    is_params: dict[str, object] = {
        "mock_inception": mock_inception,
        "splits": splits,
        "higher_is_better": True,
    }
    if classifier is not None:
        is_params["classifier"] = classifier

    config = EvaluationConfig(
        name=config_name,
        metrics=["inception_score"],
        metric_params={"inception_score": is_params},
        eval_batch_size=batch_size,
    )

    return ISMetric(rngs=rngs, config=config)
