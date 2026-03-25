"""Lightweight image modality helpers for generative models.

This package provides:
- RGB, RGBA, and grayscale image modality configuration
- synthetic image dataset helpers
- modality-local helper metrics such as MSE, PSNR, SSIM, and perceptual distance
- basic preprocessing plus retained augmentation helpers (horizontal flip and brightness jitter)

Benchmark-grade metrics such as FID, Inception Score, and LPIPS are owned by
`artifex.benchmarks.metrics.image`, not by this modality helper layer.

Example:
    >>> from flax import nnx
    >>> from artifex.generative_models.modalities.image import ImageModality, ImageModalityConfig
    >>> modality = ImageModality(config=ImageModalityConfig(height=64, width=64), rngs=nnx.Rngs(0))
    >>> images = modality.generate(n_samples=4, rngs=nnx.Rngs(1))
"""

from .adapters import ImageModalityAdapter
from .base import (
    create_image_modality,
    ImageGenerationProtocol,
    ImageModality,
    ImageModalityConfig,
    ImageRepresentation,
)
from .datasets import (
    create_image_dataset,
    generate_mnist_like_images,
    generate_synthetic_images,
)
from .evaluation import compute_image_metrics, ImageEvaluationSuite, ImageMetrics
from .representations import AugmentationProcessor, ImageProcessor, MultiScaleProcessor


__all__ = [
    # Core modality
    "ImageGenerationProtocol",
    "ImageModality",
    "ImageModalityConfig",
    "ImageRepresentation",
    "create_image_modality",
    # Adapters
    "ImageModalityAdapter",
    # Dataset handling
    "generate_synthetic_images",
    "generate_mnist_like_images",
    "create_image_dataset",
    # Evaluation
    "ImageEvaluationSuite",
    "ImageMetrics",
    "compute_image_metrics",
    # Representation processing
    "AugmentationProcessor",
    "ImageProcessor",
    "MultiScaleProcessor",
]
