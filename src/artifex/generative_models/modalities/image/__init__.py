"""Image modality for generative models.

This module provides comprehensive image generation capabilities including:
- Multiple resolution and channel support (grayscale, RGB, RGBA)
- Integration with VAE, GAN, and Diffusion models
- Comprehensive image evaluation metrics (FID, IS, LPIPS)
- Data augmentation and preprocessing pipelines
- Integration with benchmark framework

Example:
    >>> from artifex.generative_models.modalities.image import ImageModality, ImageRepresentation
    >>> modality = ImageModality(resolution=64, channels=3)
    >>> images = modality.generate(n_samples=4)
"""

from .adapters import ImageModalityAdapter
from .base import (
    create_image_modality,
    ImageGenerationProtocol,
    ImageModality,
    ImageModalityConfig,
    ImageRepresentation,
)
from .datasets import create_image_dataset, ImageDataset, SyntheticImageDataset
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
    "ImageDataset",
    "SyntheticImageDataset",
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
