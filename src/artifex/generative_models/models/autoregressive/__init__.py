"""Autoregressive models package.

This module provides implementations of various autoregressive generative models
including base classes, transformer-based models, PixelCNN for image generation,
and WaveNet for sequence generation.

Autoregressive models decompose the joint distribution using the chain rule:
p(x1, x2, ..., xn) = ∏ p(xi | x1, ..., xi-1)

This enables sequential generation while maintaining tractable likelihood computation.
"""

# Base class and utilities
from artifex.generative_models.models.autoregressive.base import AutoregressiveModel

# PixelCNN implementation for image generation
from artifex.generative_models.models.autoregressive.pixel_cnn import (
    MaskedConv2D,
    PixelCNN,
)

# Transformer-based autoregressive model
from artifex.generative_models.models.autoregressive.transformer import (
    TransformerAutoregressiveModel,
)

# WaveNet implementation for sequence generation
from artifex.generative_models.models.autoregressive.wavenet import (
    CausalConv1D,
    GatedActivationUnit,
    WaveNet,
)


__all__ = [
    # Base models and utilities
    "AutoregressiveModel",
    # PixelCNN components
    "PixelCNN",
    "MaskedConv2D",
    # Transformer components
    "TransformerAutoregressiveModel",
    # WaveNet components
    "WaveNet",
    "CausalConv1D",
    "GatedActivationUnit",
]
