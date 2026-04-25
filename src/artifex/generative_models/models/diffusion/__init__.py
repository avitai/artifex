"""Diffusion models package."""

# Base class for diffusion models
from artifex.generative_models.models.diffusion.base import DiffusionModel

# Stable Diffusion implementation
from artifex.generative_models.models.diffusion.clip_text_encoder import CLIPTextEncoder

# DDIM implementation
from artifex.generative_models.models.diffusion.ddim import DDIMModel

# DDPM implementation
from artifex.generative_models.models.diffusion.ddpm import DDPMModel

# DiT implementation
from artifex.generative_models.models.diffusion.dit import DiTModel

# Factory functions have been moved to the centralized factory
# Use: from artifex.generative_models.factory import create_model
# Guidance techniques
from artifex.generative_models.models.diffusion.guidance import (
    apply_guidance,
    ClassifierFreeGuidance,
    ClassifierGuidance,
    ConditionalDiffusionMixin,
    cosine_guidance_schedule,
    GuidedDiffusionModel,
    linear_guidance_schedule,
)

# Latent diffusion models
from artifex.generative_models.models.diffusion.latent import LDMModel

# Score-based diffusion models
from artifex.generative_models.models.diffusion.score import ScoreDiffusionModel


__all__ = [
    # Base models
    "DiffusionModel",
    "DDPMModel",
    "DDIMModel",
    "ScoreDiffusionModel",
    "LDMModel",
    "DiTModel",
    "CLIPTextEncoder",
    # Guidance
    "ClassifierFreeGuidance",
    "ClassifierGuidance",
    "ConditionalDiffusionMixin",
    "GuidedDiffusionModel",
    "apply_guidance",
    "linear_guidance_schedule",
    "cosine_guidance_schedule",
]
