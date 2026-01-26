"""Models package for generative models."""

# Import model submodules
# Ensure model registration by importing all factory modules
# Note: JAXopt package shows deprecation warnings.
# Consider alternatives per https://docs.jax.dev/en/latest/
import artifex.generative_models.models.geometric  # noqa
import artifex.generative_models.models.vae  # noqa
from artifex.generative_models.models import (
    diffusion,
    factories,  # Import all factories to ensure they're registered
    geometric,
    registry,
    vae,
)

# Factory functions have been moved to the centralized factory
# Use: from artifex.generative_models.factory import create_model

# Re-export registry
from artifex.generative_models.models.registry import (
    ModelRegistry,
    register_model,
)
# from artifex.generative_models.models.vae.factory import create_vae_model


__all__ = [
    # Submodules
    "diffusion",
    "factories",
    "geometric",
    "vae",
    # Registry
    "ModelRegistry",
    "register_model",
    "registry",
]
