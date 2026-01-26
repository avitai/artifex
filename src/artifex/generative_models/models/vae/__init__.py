"""Variational Autoencoder models module."""

from artifex.generative_models.models.vae.base import VAE
from artifex.generative_models.models.vae.beta_vae import BetaVAE
from artifex.generative_models.models.vae.conditional import ConditionalVAE

# Factory functions have been moved to the centralized factory
# Use: from artifex.generative_models.factory import create_model
from artifex.generative_models.models.vae.vq_vae import VQVAE


__all__ = ["VAE", "BetaVAE", "ConditionalVAE", "VQVAE"]
