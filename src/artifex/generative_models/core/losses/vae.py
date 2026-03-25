"""Canonical VAE loss helpers built from shared loss primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp

from artifex.generative_models.core.losses.divergence import gaussian_kl_divergence
from artifex.generative_models.core.losses.geometric import binary_cross_entropy
from artifex.generative_models.core.losses.reconstruction import mae_loss, mse_loss


VAEReconstructionLossType = Literal["mse", "mae", "bce"]


def vae_reconstruction_loss(
    reconstructed: jax.Array,
    targets: jax.Array,
    *,
    loss_type: VAEReconstructionLossType = "mse",
) -> jax.Array:
    """Compute the canonical VAE reconstruction term."""
    if loss_type == "mse":
        return mse_loss(reconstructed, targets, reduction="batch_sum")
    if loss_type == "mae":
        return mae_loss(reconstructed, targets, reduction="batch_sum")
    if loss_type == "bce":
        return binary_cross_entropy(reconstructed, targets, reduction="batch_sum")

    msg = f"Unknown VAE reconstruction loss type: {loss_type}"
    raise ValueError(msg)


def vae_kl_components(
    mean: jax.Array,
    log_var: jax.Array,
    *,
    free_bits: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Return scalar and per-sample KL terms for VAE training."""
    if free_bits <= 0.0:
        kl_per_sample = gaussian_kl_divergence(mean, log_var, reduction="none")
        return jnp.mean(kl_per_sample), kl_per_sample

    kl_per_dim = -0.5 * (1 + log_var - mean**2 - jnp.exp(log_var))
    clamped = jnp.maximum(kl_per_dim, free_bits)
    kl_per_sample = jnp.sum(clamped, axis=-1)
    return jnp.mean(kl_per_sample), kl_per_sample


def vae_elbo_terms(
    *,
    reconstructed: jax.Array,
    targets: jax.Array,
    mean: jax.Array,
    log_var: jax.Array,
    beta: float | jax.Array = 1.0,
    reconstruction_loss_type: VAEReconstructionLossType = "mse",
    reconstruction_loss_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    free_bits: float = 0.0,
) -> dict[str, jax.Array]:
    """Compute canonical VAE ELBO terms."""
    if reconstruction_loss_fn is None:
        reconstruction_loss = vae_reconstruction_loss(
            reconstructed,
            targets,
            loss_type=reconstruction_loss_type,
        )
    else:
        reconstruction_loss = reconstruction_loss_fn(reconstructed, targets)

    kl_loss, _ = vae_kl_components(mean, log_var, free_bits=free_bits)
    beta_value = jnp.asarray(beta)
    total_loss = reconstruction_loss + beta_value * kl_loss

    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
    }
