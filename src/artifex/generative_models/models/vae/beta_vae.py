"""Beta Variational Autoencoder implementation."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.vae_config import (
    BetaVAEConfig,
    BetaVAEWithCapacityConfig,
)
from artifex.generative_models.core.losses.vae import vae_elbo_terms, vae_kl_components
from artifex.generative_models.models.vae.base import _extract_vae_inputs, VAE


class BetaVAE(VAE):
    """Beta Variational Autoencoder implementation.

    Beta-VAE is a variant of VAE that introduces a hyperparameter beta
    to the KL divergence term in the loss function. This allows for better
    control over the disentanglement of latent representations.

    By setting beta > 1, the model is encouraged to learn more disentangled
    representations, but potentially at the cost of reconstruction quality.
    """

    def __init__(
        self,
        config: BetaVAEConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize a BetaVAE.

        Args:
            config: BetaVAEConfig with encoder, decoder, encoder_type, and beta settings
            rngs: Random number generator for initialization
        """
        super().__init__(config=config, rngs=rngs)

        # BetaVAE-specific settings from config
        self.beta_default = config.beta_default
        self.beta_warmup_steps = config.beta_warmup_steps
        self.reconstruction_loss_type = config.reconstruction_loss_type

    def loss_fn(
        self,
        batch: Any,
        model_outputs: dict[str, jax.Array],
        *,
        beta: float | None = None,
        **kwargs: Any,
    ) -> dict[str, jax.Array]:
        """Calculate loss for BetaVAE.

        Args:
            batch: Input batch as an array or dict containing the model inputs.
            model_outputs: Dictionary of model outputs from forward pass.
            beta: Weight for KL divergence term. If None, uses model's beta_default
                with optional warmup.
            **kwargs: Additional arguments including current training step

        Returns:
            Dictionary of loss components
        """
        x = _extract_vae_inputs(batch)

        # Extract outputs
        reconstructed = model_outputs.get("reconstructed")
        if reconstructed is None:
            raise KeyError("Missing required output key: 'reconstructed'")

        mean = model_outputs["mean"]
        log_var = model_outputs.get("log_var")
        if log_var is None:
            raise KeyError("Missing required output key: 'log_var'")

        # Get beta value and handle annealing if needed
        # If beta is None, use beta_default with optional warmup
        if beta is None:
            beta_value = self.beta_default
            # Current training step (if provided in kwargs)
            step = kwargs.get("step", 0)

            # Apply beta annealing if warmup steps > 0
            # Use jnp.where for JIT compatibility (step may be a traced value)
            warmup_ratio = jnp.where(
                self.beta_warmup_steps > 0,
                jnp.minimum(1.0, step / jnp.maximum(self.beta_warmup_steps, 1)),
                1.0,
            )
            beta = beta_value * warmup_ratio

        losses = vae_elbo_terms(
            reconstructed=reconstructed,
            targets=x,
            mean=mean,
            log_var=log_var,
            beta=beta,
            reconstruction_loss_type=self.reconstruction_loss_type,
        )
        losses["beta"] = jnp.asarray(beta)
        return losses


class BetaVAEWithCapacity(BetaVAE):
    """Beta-VAE with Burgess et al. capacity control."""

    def __init__(
        self,
        config: BetaVAEWithCapacityConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize BetaVAE with capacity control.

        Args:
            config: BetaVAEWithCapacityConfig with encoder, decoder, beta, and capacity settings
            rngs: Random number generator for initialization
        """
        super().__init__(config=config, rngs=rngs)

        # Extract capacity control parameters from config
        self.use_capacity_control = config.use_capacity_control
        self.capacity_max = config.capacity_max
        self.capacity_num_iter = config.capacity_num_iter
        self.gamma = config.gamma

    def loss_fn(
        self,
        batch: Any,
        model_outputs: dict[str, jax.Array],
        *,
        beta: float | None = None,
        step: int = 0,  # Current training step
        **kwargs: Any,
    ) -> dict[str, jax.Array]:
        """Calculate loss with optional capacity control."""
        # Get base loss components
        base_losses = super().loss_fn(
            batch=batch,
            model_outputs=model_outputs,
            beta=beta,
            step=step,
            **kwargs,
        )

        if not self.use_capacity_control:
            return base_losses

        # Extract components
        recon_loss = base_losses["reconstruction_loss"]
        mean = model_outputs["mean"]
        log_var = model_outputs.get("log_var")

        kl_loss, _ = vae_kl_components(mean, log_var)

        # Calculate current capacity
        current_capacity = jnp.minimum(
            self.capacity_max, self.capacity_max * step / self.capacity_num_iter
        )

        # Capacity loss: γ * |KL - C|
        capacity_loss = self.gamma * jnp.abs(kl_loss - current_capacity)

        # Total loss
        total_loss = recon_loss + capacity_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "capacity_loss": capacity_loss,
            "current_capacity": current_capacity,
        }
