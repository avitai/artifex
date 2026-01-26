"""Beta Variational Autoencoder implementation."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.vae_config import (
    BetaVAEConfig,
    BetaVAEWithCapacityConfig,
)
from artifex.generative_models.models.vae.base import VAE


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
        params: dict | None = None,
        batch: dict | None = None,
        rng: jax.Array | None = None,
        x: jax.Array | None = None,
        outputs: dict[str, jax.Array] | None = None,
        beta: float | None = None,
        **kwargs: Any,
    ) -> dict[str, jax.Array]:
        """Calculate loss for BetaVAE.

        Args:
            params: Model parameters (optional, for compatibility with Trainer)
            batch: Input batch (optional, for compatibility with Trainer)
            rng: Random number generator (optional, for Trainer compatibility)
            x: Input data (if not provided in batch)
            outputs: Dictionary of model outputs from forward pass
            beta: Weight for KL divergence term. If None, uses model's beta_default
                with optional warmup.
            **kwargs: Additional arguments including current training step

        Returns:
            Dictionary of loss components
        """
        # Handle different input patterns for compatibility
        if batch is not None and x is None:
            if isinstance(batch, dict) and "inputs" in batch:
                x = batch["inputs"]
            else:
                x = batch

        # Ensure x is a proper JAX array, not a dictionary
        if isinstance(x, dict):
            if "inputs" in x:
                x = x["inputs"]
            elif "input" in x:
                x = x["input"]
            else:
                # If x is still a dictionary without recognized keys, raise error
                error_msg = (
                    "Input 'x' is a dictionary without 'inputs' or 'input' keys. "
                    "Expected a JAX array."
                )
                raise ValueError(error_msg)

        if outputs is None:
            if hasattr(self, "apply") and params is not None:
                # For compatibility with Trainer
                outputs = self.apply(params, x)
            else:
                # Use direct call
                outputs = self(x)

        # Extract outputs
        recon_key1 = "reconstructed"
        recon_key2 = "reconstruction"
        reconstructed = outputs.get(recon_key1, outputs.get(recon_key2, None))
        if reconstructed is None:
            raise KeyError(f"Missing required output key: '{recon_key1}' or '{recon_key2}'")

        mean = outputs["mean"]
        log_var = outputs.get("log_var", outputs.get("logvar", None))
        if log_var is None:
            raise KeyError("Missing required output key: 'log_var' or 'logvar'")

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

        # Calculate reconstruction loss based on specified type
        if self.reconstruction_loss_type == "bce":
            # Binary cross entropy for image data (values in [0, 1])
            epsilon = 1e-8  # Small value for numerical stability
            recon_loss = -jnp.sum(
                x * jnp.log(reconstructed + epsilon)
                + (1 - x) * jnp.log(1 - reconstructed + epsilon),
                axis=tuple(range(1, x.ndim)),
            )
            recon_loss = jnp.mean(recon_loss)
        else:
            # Default to MSE loss
            recon_loss = jnp.mean((x - reconstructed) ** 2)

        # KL divergence loss
        kl_loss = -0.5 * jnp.mean(1 + log_var - mean**2 - jnp.exp(log_var))

        # Total loss with beta weighting
        total_loss = recon_loss + beta * kl_loss

        # Return dict with all loss components
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "beta": jnp.array(beta),  # Current beta value (for monitoring)
        }


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
        params: dict | None = None,
        batch: dict | None = None,
        rng: jax.Array | None = None,
        x: jax.Array | None = None,
        outputs: dict[str, jax.Array] | None = None,
        beta: float | None = None,
        step: int = 0,  # Current training step
        **kwargs,
    ) -> dict[str, jax.Array]:
        """Calculate loss with optional capacity control."""

        # Get base loss components
        base_losses = super().loss_fn(
            params=params,
            batch=batch,
            rng=rng,
            x=x,
            outputs=outputs,
            beta=beta,
            step=step,
            **kwargs,
        )

        if not self.use_capacity_control:
            return base_losses

        # Extract components
        recon_loss = base_losses["reconstruction_loss"]
        mean = outputs["mean"]
        log_var = outputs.get("log_var", outputs.get("logvar"))

        # Per-sample KL divergence (not averaged yet)
        kl_per_sample = -0.5 * jnp.sum(
            1 + log_var - mean**2 - jnp.exp(log_var),
            axis=-1,  # Sum over latent dimensions
        )

        # Average KL over batch
        kl_loss = jnp.mean(kl_per_sample)

        # Calculate current capacity
        current_capacity = jnp.minimum(
            self.capacity_max, self.capacity_max * step / self.capacity_num_iter
        )

        # Capacity loss: γ * |KL - C|
        capacity_loss = self.gamma * jnp.abs(kl_loss - current_capacity)

        # Total loss
        total_loss = recon_loss + capacity_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "capacity_loss": capacity_loss,
            "current_capacity": current_capacity,
        }
