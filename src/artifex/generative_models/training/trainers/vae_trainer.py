"""VAE-specific trainer with KL annealing and beta-VAE support.

Provides specialized training utilities for Variational Autoencoders including
KL divergence annealing schedules, beta-VAE weighting, and free bits constraints.

References:
    - Beta-VAE: https://openreview.net/forum?id=Sy2fzU9gl
    - Cyclical Annealing: https://arxiv.org/abs/1903.10145
    - Free Bits: https://arxiv.org/abs/1606.04934
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.losses import binary_cross_entropy
from artifex.generative_models.core.losses.base import reduce_loss
from artifex.generative_models.training.utils import extract_batch_data


@dataclass(slots=True)
class VAETrainingConfig:
    """Configuration for VAE-specific training.

    Attributes:
        kl_annealing: Type of KL annealing schedule.
            - "none": No annealing, use full beta from start
            - "linear": Linear warmup from 0 to beta
            - "sigmoid": Sigmoid-shaped warmup
            - "cyclical": Cyclical annealing with periodic resets
        kl_warmup_steps: Number of steps to reach full KL weight.
        beta: Final beta weight for KL term (beta-VAE).
            Higher values encourage disentanglement but may hurt reconstruction.
        free_bits: Minimum KL per latent dimension (0 = disabled).
            Prevents posterior collapse by ensuring minimum information flow.
        cyclical_period: Period for cyclical annealing (if used).
    """

    kl_annealing: Literal["none", "linear", "sigmoid", "cyclical"] = "linear"
    kl_warmup_steps: int = 10000
    beta: float = 1.0
    free_bits: float = 0.0
    cyclical_period: int = 10000


class VAETrainer:
    """VAE-specific trainer with KL annealing and beta-VAE support.

    This trainer provides a JIT-compatible interface for training VAEs with:
    - KL annealing schedules (linear, sigmoid, cyclical)
    - Beta-VAE weighting for disentanglement
    - Free bits constraint to prevent posterior collapse

    The train_step method takes model and optimizer as explicit arguments,
    allowing it to be wrapped with nnx.jit for performance.

    The trainer computes the ELBO loss with configurable KL weighting:
        L = reconstruction_loss + beta * kl_weight(step) * kl_loss

    Example (non-JIT):
        ```python
        from artifex.generative_models.training.trainers import (
            VAETrainer,
            VAETrainingConfig,
        )

        config = VAETrainingConfig(
            kl_annealing="cyclical",
            beta=4.0,
            free_bits=0.5,
        )
        trainer = VAETrainer(config)

        # Create model and optimizer separately
        model = VAEModel(config, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-4))

        # Training loop
        for step, batch in enumerate(data):
            loss, metrics = trainer.train_step(model, optimizer, batch, step=step)
        ```

    Example (JIT-compiled):
        ```python
        trainer = VAETrainer(config)
        jit_step = nnx.jit(trainer.train_step)

        for step, batch in enumerate(data):
            loss, metrics = jit_step(model, optimizer, batch, step=step)
        ```

    Note:
        The model is expected to return (reconstruction, mean, logvar) from
        its forward pass. The trainer handles loss computation and KL annealing.
    """

    __slots__ = ("config",)

    def __init__(
        self,
        config: VAETrainingConfig | None = None,
    ) -> None:
        """Initialize VAE trainer.

        Args:
            config: VAE training configuration. Uses defaults if not provided.
        """
        self.config = config or VAETrainingConfig()

    def get_kl_weight(self, step: int | jax.Array) -> jax.Array:
        """Compute KL weight based on annealing schedule.

        This method is JIT-compatible - uses JAX operations instead of Python builtins.

        Args:
            step: Current training step (can be traced array for JIT).

        Returns:
            KL weight multiplier (0.0 to beta).
        """
        step = jnp.asarray(step, dtype=jnp.float32)
        beta = self.config.beta
        warmup = jnp.maximum(1.0, self.config.kl_warmup_steps)

        if self.config.kl_annealing == "none":
            return jnp.asarray(beta)

        if self.config.kl_annealing == "linear":
            progress = jnp.minimum(1.0, step / warmup)
            return beta * progress

        if self.config.kl_annealing == "sigmoid":
            # Sigmoid annealing centered at warmup_steps/2
            center = self.config.kl_warmup_steps / 2
            scale = jnp.maximum(1.0, self.config.kl_warmup_steps / 10)
            x = (step - center) / scale
            progress = 1 / (1 + jnp.exp(-x))
            return beta * progress

        if self.config.kl_annealing == "cyclical":
            # Cyclical annealing with linear ramp within each cycle
            cycle_position = step % self.config.cyclical_period
            half_period = jnp.maximum(1.0, self.config.cyclical_period / 2)
            progress = jnp.minimum(1.0, cycle_position / half_period)
            return beta * progress

        return jnp.asarray(beta)

    def apply_free_bits(self, kl_per_dim: jax.Array) -> jax.Array:
        """Apply free bits constraint to KL divergence.

        Ensures minimum KL per latent dimension to prevent posterior collapse.

        Args:
            kl_per_dim: KL divergence per latent dimension.

        Returns:
            KL divergence with free bits applied.
        """
        if self.config.free_bits <= 0:
            return kl_per_dim
        return jnp.maximum(kl_per_dim, self.config.free_bits)

    def compute_kl_loss(
        self,
        mean: jax.Array,
        logvar: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute KL divergence from standard normal.

        Args:
            mean: Latent mean, shape (batch, latent_dim).
            logvar: Latent log-variance, shape (batch, latent_dim).

        Returns:
            Tuple of (total_kl_loss, kl_per_sample) where:
                - total_kl_loss: Scalar mean KL loss
                - kl_per_sample: KL loss per sample, shape (batch,)
        """
        # KL divergence per dimension: -0.5 * (1 + logvar - mean^2 - exp(logvar))
        kl_per_dim = -0.5 * (1 + logvar - mean**2 - jnp.exp(logvar))

        # Apply free bits constraint
        kl_per_dim = self.apply_free_bits(kl_per_dim)

        # Sum over latent dimensions, mean over batch
        kl_per_sample = jnp.sum(kl_per_dim, axis=-1)
        kl_loss = jnp.mean(kl_per_sample)

        return kl_loss, kl_per_sample

    def compute_reconstruction_loss(
        self,
        x: jax.Array,
        recon_x: jax.Array,
        loss_type: Literal["mse", "bce"] = "mse",
    ) -> jax.Array:
        """Compute reconstruction loss.

        Args:
            x: Original input, shape (batch, ...).
            recon_x: Reconstructed output, shape (batch, ...).
            loss_type: Type of reconstruction loss ("mse" or "bce").

        Returns:
            Scalar reconstruction loss.
        """
        if loss_type == "mse":
            # Mean squared error: sum over spatial, mean over batch (standard for VAE)
            mse = (x - recon_x) ** 2
            return reduce_loss(mse, reduction="batch_sum")

        if loss_type == "bce":
            # Binary cross-entropy with batch_sum reduction (standard for VAE ELBO)
            return binary_cross_entropy(recon_x, x, reduction="batch_sum")

        msg = f"Unknown loss_type: {loss_type}"
        raise ValueError(msg)

    def compute_loss(
        self,
        model: nnx.Module,
        batch: dict[str, Any],
        step: int,
        loss_type: Literal["mse", "bce"] = "mse",
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute VAE loss with KL annealing.

        Args:
            model: VAE model to evaluate.
            batch: Batch dictionary with "image" or "data" key.
            step: Current training step for annealing.
            loss_type: Type of reconstruction loss.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        # Get input data using shared utility
        x = extract_batch_data(batch)

        # Forward pass - handle both dict and tuple outputs
        outputs = model(x)
        if isinstance(outputs, dict):
            # Standard VAE model returns dict
            recon_x = outputs.get("reconstructed", outputs.get("reconstruction"))
            mean = outputs["mean"]
            logvar = outputs.get("log_var", outputs.get("logvar"))
        else:
            # Legacy tuple format: (reconstruction, mean, logvar)
            recon_x, mean, logvar = outputs

        # Reconstruction loss
        recon_loss = self.compute_reconstruction_loss(x, recon_x, loss_type)

        # KL divergence
        kl_loss, _ = self.compute_kl_loss(mean, logvar)

        # Apply KL annealing
        kl_weight = self.get_kl_weight(step)
        total_loss = recon_loss + kl_weight * kl_loss

        metrics = {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "kl_weight": kl_weight,
        }

        return total_loss, metrics

    def train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: dict[str, Any],
        step: int = 0,
        loss_type: Literal["mse", "bce"] = "mse",
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Execute a single training step.

        This method can be wrapped with nnx.jit for performance:
            jit_step = nnx.jit(trainer.train_step)
            loss, metrics = jit_step(model, optimizer, batch, step=step)

        Args:
            model: VAE model to train.
            optimizer: NNX optimizer for parameter updates.
            batch: Batch dictionary with "image" or "data" key.
            step: Current training step for annealing.
            loss_type: Type of reconstruction loss.

        Returns:
            Tuple of (loss, metrics_dict).
        """

        def loss_fn(m: nnx.Module) -> tuple[jax.Array, dict[str, Any]]:
            return self.compute_loss(m, batch, step, loss_type)

        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)

        return loss, metrics

    def create_loss_fn(
        self,
        loss_type: Literal["mse", "bce"] = "mse",
    ) -> Callable[
        [nnx.Module, dict[str, Any], jax.Array, jax.Array],
        tuple[jax.Array, dict[str, Any]],
    ]:
        """Create loss function compatible with train_epoch_staged.

        This enables DRY integration - VAE-specific training logic can be used
        with the staged training infrastructure.

        Args:
            loss_type: Type of reconstruction loss.

        Returns:
            Function with signature: (model, batch, rng, step) -> (loss, metrics)
            The step parameter is passed dynamically by train_epoch_staged,
            enabling proper KL annealing inside JIT-compiled fori_loop.
        """
        from flax import nnx as nnx_module

        def loss_fn(
            model: nnx_module.Module,
            batch: dict[str, Any],
            _rng: jax.Array,  # Accepted for API compatibility, not used by VAE
            step: jax.Array,  # Step passed dynamically for KL annealing
        ) -> tuple[jax.Array, dict[str, Any]]:
            return self.compute_loss(model, batch, step, loss_type)

        return loss_fn
