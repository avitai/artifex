"""GAN-specific trainer with multiple loss variants and regularization.

Provides specialized training utilities for Generative Adversarial Networks
including multiple loss types, gradient penalty, and R1 regularization.

References:
    - WGAN-GP: https://arxiv.org/abs/1704.00028
    - Hinge Loss: https://arxiv.org/abs/1802.05957
    - R1 Regularization: https://arxiv.org/abs/1801.04406
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.losses import (
    hinge_discriminator_loss,
    hinge_generator_loss,
    least_squares_discriminator_loss,
    least_squares_generator_loss,
    ns_vanilla_discriminator_loss,
    ns_vanilla_generator_loss,
    wasserstein_discriminator_loss,
    wasserstein_generator_loss,
)


@dataclass(slots=True)
class GANTrainingConfig:
    """Configuration for GAN-specific training.

    Attributes:
        loss_type: GAN loss variant.
            - "vanilla": Standard GAN loss (BCE)
            - "wasserstein": Wasserstein distance (requires gradient penalty)
            - "hinge": Hinge loss (used in BigGAN, StyleGAN2)
            - "lsgan": Least squares GAN
        n_critic: Discriminator updates per generator update.
        gp_weight: Gradient penalty weight (for WGAN-GP).
        gp_target: Target gradient norm (usually 1.0).
        r1_weight: R1 regularization weight.
        label_smoothing: Smooth real labels to [1-smoothing, 1].
    """

    loss_type: Literal["vanilla", "wasserstein", "hinge", "lsgan"] = "vanilla"
    n_critic: int = 1
    gp_weight: float = 10.0
    gp_target: float = 1.0
    r1_weight: float = 0.0
    label_smoothing: float = 0.0


class GANTrainer:
    """GAN-specific trainer with multiple loss variants.

    This trainer provides a JIT-compatible interface for adversarial training
    with support for multiple loss functions and regularization techniques.
    The step methods take models and optimizers as explicit arguments,
    allowing them to be wrapped with nnx.jit for performance.

    Features:
        - Multiple loss types (vanilla, wasserstein, hinge, lsgan)
        - Configurable discriminator/generator update ratio
        - WGAN-GP gradient penalty
        - R1 regularization for discriminator
        - Label smoothing

    Example (non-JIT):
        ```python
        from artifex.generative_models.training.trainers import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(
            loss_type="wasserstein",
            n_critic=5,
            gp_weight=10.0,
        )
        trainer = GANTrainer(config)

        # Create models and optimizers separately
        generator = Generator(rngs=nnx.Rngs(0))
        discriminator = Discriminator(rngs=nnx.Rngs(1))
        g_optimizer = nnx.Optimizer(generator, optax.adam(1e-4))
        d_optimizer = nnx.Optimizer(discriminator, optax.adam(1e-4))

        # Training loop
        for step in range(num_steps):
            rng, d_key, g_key = jax.random.split(rng, 3)
            d_loss, d_metrics = trainer.discriminator_step(
                generator, discriminator, d_optimizer, real_batch, z, d_key
            )
            if step % config.n_critic == 0:
                g_loss, g_metrics = trainer.generator_step(
                    generator, discriminator, g_optimizer, z
                )
        ```

    Example (JIT-compiled):
        ```python
        trainer = GANTrainer(config)
        jit_d_step = nnx.jit(trainer.discriminator_step)
        jit_g_step = nnx.jit(trainer.generator_step)

        for step in range(num_steps):
            d_loss, d_metrics = jit_d_step(
                generator, discriminator, d_optimizer, real_batch, z, d_key
            )
            if step % config.n_critic == 0:
                g_loss, g_metrics = jit_g_step(
                    generator, discriminator, g_optimizer, z
                )
        ```
    """

    __slots__ = ("config",)

    def __init__(
        self,
        config: GANTrainingConfig | None = None,
    ) -> None:
        """Initialize GAN trainer.

        Args:
            config: GAN training configuration.
        """
        self.config = config or GANTrainingConfig()

    def compute_d_loss_vanilla(
        self,
        d_real: jax.Array,
        d_fake: jax.Array,
    ) -> jax.Array:
        """Compute vanilla GAN discriminator loss.

        Uses non-saturating loss from core/losses for numerical stability.

        Args:
            d_real: Discriminator output for real samples (logits).
            d_fake: Discriminator output for fake samples (logits).

        Returns:
            Discriminator loss.
        """
        return ns_vanilla_discriminator_loss(
            d_real,
            d_fake,
            label_smoothing=self.config.label_smoothing,
        )

    def compute_d_loss_wasserstein(
        self,
        d_real: jax.Array,
        d_fake: jax.Array,
    ) -> jax.Array:
        """Compute Wasserstein discriminator loss.

        Uses wasserstein_discriminator_loss from core/losses.

        Args:
            d_real: Discriminator output for real samples.
            d_fake: Discriminator output for fake samples.

        Returns:
            Discriminator loss (negative critic loss).
        """
        return wasserstein_discriminator_loss(d_real, d_fake)

    def compute_d_loss_hinge(
        self,
        d_real: jax.Array,
        d_fake: jax.Array,
    ) -> jax.Array:
        """Compute hinge discriminator loss.

        Uses hinge_discriminator_loss from core/losses.

        Args:
            d_real: Discriminator output for real samples.
            d_fake: Discriminator output for fake samples.

        Returns:
            Discriminator loss.
        """
        return hinge_discriminator_loss(d_real, d_fake)

    def compute_d_loss_lsgan(
        self,
        d_real: jax.Array,
        d_fake: jax.Array,
    ) -> jax.Array:
        """Compute least squares GAN discriminator loss.

        Uses least_squares_discriminator_loss from core/losses.

        Args:
            d_real: Discriminator output for real samples.
            d_fake: Discriminator output for fake samples.

        Returns:
            Discriminator loss.
        """
        return least_squares_discriminator_loss(d_real, d_fake)

    def compute_discriminator_loss(
        self,
        d_real: jax.Array,
        d_fake: jax.Array,
    ) -> jax.Array:
        """Compute discriminator loss based on configured loss type.

        Args:
            d_real: Discriminator output for real samples.
            d_fake: Discriminator output for fake samples.

        Returns:
            Discriminator loss.
        """
        if self.config.loss_type == "vanilla":
            return self.compute_d_loss_vanilla(d_real, d_fake)
        if self.config.loss_type == "wasserstein":
            return self.compute_d_loss_wasserstein(d_real, d_fake)
        if self.config.loss_type == "hinge":
            return self.compute_d_loss_hinge(d_real, d_fake)
        if self.config.loss_type == "lsgan":
            return self.compute_d_loss_lsgan(d_real, d_fake)

        msg = f"Unknown loss_type: {self.config.loss_type}"
        raise ValueError(msg)

    def compute_g_loss_vanilla(self, d_fake: jax.Array) -> jax.Array:
        """Compute vanilla GAN generator loss.

        Uses ns_vanilla_generator_loss from core/losses.

        Args:
            d_fake: Discriminator output for fake samples (logits).

        Returns:
            Generator loss.
        """
        return ns_vanilla_generator_loss(d_fake)

    def compute_g_loss_wasserstein(self, d_fake: jax.Array) -> jax.Array:
        """Compute Wasserstein generator loss.

        Uses wasserstein_generator_loss from core/losses.

        Args:
            d_fake: Discriminator output for fake samples.

        Returns:
            Generator loss.
        """
        return wasserstein_generator_loss(d_fake)

    def compute_g_loss_hinge(self, d_fake: jax.Array) -> jax.Array:
        """Compute hinge generator loss.

        Uses hinge_generator_loss from core/losses.

        Args:
            d_fake: Discriminator output for fake samples.

        Returns:
            Generator loss.
        """
        return hinge_generator_loss(d_fake)

    def compute_g_loss_lsgan(self, d_fake: jax.Array) -> jax.Array:
        """Compute least squares GAN generator loss.

        Uses least_squares_generator_loss from core/losses.

        Args:
            d_fake: Discriminator output for fake samples.

        Returns:
            Generator loss.
        """
        return least_squares_generator_loss(d_fake)

    def compute_generator_loss(self, d_fake: jax.Array) -> jax.Array:
        """Compute generator loss based on configured loss type.

        Args:
            d_fake: Discriminator output for fake samples.

        Returns:
            Generator loss.
        """
        if self.config.loss_type == "vanilla":
            return self.compute_g_loss_vanilla(d_fake)
        if self.config.loss_type == "wasserstein":
            return self.compute_g_loss_wasserstein(d_fake)
        if self.config.loss_type == "hinge":
            return self.compute_g_loss_hinge(d_fake)
        if self.config.loss_type == "lsgan":
            return self.compute_g_loss_lsgan(d_fake)

        msg = f"Unknown loss_type: {self.config.loss_type}"
        raise ValueError(msg)

    def compute_gradient_penalty(
        self,
        discriminator: nnx.Module,
        real: jax.Array,
        fake: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Compute WGAN-GP gradient penalty.

        Args:
            discriminator: Discriminator model.
            real: Real samples.
            fake: Fake samples (must have same shape as real).
            key: PRNG key for interpolation.

        Returns:
            Gradient penalty loss.
        """
        batch_size = real.shape[0]

        # Random interpolation coefficient
        alpha_shape = (batch_size,) + (1,) * (len(real.shape) - 1)
        alpha = jax.random.uniform(key, alpha_shape)

        # Interpolated samples
        interpolated = alpha * real + (1 - alpha) * fake

        # Compute gradients of discriminator output w.r.t. interpolated input
        def d_interp(x: jax.Array) -> jax.Array:
            return jnp.sum(discriminator(x))

        grads = jax.grad(d_interp)(interpolated)

        # Gradient norm
        grad_norm = jnp.sqrt(jnp.sum(grads**2, axis=tuple(range(1, len(grads.shape)))) + 1e-8)

        # Penalty: (||grad|| - target)^2
        return jnp.mean((grad_norm - self.config.gp_target) ** 2)

    def compute_r1_penalty(
        self,
        discriminator: nnx.Module,
        real: jax.Array,
    ) -> jax.Array:
        """Compute R1 regularization penalty.

        Args:
            discriminator: Discriminator model.
            real: Real samples.

        Returns:
            R1 penalty.
        """

        def d_real_sum(x: jax.Array) -> jax.Array:
            return jnp.sum(discriminator(x))

        grads = jax.grad(d_real_sum)(real)
        grad_norm_sq = jnp.sum(grads**2, axis=tuple(range(1, len(grads.shape))))
        return jnp.mean(grad_norm_sq)

    def discriminator_step(
        self,
        generator: nnx.Module,
        discriminator: nnx.Module,
        d_optimizer: nnx.Optimizer,
        real: jax.Array,
        z: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Execute a discriminator training step.

        This method can be wrapped with nnx.jit for performance:
            jit_step = nnx.jit(trainer.discriminator_step)
            loss, metrics = jit_step(generator, discriminator, d_optimizer, real, z, key)

        Args:
            generator: Generator model.
            discriminator: Discriminator model.
            d_optimizer: Optimizer for discriminator.
            real: Real samples.
            z: Latent vectors for generator.
            key: PRNG key for gradient penalty.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        # Generate fake samples (no gradient through generator)
        fake = jax.lax.stop_gradient(generator(z))

        def d_loss_fn(disc: nnx.Module) -> tuple[jax.Array, dict[str, Any]]:
            d_real = disc(real)
            d_fake = disc(fake)

            # Base loss
            loss = self.compute_discriminator_loss(d_real, d_fake)
            metrics: dict[str, Any] = {
                "d_loss": loss,
                "d_real": jnp.mean(d_real),
                "d_fake": jnp.mean(d_fake),
            }

            # Gradient penalty (WGAN-GP)
            if self.config.gp_weight > 0 and self.config.loss_type == "wasserstein":
                gp = self.compute_gradient_penalty(disc, real, fake, key)
                loss = loss + self.config.gp_weight * gp
                metrics["gp_loss"] = gp

            # R1 regularization
            if self.config.r1_weight > 0:
                r1 = self.compute_r1_penalty(disc, real)
                loss = loss + self.config.r1_weight * r1
                metrics["r1_loss"] = r1

            metrics["d_loss_total"] = loss
            return loss, metrics

        (loss, metrics), grads = nnx.value_and_grad(d_loss_fn, has_aux=True)(discriminator)
        d_optimizer.update(discriminator, grads)

        return loss, metrics

    def generator_step(
        self,
        generator: nnx.Module,
        discriminator: nnx.Module,
        g_optimizer: nnx.Optimizer,
        z: jax.Array,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Execute a generator training step.

        This method can be wrapped with nnx.jit for performance:
            jit_step = nnx.jit(trainer.generator_step)
            loss, metrics = jit_step(generator, discriminator, g_optimizer, z)

        Args:
            generator: Generator model.
            discriminator: Discriminator model.
            g_optimizer: Optimizer for generator.
            z: Latent vectors for generator.

        Returns:
            Tuple of (loss, metrics_dict).
        """

        def g_loss_fn(gen: nnx.Module) -> tuple[jax.Array, dict[str, Any]]:
            fake = gen(z)
            d_fake = discriminator(fake)

            loss = self.compute_generator_loss(d_fake)
            metrics = {
                "g_loss": loss,
                "d_fake_g": jnp.mean(d_fake),
            }
            return loss, metrics

        (loss, metrics), grads = nnx.value_and_grad(g_loss_fn, has_aux=True)(generator)
        g_optimizer.update(generator, grads)

        return loss, metrics
