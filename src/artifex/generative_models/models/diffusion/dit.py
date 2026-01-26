"""DiT (Diffusion Transformer) Model implementation.

Integrates the Diffusion Transformer backbone with the diffusion framework.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel
from artifex.generative_models.core.configuration import DiTConfig
from artifex.generative_models.core.noise_schedule import create_noise_schedule, NoiseSchedule


class DiTModel(GenerativeModel):
    """Diffusion model using Transformer backbone instead of U-Net.

    Implements Diffusion Transformers (DiT) which replace the U-Net backbone
    with a Vision Transformer for improved scalability and performance.

    Uses nested DiTConfig with:
    - noise_schedule: NoiseScheduleConfig for the diffusion schedule
    - patch_size, hidden_size, depth, num_heads, mlp_ratio: Transformer architecture
    - num_classes: Number of classes for conditional generation
    - cfg_scale: Classifier-free guidance scale
    """

    def __init__(
        self,
        config: DiTConfig,
        backbone_fn: Callable[[DiTConfig, nnx.Rngs], nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize DiT model.

        Args:
            config: DiTConfig with DiT-specific parameters and noise_schedule
            rngs: Random number generators
            backbone_fn: Optional custom backbone function
        """
        super().__init__(rngs=rngs)
        self.config = config

        # Store DiT-specific parameters from config
        self.num_classes = config.num_classes
        self.cfg_scale = config.cfg_scale
        self.learn_sigma = config.learn_sigma
        self.input_dim = config.input_shape

        # Pre-compute image dimensions from input_shape (C, H, W format)
        if len(config.input_shape) >= 3:
            self.in_channels = config.input_shape[0]
            self.img_size = config.input_shape[1]
        elif len(config.input_shape) >= 1:
            self.in_channels = config.input_shape[0]
            self.img_size = 32
        else:
            self.in_channels = 3
            self.img_size = 32

        # Create noise schedule from nested config
        self.noise_schedule: NoiseSchedule = create_noise_schedule(config.noise_schedule)

        # Expose schedule attributes for backward compatibility
        self.betas = self.noise_schedule.betas
        self.alphas = self.noise_schedule.alphas
        self.alphas_cumprod = self.noise_schedule.alphas_cumprod

        # If no custom backbone provided, use DiT
        if backbone_fn is None:
            from artifex.generative_models.models.backbones.dit import DiffusionTransformer

            def create_dit_backbone(cfg: DiTConfig, rngs: nnx.Rngs) -> nnx.Module:
                """Create DiT backbone from config."""
                # Extract input channels from input_shape (C, H, W format)
                if len(cfg.input_shape) >= 1:
                    in_channels = cfg.input_shape[0]
                else:
                    in_channels = 3

                # Compute img_size from input_shape
                if len(cfg.input_shape) >= 3:
                    img_size = cfg.input_shape[1]  # Height from (C, H, W)
                else:
                    img_size = 32

                return DiffusionTransformer(
                    img_size=img_size,
                    patch_size=cfg.patch_size,
                    in_channels=in_channels,
                    hidden_size=cfg.hidden_size,
                    depth=cfg.depth,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    num_classes=cfg.num_classes,
                    dropout_rate=0.0,
                    learn_sigma=cfg.learn_sigma,
                    rngs=rngs,
                )

            backbone_fn = create_dit_backbone

        # Create backbone
        self.backbone = backbone_fn(config, rngs)

    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        y: jax.Array | None = None,
        *,
        deterministic: bool = False,
        cfg_scale: float | None = None,
    ) -> jax.Array:
        """Forward pass through DiT model.

        Args:
            x: Input images [batch, height, width, channels]
            t: Timesteps [batch]
            y: Optional class labels [batch]
            deterministic: Whether to apply dropout
            cfg_scale: Classifier-free guidance scale

        Returns:
            Predicted noise [batch, height, width, channels]
        """
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        return self.backbone(x, t, y, deterministic=deterministic, cfg_scale=cfg_scale)

    def sample_step(
        self,
        x_t: jax.Array,
        t: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
        y: jax.Array | None = None,
        cfg_scale: float | None = None,
    ) -> jax.Array:
        """Single sampling step with optional classifier-free guidance.

        Args:
            x_t: Current noisy sample [batch, height, width, channels]
            t: Current timestep [batch]
            rngs: Random number generators
            y: Optional class labels for conditional generation
            cfg_scale: Classifier-free guidance scale

        Returns:
            Denoised sample [batch, height, width, channels]
        """
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        # Get noise prediction
        if cfg_scale > 1.0 and y is not None:
            # Classifier-free guidance
            # Predict noise with and without conditioning
            x_t.shape[0]

            # Stack conditional and unconditional inputs
            x_combined = jnp.concatenate([x_t, x_t], axis=0)
            t_combined = jnp.concatenate([t, t], axis=0)

            # Create labels: actual labels and null labels
            null_labels = jnp.full_like(y, self.num_classes) if self.num_classes else None
            y_combined = jnp.concatenate([y, null_labels], axis=0) if y is not None else None

            # Get predictions
            noise_pred = self.backbone(x_combined, t_combined, y_combined, deterministic=True)

            # Split conditional and unconditional predictions
            noise_cond, noise_uncond = jnp.split(noise_pred, 2, axis=0)

            # Apply classifier-free guidance
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            # Standard prediction
            noise_pred = self.backbone(x_t, t, y, deterministic=True)

        # If learning sigma, split the output
        if self.learn_sigma:
            noise_pred, learned_var = jnp.split(noise_pred, 2, axis=-1)

        # Use noise schedule from config
        noise_steps = self.noise_schedule.num_timesteps
        beta_start = self.config.noise_schedule.beta_start
        beta_end = self.config.noise_schedule.beta_end

        # Linear beta schedule
        timestep = t[0]
        beta = beta_start + (beta_end - beta_start) * timestep / noise_steps
        alpha = 1.0 - beta
        alpha_bar = jnp.prod(
            jnp.array(
                [
                    1.0 - beta_start - (beta_end - beta_start) * i / noise_steps
                    for i in range(timestep + 1)
                ]
            )
        )

        # Compute mean
        mean = (x_t - beta * noise_pred / jnp.sqrt(1 - alpha_bar)) / jnp.sqrt(alpha)

        # Add noise for non-final steps
        if t[0] > 0:
            key = (rngs or self.rngs).sample()

            noise = jax.random.normal(key, x_t.shape)

            if self.learn_sigma:
                # Use learned variance
                log_var = learned_var
                std = jnp.exp(0.5 * log_var)
            else:
                # Fixed variance
                std = jnp.sqrt(beta)

            x_t_minus_1 = mean + std * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1

    def generate(
        self,
        n_samples: int = 1,
        *,
        rngs: nnx.Rngs,
        num_steps: int = 1000,
        y: jax.Array | None = None,
        cfg_scale: float | None = None,
        img_size: int | None = None,
    ) -> jax.Array:
        """Generate samples using DiT.

        Args:
            n_samples: Number of samples to generate
            rngs: Random number generators
            num_steps: Number of diffusion steps
            y: Optional class labels for conditional generation
            cfg_scale: Classifier-free guidance scale
            img_size: Image size (uses config default if not specified)

        Returns:
            Generated samples [n_samples, height, width, channels]
        """
        # Use precomputed values or override
        actual_img_size = img_size if img_size is not None else self.img_size
        channels = self.in_channels

        # Initialize from noise
        key = (rngs or self.rngs).sample()

        x = jax.random.normal(key, (n_samples, actual_img_size, actual_img_size, channels))

        # Reverse diffusion process
        for t in reversed(range(num_steps)):
            t_batch = jnp.full((n_samples,), t)
            x = self.sample_step(x, t_batch, rngs=rngs, y=y, cfg_scale=cfg_scale)

        return x
