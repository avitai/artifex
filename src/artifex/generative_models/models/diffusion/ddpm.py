"""DDPM (Denoising Diffusion Probabilistic Models) implementation."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import DDPMConfig
from artifex.generative_models.models.diffusion.base import DiffusionModel
from artifex.generative_models.training.utils import extract_model_prediction


class DDPMModel(DiffusionModel):
    """DDPM (Denoising Diffusion Probabilistic Models) implementation.

    This model implements the denoising diffusion probabilistic model
    as described in the DDPM paper by Ho et al.

    Uses nested DDPMConfig with:
    - backbone: BackboneConfig (polymorphic) for the denoising network
    - noise_schedule: NoiseScheduleConfig for the diffusion schedule
    - loss_type: Loss function type (mse, l1, huber)
    - clip_denoised: Whether to clip denoised samples

    Backbone type is determined by config.backbone.backbone_type discriminator.
    """

    def __init__(
        self,
        config: DDPMConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the DDPM model.

        Args:
            config: DDPMConfig with nested backbone and noise_schedule configs.
                    The backbone field accepts any BackboneConfig type and the
                    appropriate backbone is created based on backbone_type.
            rngs: Random number generators
        """
        # Store DDPM-specific parameters
        self.loss_type = config.loss_type
        self.clip_denoised = config.clip_denoised

        # Store input shape from config
        self.input_dim = config.input_shape
        # Extract in_channels from input_shape (C, H, W format)
        if len(config.input_shape) > 0:
            self.in_channels = config.input_shape[0]
        else:
            self.in_channels = 1

        # Store noise schedule parameters for convenience
        self.noise_steps = config.noise_schedule.num_timesteps
        self.beta_start = config.noise_schedule.beta_start
        self.beta_end = config.noise_schedule.beta_end
        self.beta_schedule = config.noise_schedule.schedule_type

        # Initialize parent class - handles backbone and noise schedule
        super().__init__(config, rngs=rngs)

    def forward_diffusion(self, x: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward diffusion process q(x_t | x_0).

        Args:
            x: Input data tensor
            t: Timestep indices

        Returns:
            Tuple of (noisy_x, noise)
        """
        # Generate noise using self.rngs
        noise = jax.random.normal(self.rngs.noise(), x.shape)

        # Apply noise according to schedule
        noisy_x = self.q_sample(x, t, noise)

        return noisy_x, noise

    def denoise_step(
        self,
        x_t: jax.Array,
        t: jax.Array,
        predicted_noise: jax.Array,
        clip_denoised: bool = True,
    ) -> jax.Array:
        """Perform a single denoising step: x_{t-1} = f(x_t, t, noise).

        Args:
            x_t: Noisy input at timestep t
            t: Current timestep indices
            predicted_noise: Predicted noise from the model
            clip_denoised: Whether to clip values to [-1, 1]

        Returns:
            Denoised x_{t-1}
        """
        # Predict x_0 from x_t and predicted noise
        pred_x0 = self.predict_start_from_noise(x_t, t, predicted_noise)

        # Clip if requested
        if clip_denoised:
            pred_x0 = jnp.clip(pred_x0, -1.0, 1.0)

        # Get posterior parameters
        posterior_mean, _, _ = self.q_posterior_mean_variance(pred_x0, x_t, t)

        # Return posterior mean (deterministic step)
        return posterior_mean

    def sample(
        self,
        n_samples_or_shape: int | tuple[int, ...],
        scheduler: str = "ddpm",
        steps: int | None = None,
    ) -> jax.Array:
        """Sample from the diffusion model.

        Args:
            n_samples_or_shape: Number of samples or full shape including batch dimension
            scheduler: Sampling scheduler to use ('ddpm', 'ddim')
            steps: Number of sampling steps (if None, use default)

        Returns:
            Generated samples
        """
        # Handle both int (n_samples) and tuple (full_shape) inputs
        if isinstance(n_samples_or_shape, int):
            n_samples = n_samples_or_shape
            input_shape = self._get_sample_shape()
        else:
            # Extract n_samples and shape from the full shape tuple
            full_shape = n_samples_or_shape
            n_samples = full_shape[0]
            input_shape = full_shape[1:]  # Remove batch dimension

        # Set default steps if not provided
        if steps is None:
            steps = self.noise_steps

        # Sample using selected scheduler
        if scheduler == "ddpm":
            # Default DDPM sampling
            return self.generate(n_samples, shape=input_shape)
        elif scheduler == "ddim":
            # DDIM sampling (deterministic)
            return self._sample_ddim(n_samples, input_shape, steps)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

    def _get_sample_shape(self) -> tuple[int, ...]:
        """Get the shape for sampling.

        Returns:
            Shape tuple for samples
        """
        if self.input_dim is not None:
            if isinstance(self.input_dim, (tuple, list)):
                return tuple(self.input_dim)
            else:
                raise ValueError(f"Cannot determine sample shape from input_dim: {self.input_dim}")
        else:
            # Default fallback
            return (32, 32, 3)

    def _sample_ddim(
        self,
        n_samples: int,
        shape: tuple[int, ...],
        steps: int,
        eta: float = 0.0,
    ) -> jax.Array:
        """DDIM sampling implementation.

        Args:
            n_samples: Number of samples
            shape: Sample shape
            steps: Number of steps
            eta: DDIM parameter (0 = deterministic, 1 = DDPM)

        Returns:
            Generated samples
        """
        # Initialize noise using self.rngs
        x = jax.random.normal(self.rngs.sample(), (n_samples, *shape))

        # Create timestep schedule for DDIM
        timesteps = jnp.linspace(self.noise_steps - 1, 0, steps, dtype=jnp.int32)

        # DDIM sampling loop
        for i, t in enumerate(timesteps):
            # Current timestep for all samples
            t_batch = jnp.full((n_samples,), t, dtype=jnp.int32)

            # Get model prediction
            model_output_dict = self(x, t_batch)
            predicted_noise = extract_model_prediction(model_output_dict)

            # DDIM step
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                t_prev_batch = jnp.full((n_samples,), t_prev, dtype=jnp.int32)
                x = self._ddim_step(x, t_batch, t_prev_batch, predicted_noise, eta)
            else:
                # Last step - predict x_0 directly
                x = self.predict_start_from_noise(x, t_batch, predicted_noise)
                x = jnp.clip(x, -1.0, 1.0)

        return x

    def _ddim_step(
        self,
        x_t: jax.Array,
        t: jax.Array,
        t_prev: jax.Array,
        predicted_noise: jax.Array,
        eta: float = 0.0,
    ) -> jax.Array:
        """Single DDIM step.

        Args:
            x_t: Current sample
            t: Current timestep
            t_prev: Previous timestep
            predicted_noise: Predicted noise
            eta: DDIM interpolation parameter

        Returns:
            Next sample x_{t-1}
        """
        # Get alpha values
        alpha_t = self._extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
        alpha_t_prev = self._extract_into_tensor(self.alphas_cumprod, t_prev, x_t.shape)

        # Predict x_0
        pred_x0 = self.predict_start_from_noise(x_t, t, predicted_noise)
        pred_x0 = jnp.clip(pred_x0, -1.0, 1.0)

        # DDIM formula
        sigma_t = (
            eta
            * jnp.sqrt((1 - alpha_t_prev) / (1 - alpha_t))
            * jnp.sqrt(1 - alpha_t / alpha_t_prev)
        )

        # Deterministic part
        pred_dir = jnp.sqrt(1 - alpha_t_prev - sigma_t**2) * predicted_noise

        # Add noise if eta > 0 using self.rngs
        if eta > 0:
            noise = jax.random.normal(self.rngs.sample(), x_t.shape)
            random_noise = sigma_t * noise
        else:
            random_noise = 0.0

        # Compute next sample
        x_prev = jnp.sqrt(alpha_t_prev) * pred_x0 + pred_dir + random_noise

        return x_prev

    def __call__(
        self,
        x: jax.Array,
        timesteps: jax.Array,
    ) -> dict[str, Any]:
        """Forward pass through the DDPM model.

        Args:
            x: Input data
            timesteps: Diffusion timesteps

        Returns:
            Dictionary containing model outputs

        Note:
            Train/eval mode controlled via model.train()/model.eval().
            RNGs stored at init time per NNX best practices.
        """
        return super().__call__(x, timesteps)
