"""Base Diffusion Model implementation."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel
from artifex.generative_models.core.configuration import DiffusionConfig
from artifex.generative_models.core.configuration.backbone_config import create_backbone
from artifex.generative_models.core.noise_schedule import create_noise_schedule, NoiseSchedule
from artifex.generative_models.training.utils import extract_model_prediction


class DiffusionModel(GenerativeModel):
    """Base class for diffusion models.

    This implements a general diffusion model that can support various diffusion
    processes like DDPM (Denoising Diffusion Probabilistic Models) and
    DDIM (Denoising Diffusion Implicit Models).

    Uses the nested DiffusionConfig architecture with:
    - backbone: BackboneConfig (polymorphic) for the denoising network
    - noise_schedule: NoiseScheduleConfig for the diffusion schedule

    Backbone type is determined by config.backbone.backbone_type discriminator.
    Supported backbones: UNet, DiT, U-ViT, UNet2DCondition.

    Attributes:
        config: DiffusionConfig for the model
        backbone: The denoising network (created from config.backbone)
        noise_schedule: NoiseSchedule instance for diffusion process
    """

    def __init__(
        self,
        config: DiffusionConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize diffusion model.

        Args:
            config: DiffusionConfig with nested backbone and noise_schedule configs.
                    The backbone field accepts any BackboneConfig type (UNetBackboneConfig,
                    DiTBackboneConfig, etc.) and the appropriate backbone is created
                    based on the backbone_type discriminator.
            rngs: Random number generators
        """
        super().__init__(
            rngs=rngs,
        )
        self.config = config

        # Initialize backbone network from polymorphic config
        # Uses create_backbone factory which dispatches on backbone_type
        self.backbone = create_backbone(config.backbone, rngs=rngs)

        # Create noise schedule from nested config
        # Supports linear, cosine, and quadratic schedules based on schedule_type
        self.noise_schedule: NoiseSchedule = create_noise_schedule(config.noise_schedule)

        # Expose schedule attributes directly for backward compatibility with subclasses
        # that access self.betas, self.alphas, etc.
        self.betas = self.noise_schedule.betas
        self.alphas = self.noise_schedule.alphas
        self.alphas_cumprod = self.noise_schedule.alphas_cumprod
        self.alphas_cumprod_prev = self.noise_schedule.alphas_cumprod_prev
        self.sqrt_alphas_cumprod = self.noise_schedule.sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = self.noise_schedule.sqrt_one_minus_alphas_cumprod
        self.log_one_minus_alphas_cumprod = self.noise_schedule.log_one_minus_alphas_cumprod
        self.sqrt_recip_alphas_cumprod = self.noise_schedule.sqrt_recip_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = self.noise_schedule.sqrt_recipm1_alphas_cumprod
        self.posterior_variance = self.noise_schedule.posterior_variance
        self.posterior_log_variance_clipped = self.noise_schedule.posterior_log_variance_clipped
        self.posterior_mean_coef1 = self.noise_schedule.posterior_mean_coef1
        self.posterior_mean_coef2 = self.noise_schedule.posterior_mean_coef2

    def __call__(
        self,
        x: jax.Array,
        timesteps: jax.Array,
        *,
        conditioning: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Forward pass through the diffusion model.

        Args:
            x: Input data
            timesteps: Diffusion timesteps
            conditioning: Optional conditioning information (for guided generation)
            **kwargs: Additional arguments passed to the backbone

        Returns:
            Dictionary containing model outputs

        Note:
            Train/eval mode controlled via model.train()/model.eval().
            RNGs stored at init time per NNX best practices.
        """
        # Apply the backbone network - always pass conditioning, let backbone handle None
        output = self.backbone(x, timesteps, conditioning=conditioning, **kwargs)

        # Return as dictionary for consistency with protocol
        if isinstance(output, dict):
            return output
        else:
            return {"predicted_noise": output}

    def q_sample(
        self,
        x_start: jax.Array,
        t: jax.Array,
        noise: jax.Array | None = None,
    ) -> jax.Array:
        """Sample from the forward diffusion process q(x_t | x_0).

        Args:
            x_start: Starting clean data (x_0)
            t: Timesteps
            noise: Optional pre-generated noise

        Returns:
            Noisy samples x_t
        """
        # Generate noise if not provided using self.rngs
        if noise is None:
            noise = jax.random.normal(self.rngs.noise(), x_start.shape)

        # Get the appropriate amount of noise for timesteps t
        sqrt_alphas_cumprod_t = self._extract_into_tensor(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        # Apply noise according to diffusion equation
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _extract_into_tensor(
        self,
        arr: jax.Array,
        timesteps: jax.Array,
        broadcast_shape: tuple[int, ...],
    ) -> jax.Array:
        """Extract values from a 1D array for a batch of indices.

        Args:
            arr: 1D array to extract from
            timesteps: Indices to extract
            broadcast_shape: Shape to broadcast the extracted values to

        Returns:
            Array of extracted values
        """
        # Extract values at specified indices
        res = arr[timesteps]

        # Ensure we have the right batch size
        batch_size = broadcast_shape[0]

        # Ensure timesteps has the same batch size as the input
        if timesteps.shape[0] != batch_size:
            # Repeat timesteps to match batch size
            if timesteps.shape[0] == 1:
                timesteps = jnp.repeat(timesteps, batch_size, axis=0)
                res = arr[timesteps]
            else:
                # Truncate or pad to match batch size
                if timesteps.shape[0] < batch_size:
                    # Pad by repeating the last value
                    padding_needed = batch_size - timesteps.shape[0]
                    last_value = timesteps[-1:] if timesteps.size > 0 else jnp.array([0])
                    padding = jnp.repeat(last_value, padding_needed)
                    timesteps = jnp.concatenate([timesteps, padding])
                else:
                    # Truncate to batch size
                    timesteps = timesteps[:batch_size]
                res = arr[timesteps]

        # Reshape to (batch_size, 1, 1, ...) to match broadcast_shape dimensions
        # The first dimension is batch_size, the rest should be 1s
        target_shape = (batch_size,) + (1,) * (len(broadcast_shape) - 1)
        res = res.reshape(target_shape)

        return jnp.broadcast_to(res, broadcast_shape)

    def predict_start_from_noise(self, x_t: jax.Array, t: jax.Array, noise: jax.Array) -> jax.Array:
        """Predict x_0 from noise model output.

        Args:
            x_t: Noisy input at timestep t
            t: Timesteps
            noise: Predicted noise

        Returns:
            Predicted x_0
        """
        sqrt_recip_alphas_cumprod_t = self._extract_into_tensor(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def q_posterior_mean_variance(
        self, x_start: jax.Array, x_t: jax.Array, t: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Clean data (x_0)
            x_t: Noisy data (x_t)
            t: Timesteps

        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance_clipped)
        """
        posterior_mean_coef1_t = self._extract_into_tensor(
            self.posterior_mean_coef1, t, x_start.shape
        )
        posterior_mean_coef2_t = self._extract_into_tensor(
            self.posterior_mean_coef2, t, x_start.shape
        )

        # Compute posterior mean
        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t

        # Extract and broadcast posterior variance and log variance
        posterior_variance_t = self._extract_into_tensor(self.posterior_variance, t, x_start.shape)
        posterior_log_variance_clipped_t = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_start.shape
        )

        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t

    def p_mean_variance(
        self,
        model_output: jax.Array,
        x_t: jax.Array,
        t: jax.Array,
        clip_denoised: bool = True,
    ) -> dict[str, jax.Array]:
        """Compute the model's predicted mean and variance for x_{t-1}.

        Args:
            model_output: Predicted noise or x_0
            x_t: Noisy input at timestep t
            t: Timesteps
            clip_denoised: Whether to clip the denoised signal to [-1, 1]

        Returns:
            dictionary with predicted mean and variance
        """
        # The model predicts epsilon (noise)
        pred_noise = model_output

        # Get predicted x_0
        pred_x_start = self.predict_start_from_noise(x_t, t, pred_noise)

        # Clip if needed
        if clip_denoised:
            pred_x_start = jnp.clip(pred_x_start, -1.0, 1.0)

        # Get the parameters for q(x_{t-1} | x_t, x_0)
        posterior_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            pred_x_start, x_t, t
        )

        return {
            "mean": posterior_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_x_start": pred_x_start,
        }

    def p_sample(
        self,
        model_output: jax.Array,
        x_t: jax.Array,
        t: jax.Array,
        clip_denoised: bool = True,
    ) -> jax.Array:
        """Sample from the denoising process p(x_{t-1} | x_t).

        Args:
            model_output: Predicted noise
            x_t: Noisy input at timestep t
            t: Timesteps
            clip_denoised: Whether to clip the denoised signal to [-1, 1]

        Returns:
            Denoised x_{t-1}
        """
        # Get predicted mean and variance
        out = self.p_mean_variance(model_output, x_t, t, clip_denoised=clip_denoised)

        # Create noise for sampling using self.rngs
        noise = jax.random.normal(self.rngs.sample(), x_t.shape)

        # Don't add noise for the last step (t=0)
        nonzero_mask = (t != 0).reshape((-1,) + (1,) * (len(x_t.shape) - 1))
        nonzero_mask = jnp.broadcast_to(nonzero_mask, x_t.shape)

        # Compute x_{t-1} = mean + sigma * z where z is random noise
        pred_prev_sample = out["mean"] + nonzero_mask * jnp.exp(0.5 * out["log_variance"]) * noise

        return pred_prev_sample

    def generate(
        self,
        n_samples: int = 1,
        *,
        shape: tuple[int, ...] | None = None,
        clip_denoised: bool = True,
    ) -> jax.Array:
        """Generate samples from random noise.

        Args:
            n_samples: Number of samples to generate
            shape: Shape of samples to generate (excluding batch dimension)
            clip_denoised: Whether to clip the denoised signal to [-1, 1]

        Returns:
            Generated samples
        """
        # Determine shape of samples - use input_shape from config
        if shape is None:
            shape = self.config.input_shape

        # Initialize noise — extract RNG key before entering lax.scan
        # to avoid mutating NNX RngCount inside a different trace level
        key = self.rngs.sample()
        key, init_key = jax.random.split(key)
        img = jax.random.normal(init_key, (n_samples, *shape))

        # Get number of timesteps from noise schedule
        num_timesteps = self.noise_schedule.num_timesteps

        # Create reversed timestep array for lax.scan
        # scan processes timesteps from num_timesteps-1 down to 0
        reversed_timesteps = jnp.arange(num_timesteps - 1, -1, -1, dtype=jnp.int32)

        def denoise_step(
            carry: tuple[jax.Array, jax.Array],
            t_scalar: jax.Array,
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            img, rng = carry
            rng, step_key = jax.random.split(rng)
            timesteps = jnp.full((n_samples,), t_scalar, dtype=jnp.int32)

            # Get model output for current timestep
            model_output_dict = self(img, timesteps)
            model_output = extract_model_prediction(model_output_dict)

            # Inline p_sample logic with explicit RNG key (avoids self.rngs mutation)
            out = self.p_mean_variance(
                model_output,
                img,
                timesteps,
                clip_denoised=clip_denoised,
            )
            noise = jax.random.normal(step_key, img.shape)
            nonzero_mask = (timesteps != 0).reshape((-1,) + (1,) * (len(img.shape) - 1))
            nonzero_mask = jnp.broadcast_to(nonzero_mask, img.shape)
            img = out["mean"] + nonzero_mask * jnp.exp(0.5 * out["log_variance"]) * noise

            return (img, rng), None

        (img, _), _ = jax.lax.scan(denoise_step, (img, key), reversed_timesteps)

        return img

    def loss_fn(
        self,
        batch: Any,
        model_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute loss for training.

        Args:
            batch: Input batch (should contain 'x' key with data)
            model_outputs: Model outputs from forward pass

        Returns:
            Dictionary containing loss and metrics
        """
        # Extract data from batch
        if isinstance(batch, dict):
            x = batch.get("x", batch.get("data", batch))
        else:
            x = batch

        # Get batch size
        batch_size = x.shape[0]

        # Sample timesteps using self.rngs
        num_timesteps = self.noise_schedule.num_timesteps
        t = jax.random.randint(self.rngs.timestep(), (batch_size,), 0, num_timesteps)

        # Generate noise using self.rngs
        noise = jax.random.normal(self.rngs.noise(), x.shape)

        # Add noise to inputs (for target we use the original noise)
        # noisy_x = self.q_sample(x, t, noise=noise, rngs=rngs)  # Not needed for loss calculation

        # Get prediction from model outputs
        predicted = extract_model_prediction(model_outputs)

        # Compute loss
        loss = jnp.mean((predicted - noise) ** 2)

        # Return loss and metrics as dictionary
        return {
            "loss": loss,
            "mse_loss": loss,
            "avg_timestep": jnp.mean(t.astype(jnp.float32)),
        }
