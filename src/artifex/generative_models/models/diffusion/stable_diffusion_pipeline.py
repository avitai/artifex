"""Stable Diffusion Pipeline implementation.

This module implements a production-ready Stable Diffusion pipeline that wraps
StableDiffusionModel and adds generation/inference capabilities:
- DDPM scheduler for diffusion sampling
- Full generation loop with classifier-free guidance
- Train/eval mode switching

The Pipeline composes StableDiffusionModel to avoid code duplication (DRY).
"""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import StableDiffusionConfig
from artifex.generative_models.models.diffusion.stable_diffusion import StableDiffusionModel


class DDPMScheduler:
    """DDPM noise scheduler for stable diffusion.

    This implements the DDPM noise scheduling used in Stable Diffusion.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
    ):
        """Initialize DDPM scheduler.

        Args:
            num_train_timesteps: Number of diffusion steps
            beta_start: Start value for beta schedule
            beta_end: End value for beta schedule
            beta_schedule: Type of beta schedule
        """
        self.num_train_timesteps = num_train_timesteps

        # Create beta schedule
        if beta_schedule == "linear":
            betas = jnp.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            # Stable Diffusion uses scaled linear schedule
            betas = jnp.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(
        self,
        original_samples: jax.Array,
        noise: jax.Array,
        timesteps: jax.Array,
    ) -> jax.Array:
        """Add noise to the original samples according to the noise magnitude at each timestep.

        Args:
            original_samples: Original samples
            noise: Random noise to add
            timesteps: Timesteps for each sample

        Returns:
            Noisy samples
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod[..., None]
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[..., None]

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(
        self,
        model_output: jax.Array,
        timestep: int,
        sample: jax.Array,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Predict the sample at the previous timestep.

        Args:
            model_output: Direct output from learned diffusion model
            timestep: Current discrete timestep
            sample: Current instance of sample
            key: Random key for sampling

        Returns:
            Predicted previous sample
        """
        t = timestep

        # 1. Compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t] if t > 0 else jnp.array(1.0)
        beta_prod_t = 1 - alpha_prod_t

        # 2. Compute predicted original sample from predicted noise
        pred_original_sample = (sample - jnp.sqrt(beta_prod_t) * model_output) / jnp.sqrt(
            alpha_prod_t
        )

        # 3. Clip predicted original sample
        pred_original_sample = jnp.clip(pred_original_sample, -1, 1)

        # 4. Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = (jnp.sqrt(alpha_prod_t_prev) * self.betas[t]) / beta_prod_t
        current_sample_coeff = jnp.sqrt(self.alphas[t]) * (1 - alpha_prod_t_prev) / beta_prod_t

        # 5. Compute predicted previous sample mean
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        )

        # 6. Add noise
        variance = 0
        if t > 0:
            variance_noise = (
                jax.random.normal(key, sample.shape) if key is not None else jnp.zeros_like(sample)
            )
            variance = jnp.sqrt(self.posterior_variance[t]) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def set_timesteps(self, num_inference_steps: int) -> jax.Array:
        """Set the discrete timesteps used for the diffusion chain.

        Args:
            num_inference_steps: Number of diffusion steps during inference

        Returns:
            Array of timesteps
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = jnp.arange(0, num_inference_steps) * step_ratio
        timesteps = jnp.flip(timesteps)
        return timesteps.astype(jnp.int32)


class StableDiffusionPipeline(nnx.Module):
    """Stable Diffusion Pipeline.

    This pipeline wraps StableDiffusionModel and adds generation capabilities:
    - DDPM scheduler for step-by-step denoising
    - Full generation loop with classifier-free guidance
    - Train/eval mode switching

    Uses composition to avoid duplicating model code (DRY principle).
    """

    def __init__(self, config: StableDiffusionConfig, *, rngs: nnx.Rngs):
        """Initialize Stable Diffusion Pipeline.

        Args:
            config: StableDiffusionConfig with nested configs for all components
            rngs: Random number generators
        """
        super().__init__()

        self.config = config
        self.rngs = rngs

        # Compose StableDiffusionModel (DRY - avoid duplicating model code)
        self.model = StableDiffusionModel(config, rngs=rngs)

        # Initialize scheduler for generation loop
        # Use scaled_linear schedule (Stable Diffusion default)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=config.noise_schedule.num_timesteps,
            beta_start=config.noise_schedule.beta_start,
            beta_end=config.noise_schedule.beta_end,
            beta_schedule="scaled_linear",
        )

        # Generation defaults from config
        self.default_guidance_scale = config.guidance_scale
        self.default_num_inference_steps = 50

        # VAE scaling factor
        self.vae_scaling_factor = config.latent_scale_factor

    # Delegate core functionality to model
    def encode_text(
        self, token_ids: jax.Array, attention_mask: jax.Array | None = None
    ) -> jax.Array:
        """Encode text tokens to embeddings.

        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask

        Returns:
            Text embeddings [batch_size, seq_len, embedding_dim]
        """
        return self.model.encode_text(token_ids, attention_mask=attention_mask)

    def encode_image(self, images: jax.Array) -> jax.Array:
        """Encode images to latent representations.

        Args:
            images: Images [batch_size, height, width, channels]

        Returns:
            Latent representations [batch_size, latent_height, latent_width, latent_channels]
        """
        # Use model's encode_image which returns (mean, log_var)
        mean, _ = self.model.encode_image(images)
        # Scale latents
        latents = mean * self.vae_scaling_factor
        return latents

    def decode_latents(self, latents: jax.Array) -> jax.Array:
        """Decode latent representations to images.

        Args:
            latents: Latent representations [batch_size, latent_height,
                latent_width, latent_channels]

        Returns:
            Decoded images [batch_size, height, width, channels]
        """
        # Unscale latents
        unscaled_latents = latents / self.vae_scaling_factor
        # Decode using model
        images = self.model.decode_latent(unscaled_latents)
        # Clip to [0, 1]
        images = jnp.clip(images, 0.0, 1.0)
        return images

    def generate_from_text(
        self,
        token_ids: jax.Array,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        height: int = 512,
        width: int = 512,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Generate images from text prompts.

        Args:
            token_ids: Text token IDs [batch_size, seq_len]
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            height: Output image height
            width: Output image width
            rngs: Random number generators

        Returns:
            Generated images [batch_size, height, width, 3]
        """
        if num_inference_steps is None:
            num_inference_steps = self.default_num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.default_guidance_scale

        batch_size = token_ids.shape[0]

        # Encode text
        text_embeddings = self.encode_text(token_ids)

        # For classifier-free guidance, we need unconditional embeddings
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            # Create unconditional (empty) token IDs
            uncond_tokens = jnp.zeros_like(token_ids)
            uncond_embeddings = self.encode_text(uncond_tokens)

            # Concatenate for parallel processing
            text_embeddings = jnp.concatenate([uncond_embeddings, text_embeddings], axis=0)

        # Calculate latent dimensions (SD uses 8x downsampling)
        latent_height = height // 8
        latent_width = width // 8
        latent_channels = self.model.latent_channels

        # Initialize random latents
        latent_shape = (batch_size, latent_height, latent_width, latent_channels)
        sample_key = rngs()
        latents = jax.random.normal(sample_key, latent_shape)

        # Set timesteps
        timesteps = self.scheduler.set_timesteps(num_inference_steps)

        # Denoising loop
        for t in timesteps:
            # Expand latents if doing classifier-free guidance
            latent_model_input = (
                jnp.concatenate([latents, latents], axis=0)
                if do_classifier_free_guidance
                else latents
            )

            # Create timestep array
            t_batch = jnp.full((latent_model_input.shape[0],), t, dtype=jnp.int32)

            # Predict noise residual using model's UNet
            noise_pred = self.model.unet(latent_model_input, t_batch, conditioning=text_embeddings)

            # Perform classifier-free guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = jnp.split(noise_pred, 2, axis=0)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # Compute previous noisy sample
            step_key = rngs()
            latents = self.scheduler.step(noise_pred, t, latents, key=step_key)

        # Decode latents to images
        images = self.decode_latents(latents)

        return images

    def train_step(self, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """Perform a single training step.

        Delegates to the underlying model's train_step.

        Args:
            batch: Batch containing "images" and "token_ids"

        Returns:
            Dictionary containing loss and other metrics
        """
        # Delegate to model's train_step
        return self.model.train_step(
            images=batch["images"],
            text_tokens=batch["token_ids"],
        )

    def train(self):
        """Set pipeline to training mode."""
        self.model.text_encoder.train()
        self.model.unet.train()
        self.model.encoder.train()
        self.model.decoder.train()

    def eval(self):
        """Set pipeline to evaluation mode."""
        self.model.text_encoder.eval()
        self.model.unet.eval()
        self.model.encoder.eval()
        self.model.decoder.eval()

    # Expose model's components for direct access
    @property
    def encoder(self):
        """Access to image encoder."""
        return self.model.encoder

    @property
    def decoder(self):
        """Access to image decoder."""
        return self.model.decoder

    @property
    def text_encoder(self):
        """Access to text encoder."""
        return self.model.text_encoder

    @property
    def unet(self):
        """Access to UNet."""
        return self.model.unet
