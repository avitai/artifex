"""Stable Diffusion Model implementation.

This module implements Stable Diffusion using proper components:
- UNet2DCondition: UNet with cross-attention for text conditioning
- CLIPTextEncoder: CLIP-style text encoder
- SpatialEncoder/SpatialDecoder: VAE for spatial latent encoding

The design follows HuggingFace Diffusers' UNet2DConditionModel patterns.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import StableDiffusionConfig
from artifex.generative_models.models.backbones.unet_cross_attention import UNet2DCondition
from artifex.generative_models.models.diffusion.clip_text_encoder import CLIPTextEncoder
from artifex.generative_models.models.vae.spatial_autoencoder import (
    SpatialDecoder,
    SpatialEncoder,
)


class StableDiffusionModel(nnx.Module):
    """Stable Diffusion Model.

    This implements Stable Diffusion with proper architecture:
    - UNet with cross-attention for text-conditioned denoising
    - CLIP-style text encoder for text embeddings
    - Spatial VAE for latent space encoding/decoding
    - Classifier-free guidance support

    Uses StableDiffusionConfig which extends LatentDiffusionConfig -> DDPMConfig.
    """

    def __init__(
        self,
        config: StableDiffusionConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize Stable Diffusion model.

        Args:
            config: StableDiffusionConfig with nested configs for backbone,
                    noise_schedule, encoder, decoder, and text parameters.
            rngs: Random number generators
        """
        super().__init__()

        self.config = config
        self.rngs = rngs

        # Extract configuration
        self.text_embedding_dim = config.text_embedding_dim
        self.text_max_length = config.text_max_length
        self.vocab_size = config.vocab_size
        self.guidance_scale = config.guidance_scale
        self.use_guidance = config.use_guidance
        self.noise_steps = config.noise_schedule.num_timesteps

        # Input/output dimensions
        self.input_shape = config.input_shape
        latent_channels = config.encoder.latent_dim
        self.latent_channels = latent_channels

        # Build noise schedule
        betas = jnp.linspace(
            config.noise_schedule.beta_start,
            config.noise_schedule.beta_end,
            self.noise_steps,
        )
        alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(alphas)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

        # Initialize Text Encoder (CLIP-style)
        self.text_encoder = CLIPTextEncoder(
            vocab_size=self.vocab_size,
            max_length=self.text_max_length,
            embedding_dim=self.text_embedding_dim,
            num_layers=6,  # Smaller for testing
            num_heads=8,
            rngs=rngs,
        )

        # Initialize UNet with cross-attention
        # UNet operates in latent space, so use latent_channels for in/out
        hidden_dims = config.backbone.hidden_dims
        self.unet = UNet2DCondition(
            in_channels=latent_channels,
            out_channels=latent_channels,
            hidden_dims=list(hidden_dims),
            num_res_blocks=2,
            attention_levels=[0, 1, 2] if len(hidden_dims) >= 3 else list(range(len(hidden_dims))),
            cross_attention_dim=self.text_embedding_dim,
            num_heads=8,
            rngs=rngs,
        )

        # Initialize spatial encoder/decoder for VAE using config objects
        self.encoder = SpatialEncoder(config=config.encoder, rngs=rngs)
        self.decoder = SpatialDecoder(config=config.decoder, rngs=rngs)

        # Latent scaling factor (Stable Diffusion uses 0.18215)
        self.latent_scale_factor = config.latent_scale_factor

        # Guidance setup
        if self.use_guidance:
            self.unconditional_token = jnp.zeros((1, self.text_max_length), dtype=jnp.int32)

    def encode_text(
        self,
        text_tokens: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Encode text tokens to embeddings.

        Args:
            text_tokens: Text token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            Text embeddings [batch_size, seq_len, text_embedding_dim]
        """
        return self.text_encoder(text_tokens, attention_mask=attention_mask)

    def encode_image(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Encode images to latent space.

        Args:
            x: Images [batch_size, height, width, channels]

        Returns:
            Tuple of (mean, log_var) for latent distribution
        """
        return self.encoder(x)

    def decode_latent(self, z: jax.Array) -> jax.Array:
        """Decode latent codes to images.

        Args:
            z: Latent codes [batch_size, latent_height, latent_width, latent_channels]

        Returns:
            Decoded images [batch_size, height, width, channels]
        """
        return self.decoder(z)

    def __call__(
        self,
        x: jax.Array,
        timesteps: jax.Array,
        *,
        text_embeddings: jax.Array | None = None,
    ) -> dict[str, Any]:
        """Forward pass for noise prediction in latent space.

        This method predicts noise for the given noisy latent codes at the specified
        timesteps, conditioned on text embeddings via cross-attention.

        Note: The UNet operates in latent space. Input should be latent codes
              with shape [batch, latent_height, latent_width, latent_channels].
              Use encode_image() to convert images to latent space first.

        Args:
            x: Latent codes [batch_size, latent_height, latent_width, latent_channels]
            timesteps: Diffusion timesteps [batch_size]
            text_embeddings: Text embeddings [batch_size, seq_len, text_embedding_dim]
                             If None, uses unconditional embedding.

        Returns:
            Dictionary containing 'predicted_noise' in latent space
        """
        # If no text embeddings provided, create unconditional ones
        if text_embeddings is None:
            batch_size = x.shape[0]
            uncond_tokens = jnp.tile(self.unconditional_token, (batch_size, 1))
            text_embeddings = self.encode_text(uncond_tokens)

        # Predict noise using UNet with cross-attention
        predicted_noise = self.unet(x, timesteps, conditioning=text_embeddings)

        return {"predicted_noise": predicted_noise}

    def forward_diffusion(
        self,
        x: jax.Array,
        t: jax.Array,
        noise: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Add noise to samples according to diffusion schedule.

        Args:
            x: Clean samples
            t: Timestep indices
            noise: Optional pre-generated noise

        Returns:
            Tuple of (noisy_samples, noise)
        """
        if noise is None:
            noise = jax.random.normal(self.rngs.noise(), x.shape)

        # Get schedule values
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        # Broadcast for spatial dimensions
        while len(sqrt_alpha.shape) < len(x.shape):
            sqrt_alpha = sqrt_alpha[..., None]
            sqrt_one_minus_alpha = sqrt_one_minus_alpha[..., None]

        noisy = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        return noisy, noise

    def compute_text_similarity(
        self,
        text_tokens_1: jax.Array,
        text_tokens_2: jax.Array,
    ) -> jax.Array:
        """Compute cosine similarity between text embeddings.

        Args:
            text_tokens_1: First text tokens [batch_size, seq_len]
            text_tokens_2: Second text tokens [batch_size, seq_len]

        Returns:
            Cosine similarity [batch_size]
        """
        emb_1 = self.encode_text(text_tokens_1)
        emb_2 = self.encode_text(text_tokens_2)

        # Average pool embeddings
        emb_1_pooled = jnp.mean(emb_1, axis=1)
        emb_2_pooled = jnp.mean(emb_2, axis=1)

        # Compute cosine similarity
        norm_1 = jnp.linalg.norm(emb_1_pooled, axis=-1, keepdims=True)
        norm_2 = jnp.linalg.norm(emb_2_pooled, axis=-1, keepdims=True)

        similarity = jnp.sum(emb_1_pooled * emb_2_pooled, axis=-1) / (
            norm_1.squeeze(-1) * norm_2.squeeze(-1) + 1e-8
        )

        return similarity

    def train_step(
        self,
        images: jax.Array,
        text_tokens: jax.Array,
    ) -> dict[str, jax.Array]:
        """Single training step.

        Args:
            images: Training images [batch_size, height, width, channels]
            text_tokens: Text token IDs [batch_size, seq_len]

        Returns:
            Dictionary containing 'loss'
        """
        batch_size = images.shape[0]

        # Encode images to latent space
        mean, _ = self.encode_image(images)
        latents = mean * self.latent_scale_factor

        # Sample noise and timesteps
        noise = jax.random.normal(self.rngs.noise(), latents.shape)
        timesteps = jax.random.randint(
            self.rngs.time(),
            (batch_size,),
            0,
            self.noise_steps,
        )

        # Add noise to latents
        noisy_latents, _ = self.forward_diffusion(latents, timesteps, noise)

        # Encode text
        text_embeddings = self.encode_text(text_tokens)

        # Predict noise
        output = self(noisy_latents, timesteps, text_embeddings=text_embeddings)
        noise_pred = output["predicted_noise"]

        # Compute MSE loss
        loss = jnp.mean((noise_pred - noise) ** 2)

        return {"loss": loss}
