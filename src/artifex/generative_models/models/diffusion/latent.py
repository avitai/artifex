"""Latent Diffusion Model implementation.

This module provides a Latent Diffusion Model (LDM) that combines a VAE
encoder/decoder with a diffusion model operating in the compressed latent space.

The implementation supports both:
- Flat latent codes with 1D backbones (UNet1D)
- Spatial latent codes with 2D backbones (UNet, DiT, etc.)

For 2D backbones, flat latent codes are automatically reshaped to spatial format
(batch, H, W, C) before being passed to the backbone.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import LatentDiffusionConfig
from artifex.generative_models.models.diffusion.ddpm import DDPMModel
from artifex.generative_models.models.vae.decoders import MLPDecoder
from artifex.generative_models.models.vae.encoders import MLPEncoder


class SimpleEncoder(nnx.Module):
    """Simple encoder for latent diffusion model."""

    def __init__(
        self,
        input_dim: tuple[int, ...],
        latent_dim: int,
        hidden_dims: list | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the encoder.

        Args:
            input_dim: Input dimensions (H, W, C) for images or (D,) for vectors
            latent_dim: Latent dimension
            hidden_dims: Hidden layer dimensions
            rngs: Random number generators
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [32, 64]

        # Determine if input is image or vector
        self.is_image = isinstance(input_dim, (tuple, list)) and len(input_dim) >= 2

        # Calculate flattened dimension
        if self.is_image:
            # Image input: calculate flattened dimension
            if len(input_dim) == 3:  # (H, W, C)
                self.flat_dim = input_dim[0] * input_dim[1] * input_dim[2]
            elif len(input_dim) == 2:  # (H, W) - grayscale
                self.flat_dim = input_dim[0] * input_dim[1]
            else:
                raise ValueError(f"Unsupported input_dim shape: {input_dim}")
        else:
            # Vector input
            self.flat_dim = input_dim if isinstance(input_dim, int) else input_dim[0]

        # Create encoder layers
        self.layers = nnx.List([])
        in_features = self.flat_dim
        for hidden_dim in self.hidden_dims:
            self.layers.append(nnx.Linear(in_features, hidden_dim, rngs=rngs))
            in_features = hidden_dim

        # Output layers for mean and log variance
        self.mean_layer = nnx.Linear(in_features, latent_dim, rngs=rngs)
        self.logvar_layer = nnx.Linear(in_features, latent_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Encode input to latent space.

        Args:
            x: Input tensor

        Returns:
            Tuple of (mean, log_variance)
        """
        # Flatten input if needed
        if self.is_image:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)

        # Pass through encoder layers
        h = x
        for layer in self.layers:
            h = nnx.relu(layer(h))

        # Get mean and log variance
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)

        return mean, logvar


class SimpleDecoder(nnx.Module):
    """Simple decoder for latent diffusion model."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: tuple[int, ...],
        hidden_dims: list | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the decoder.

        Args:
            latent_dim: Latent dimension
            output_dim: Output dimensions (H, W, C) for images or (D,) for vectors
            hidden_dims: Hidden layer dimensions (in reverse order)
            rngs: Random number generators
        """
        super().__init__()
        if rngs is None:
            raise ValueError("rngs must be provided for SimpleDecoder")

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [64, 32]

        # Determine if output is image or vector
        self.is_image = isinstance(output_dim, (tuple, list)) and len(output_dim) >= 2

        # Calculate flattened dimension
        if self.is_image:
            if len(output_dim) == 3:  # (H, W, C)
                self.flat_dim = output_dim[0] * output_dim[1] * output_dim[2]
            elif len(output_dim) == 2:  # (H, W) - grayscale
                self.flat_dim = output_dim[0] * output_dim[1]
            else:
                raise ValueError(f"Unsupported output_dim shape: {output_dim}")
        else:
            self.flat_dim = output_dim if isinstance(output_dim, int) else output_dim[0]

        # Create decoder layers
        self.layers = nnx.List([])
        in_features = latent_dim
        for hidden_dim in self.hidden_dims:
            self.layers.append(nnx.Linear(in_features, hidden_dim, rngs=rngs))
            in_features = hidden_dim

        # Output layer
        self.output_layer = nnx.Linear(in_features, self.flat_dim, rngs=rngs)

    def __call__(self, z: jax.Array) -> jax.Array:
        """Decode latent code to output space.

        Args:
            z: Latent code

        Returns:
            Decoded output
        """
        # Pass through decoder layers
        h = z
        for layer in self.layers:
            h = nnx.relu(layer(h))

        # Get output
        output = self.output_layer(h)

        # Reshape to image if needed
        if self.is_image:
            batch_size = output.shape[0]
            output = output.reshape(batch_size, *self.output_dim)

        return output


class LDMModel(DDPMModel):
    """Latent Diffusion Model implementation.

    This model combines a VAE for encoding/decoding with a diffusion model
    that operates in the latent space.

    Uses nested LatentDiffusionConfig with:
    - backbone: BackboneConfig (polymorphic) for the denoising network
    - noise_schedule: NoiseScheduleConfig for the diffusion schedule
    - encoder: EncoderConfig for encoding to latent space
    - decoder: DecoderConfig for decoding from latent space
    - latent_scale_factor: Scaling factor for latent codes

    Supports both 1D and 2D backbones:
    - UNet1D: Works with (batch, sequence, channels) format
    - UNet/DiT: Works with (batch, H, W, C) spatial format

    Flat latent codes from the MLP encoder are automatically reshaped to the
    appropriate format for the backbone.
    """

    def __init__(
        self,
        config: LatentDiffusionConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the Latent Diffusion Model.

        Args:
            config: LatentDiffusionConfig with nested configs for backbone,
                    noise_schedule, encoder, and decoder.
                    The backbone field accepts any BackboneConfig type and the
                    appropriate backbone is created based on backbone_type.
            rngs: Random number generators
        """
        # Store config and LDM-specific parameters
        self.original_input_dim = config.input_shape  # Store original for output_dim property
        self.latent_dim = config.encoder.latent_dim
        self.scale_factor = config.latent_scale_factor

        # Determine backbone type and compute latent spatial shape
        self._backbone_type = config.backbone.backbone_type
        self._uses_spatial_backbone = self._backbone_type not in ("unet_1d",)

        # Get expected channels from backbone config if using spatial backbone
        if self._uses_spatial_backbone and hasattr(config.backbone, "in_channels"):
            expected_channels = config.backbone.in_channels
        else:
            expected_channels = None

        self._latent_spatial_shape = self._compute_latent_spatial_shape(
            self.latent_dim, expected_channels
        )

        # Create encoder and decoder from nested configs
        self.encoder = MLPEncoder(config=config.encoder, rngs=rngs)
        self.decoder = MLPDecoder(config=config.decoder, rngs=rngs)
        self.use_pretrained_vae = False

        # Initialize parent DDPM model - operates in latent space
        # The parent will use config.backbone for the UNet
        super().__init__(config, rngs=rngs)

        # Override input_dim for latent space operations
        # For spatial backbones, use spatial shape; otherwise use flat
        if self._uses_spatial_backbone:
            self.input_dim = self._latent_spatial_shape
        else:
            self.input_dim = (self.latent_dim,)

    def _compute_latent_spatial_shape(
        self, latent_dim: int, expected_channels: int | None = None
    ) -> tuple[int, int, int]:
        """Compute spatial shape (H, W, C) for latent codes.

        For 2D backbones like UNet, we need to reshape flat latent codes
        to spatial format. This method finds suitable (H, W, C) dimensions
        such that H * W * C = latent_dim.

        Args:
            latent_dim: Total dimension of latent codes
            expected_channels: If provided, use this exact number of channels
                             (from backbone's in_channels)

        Returns:
            Tuple of (H, W, C) spatial dimensions
        """
        # If expected channels are specified, use them
        if expected_channels is not None and latent_dim % expected_channels == 0:
            remaining = latent_dim // expected_channels
            sqrt_remaining = int(math.sqrt(remaining))

            # Check for perfect square
            if sqrt_remaining * sqrt_remaining == remaining:
                return (sqrt_remaining, sqrt_remaining, expected_channels)

            # Try to find H, W factors that are close to square
            for h in range(sqrt_remaining, 0, -1):
                if remaining % h == 0:
                    w = remaining // h
                    if w <= h * 4:  # Allow aspect ratio up to 1:4
                        return (h, w, expected_channels)

        # Fallback: Try common channel counts
        for c in [4, 3, 2, 1]:
            if latent_dim % c == 0:
                remaining = latent_dim // c
                sqrt_remaining = int(math.sqrt(remaining))

                # Check for perfect square
                if sqrt_remaining * sqrt_remaining == remaining:
                    return (sqrt_remaining, sqrt_remaining, c)

                # Try to find H, W factors that are close to square
                for h in range(sqrt_remaining, 0, -1):
                    if remaining % h == 0:
                        w = remaining // h
                        # Prefer roughly square dimensions
                        if w <= h * 4:  # Allow aspect ratio up to 1:4
                            return (h, w, c)

        # Fallback: treat as 1D with single channel
        return (latent_dim, 1, 1)

    def _reshape_to_spatial(self, z: jax.Array) -> jax.Array:
        """Reshape flat latent codes to spatial format for 2D backbones.

        Args:
            z: Flat latent codes with shape (batch, latent_dim)

        Returns:
            Spatial latent codes with shape (batch, H, W, C)
        """
        batch_size = z.shape[0]
        return z.reshape(batch_size, *self._latent_spatial_shape)

    def _reshape_from_spatial(self, z: jax.Array) -> jax.Array:
        """Reshape spatial latent codes back to flat format.

        Args:
            z: Spatial latent codes with shape (batch, H, W, C)

        Returns:
            Flat latent codes with shape (batch, latent_dim)
        """
        batch_size = z.shape[0]
        return z.reshape(batch_size, -1)

    @property
    def output_dim(self):
        """Get output dimensions."""
        return self.original_input_dim

    def encode(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Encode input to latent space.

        Args:
            x: Input tensor

        Returns:
            Tuple of (latent_code, posterior_params)
        """
        if self.use_pretrained_vae:
            raise NotImplementedError("Pre-trained VAE encoding not implemented")

        mean, logvar = self.encoder(x)
        return mean, logvar

    def decode(self, z: jax.Array) -> jax.Array:
        """Decode latent code to output space.

        Args:
            z: Latent code

        Returns:
            Decoded output
        """
        if self.use_pretrained_vae:
            raise NotImplementedError("Pre-trained VAE decoding not implemented")

        return self.decoder(z)

    def reparameterize(self, mean: jax.Array, logvar: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        """Reparameterization trick.

        Args:
            mean: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            rngs: Random number generators

        Returns:
            Sampled latent code
        """
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rngs.sample(), mean.shape)
        return mean + eps * std

    def denoise(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Predict noise from noisy input using the backbone.

        Handles reshaping between flat and spatial formats as needed
        by the backbone type.

        Args:
            x: Noisy input (flat format: batch, latent_dim)
            t: Timestep indices

        Returns:
            Predicted noise (same shape as input)
        """
        if self._uses_spatial_backbone:
            # Reshape to spatial format for 2D backbone
            x_spatial = self._reshape_to_spatial(x)
            noise_pred_spatial = self.backbone(x_spatial, t)
            # Reshape back to flat format
            return self._reshape_from_spatial(noise_pred_spatial)
        else:
            # For 1D backbone, pass through directly
            return self.backbone(x, t)

    def __call__(
        self,
        x: jax.Array,
        t: jax.Array | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> dict[str, jax.Array]:
        """Forward pass through the model.

        Args:
            x: Input images
            t: Timesteps (optional)
            rngs: Random number generators

        Returns:
            Dictionary with model outputs
        """
        if rngs is None:
            rngs = self.rngs

        # Encode to latent space
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar, rngs=rngs)

        # Scale latent code
        z = z * self.scale_factor

        # Apply diffusion in latent space
        if t is None:
            batch_size = x.shape[0]
            t = jax.random.randint(rngs.time(), (batch_size,), 0, self.noise_steps)

        # Forward diffusion (uses self.rngs internally)
        z_noisy, noise = self.forward_diffusion(z, t)

        # Predict noise
        predicted_noise = self.denoise(z_noisy, t)

        # Decode back to image space
        z_recon = z_noisy - predicted_noise
        x_recon = self.decode(z_recon / self.scale_factor)

        return {
            "reconstruction": x_recon,
            "mean": mean,
            "logvar": logvar,
            "latent": z,
            "noisy_latent": z_noisy,
            "predicted_noise": predicted_noise,
            "true_noise": noise,
        }

    def sample(
        self,
        num_samples: int,
        *,
        return_trajectory: bool = False,
    ) -> jax.Array | list[jax.Array]:
        """Sample from the model by generating in latent space.

        Args:
            num_samples: Number of samples to generate
            return_trajectory: If True, return full trajectory

        Returns:
            Generated samples (decoded to image space) or trajectory
        """
        # For spatial backbones, sample in spatial latent format
        # For 1D backbones, sample in flat format
        if self._uses_spatial_backbone:
            latent_shape = self._latent_spatial_shape
        else:
            latent_shape = (self.latent_dim,)

        # Initialize noise in latent space
        z = jax.random.normal(self.rngs.sample(), (num_samples, *latent_shape))

        # Get number of timesteps
        num_timesteps = self.noise_schedule.num_timesteps

        trajectory = [] if return_trajectory else None

        # Reverse diffusion loop in latent space
        for t in range(num_timesteps - 1, -1, -1):
            timesteps = jnp.full((num_samples,), t, dtype=jnp.int32)

            # Predict noise in latent space
            # For spatial backbones, z is already spatial
            # For flat backbones, z is flat
            if self._uses_spatial_backbone:
                noise_pred_spatial = self.backbone(z, timesteps)
                noise_pred = noise_pred_spatial  # Keep spatial for diffusion step
            else:
                noise_pred = self.backbone(z, timesteps)

            # Denoise step using parent's logic
            z = self.denoise_step(z, timesteps, noise_pred, clip_denoised=True)

            if return_trajectory:
                # Convert to flat for decoding
                z_flat = self._reshape_from_spatial(z) if self._uses_spatial_backbone else z
                x = self.decode(z_flat / self.scale_factor)
                trajectory.append(x)

        # Final conversion to flat format for decoding
        if self._uses_spatial_backbone:
            z = self._reshape_from_spatial(z)

        # Decode final latent to image space
        samples = self.decode(z / self.scale_factor)

        if return_trajectory:
            return trajectory
        return samples

    def loss(
        self, x: jax.Array, t: jax.Array | None = None, *, rngs: nnx.Rngs | None = None
    ) -> jax.Array:
        """Compute LDM loss.

        Args:
            x: Input images
            t: Timesteps (optional)
            rngs: Random number generators

        Returns:
            Loss value
        """
        outputs = self(x, t, rngs=rngs)

        # Diffusion loss in latent space
        diffusion_loss = jnp.mean((outputs["predicted_noise"] - outputs["true_noise"]) ** 2)

        # KL divergence loss
        kl_loss = -0.5 * jnp.mean(
            1 + outputs["logvar"] - outputs["mean"] ** 2 - jnp.exp(outputs["logvar"])
        )

        # Total loss
        kl_weight = 1e-6  # Small weight for KL loss
        return diffusion_loss + kl_weight * kl_loss
