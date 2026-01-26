"""PixelCNN implementation for autoregressive image generation."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import PixelCNNConfig
from artifex.generative_models.core.layers import PixelCNNResidualBlock
from artifex.generative_models.models.autoregressive.base import AutoregressiveModel


class MaskedConv2D(nnx.Module):
    """Masked 2D convolution for autoregressive image modeling.

    Implements spatial masking to ensure the autoregressive property:
    each pixel can only depend on previously generated pixels.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, int] = (3, 3),
        mask_type: str = "A",
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize masked convolution.

        Args:
            in_features: Number of input channels
            out_features: Number of output channels
            kernel_size: Size of the convolution kernel
            mask_type: Type of mask ("A" for first layer, "B" for subsequent layers)
            rngs: Random number generators
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.mask_type = mask_type

        # Create the base convolution
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )

        # Create the mask
        self.mask = self._create_mask()

    def _create_mask(self) -> jax.Array:
        """Create the autoregressive mask for the convolution kernel.

        Returns:
            Mask array with same shape as kernel
        """
        kh, kw = self.kernel_size
        mask = jnp.ones((kh, kw, self.in_features, self.out_features))

        # Center of the kernel
        center_h, center_w = kh // 2, kw // 2

        # Mask future pixels (below and to the right)
        # Mask all pixels below center
        mask = mask.at[center_h + 1 :, :, :, :].set(0)

        # Mask pixels to the right of center on center row
        mask = mask.at[center_h, center_w + 1 :, :, :].set(0)

        # For mask type A, also mask the center pixel
        if self.mask_type == "A":
            mask = mask.at[center_h, center_w, :, :].set(0)

        return mask

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        """Apply masked convolution.

        Args:
            x: Input tensor [batch, height, width, channels]
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after masked convolution
        """
        # Apply mask to the kernel weights
        # Use indexing [...] to access the underlying array (new NNX API)
        masked_kernel = self.conv.kernel[...] * self.mask

        # Save original kernel and temporarily replace for this forward pass
        original_kernel = self.conv.kernel[...]
        self.conv.kernel[...] = masked_kernel

        # Apply convolution
        output = self.conv(x)

        # Restore original kernel
        self.conv.kernel[...] = original_kernel

        return output


# ResidualBlock is now imported from core layers as PixelCNNResidualBlock


class PixelCNN(AutoregressiveModel):
    """PixelCNN model for autoregressive image generation.

    Generates images pixel by pixel using masked convolutions to maintain
    the autoregressive property in 2D spatial coordinates.
    """

    def __init__(
        self,
        config: PixelCNNConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize PixelCNN model.

        Args:
            config: PixelCNNConfig with model configuration
            rngs: Random number generators

        Raises:
            TypeError: If config is not a PixelCNNConfig
        """
        if not isinstance(config, PixelCNNConfig):
            raise TypeError(f"config must be PixelCNNConfig, got {type(config).__name__}")

        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.key(0))

        # Extract config values
        image_shape = config.image_shape
        height, width, channels = image_shape

        # For discrete pixel values, vocab_size is typically 256 per channel
        vocab_size = 256
        sequence_length = height * width * channels

        super().__init__(
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            rngs=rngs,
        )

        # Store config
        self.config = config

        # Extract values from config for convenience
        self.image_shape = image_shape
        self.num_layers = config.num_layers
        self.hidden_channels = config.hidden_channels
        self.num_residual_blocks = config.num_residual_blocks

        # First layer with mask type A
        self.first_conv = MaskedConv2D(
            in_features=channels,
            out_features=self.hidden_channels,
            kernel_size=(7, 7),
            mask_type="A",
            rngs=rngs,
        )

        # Hidden layers with mask type B
        self.hidden_convs = nnx.List([])
        for _ in range(self.num_layers - 1):
            self.hidden_convs.append(
                MaskedConv2D(
                    in_features=self.hidden_channels,
                    out_features=self.hidden_channels,
                    kernel_size=(3, 3),
                    mask_type="B",
                    rngs=rngs,
                )
            )

        # Residual blocks
        self.residual_blocks = nnx.List([])
        for _ in range(self.num_residual_blocks):
            self.residual_blocks.append(
                PixelCNNResidualBlock(
                    channels=self.hidden_channels,
                    kernel_size=(3, 3),
                    mask_type="B",  # PixelCNN uses mask type B for residual blocks
                    rngs=rngs,
                )
            )

        # Output layers - one for each channel to predict pixel values
        self.output_convs = nnx.List([])
        for _ in range(channels):
            self.output_convs.append(
                nnx.Conv(
                    in_features=self.hidden_channels,
                    out_features=vocab_size,  # 256 possible values per channel
                    kernel_size=(1, 1),
                    rngs=rngs,
                )
            )

    def __call__(
        self, x: jax.Array, *, rngs: nnx.Rngs | None = None, training: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """Forward pass through PixelCNN.

        Args:
            x: Input images [batch, height, width, channels] with integer values [0, 255]
            rngs: Random number generators
            training: Whether in training mode
            **kwargs: Additional keyword arguments

        Returns:
            dictionary containing logits for each pixel and channel
        """
        # Convert to float and normalize to [0, 1]
        x_float = x / 255.0

        # First masked convolution
        h = self.first_conv(x_float)
        h = nnx.relu(h)

        # Hidden layers
        for conv in self.hidden_convs:
            h = conv(h)
            h = nnx.relu(h)

        # Residual blocks
        for block in self.residual_blocks:
            h = block(h)

        # Output predictions for each channel
        channel_logits = []
        for output_conv in self.output_convs:
            channel_logit = output_conv(h)
            channel_logits.append(channel_logit)

        # Stack channel predictions
        # Output shape: [batch, height, width, channels, vocab_size]
        logits = jnp.stack(channel_logits, axis=-2)

        # Reshape for compatibility with autoregressive base class
        # [batch, height * width * channels, vocab_size]
        batch_size = logits.shape[0]
        logits_flat = logits.reshape(batch_size, -1, self.vocab_size)

        return {
            "logits": logits_flat,
            "logits_spatial": logits,  # Keep spatial format for convenience
        }

    def generate(
        self,
        n_samples: int = 1,
        *,
        rngs: nnx.Rngs | None = None,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> jax.Array:
        """Generate images pixel by pixel.

        Args:
            n_samples: Number of images to generate
            rngs: Random number generators
            temperature: Sampling temperature
            **kwargs: Additional keyword arguments

        Returns:
            Generated images [n_samples, height, width, channels]
        """
        if rngs is None:
            rngs = self._rngs

        height, width, channels = self.image_shape

        # Initialize images with zeros
        images = jnp.zeros((n_samples, height, width, channels))

        # Get sampling key
        sample_key = self._get_rng_key(rngs, "sample", 0)

        # Generate pixel by pixel in raster scan order
        for h in range(height):
            for w in range(width):
                for c in range(channels):
                    # Get predictions for current state
                    outputs = self(images, rngs=rngs, training=False, **kwargs)
                    logits_spatial = outputs["logits_spatial"]

                    # Extract logits for current pixel and channel
                    current_logits = logits_spatial[:, h, w, c, :]  # [n_samples, vocab_size]

                    # Apply temperature
                    if temperature != 1.0:
                        current_logits = current_logits / temperature

                    # Sample pixel value
                    sample_key, subkey = jax.random.split(sample_key)
                    pixel_values = jax.random.categorical(subkey, current_logits, axis=-1)

                    # Update images
                    images = images.at[:, h, w, c].set(pixel_values)

        return images

    def loss_fn(
        self,
        batch: Any,
        model_outputs: dict[str, Any],
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute loss for PixelCNN.

        Args:
            batch: Batch of images
            model_outputs: Model outputs
            rngs: Random number generators
            **kwargs: Additional keyword arguments

        Returns:
            dictionary containing loss and metrics
        """
        # Extract images from batch
        if isinstance(batch, dict):
            images = batch.get("x", batch.get("images", batch))
        else:
            images = batch

        # Get logits in spatial format
        logits_spatial = model_outputs["logits_spatial"]

        # Vectorized cross-entropy loss over all pixels
        # logits_spatial: (batch, H, W, C, num_values)
        # images: (batch, H, W, C) with integer pixel values

        # Compute log softmax over the value dimension
        log_probs_all = nnx.log_softmax(logits_spatial)  # (batch, H, W, C, num_values)

        # Gather log-prob of the target pixel values
        targets = images[..., None]  # (batch, H, W, C, 1)
        per_pixel_log_prob = jnp.take_along_axis(log_probs_all, targets, axis=-1)  # (batch,H,W,C,1)
        per_pixel_log_prob = per_pixel_log_prob.squeeze(-1)  # (batch, H, W, C)

        # Cross-entropy loss is negative log probability
        loss = -jnp.mean(per_pixel_log_prob)

        # Compute accuracy
        predictions = jnp.argmax(logits_spatial, axis=-1)
        accuracy = jnp.mean(predictions == images)

        # Compute bits per dimension (common metric for image models)
        bits_per_dim = loss / jnp.log(2.0)

        return {
            "loss": loss,
            "nll_loss": loss,
            "accuracy": accuracy,
            "bits_per_dim": bits_per_dim,
        }

    def log_prob(self, x: jax.Array, *, rngs: nnx.Rngs | None = None, **kwargs: Any) -> jax.Array:
        """Compute log probability of images.

        Args:
            x: Input images [batch, height, width, channels]
            rngs: Random number generators
            **kwargs: Additional keyword arguments

        Returns:
            Log probabilities [batch]
        """
        # Get model outputs
        outputs = self(x, rngs=rngs, training=False, **kwargs)
        logits_spatial = outputs["logits_spatial"]

        # Vectorized log probability over all pixels
        # logits_spatial: (batch, H, W, C, num_values)
        # x: (batch, H, W, C) with integer pixel values
        log_probs_all = nnx.log_softmax(logits_spatial)  # (batch, H, W, C, num_values)

        targets = x[..., None]  # (batch, H, W, C, 1)
        per_pixel_log_prob = jnp.take_along_axis(log_probs_all, targets, axis=-1).squeeze(-1)

        # Sum over spatial and channel dims to get per-image log probability
        image_log_probs = jnp.sum(per_pixel_log_prob, axis=(1, 2, 3))  # (batch,)

        return image_log_probs

    def inpaint(
        self,
        conditioning: jax.Array,
        mask: jax.Array,
        n_samples: int = 1,
        *,
        rngs: nnx.Rngs | None = None,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> jax.Array:
        """Generate images with spatial conditioning (inpainting).

        Args:
            conditioning: Conditioning image [height, width, channels]
            mask: Binary mask [height, width] where 1 = keep, 0 = generate
            n_samples: Number of samples to generate
            rngs: Random number generators
            temperature: Sampling temperature
            **kwargs: Additional keyword arguments

        Returns:
            Generated images [n_samples, height, width, channels]
        """
        if rngs is None:
            rngs = self._rngs

        height, width, channels = self.image_shape

        # Expand conditioning for multiple samples
        images = jnp.tile(conditioning[None], (n_samples, 1, 1, 1))

        # Get sampling key
        sample_key = self._get_rng_key(rngs, "sample", 0)

        # Generate only unmasked pixels
        for h in range(height):
            for w in range(width):
                # Skip if this pixel should be kept from conditioning
                if mask[h, w] == 1:
                    continue

                for c in range(channels):
                    # Get predictions for current state
                    outputs = self(images, rngs=rngs, training=False, **kwargs)
                    logits_spatial = outputs["logits_spatial"]

                    # Extract logits for current pixel and channel
                    current_logits = logits_spatial[:, h, w, c, :]

                    # Apply temperature
                    if temperature != 1.0:
                        current_logits = current_logits / temperature

                    # Sample pixel value
                    sample_key, subkey = jax.random.split(sample_key)
                    pixel_values = jax.random.categorical(subkey, current_logits, axis=-1)

                    # Update images
                    images = images.at[:, h, w, c].set(pixel_values)

        return images
