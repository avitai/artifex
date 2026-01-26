"""Base classes and protocols for image modality.

This module defines the core interfaces and base classes for image generation,
following the established modality patterns in the framework.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel


class ImageRepresentation(Enum):
    """Image representation formats supported by the modality."""

    RGB = "rgb"
    GRAYSCALE = "grayscale"
    RGBA = "rgba"


@dataclass
class ImageModalityConfig:
    """Configuration for image modality processing.

    Args:
        representation: Image representation format to use
        height: Image height in pixels
        width: Image width in pixels (defaults to height for square images)
        channels: Number of channels (auto-determined by representation if None)
        normalize: Whether to normalize pixel values to [0, 1] or [-1, 1]
        augmentation: Whether to enable data augmentation
        resize_method: Method for resizing ('bilinear', 'nearest')
    """

    representation: ImageRepresentation = ImageRepresentation.RGB
    height: int = 64
    width: int | None = None
    channels: int | None = None
    normalize: bool = True
    augmentation: bool = False
    resize_method: str = "bilinear"

    def __post_init__(self):
        """Set defaults and validate configuration."""
        # Set width to height for square images if not specified
        if self.width is None:
            self.width = self.height

        # Auto-determine channels based on representation
        if self.channels is None:
            if self.representation == ImageRepresentation.RGB:
                self.channels = 3
            elif self.representation == ImageRepresentation.RGBA:
                self.channels = 4
            else:  # GRAYSCALE
                self.channels = 1


class ImageGenerationProtocol(Protocol):
    """Protocol for image generation models."""

    def generate_images(
        self,
        n_samples: int = 1,
        height: int | None = None,
        width: int | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> jax.Array:
        """Generate image samples.

        Args:
            n_samples: Number of image samples to generate
            height: Image height override (uses config default if None)
            width: Image width override (uses config default if None)
            rngs: Random number generators
            **kwargs: Additional generation parameters

        Returns:
            Generated image array of shape (n_samples, height, width, channels)
        """
        ...

    def compute_likelihood(self, images: jax.Array) -> jax.Array:
        """Compute likelihood of image samples.

        Args:
            images: Image data to evaluate

        Returns:
            Log-likelihood value
        """
        ...


class ImageModality(GenerativeModel):
    """Base image modality class providing unified interface for image generation.

    This class provides a unified interface for different image generation approaches
    while supporting multiple resolution and channel configurations.
    """

    name = "image"

    def __init__(
        self,
        config: ImageModalityConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize image modality.

        Args:
            config: Image modality configuration
            rngs: Random number generators
        """
        super().__init__(rngs=rngs)
        self.config = config or ImageModalityConfig()

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Image shape (height, width, channels)."""
        return (self.config.height, self.config.width, self.config.channels)  # type: ignore[return-value]

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """Output shape for generated images."""
        return self.image_shape

    def generate(
        self,
        n_samples: int = 1,
        height: int | None = None,
        width: int | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> jax.Array:
        """Generate image samples using the configured model.

        Args:
            n_samples: Number of image samples to generate
            height: Height override (uses config default if None)
            width: Width override (uses config default if None)
            rngs: Random number generators
            **kwargs: Additional generation parameters

        Returns:
            Generated image array of shape (n_samples, height, width, channels)
        """
        # Default implementation - subclasses should override
        actual_height = height if height is not None else self.config.height
        actual_width = width if width is not None else self.config.width
        actual_channels = self.config.channels

        if actual_height is None or actual_width is None or actual_channels is None:
            raise ValueError("Image dimensions (height, width, channels) cannot be None")

        # Generate simple synthetic images for base implementation
        if rngs is None:
            raise ValueError("rngs must be provided for sample generation")

        key = rngs.sample()

        if self.config.representation == ImageRepresentation.GRAYSCALE:
            # Generate gradient patterns
            y, x = jnp.meshgrid(
                jnp.linspace(0, 1, actual_height),
                jnp.linspace(0, 1, actual_width),
                indexing="ij",  # type: ignore[arg-type]
            )
            gradient = (x + y) / 2
            base_image = gradient[..., None]  # Add channel dimension
            noise_shape = (n_samples, actual_height, actual_width, actual_channels)
            noise = 0.1 * jax.random.normal(key, noise_shape)  # type: ignore[arg-type]
            generated = base_image[None, ...] + noise
            # Clip to ensure values stay in valid range
            return jnp.clip(generated, 0.0, 1.0)
        else:
            # Generate random colored images for RGB/RGBA
            shape = (n_samples, actual_height, actual_width, actual_channels)
            base_images = jax.random.uniform(key, shape)  # type: ignore[arg-type]
            if self.config.normalize:
                return base_images  # Already in [0, 1]
            else:
                return base_images * 2 - 1  # Scale to [-1, 1]

    def loss_fn(
        self,
        batch: dict[str, jax.Array],
        model_outputs: dict[str, Any],
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Compute loss for image generation training.

        Args:
            batch: Training batch containing 'images' key
            model_outputs: Model predictions
            rngs: Random number generators
            **kwargs: Additional loss parameters

        Returns:
            Dictionary containing loss and metrics
        """
        # Default MSE loss - subclasses should override for specific losses
        target_images = batch["images"]
        predicted_images = model_outputs.get("images", model_outputs.get("predictions"))

        if predicted_images is None:
            raise ValueError("Model outputs must contain 'images' or 'predictions' key")

        mse_loss = jnp.mean((target_images - predicted_images) ** 2)

        # Return dictionary with primary loss and any metrics
        return {"loss": mse_loss, "mse": mse_loss}

    def process(self, data: jax.Array, **kwargs) -> jax.Array:
        """Process image data for multi-modal fusion.

        Args:
            data: Image data with shape (height, width, channels) or (batch, height,
                width, channels)
            **kwargs: Additional processing arguments

        Returns:
            Processed image features as flattened array
        """
        # Ensure we have a batch dimension
        if data.ndim == 3:
            data = data[jnp.newaxis, ...]

        # Normalize to expected range
        if self.config.normalize:
            # Ensure data is in [0, 1] range
            data = jnp.clip(data, 0.0, 1.0)
        else:
            # Data might be in [-1, 1], convert to [0, 1]
            data = (data + 1.0) / 2.0
            data = jnp.clip(data, 0.0, 1.0)

        # For multi-modal processing, flatten the spatial dimensions
        # Keep batch dimension, flatten spatial and channel dimensions
        batch_size = data.shape[0]
        flattened = data.reshape(batch_size, -1)

        # If batch size is 1, return without batch dimension for compatibility
        if batch_size == 1:
            return flattened[0]

        return flattened

    def get_adapter(self, model_type: str | Any) -> Any:
        """Get an adapter for the specified model type.

        Args:
            model_type: The model type or class to adapt

        Returns:
            An ImageModalityAdapter instance
        """
        # Import here to avoid circular dependencies
        from artifex.generative_models.modalities.image.adapters import ImageModalityAdapter

        return ImageModalityAdapter()

    def get_extensions(self, config: Any, *, rngs: nnx.Rngs | None = None) -> dict[str, Any]:
        """Get image-specific extensions.

        Args:
            config: Extension configuration
            rngs: Random number generators

        Returns:
            Dictionary of extensions (empty for now as image doesn't need special extensions)
        """
        return {}


def create_image_modality(
    resolution: int = 64,
    channels: int | None = None,
    representation: ImageRepresentation | str = ImageRepresentation.RGB,
    *,
    rngs: nnx.Rngs,
    **kwargs,
) -> ImageModality:
    """Factory function to create image modality with common configurations.

    Args:
        resolution: Image resolution (height and width)
        channels: Number of channels (auto-determined if None)
        representation: Image representation format
        rngs: Random number generators
        **kwargs: Additional configuration parameters

    Returns:
        Configured ImageModality instance
    """
    if isinstance(representation, str):
        representation = ImageRepresentation(representation)

    config = ImageModalityConfig(
        representation=representation,
        height=resolution,
        width=resolution,
        channels=channels,
        **kwargs,
    )

    return ImageModality(config=config, rngs=rngs)
