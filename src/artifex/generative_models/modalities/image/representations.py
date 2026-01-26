"""Image representation processing for the image modality.

This module provides utilities for processing images in different representations,
including multi-scale processing and augmentation pipelines.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .base import ImageModalityConfig


class ImageProcessor(nnx.Module):
    """Base image processor for format conversions and normalization."""

    def __init__(self, config: ImageModalityConfig, *, rngs: nnx.Rngs):
        """Initialize image processor.

        Args:
            config: Image modality configuration
            rngs: Random number generators
        """
        self.config = config
        self.rngs = rngs

    def normalize(self, images: jax.Array) -> jax.Array:
        """Normalize images to [0, 1] or [-1, 1] range.

        Args:
            images: Input images, assumed to be in [0, 255] range

        Returns:
            Normalized images
        """
        # Normalize to [0, 1]
        normalized = images / 255.0

        if not self.config.normalize:
            # Scale to [-1, 1] for some models (e.g., GANs)
            normalized = normalized * 2.0 - 1.0

        return normalized

    def denormalize(self, images: jax.Array) -> jax.Array:
        """Denormalize images back to [0, 255] range.

        Args:
            images: Normalized images

        Returns:
            Images in [0, 255] range, clipped and cast to uint8
        """
        if not self.config.normalize:
            # Scale from [-1, 1] to [0, 1]
            images = (images + 1.0) / 2.0

        # Scale to [0, 255] and clip
        images = jnp.clip(images * 255.0, 0, 255)
        return images.astype(jnp.uint8)

    def rgb_to_grayscale(self, images: jax.Array) -> jax.Array:
        """Convert RGB images to grayscale.

        Args:
            images: RGB images of shape (..., 3)

        Returns:
            Grayscale images of shape (..., 1)
        """
        # Standard luminance weights
        weights = jnp.array([0.299, 0.587, 0.114])
        gray = jnp.sum(images * weights, axis=-1, keepdims=True)
        return gray

    def grayscale_to_rgb(self, images: jax.Array) -> jax.Array:
        """Convert grayscale images to RGB by repeating channels.

        Args:
            images: Grayscale images of shape (..., 1)

        Returns:
            RGB images of shape (..., 3)
        """
        return jnp.repeat(images, 3, axis=-1)


class AugmentationProcessor(nnx.Module):
    """Image augmentation processor for data augmentation."""

    def __init__(
        self,
        config: ImageModalityConfig,
        rotation_range: float = 15.0,
        zoom_range: float = 0.1,
        flip_horizontal: bool = True,
        brightness_range: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize augmentation processor.

        Args:
            config: Image modality configuration
            rotation_range: Maximum rotation angle in degrees
            zoom_range: Maximum zoom factor
            flip_horizontal: Whether to enable horizontal flipping
            brightness_range: Maximum brightness adjustment
            rngs: Random number generators
        """
        self.config = config
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.flip_horizontal = flip_horizontal
        self.brightness_range = brightness_range
        self.rngs = rngs

    def random_flip(self, images: jax.Array, key: jax.Array) -> jax.Array:
        """Randomly flip images horizontally.

        Args:
            images: Input images
            key: Random key

        Returns:
            Potentially flipped images
        """
        if not self.flip_horizontal:
            return images

        # Random boolean for each image in batch
        flip_mask = jax.random.bernoulli(key, 0.5, shape=(images.shape[0],))

        # Apply flip where mask is True
        def flip_image(img, should_flip):
            return jnp.where(should_flip, jnp.fliplr(img), img)

        return jax.vmap(flip_image)(images, flip_mask)

    def random_brightness(self, images: jax.Array, key: jax.Array) -> jax.Array:
        """Randomly adjust image brightness.

        Args:
            images: Input images
            key: Random key

        Returns:
            Brightness-adjusted images
        """
        if self.brightness_range <= 0:
            return images

        # Random brightness factor for each image
        brightness_factors = jax.random.uniform(
            key,
            shape=(images.shape[0], 1, 1, 1),
            minval=1.0 - self.brightness_range,
            maxval=1.0 + self.brightness_range,
        )

        adjusted = images * brightness_factors
        return jnp.clip(adjusted, 0.0, 1.0)

    def augment_batch(self, images: jax.Array) -> jax.Array:
        """Apply augmentation to a batch of images.

        Args:
            images: Batch of images

        Returns:
            Augmented images
        """
        if not self.config.augmentation:
            return images

        key = self.rngs.sample()
        keys = jax.random.split(key, 3)

        # Apply augmentations sequentially
        images = self.random_flip(images, keys[0])
        images = self.random_brightness(images, keys[1])

        return images


class MultiScaleProcessor(nnx.Module):
    """Multi-scale image processor for hierarchical representations."""

    def __init__(
        self,
        config: ImageModalityConfig,
        scales: list[int] | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize multi-scale processor.

        Args:
            config: Image modality configuration
            scales: List of scales (resolutions) to process
            rngs: Random number generators
        """
        self.config = config
        self.scales = scales or [32, 64, 128, 256]
        self.rngs = rngs

    def resize_image(self, image: jax.Array, new_size: tuple[int, int]) -> jax.Array:
        """Resize image to new size using bilinear interpolation.

        Args:
            image: Input image of shape (height, width, channels)
            new_size: Target size (height, width)

        Returns:
            Resized image
        """
        # Simple implementation using JAX's image resize
        # Note: This is a basic implementation. For production use,
        # consider using more sophisticated resizing methods

        height, width = new_size
        current_height, current_width = image.shape[:2]

        # Create coordinate grids for the new size
        y_coords = jnp.linspace(0, current_height - 1, height)
        x_coords = jnp.linspace(0, current_width - 1, width)

        # Create meshgrid for interpolation coordinates
        y_grid, x_grid = jnp.meshgrid(y_coords, x_coords, indexing="ij")

        # Stack coordinates for map_coordinates
        coords = jnp.stack([y_grid, x_grid], axis=0)

        # Interpolate each channel separately
        if image.ndim == 3:
            resized_channels = []
            for c in range(image.shape[2]):
                resized_channel = jax.scipy.ndimage.map_coordinates(
                    image[:, :, c], coords, order=1, mode="nearest"
                )
                resized_channels.append(resized_channel)
            return jnp.stack(resized_channels, axis=2)
        else:
            return jax.scipy.ndimage.map_coordinates(image, coords, order=1, mode="nearest")

    def create_multi_scale_pyramid(self, images: jax.Array) -> dict[int, jax.Array]:
        """Create multi-scale pyramid of images.

        Args:
            images: Batch of images

        Returns:
            Dictionary mapping scales to resized image batches
        """
        pyramid = {}

        for scale in self.scales:
            if scale == self.config.height and scale == self.config.width:
                # Original resolution
                pyramid[scale] = images
            else:
                # Resize images to this scale
                resized_batch = []
                for img in images:
                    resized_img = self.resize_image(img, (scale, scale))
                    resized_batch.append(resized_img)
                pyramid[scale] = jnp.stack(resized_batch)

        return pyramid

    def combine_scales(self, pyramid: dict[int, jax.Array]) -> jax.Array:
        """Combine multi-scale representations into single tensor.

        Args:
            pyramid: Multi-scale pyramid

        Returns:
            Combined tensor with all scales
        """
        # Sort scales and concatenate along a new axis
        sorted_scales = sorted(pyramid.keys())
        scale_tensors = [pyramid[scale] for scale in sorted_scales]

        # Pad smaller scales to match the largest scale
        max_scale = max(sorted_scales)
        padded_tensors = []

        for scale, tensor in zip(sorted_scales, scale_tensors):
            if scale < max_scale:
                # Calculate padding needed
                pad_size = (max_scale - scale) // 2
                padding = [(0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)]
                tensor = jnp.pad(tensor, padding, mode="constant")
            padded_tensors.append(tensor)

        return jnp.stack(padded_tensors, axis=1)  # New axis for scales
