"""Image dataset handling for the image modality.

This module provides utilities for loading and processing image datasets,
including synthetic data generation for testing and development.
"""

from typing import Iterator

import jax
import jax.numpy as jnp
from flax import nnx

from .base import ImageModalityConfig, ImageRepresentation


class ImageDataset(nnx.Module):
    """Base class for image datasets."""

    def __init__(
        self,
        config: ImageModalityConfig,
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize image dataset.

        Args:
            config: Image modality configuration
            split: Dataset split ('train', 'val', 'test')
            rngs: Random number generators
        """
        self.config = config
        self.split = split
        self.rngs = rngs

    def __len__(self) -> int:
        """Return dataset size."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over dataset samples."""
        raise NotImplementedError

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Get a batch of samples.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Batch dictionary with 'images' and potentially 'labels'
        """
        raise NotImplementedError


class SyntheticImageDataset(ImageDataset):
    """Synthetic image dataset for testing and development."""

    def __init__(
        self,
        config: ImageModalityConfig,
        dataset_size: int = 1000,
        pattern_type: str = "random",
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize synthetic image dataset.

        Args:
            config: Image modality configuration
            dataset_size: Number of synthetic samples
            pattern_type: Type of pattern to generate
                ('random', 'gradient', 'checkerboard', 'circles')
            split: Dataset split
            rngs: Random number generators
        """
        super().__init__(config, split, rngs=rngs)
        self.dataset_size = dataset_size
        self.pattern_type = pattern_type

        # Ensure height, width, and channels are not None
        assert self.config.height is not None, "Height cannot be None"
        assert self.config.width is not None, "Width cannot be None"
        assert self.config.channels is not None, "Channels cannot be None"

    def __len__(self) -> int:
        """Return dataset size."""
        return self.dataset_size

    def _generate_random_pattern(self, n_samples: int) -> jax.Array:
        """Generate random noise patterns."""
        key = self.rngs.sample()
        # Use assertion to ensure these are not None
        assert self.config.height is not None
        assert self.config.width is not None
        assert self.config.channels is not None

        shape = (n_samples, self.config.height, self.config.width, self.config.channels)
        return jax.random.uniform(key, shape, minval=0.0, maxval=1.0)

    def _generate_gradient_pattern(self, n_samples: int) -> jax.Array:
        """Generate gradient patterns."""
        # Use assertion to ensure these are not None
        assert self.config.height is not None
        assert self.config.width is not None
        assert self.config.channels is not None

        # Create coordinate grids
        y, x = jnp.meshgrid(
            jnp.linspace(0, 1, self.config.height),
            jnp.linspace(0, 1, self.config.width),
            indexing="ij",
        )

        patterns = []
        key = self.rngs.sample()
        keys = jax.random.split(key, n_samples)

        for i in range(n_samples):
            # Random gradient direction
            angle = jax.random.uniform(keys[i], minval=0, maxval=2 * jnp.pi)
            gradient = jnp.cos(angle) * x + jnp.sin(angle) * y

            # Expand to channels
            if self.config.channels == 1:
                pattern = gradient[..., None]
            elif self.config.channels == 3:
                # Create RGB gradient with different color combinations
                r = gradient
                g = jnp.sin(gradient * jnp.pi)
                b = jnp.cos(gradient * jnp.pi)
                pattern = jnp.stack([r, g, b], axis=-1)
            else:  # RGBA
                r = gradient
                g = jnp.sin(gradient * jnp.pi)
                b = jnp.cos(gradient * jnp.pi)
                a = jnp.ones_like(gradient)
                pattern = jnp.stack([r, g, b, a], axis=-1)

            patterns.append(pattern)

        return jnp.stack(patterns)

    def _generate_checkerboard_pattern(self, n_samples: int) -> jax.Array:
        """Generate checkerboard patterns."""
        # Use assertion to ensure these are not None
        assert self.config.height is not None
        assert self.config.width is not None
        assert self.config.channels is not None

        patterns = []
        key = self.rngs.sample()
        keys = jax.random.split(key, n_samples)

        for i in range(n_samples):
            # Random checkerboard size
            size = jax.random.randint(keys[i], minval=4, maxval=16, shape=())

            # Create checkerboard
            y_indices = jnp.arange(self.config.height) // size
            x_indices = jnp.arange(self.config.width) // size
            y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing="ij")
            checkerboard = (y_grid + x_grid) % 2

            # Expand to channels
            if self.config.channels == 1:
                pattern = checkerboard[..., None].astype(jnp.float32)
            else:
                pattern = jnp.repeat(checkerboard[..., None], self.config.channels, axis=-1).astype(
                    jnp.float32
                )

            patterns.append(pattern)

        return jnp.stack(patterns)

    def _generate_circles_pattern(self, n_samples: int) -> jax.Array:
        """Generate circular patterns."""
        # Use assertion to ensure these are not None
        assert self.config.height is not None
        assert self.config.width is not None
        assert self.config.channels is not None

        patterns = []
        key = self.rngs.sample()
        keys = jax.random.split(key, n_samples)

        # Create coordinate grids
        y, x = jnp.meshgrid(
            jnp.arange(self.config.height) - self.config.height // 2,
            jnp.arange(self.config.width) - self.config.width // 2,
            indexing="ij",
        )

        for i in range(n_samples):
            # Random circle parameters
            center_y = jax.random.uniform(
                keys[i], minval=-self.config.height // 4, maxval=self.config.height // 4
            )
            center_x = jax.random.uniform(
                keys[i], minval=-self.config.width // 4, maxval=self.config.width // 4
            )
            radius = jax.random.uniform(
                keys[i], minval=10, maxval=min(self.config.height, self.config.width) // 3
            )

            # Create circle
            distance = jnp.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
            circle = (distance <= radius).astype(jnp.float32)

            # Add some noise for variation
            noise_key = jax.random.fold_in(keys[i], 1)
            noise = 0.1 * jax.random.normal(
                noise_key, shape=(self.config.height, self.config.width)
            )
            circle = jnp.clip(circle + noise, 0.0, 1.0)

            # Expand to channels
            if self.config.channels == 1:
                pattern = circle[..., None]
            else:
                pattern = jnp.repeat(circle[..., None], self.config.channels, axis=-1)

            patterns.append(pattern)

        return jnp.stack(patterns)

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Generate a batch of synthetic images.

        Args:
            batch_size: Number of images to generate

        Returns:
            Batch dictionary with 'images' key
        """
        if self.pattern_type == "random":
            images = self._generate_random_pattern(batch_size)
        elif self.pattern_type == "gradient":
            images = self._generate_gradient_pattern(batch_size)
        elif self.pattern_type == "checkerboard":
            images = self._generate_checkerboard_pattern(batch_size)
        elif self.pattern_type == "circles":
            images = self._generate_circles_pattern(batch_size)
        else:
            raise ValueError(f"Unknown pattern type: {self.pattern_type}")

        return {"images": images}

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over synthetic samples."""
        for _ in range(self.dataset_size):
            batch = self.get_batch(1)
            yield {k: v[0] for k, v in batch.items()}  # Remove batch dimension


class MNISTLikeDataset(ImageDataset):
    """MNIST-like synthetic dataset for digit-like patterns."""

    def __init__(
        self,
        config: ImageModalityConfig,
        dataset_size: int = 1000,
        num_classes: int = 10,
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize MNIST-like dataset.

        Args:
            config: Image modality configuration (should be grayscale, 28x28)
            dataset_size: Number of synthetic samples
            num_classes: Number of classes to generate
            split: Dataset split
            rngs: Random number generators
        """
        super().__init__(config, split, rngs=rngs)
        self.dataset_size = dataset_size
        self.num_classes = num_classes

        # Ensure height, width, and channels are not None
        assert self.config.height is not None, "Height cannot be None"
        assert self.config.width is not None, "Width cannot be None"
        assert self.config.channels is not None, "Channels cannot be None"

        # Ensure appropriate config for MNIST-like data
        if config.representation != ImageRepresentation.GRAYSCALE:
            print(
                f"Warning: MNIST-like dataset expects grayscale images, got {config.representation}"
            )

    def __len__(self) -> int:
        """Return dataset size."""
        return self.dataset_size

    def _generate_digit_like_pattern(self, digit_class: int) -> jax.Array:
        """Generate a digit-like pattern for the given class."""
        key = self.rngs.sample()

        # Use assertion to ensure these are not None
        assert self.config.height is not None
        assert self.config.width is not None
        assert self.config.channels is not None

        # Simple patterns for each "digit"
        pattern = jnp.zeros((self.config.height, self.config.width))

        if digit_class == 0:  # Circle
            center_y, center_x = self.config.height // 2, self.config.width // 2
            y, x = jnp.meshgrid(
                jnp.arange(self.config.height) - center_y,
                jnp.arange(self.config.width) - center_x,
                indexing="ij",
            )
            radius = min(self.config.height, self.config.width) // 3
            distance = jnp.sqrt(y**2 + x**2)
            pattern = ((distance <= radius) & (distance >= radius - 3)).astype(jnp.float32)

        elif digit_class == 1:  # Vertical line
            center_x = self.config.width // 2
            pattern = pattern.at[:, center_x - 1 : center_x + 2].set(1.0)

        elif digit_class == 2:  # Horizontal line
            center_y = self.config.height // 2
            pattern = pattern.at[center_y - 1 : center_y + 2, :].set(1.0)

        # Add more patterns for other classes as needed...

        # Add some noise for realism
        noise = 0.1 * jax.random.normal(key, pattern.shape)
        pattern = jnp.clip(pattern + noise, 0.0, 1.0)

        # Add channel dimension
        if self.config.channels == 1:
            return pattern[..., None]
        else:
            return jnp.repeat(pattern[..., None], self.config.channels, axis=-1)

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Generate a batch of digit-like images with labels.

        Args:
            batch_size: Number of images to generate

        Returns:
            Batch dictionary with 'images' and 'labels' keys
        """
        images = []
        labels = []

        key = self.rngs.sample()
        class_keys = jax.random.split(key, batch_size)

        for i in range(batch_size):
            # Random class for each image
            digit_class = jax.random.randint(
                class_keys[i], minval=0, maxval=self.num_classes, shape=()
            )
            image = self._generate_digit_like_pattern(int(digit_class))

            images.append(image)
            labels.append(digit_class)

        return {
            "images": jnp.stack(images),
            "labels": jnp.array(labels),
        }

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over synthetic samples."""
        for _ in range(self.dataset_size):
            batch = self.get_batch(1)
            yield {
                "images": batch["images"][0],  # Remove batch dimension
                "labels": batch["labels"][0],
            }


def create_image_dataset(
    dataset_type: str = "synthetic",
    config: ImageModalityConfig | None = None,
    *,
    rngs: nnx.Rngs,
    **kwargs,
) -> ImageDataset:
    """Factory function to create image datasets.

    Args:
        dataset_type: Type of dataset ('synthetic', 'mnist_like')
        config: Image modality configuration
        rngs: Random number generators
        **kwargs: Additional dataset parameters

    Returns:
        Created dataset instance
    """
    if config is None:
        config = ImageModalityConfig()

    if dataset_type == "synthetic":
        return SyntheticImageDataset(config, rngs=rngs, **kwargs)
    elif dataset_type == "mnist_like":
        return MNISTLikeDataset(config, rngs=rngs, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
