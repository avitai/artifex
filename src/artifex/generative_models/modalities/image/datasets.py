"""Image datasets backed by datarax MemorySource.

Provides pure data generation functions and factory functions that wrap
generated data in datarax MemorySource for pipeline integration.
"""

from typing import Any

import jax
import jax.numpy as jnp
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx

from .base import ImageModalityConfig


# ---------------------------------------------------------------------------
# Pattern generation helpers (private)
# ---------------------------------------------------------------------------


def _gradient_pattern(key: jax.Array, h: int, w: int, c: int) -> jax.Array:
    """Generate a gradient pattern."""
    y, x = jnp.meshgrid(jnp.linspace(0, 1, h), jnp.linspace(0, 1, w), indexing="ij")
    angle = jax.random.uniform(key, minval=0, maxval=2 * jnp.pi)
    gradient = jnp.cos(angle) * x + jnp.sin(angle) * y

    if c == 1:
        return gradient[..., None]
    elif c == 3:
        return jnp.stack(
            [gradient, jnp.sin(gradient * jnp.pi), jnp.cos(gradient * jnp.pi)],
            axis=-1,
        )
    else:
        rgb = jnp.stack(
            [gradient, jnp.sin(gradient * jnp.pi), jnp.cos(gradient * jnp.pi)],
            axis=-1,
        )
        alpha = jnp.ones((h, w, 1))
        return jnp.concatenate([rgb, alpha], axis=-1)


def _checkerboard_pattern(key: jax.Array, h: int, w: int, c: int) -> jax.Array:
    """Generate a checkerboard pattern."""
    size = jax.random.randint(key, minval=4, maxval=16, shape=())
    y_indices = jnp.arange(h) // size
    x_indices = jnp.arange(w) // size
    y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing="ij")
    checkerboard = ((y_grid + x_grid) % 2).astype(jnp.float32)
    return jnp.repeat(checkerboard[..., None], c, axis=-1)


def _circles_pattern(key: jax.Array, h: int, w: int, c: int) -> jax.Array:
    """Generate a circles pattern."""
    y, x = jnp.meshgrid(jnp.arange(h) - h // 2, jnp.arange(w) - w // 2, indexing="ij")
    center_y = jax.random.uniform(key, minval=-h // 4, maxval=h // 4)
    center_x = jax.random.uniform(key, minval=-w // 4, maxval=w // 4)
    radius = jax.random.uniform(key, minval=3, maxval=min(h, w) // 3)
    distance = jnp.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    circle = (distance <= radius).astype(jnp.float32)

    noise_key = jax.random.fold_in(key, 1)
    noise = 0.1 * jax.random.normal(noise_key, shape=(h, w))
    circle = jnp.clip(circle + noise, 0.0, 1.0)
    return jnp.repeat(circle[..., None], c, axis=-1)


def _generate_pattern(
    key: jax.Array,
    h: int,
    w: int,
    c: int,
    pattern_type: str,
) -> jax.Array:
    """Generate a pattern image.

    Args:
        key: RNG key for generation.
        h: Image height.
        w: Image width.
        c: Number of channels.
        pattern_type: Pattern type.

    Returns:
        Image array of shape (h, w, c).

    Raises:
        ValueError: If pattern_type is unknown.
    """
    if pattern_type == "random":
        return jax.random.uniform(key, (h, w, c), minval=0.0, maxval=1.0)
    elif pattern_type == "gradient":
        return _gradient_pattern(key, h, w, c)
    elif pattern_type == "checkerboard":
        return _checkerboard_pattern(key, h, w, c)
    elif pattern_type == "circles":
        return _circles_pattern(key, h, w, c)
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")


def _generate_digit_pattern(
    key: jax.Array,
    digit_class: int,
    h: int,
    w: int,
    c: int,
) -> jax.Array:
    """Generate a digit-like pattern for the given class.

    Args:
        key: RNG key for noise generation.
        digit_class: Which digit class to generate.
        h: Image height.
        w: Image width.
        c: Number of channels.

    Returns:
        Image array of shape (h, w, c).
    """
    pattern = jnp.zeros((h, w))

    if digit_class == 0:  # Circle
        center_y, center_x = h // 2, w // 2
        y, x = jnp.meshgrid(jnp.arange(h) - center_y, jnp.arange(w) - center_x, indexing="ij")
        radius = min(h, w) // 3
        distance = jnp.sqrt(y**2 + x**2)
        pattern = ((distance <= radius) & (distance >= radius - 3)).astype(jnp.float32)
    elif digit_class == 1:  # Vertical line
        center_x = w // 2
        pattern = pattern.at[:, center_x - 1 : center_x + 2].set(1.0)
    elif digit_class == 2:  # Horizontal line
        center_y = h // 2
        pattern = pattern.at[center_y - 1 : center_y + 2, :].set(1.0)

    noise = 0.1 * jax.random.normal(key, pattern.shape)
    pattern = jnp.clip(pattern + noise, 0.0, 1.0)

    if c == 1:
        return pattern[..., None]
    return jnp.repeat(pattern[..., None], c, axis=-1)


# ---------------------------------------------------------------------------
# Data generation (pure functions)
# ---------------------------------------------------------------------------


def generate_synthetic_images(
    num_samples: int,
    *,
    height: int = 64,
    width: int = 64,
    channels: int = 3,
    pattern_type: str = "random",
) -> dict[str, jnp.ndarray]:
    """Generate synthetic image data.

    Args:
        num_samples: Number of images to generate.
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of color channels.
        pattern_type: Pattern for generation
            ('random', 'gradient', 'checkerboard', 'circles').

    Returns:
        Dictionary with 'images' array of shape (num_samples, H, W, C).

    Raises:
        ValueError: If height, width, or channels is non-positive.
        ValueError: If pattern_type is unknown.
    """
    if height <= 0:
        raise ValueError(f"Height must be positive, got {height}")
    if width <= 0:
        raise ValueError(f"Width must be positive, got {width}")
    if channels <= 0:
        raise ValueError(f"Channels must be positive, got {channels}")

    images = []
    for i in range(num_samples):
        key = jax.random.key(i)
        image = _generate_pattern(key, height, width, channels, pattern_type)
        images.append(image)

    return {"images": jnp.stack(images)}


def generate_mnist_like_images(
    num_samples: int,
    *,
    height: int = 28,
    width: int = 28,
    channels: int = 1,
    num_classes: int = 10,
) -> dict[str, jnp.ndarray]:
    """Generate MNIST-like digit pattern data.

    Args:
        num_samples: Number of images to generate.
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of color channels.
        num_classes: Number of digit classes.

    Returns:
        Dictionary with 'images' and 'labels' arrays.

    Raises:
        ValueError: If height, width, or channels is non-positive.
    """
    if height <= 0:
        raise ValueError(f"Height must be positive, got {height}")
    if width <= 0:
        raise ValueError(f"Width must be positive, got {width}")
    if channels <= 0:
        raise ValueError(f"Channels must be positive, got {channels}")

    images = []
    labels = []
    for i in range(num_samples):
        key = jax.random.key(i)
        digit_class = i % num_classes
        image = _generate_digit_pattern(key, digit_class, height, width, channels)
        images.append(image)
        labels.append(digit_class)

    return {
        "images": jnp.stack(images),
        "labels": jnp.array(labels),
    }


# ---------------------------------------------------------------------------
# Factory functions — return MemorySource instances
# ---------------------------------------------------------------------------


def create_image_dataset(
    dataset_type: str = "synthetic",
    config: ImageModalityConfig | None = None,
    *,
    rngs: nnx.Rngs,
    shuffle: bool = False,
    **kwargs: Any,
) -> MemorySource:
    """Create an image dataset as a MemorySource.

    Args:
        dataset_type: Type of dataset ('synthetic', 'mnist_like').
        config: Optional modality configuration. If provided,
            height/width/channels are extracted from it.
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.
        **kwargs: Additional generation parameters
            (height, width, channels, dataset_size, pattern_type, num_classes).

    Returns:
        MemorySource backed by generated image data.

    Raises:
        ValueError: If dataset_type is unknown.
    """
    if config is not None:
        kwargs.setdefault("height", config.height)
        kwargs.setdefault("width", config.width or config.height)
        kwargs.setdefault("channels", config.channels or 3)

    num_samples = kwargs.pop("dataset_size", kwargs.pop("num_samples", 1000))

    if dataset_type == "synthetic":
        data = generate_synthetic_images(num_samples, **kwargs)
    elif dataset_type == "mnist_like":
        data = generate_mnist_like_images(num_samples, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    source_config = MemorySourceConfig(shuffle=shuffle)
    return MemorySource(source_config, data, rngs=rngs)
