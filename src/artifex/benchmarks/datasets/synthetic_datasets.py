"""Synthetic dataset generators for benchmarks.

This module provides functions to generate synthetic datasets for benchmarking
generative models, including Gaussian mixtures and simple image datasets.
"""

import jax
import jax.numpy as jnp

from artifex.benchmarks.datasets.dataset_loaders import (
    BenchmarkDataset,
    BenchmarkDatasetConfig,
)


def create_gaussian_mixture(
    num_samples: int = 1000,
    num_components: int = 5,
    dimension: int = 2,
    component_std: float = 0.1,
    random_seed: int | None = None,
) -> BenchmarkDataset:
    """Create a synthetic dataset from a Gaussian mixture model.

    Args:
        num_samples: Number of samples to generate.
        num_components: Number of Gaussian components.
        dimension: Dimensionality of the data.
        component_std: Standard deviation of each component.
        random_seed: Random seed for reproducibility.

    Returns:
        A BenchmarkDataset containing the generated data.
    """
    # Set random seed
    if random_seed is not None:
        key = jax.random.PRNGKey(random_seed)
    else:
        key = jax.random.PRNGKey(0)

    # Generate component means
    key, subkey = jax.random.split(key)
    component_means = jax.random.uniform(
        subkey, shape=(num_components, dimension), minval=-2.0, maxval=2.0
    )

    # Sample component assignments
    key, subkey = jax.random.split(key)
    component_assignments = jax.random.categorical(
        subkey, jnp.zeros(num_components), shape=(num_samples,)
    )

    # Generate samples for each component
    key, subkey = jax.random.split(key)
    samples = jnp.zeros((num_samples, dimension))

    for i in range(num_components):
        # Get indices for this component
        indices = jnp.where(component_assignments == i)[0]
        if len(indices) == 0:
            continue

        # Generate samples for this component
        component_key = jax.random.fold_in(subkey, i)
        component_samples = (
            jax.random.normal(component_key, shape=(len(indices), dimension)) * component_std
            + component_means[i]
        )

        # Update samples
        samples = samples.at[indices].set(component_samples)

    # Create dataset configuration
    config = BenchmarkDatasetConfig(
        name=f"gaussian_mixture_{num_components}_{dimension}d",
        description=(f"Synthetic Gaussian mixture with {num_components} components"),
        data_type="continuous",
        dimensions=[dimension],
        metadata={
            "num_components": num_components,
            "component_std": component_std,
            "random_seed": random_seed,
        },
    )

    # Create and return dataset
    return BenchmarkDataset(config=config, data=samples)


def _create_square(size: int, pos_x: int, pos_y: int, image_size: int) -> jnp.ndarray:
    """Create a square in an image.

    Args:
        size: Size of the square.
        pos_x: X position of the square center.
        pos_y: Y position of the square center.
        image_size: Size of the image.

    Returns:
        2D array with the square.
    """
    img = jnp.zeros((image_size, image_size))
    half_size = size // 2

    start_x = max(0, pos_x - half_size)
    end_x = min(image_size, pos_x + half_size)
    start_y = max(0, pos_y - half_size)
    end_y = min(image_size, pos_y + half_size)

    # Create selection indices
    x_indices = jnp.arange(start_x, end_x)
    y_indices = jnp.arange(start_y, end_y)

    # Create a grid of indices
    xx, yy = jnp.meshgrid(x_indices, y_indices)

    # Set values at the square region to 1
    img = img.at[yy, xx].set(1.0)

    return img


def create_image_dataset(
    num_samples: int = 1000,
    image_size: int = 32,
    num_shapes_per_image: int = 3,
    shape_size_range: tuple[int, int] = (4, 8),
    random_seed: int | None = None,
) -> BenchmarkDataset:
    """Create a synthetic image dataset with simple shapes.

    Args:
        num_samples: Number of images to generate.
        image_size: Size of each image (square).
        num_shapes_per_image: Number of shapes per image.
        shape_size_range: Range of shape sizes (min, max).
        random_seed: Random seed for reproducibility.

    Returns:
        A BenchmarkDataset containing the generated images.
    """
    # Set random seed
    if random_seed is not None:
        key = jax.random.PRNGKey(random_seed)
    else:
        key = jax.random.PRNGKey(0)

    # Initialize empty array for images
    images = jnp.zeros((num_samples, image_size, image_size))

    for i in range(num_samples):
        # Create a new image
        img = jnp.zeros((image_size, image_size))

        # Generate shapes for this image
        for j in range(num_shapes_per_image):
            # Get random parameters for the shape
            key, subkey = jax.random.split(key)
            size = jax.random.randint(
                subkey, shape=(1,), minval=shape_size_range[0], maxval=shape_size_range[1]
            ).item()

            key, subkey = jax.random.split(key)
            pos_x = jax.random.randint(subkey, shape=(1,), minval=0, maxval=image_size).item()

            key, subkey = jax.random.split(key)
            pos_y = jax.random.randint(subkey, shape=(1,), minval=0, maxval=image_size).item()

            # Create and add shape to the image
            shape = _create_square(size, pos_x, pos_y, image_size)
            img = jnp.maximum(img, shape)  # Use maximum to avoid overlap issues

        # Add image to the dataset
        images = images.at[i].set(img)

    # Create dataset configuration
    config = BenchmarkDatasetConfig(
        name=f"simple_images_{image_size}x{image_size}",
        description="Synthetic image dataset with simple shapes",
        data_type="continuous",
        dimensions=[image_size, image_size],
        metadata={
            "num_shapes_per_image": num_shapes_per_image,
            "shape_size_range": shape_size_range,
            "random_seed": random_seed,
        },
    )

    # Create and return dataset
    return BenchmarkDataset(config=config, data=images)
