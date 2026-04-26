from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from artifex.benchmarks.datasets.synthetic_datasets import (
    _create_square,
    create_gaussian_mixture,
    create_image_dataset,
)


def test_gaussian_mixture_dataset_is_reproducible_and_described() -> None:
    """Gaussian mixture generation should preserve shape, metadata, and seed behavior."""
    first = create_gaussian_mixture(
        num_samples=16,
        num_components=4,
        dimension=3,
        component_std=0.25,
        random_seed=7,
    )
    second = create_gaussian_mixture(
        num_samples=16,
        num_components=4,
        dimension=3,
        component_std=0.25,
        random_seed=7,
    )

    assert first.config.name == "gaussian_mixture_4_3d"
    assert first.config.data_type == "continuous"
    assert first.config.dimensions == [3]
    assert first.config.metadata == {
        "num_components": 4,
        "component_std": 0.25,
        "random_seed": 7,
    }
    assert len(first) == 16
    assert first[0].shape == (3,)
    np.testing.assert_allclose(first.data, second.data)


def test_gaussian_mixture_handles_default_seed_and_empty_components() -> None:
    """Small sample counts should still produce a complete dataset if components are empty."""
    dataset = create_gaussian_mixture(
        num_samples=2,
        num_components=8,
        dimension=2,
        component_std=0.1,
        random_seed=None,
    )

    assert dataset.config.metadata["random_seed"] is None
    assert dataset.data.shape == (2, 2)
    assert jnp.isfinite(dataset.data).all()


def test_create_square_clips_to_image_boundaries() -> None:
    """Squares near an edge should be clipped rather than indexing out of bounds."""
    square = _create_square(size=4, pos_x=0, pos_y=0, image_size=6)

    assert square.shape == (6, 6)
    assert jnp.sum(square) == 4.0
    np.testing.assert_allclose(square[:2, :2], jnp.ones((2, 2)))
    np.testing.assert_allclose(square[2:, :], jnp.zeros((4, 6)))


def test_image_dataset_contains_bounded_shape_images_and_metadata() -> None:
    """Synthetic image generation should return bounded arrays and retained metadata."""
    dataset = create_image_dataset(
        num_samples=3,
        image_size=8,
        num_shapes_per_image=2,
        shape_size_range=(2, 4),
        random_seed=11,
    )

    assert dataset.config.name == "simple_images_8x8"
    assert dataset.config.dimensions == [8, 8]
    assert dataset.config.metadata == {
        "num_shapes_per_image": 2,
        "shape_size_range": (2, 4),
        "random_seed": 11,
    }
    assert dataset.data.shape == (3, 8, 8)
    assert float(jnp.min(dataset.data)) >= 0.0
    assert float(jnp.max(dataset.data)) <= 1.0
    assert bool(jnp.any(dataset.data > 0.0))
