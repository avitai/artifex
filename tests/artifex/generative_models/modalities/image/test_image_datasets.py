"""Tests for image datasets backed by datarax MemorySource."""

import jax.numpy as jnp
import pytest
from datarax import Pipeline
from datarax.core.data_source import DataSourceModule
from datarax.sources import MemorySource
from flax import nnx

from artifex.generative_models.modalities.image.datasets import (
    create_image_dataset,
    generate_mnist_like_images,
    generate_synthetic_images,
)


@pytest.fixture
def rngs():
    """Random number generators for testing."""
    return nnx.Rngs(42)


# --- Data generation functions ---


class TestGenerateSyntheticImages:
    """Test pure image generation function."""

    def test_basic_generation(self) -> None:
        data = generate_synthetic_images(10, height=8, width=8, channels=3)
        assert "images" in data
        assert data["images"].shape == (10, 8, 8, 3)

    def test_single_channel(self) -> None:
        data = generate_synthetic_images(5, height=16, width=16, channels=1)
        assert data["images"].shape == (5, 16, 16, 1)

    def test_pattern_types(self) -> None:
        for pattern in ["random", "gradient", "checkerboard", "circles"]:
            data = generate_synthetic_images(
                2, height=16, width=16, channels=3, pattern_type=pattern
            )
            assert data["images"].shape == (2, 16, 16, 3)

    def test_invalid_pattern_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown pattern type"):
            generate_synthetic_images(1, pattern_type="invalid")

    def test_invalid_dimensions_raise(self) -> None:
        with pytest.raises(ValueError, match="Height must be positive"):
            generate_synthetic_images(1, height=0)
        with pytest.raises(ValueError, match="Width must be positive"):
            generate_synthetic_images(1, width=0)
        with pytest.raises(ValueError, match="Channels must be positive"):
            generate_synthetic_images(1, channels=0)

    def test_values_are_finite(self) -> None:
        data = generate_synthetic_images(5, height=8, width=8, channels=3)
        assert jnp.all(jnp.isfinite(data["images"]))


class TestGenerateMnistLikeImages:
    """Test MNIST-like image generation function."""

    def test_basic_generation(self) -> None:
        data = generate_mnist_like_images(10, height=28, width=28, channels=1)
        assert "images" in data
        assert "labels" in data
        assert data["images"].shape == (10, 28, 28, 1)
        assert data["labels"].shape == (10,)

    def test_labels_cycle_through_classes(self) -> None:
        data = generate_mnist_like_images(15, num_classes=5)
        # Labels should cycle: 0,1,2,3,4,0,1,2,3,4,...
        for i in range(15):
            assert int(data["labels"][i]) == i % 5

    def test_invalid_dimensions_raise(self) -> None:
        with pytest.raises(ValueError, match="Height must be positive"):
            generate_mnist_like_images(1, height=0)


# --- MemorySource factory ---


class TestCreateImageDataset:
    """Test factory function returns MemorySource."""

    def test_returns_memory_source(self, rngs) -> None:
        source = create_image_dataset("synthetic", rngs=rngs, dataset_size=5)
        assert isinstance(source, MemorySource)
        assert isinstance(source, DataSourceModule)

    def test_correct_length(self, rngs) -> None:
        source = create_image_dataset("synthetic", rngs=rngs, dataset_size=10, height=8, width=8)
        assert len(source) == 10

    def test_getitem(self, rngs) -> None:
        source = create_image_dataset(
            "synthetic", rngs=rngs, dataset_size=5, height=8, width=8, channels=3
        )
        sample = source[0]
        assert "images" in sample
        assert sample["images"].shape == (8, 8, 3)

    def test_negative_index(self, rngs) -> None:
        source = create_image_dataset("synthetic", rngs=rngs, dataset_size=5, height=8, width=8)
        sample = source[-1]
        assert "images" in sample

    def test_out_of_bounds_raises(self, rngs) -> None:
        source = create_image_dataset("synthetic", rngs=rngs, dataset_size=5, height=8, width=8)
        with pytest.raises(IndexError):
            source[5]

    def test_iteration(self, rngs) -> None:
        source = create_image_dataset(
            "synthetic", rngs=rngs, dataset_size=3, height=8, width=8, channels=1
        )
        elements = list(source)
        assert len(elements) == 3
        for el in elements:
            assert "images" in el
            assert el["images"].shape == (8, 8, 1)

    def test_get_batch(self, rngs) -> None:
        source = create_image_dataset(
            "synthetic", rngs=rngs, dataset_size=10, height=8, width=8, channels=1
        )
        batch = source.get_batch(4)
        assert "images" in batch
        assert batch["images"].shape == (4, 8, 8, 1)

    def test_pattern_types(self, rngs) -> None:
        for pattern in ["random", "gradient", "checkerboard", "circles"]:
            source = create_image_dataset(
                "synthetic",
                rngs=rngs,
                dataset_size=2,
                height=16,
                width=16,
                channels=3,
                pattern_type=pattern,
            )
            batch = source.get_batch(2)
            assert batch["images"].shape == (2, 16, 16, 3)

    def test_mnist_like(self, rngs) -> None:
        source = create_image_dataset(
            "mnist_like",
            rngs=rngs,
            dataset_size=5,
            height=28,
            width=28,
            channels=1,
            num_classes=5,
        )
        assert isinstance(source, MemorySource)
        sample = source[0]
        assert "images" in sample
        assert "labels" in sample
        assert sample["images"].shape == (28, 28, 1)
        batch = source.get_batch(3)
        assert batch["images"].shape == (3, 28, 28, 1)
        assert batch["labels"].shape == (3,)

    def test_unknown_type_raises(self, rngs) -> None:
        with pytest.raises(ValueError, match="Unknown dataset type"):
            create_image_dataset("unknown", rngs=rngs)


# --- Pipeline integration ---


class TestImagePipeline:
    """Test datarax pipeline integration."""

    def test_batched_pipeline(self, rngs) -> None:
        source = create_image_dataset(
            "synthetic", rngs=rngs, dataset_size=6, height=8, width=8, channels=1
        )
        pipeline = Pipeline(source=source, stages=[], batch_size=3, rngs=nnx.Rngs(0))
        batch = next(iter(pipeline))
        assert "images" in batch
        assert batch["images"].shape[0] == 3
