"""Tests for timeseries datasets backed by datarax MemorySource."""

import jax.numpy as jnp
import pytest
from datarax import from_source
from datarax.core.data_source import DataSourceModule
from datarax.sources import MemorySource
from flax import nnx

from artifex.generative_models.modalities.timeseries.datasets import (
    create_simple_timeseries_dataset,
    create_synthetic_timeseries_dataset,
    generate_synthetic_timeseries,
)


@pytest.fixture
def rngs():
    """Random number generators for testing."""
    return nnx.Rngs(42)


# --- Data generation functions ---


class TestGenerateSyntheticTimeseries:
    """Test pure timeseries generation function."""

    def test_basic_generation(self) -> None:
        data = generate_synthetic_timeseries(10, sequence_length=20, num_features=2)
        assert "timeseries" in data
        assert data["timeseries"].shape == (10, 20, 2)

    def test_pattern_types(self) -> None:
        for pattern in ["sinusoidal", "random_walk", "ar", "seasonal", "mixed"]:
            data = generate_synthetic_timeseries(
                3, sequence_length=20, num_features=1, pattern_type=pattern
            )
            assert data["timeseries"].shape == (3, 20, 1)
            assert jnp.all(jnp.isfinite(data["timeseries"]))

    def test_invalid_pattern_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown pattern type"):
            generate_synthetic_timeseries(5, sequence_length=20, pattern_type="invalid")

    def test_validation_sequence_length(self) -> None:
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            generate_synthetic_timeseries(5, sequence_length=0)

    def test_validation_num_features(self) -> None:
        with pytest.raises(ValueError, match="num_features must be positive"):
            generate_synthetic_timeseries(5, num_features=0)

    def test_validation_num_samples(self) -> None:
        with pytest.raises(ValueError, match="num_samples must be positive"):
            generate_synthetic_timeseries(0)

    def test_validation_noise_level(self) -> None:
        with pytest.raises(ValueError, match="noise_level must be non-negative"):
            generate_synthetic_timeseries(5, noise_level=-0.1)

    def test_with_trend(self) -> None:
        data = generate_synthetic_timeseries(5, sequence_length=20, trend_strength=1.0)
        assert data["timeseries"].shape == (5, 20, 1)

    def test_with_seasonal_period(self) -> None:
        data = generate_synthetic_timeseries(
            5,
            sequence_length=40,
            pattern_type="seasonal",
            seasonal_period=10,
        )
        assert data["timeseries"].shape == (5, 40, 1)


# --- MemorySource factory ---


class TestCreateSyntheticTimeseriesDataset:
    """Test factory function returns MemorySource."""

    def test_returns_memory_source(self, rngs) -> None:
        source = create_synthetic_timeseries_dataset(sequence_length=20, num_samples=5, rngs=rngs)
        assert isinstance(source, MemorySource)
        assert isinstance(source, DataSourceModule)

    def test_correct_length(self, rngs) -> None:
        source = create_synthetic_timeseries_dataset(sequence_length=20, num_samples=50, rngs=rngs)
        assert len(source) == 50

    def test_getitem(self, rngs) -> None:
        source = create_synthetic_timeseries_dataset(
            sequence_length=20, num_features=2, num_samples=10, rngs=rngs
        )
        item = source[0]
        assert isinstance(item, dict)
        assert "timeseries" in item
        assert item["timeseries"].shape == (20, 2)

    def test_negative_index(self, rngs) -> None:
        source = create_synthetic_timeseries_dataset(sequence_length=20, num_samples=5, rngs=rngs)
        item = source[-1]
        assert "timeseries" in item

    def test_out_of_bounds_raises(self, rngs) -> None:
        source = create_synthetic_timeseries_dataset(sequence_length=20, num_samples=5, rngs=rngs)
        with pytest.raises(IndexError):
            source[5]

    def test_iteration(self, rngs) -> None:
        source = create_synthetic_timeseries_dataset(
            sequence_length=20, num_features=1, num_samples=3, rngs=rngs
        )
        elements = list(source)
        assert len(elements) == 3
        for el in elements:
            assert "timeseries" in el
            assert el["timeseries"].shape == (20, 1)

    def test_get_batch(self, rngs) -> None:
        source = create_synthetic_timeseries_dataset(
            sequence_length=20, num_features=2, num_samples=10, rngs=rngs
        )
        batch = source.get_batch(4)
        assert "timeseries" in batch
        assert batch["timeseries"].shape == (4, 20, 2)

    def test_pattern_types(self, rngs) -> None:
        for pattern in ["sinusoidal", "random_walk", "ar", "seasonal", "mixed"]:
            source = create_synthetic_timeseries_dataset(
                sequence_length=20,
                num_features=1,
                num_samples=3,
                pattern_type=pattern,
                rngs=rngs,
            )
            item = source[0]
            assert item["timeseries"].shape == (20, 1)
            assert jnp.all(jnp.isfinite(item["timeseries"]))


class TestCreateSimpleTimeseriesDataset:
    """Test simple factory function."""

    def test_returns_memory_source(self, rngs) -> None:
        source = create_simple_timeseries_dataset(sequence_length=20, num_samples=10, rngs=rngs)
        assert isinstance(source, MemorySource)
        assert isinstance(source, DataSourceModule)

    def test_defaults(self, rngs) -> None:
        source = create_simple_timeseries_dataset(rngs=rngs)
        assert len(source) == 100
        item = source[0]
        assert item["timeseries"].shape == (50, 1)


# --- Pipeline integration ---


class TestTimeseriesPipeline:
    """Test datarax pipeline integration."""

    def test_from_source_pipeline(self, rngs) -> None:
        source = create_synthetic_timeseries_dataset(
            sequence_length=20, num_features=1, num_samples=6, rngs=rngs
        )
        pipeline = from_source(source, batch_size=3)
        batch = next(iter(pipeline))
        assert "timeseries" in batch
        assert batch["timeseries"].shape[0] == 3
