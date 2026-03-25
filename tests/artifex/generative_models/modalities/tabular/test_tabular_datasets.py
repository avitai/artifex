"""Tests for tabular datasets backed by datarax MemorySource."""

import pytest
from datarax import from_source
from datarax.core.data_source import DataSourceModule
from datarax.sources import MemorySource
from flax import nnx

from artifex.generative_models.modalities.tabular import (
    create_simple_tabular_dataset,
    create_synthetic_tabular_dataset,
    TabularModalityConfig,
)
from artifex.generative_models.modalities.tabular.datasets import (
    compute_feature_statistics,
    generate_synthetic_tabular_data,
)


@pytest.fixture
def rngs():
    """Random number generators for testing."""
    return nnx.Rngs(42)


@pytest.fixture
def simple_modality_config():
    """Simple tabular modality config for testing."""
    return TabularModalityConfig(
        num_features=5,
        numerical_features=["age", "income"],
        categorical_features=["category"],
        ordinal_features=["education"],
        binary_features=["is_member"],
        categorical_vocab_sizes={"category": 4},
        ordinal_orders={"education": ["high_school", "bachelor", "master", "phd"]},
        normalization_type="standard",
        handle_missing="impute",
        max_categorical_cardinality=10,
    )


# --- Data generation functions ---


class TestGenerateSyntheticTabularData:
    """Test pure tabular data generation function."""

    def test_basic_generation(self, simple_modality_config) -> None:
        data = generate_synthetic_tabular_data(simple_modality_config, 50)
        assert isinstance(data, dict)
        assert set(data.keys()) == {"age", "income", "category", "education", "is_member"}

    def test_correct_shapes(self, simple_modality_config) -> None:
        data = generate_synthetic_tabular_data(simple_modality_config, 100)
        for value in data.values():
            assert value.shape == (100,)


class TestComputeFeatureStatistics:
    """Test feature statistics computation."""

    def test_statistics(self, simple_modality_config) -> None:
        data = generate_synthetic_tabular_data(simple_modality_config, 100)
        stats = compute_feature_statistics(data, simple_modality_config, 100)
        assert "age" in stats
        assert stats["age"]["type"] == "numerical"
        assert "mean" in stats["age"]
        assert "category" in stats
        assert stats["category"]["type"] == "categorical"
        assert "is_member" in stats
        assert stats["is_member"]["type"] == "binary"


# --- MemorySource factory ---


class TestCreateSyntheticTabularDataset:
    """Test factory function returns MemorySource."""

    def test_returns_memory_source(self, rngs) -> None:
        source, config = create_synthetic_tabular_dataset(
            num_features=10, num_samples=50, rngs=rngs
        )
        assert isinstance(source, MemorySource)
        assert isinstance(source, DataSourceModule)
        assert isinstance(config, TabularModalityConfig)

    def test_correct_length(self, rngs) -> None:
        source, _ = create_synthetic_tabular_dataset(num_features=10, num_samples=50, rngs=rngs)
        assert len(source) == 50

    def test_getitem(self, rngs) -> None:
        source, config = create_synthetic_tabular_dataset(
            num_features=5,
            num_samples=10,
            numerical_ratio=0.4,
            categorical_ratio=0.2,
            ordinal_ratio=0.2,
            binary_ratio=0.2,
            rngs=rngs,
        )
        item = source[0]
        assert isinstance(item, dict)
        for value in item.values():
            assert value.shape == ()  # Scalar

    def test_negative_index(self, rngs) -> None:
        source, _ = create_synthetic_tabular_dataset(num_features=5, num_samples=5, rngs=rngs)
        item = source[-1]
        assert isinstance(item, dict)

    def test_out_of_bounds_raises(self, rngs) -> None:
        source, _ = create_synthetic_tabular_dataset(num_features=5, num_samples=5, rngs=rngs)
        with pytest.raises(IndexError):
            source[5]

    def test_iteration(self, rngs) -> None:
        source, _ = create_synthetic_tabular_dataset(num_features=5, num_samples=3, rngs=rngs)
        elements = list(source)
        assert len(elements) == 3

    def test_get_batch(self, rngs) -> None:
        source, _ = create_synthetic_tabular_dataset(
            num_features=5,
            num_samples=10,
            numerical_ratio=0.4,
            categorical_ratio=0.2,
            ordinal_ratio=0.2,
            binary_ratio=0.2,
            rngs=rngs,
        )
        batch = source.get_batch(4)
        # Should have feature keys
        assert len(batch) > 0
        for value in batch.values():
            assert value.shape[0] == 4

    def test_invalid_ratios(self, rngs) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            create_synthetic_tabular_dataset(
                numerical_ratio=0.5,
                categorical_ratio=0.3,
                ordinal_ratio=0.1,
                binary_ratio=0.05,
                rngs=rngs,
            )


class TestCreateSimpleTabularDataset:
    """Test simple factory function."""

    def test_returns_memory_source(self, rngs) -> None:
        source, config = create_simple_tabular_dataset(num_samples=20, rngs=rngs)
        assert isinstance(source, MemorySource)
        assert isinstance(source, DataSourceModule)
        assert config.num_features == 5

    def test_correct_features(self, rngs) -> None:
        source, _ = create_simple_tabular_dataset(num_samples=10, rngs=rngs)
        item = source[0]
        assert set(item.keys()) == {"age", "income", "category", "education", "is_member"}

    def test_get_batch(self, rngs) -> None:
        source, _ = create_simple_tabular_dataset(num_samples=10, rngs=rngs)
        batch = source.get_batch(4)
        assert "age" in batch
        assert "income" in batch
        assert batch["age"].shape == (4,)


# --- Pipeline integration ---


class TestTabularPipeline:
    """Test datarax pipeline integration."""

    def test_from_source_pipeline(self, rngs) -> None:
        source, _ = create_simple_tabular_dataset(num_samples=6, rngs=rngs)
        pipeline = from_source(source, batch_size=3)
        batch = next(iter(pipeline))
        assert "age" in batch
        assert batch["age"].shape[0] == 3
