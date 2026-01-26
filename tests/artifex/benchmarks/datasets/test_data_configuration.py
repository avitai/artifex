"""Tests for DataConfig usage in dataset classes.

This test file validates that dataset classes properly use the unified
DataConfig system and reject dict configs.
"""

from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from artifex.benchmarks.datasets.base import DatasetProtocol
from artifex.generative_models.core.configuration import DataConfig


class MockDataset(DatasetProtocol):
    """Mock dataset for testing DataConfig usage."""

    def __init__(self, data_path: str, config: DataConfig, *, rngs: nnx.Rngs):
        """Initialize with DataConfig instead of dict."""
        if not isinstance(config, DataConfig):
            raise TypeError(f"config must be DataConfig, got {type(config).__name__}")

        # Store typed_config and pass it directly to base class
        self.typed_config = config
        # Pass DataConfig directly - no dict conversion needed
        super().__init__(data_path, config, rngs=rngs)

    def _validate_dataset_path(self):
        """Validate dataset path exists."""
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)

    def _load_dataset(self):
        """Load mock dataset."""
        # Create mock data based on config
        self.typed_config.metadata.get("batch_size", 32)
        feature_dim = self.typed_config.metadata.get("feature_dim", 10)
        self.data = jnp.ones((100, feature_dim))

    def get_batch(self, batch_size=None):
        """Get a batch of mock data."""
        batch_size = batch_size or self.typed_config.metadata.get("batch_size", 32)
        indices = jnp.arange(min(batch_size, len(self.data)))
        return {"data": self.data[indices]}

    def get_dataset_info(self):
        """Get dataset information."""
        return {
            "name": self.typed_config.dataset_name,
            "modality": "test",
            "n_samples": len(self.data),
            "data_shape": self.data.shape,
        }


@pytest.fixture
def rngs():
    """Standard fixture for rngs."""
    return nnx.Rngs(42)


@pytest.fixture
def data_config():
    """Create a standard DataConfig for testing."""
    return DataConfig(
        name="test_data_config",
        dataset_name="mock_dataset",
        data_dir=Path("./test_data"),
        split="train",
        num_workers=2,
        prefetch_factor=1,
        pin_memory=False,
        augmentation=False,
        metadata={
            "batch_size": 16,
            "feature_dim": 20,
        },
    )


class TestDataConfigUsage:
    """Test that datasets properly use DataConfig."""

    def test_dataset_with_data_configuration(self, data_config, rngs):
        """Test creating dataset with DataConfig works."""
        dataset = MockDataset(data_path="./test_data", config=data_config, rngs=rngs)

        assert isinstance(dataset.typed_config, DataConfig)
        assert dataset.typed_config.dataset_name == "mock_dataset"
        assert dataset.typed_config.split == "train"

        # Test batch retrieval
        batch = dataset.get_batch()
        assert "data" in batch
        assert batch["data"].shape == (16, 20)  # batch_size x feature_dim

    def test_dataset_rejects_dict_config(self, rngs):
        """Test that dataset raises TypeError for dict config."""
        dict_config = {"dataset_name": "mock_dataset", "batch_size": 32, "split": "train"}

        with pytest.raises(TypeError, match="config must be DataConfig"):
            MockDataset(data_path="./test_data", config=dict_config, rngs=rngs)

    def test_dataset_rejects_none_config(self, rngs):
        """Test that dataset raises TypeError for None config."""
        with pytest.raises(TypeError, match="config must be DataConfig"):
            MockDataset(data_path="./test_data", config=None, rngs=rngs)

    def test_dataset_info_uses_typed_config(self, data_config, rngs):
        """Test that dataset info uses typed configuration."""
        dataset = MockDataset(data_path="./test_data", config=data_config, rngs=rngs)

        info = dataset.get_dataset_info()
        assert info["name"] == data_config.dataset_name
        assert info["n_samples"] == 100

    def test_data_configuration_fields(self, data_config):
        """Test DataConfig has all required fields."""
        assert data_config.dataset_name == "mock_dataset"
        assert data_config.data_dir == Path("./test_data")
        assert data_config.split == "train"
        assert data_config.num_workers == 2
        assert data_config.prefetch_factor == 1
        assert data_config.pin_memory is False
        assert data_config.augmentation is False
        assert data_config.augmentation_params == {}

    def test_data_configuration_validation(self):
        """Test DataConfig validation."""
        # Test invalid num_workers
        with pytest.raises(ValueError):
            DataConfig(
                name="invalid_config",
                dataset_name="test",
                num_workers=-1,  # Must be >= 0
            )

        # Test invalid prefetch_factor
        with pytest.raises(ValueError):
            DataConfig(
                name="invalid_config",
                dataset_name="test",
                prefetch_factor=0,  # Must be >= 1
            )


class TestDatasetLoader:
    """Test DatasetLoader with typed configurations."""

    def test_updated_loader_with_data_configuration(self, data_config, rngs):
        """Test that updated DatasetLoader should accept DataConfig."""

        # Create a mock updated loader that uses DataConfig
        class UpdatedDatasetLoader:
            def __init__(self):
                self.dataset_types = {}

            def register_dataset_type(self, type_name: str, dataset_class):
                self.dataset_types[type_name] = dataset_class

            def load_from_config(self, config: DataConfig, *, rngs: nnx.Rngs):
                """Updated load_from_config that requires DataConfig."""
                if not isinstance(config, DataConfig):
                    raise TypeError(f"config must be DataConfig, got {type(config).__name__}")

                # Get dataset type from metadata
                dataset_type = config.metadata.get("type", "mock")
                if dataset_type not in self.dataset_types:
                    raise ValueError(f"Dataset type '{dataset_type}' not registered")

                dataset_class = self.dataset_types[dataset_type]
                return dataset_class(data_path=str(config.data_dir), config=config, rngs=rngs)

        loader = UpdatedDatasetLoader()
        loader.register_dataset_type("mock", MockDataset)

        # Add type to metadata
        data_config.metadata["type"] = "mock"

        # This should work with updated loader
        dataset = loader.load_from_config(data_config, rngs=rngs)
        assert isinstance(dataset.typed_config, DataConfig)
        assert dataset.typed_config.dataset_name == "mock_dataset"

    def test_updated_loader_rejects_dict_config(self, rngs):
        """Test that updated loader rejects dict config."""

        # Create a mock updated loader
        class UpdatedDatasetLoader:
            def load_from_config(self, config, *, rngs: nnx.Rngs):
                if not isinstance(config, DataConfig):
                    raise TypeError(f"config must be DataConfig, got {type(config).__name__}")

        loader = UpdatedDatasetLoader()

        # Old style dict config
        dict_config = {
            "type": "mock",
            "data_path": "./test_data",
            "dataset_name": "mock_dataset",
            "batch_size": 32,
        }

        # Should raise TypeError
        with pytest.raises(TypeError, match="config must be DataConfig"):
            loader.load_from_config(dict_config, rngs=rngs)
