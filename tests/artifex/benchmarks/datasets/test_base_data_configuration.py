"""Tests for base.py using DataConfig."""

from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from artifex.benchmarks.datasets.base import (
    DatasetLoader,
    DatasetProtocol,
    DatasetValidator,
)
from artifex.generative_models.core.configuration import DataConfig


class ConcreteDataset(DatasetProtocol):
    """Concrete implementation for testing."""

    def _validate_dataset_path(self):
        """Create path if it doesn't exist."""
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)

    def _load_dataset(self):
        """Load mock data."""
        self.config.metadata.get("batch_size", 32)
        self.data = jnp.ones((100, 10))

    def get_batch(self, batch_size=None):
        """Get a batch of data."""
        batch_size = batch_size or self.config.metadata.get("batch_size", 32)
        indices = jnp.arange(min(batch_size, len(self.data)))
        return {"data": self.data[indices]}

    def get_dataset_info(self):
        """Get dataset information."""
        return {
            "name": self.config.dataset_name,
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
    """Create a DataConfig for testing."""
    return DataConfig(
        name="test_dataset_config",
        dataset_name="concrete_dataset",
        data_dir=Path("./test_data"),
        split="train",
        num_workers=4,
        metadata={
            "batch_size": 16,
            "type": "concrete",  # Required for DatasetLoader
        },
    )


class TestDatasetProtocolWithDataConfig:
    """Test DatasetProtocol with DataConfig."""

    def test_dataset_init_with_data_configuration(self, data_config, rngs):
        """Test creating dataset with DataConfig."""
        dataset = ConcreteDataset(data_path="./test_data", config=data_config, rngs=rngs)

        assert isinstance(dataset.config, DataConfig)
        assert dataset.config.dataset_name == "concrete_dataset"
        assert dataset.data.shape == (100, 10)

    def test_dataset_rejects_dict_config(self, rngs):
        """Test that DatasetProtocol rejects dict config."""
        dict_config = {"dataset_name": "test", "batch_size": 32}

        with pytest.raises(TypeError, match="config must be DataConfig"):
            ConcreteDataset(data_path="./test_data", config=dict_config, rngs=rngs)

    def test_get_batch_uses_config_metadata(self, data_config, rngs):
        """Test that get_batch uses config metadata."""
        dataset = ConcreteDataset(data_path="./test_data", config=data_config, rngs=rngs)

        batch = dataset.get_batch()
        assert batch["data"].shape == (16, 10)  # Uses batch_size from metadata

    def test_validate_batch_with_data_configuration(self, data_config, rngs):
        """Test validate_batch with DataConfig."""
        # Add validation config to metadata
        data_config.metadata["validation"] = {"required_fields": ["data", "labels"]}

        dataset = ConcreteDataset(data_path="./test_data", config=data_config, rngs=rngs)

        # Invalid batch - missing required field
        batch = {"data": jnp.ones((16, 10))}
        assert not dataset.validate_batch(batch)

        # Valid batch
        batch = {"data": jnp.ones((16, 10)), "labels": jnp.zeros(16)}
        assert dataset.validate_batch(batch)


class TestDatasetLoaderWithDataConfig:
    """Test DatasetLoader with DataConfig."""

    def test_loader_with_data_configuration(self, data_config, rngs):
        """Test DatasetLoader.load_from_config with DataConfig."""
        loader = DatasetLoader()
        loader.register_dataset_type("concrete", ConcreteDataset)

        dataset = loader.load_from_config(data_config, rngs=rngs)

        assert isinstance(dataset, ConcreteDataset)
        assert isinstance(dataset.config, DataConfig)
        assert dataset.config.dataset_name == "concrete_dataset"

    def test_loader_rejects_dict_config(self, rngs):
        """Test that DatasetLoader rejects dict config."""
        loader = DatasetLoader()
        loader.register_dataset_type("concrete", ConcreteDataset)

        dict_config = {"type": "concrete", "data_path": "./test_data", "dataset_name": "test"}

        with pytest.raises(TypeError, match="config must be DataConfig"):
            loader.load_from_config(dict_config, rngs=rngs)

    def test_loader_missing_type_in_metadata(self, rngs):
        """Test error when type is missing from metadata."""
        loader = DatasetLoader()

        config = DataConfig(
            name="test_config",
            dataset_name="test",
            data_dir=Path("./test_data"),
            metadata={},  # No type field
        )

        with pytest.raises(KeyError, match="metadata must specify 'type'"):
            loader.load_from_config(config, rngs=rngs)

    def test_loader_unregistered_type(self, rngs):
        """Test error for unregistered dataset type."""
        loader = DatasetLoader()

        config = DataConfig(
            name="test_config",
            dataset_name="test",
            data_dir=Path("./test_data"),
            metadata={"type": "unknown"},
        )

        with pytest.raises(ValueError, match="Dataset type 'unknown' not registered"):
            loader.load_from_config(config, rngs=rngs)


class TestDatasetValidatorWithDataConfig:
    """Test DatasetValidator with DataConfig."""

    def test_validate_config_with_data_configuration(self):
        """Test validating DataConfig."""
        validator = DatasetValidator()

        # Valid config
        config = DataConfig(
            name="valid_config",
            dataset_name="test_dataset",
            num_workers=4,
            prefetch_factor=2,
            metadata={"batch_size": 32},
        )

        is_valid, errors = validator.validate_config(config)
        assert is_valid
        assert len(errors) == 0

    def test_validate_config_rejects_dict(self):
        """Test that validate_config rejects dict."""
        validator = DatasetValidator()

        dict_config = {"dataset_name": "test", "batch_size": 32}

        is_valid, errors = validator.validate_config(dict_config)
        assert not is_valid
        assert "config must be DataConfig" in errors[0]

    def test_validate_config_invalid_fields(self):
        """Test that dataclass validation catches invalid field values at creation time.

        With frozen dataclasses, validation happens in __post_init__, so invalid
        configs cannot be created. This is the correct behavior.
        """
        # Empty dataset_name should raise ValueError at creation time
        with pytest.raises(ValueError, match="dataset_name cannot be empty"):
            DataConfig(
                name="invalid_config",
                dataset_name="",  # Empty string - raises immediately
                metadata={"batch_size": -1},
            )

        # Test validator with a valid config but invalid metadata
        validator = DatasetValidator()
        config = DataConfig(
            name="test_config",
            dataset_name="test_dataset",
            metadata={"batch_size": -1},  # Invalid batch size in metadata
        )
        is_valid, errors = validator.validate_config(config)
        # Validator should catch invalid metadata
        assert not is_valid or any("batch_size" in str(e) for e in errors)
