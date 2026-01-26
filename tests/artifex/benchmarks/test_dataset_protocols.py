"""Tests for dataset management protocols."""

import shutil
import tempfile
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from artifex.benchmarks.datasets.base import (
    DatasetLoader,
    DatasetProtocol,
    DatasetRegistry,
    DatasetValidator,
)
from artifex.generative_models.core.configuration import DataConfig


@pytest.fixture
def rngs():
    """Standard fixture for rngs."""
    return nnx.Rngs(42)


@pytest.fixture
def temp_dataset_dir():
    """Create temporary dataset directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataset_config():
    """Sample dataset configuration."""
    return DataConfig(
        name="test_dataset_config",
        dataset_name="test_dataset",
        data_dir=Path("./test_data"),
        split="train",
        num_workers=2,
        metadata={
            "modality": "geometric",
            "batch_size": 32,
            "shuffle": True,
            "n_samples": 1000,
            "point_dim": 3,
            "n_points": 512,
            "preprocessing": {"normalize": True, "augment": False},
            "validation": {"min_samples": 100, "required_fields": ["points", "labels"]},
            "type": "mock",  # For DatasetLoader
        },
    )


class MockDataset(DatasetProtocol):
    """Mock dataset implementation for testing."""

    def _validate_dataset_path(self):
        """Validate dataset path and structure."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path {self.data_path} does not exist")

    def _load_dataset(self):
        """Load the dataset."""
        # Create mock data
        n_samples = self.config.metadata.get("n_samples", 1000)
        point_dim = self.config.metadata.get("point_dim", 3)
        n_points = self.config.metadata.get("n_points", 512)

        key = self.rngs.default() if "default" in self.rngs else jax.random.key(0)
        keys = jax.random.split(key, 3)

        self.data = {
            "points": jax.random.normal(keys[0], (n_samples, n_points, point_dim)),
            "labels": jax.random.randint(keys[1], (n_samples,), 0, 10),
            "metadata": {"n_samples": n_samples, "point_dim": point_dim, "n_points": n_points},
        }

    def get_batch(self, batch_size: int | None = None) -> dict[str, jax.Array]:
        """Get a batch of data."""
        batch_size = batch_size or self.config.metadata.get("batch_size", 32)
        n_samples = self.data["metadata"]["n_samples"]

        # Simple random sampling
        key = self.rngs.default() if "default" in self.rngs else jax.random.key(0)
        indices = jax.random.choice(key, n_samples, (batch_size,), replace=False)

        return {"points": self.data["points"][indices], "labels": self.data["labels"][indices]}

    def get_dataset_info(self) -> dict[str, Any]:
        """Get dataset information."""
        return {
            "name": self.config.dataset_name,
            "modality": self.config.metadata.get("modality", "unknown"),
            "n_samples": self.data["metadata"]["n_samples"],
            "data_shape": {
                "points": self.data["points"].shape,
                "labels": self.data["labels"].shape,
            },
        }


class TestDatasetProtocol:
    """Test DatasetProtocol abstract base class."""

    def test_dataset_initialization(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test dataset initialization."""
        # Create dummy dataset directory
        temp_dataset_dir.mkdir(exist_ok=True)

        dataset = MockDataset(
            data_path=str(temp_dataset_dir), config=sample_dataset_config, rngs=rngs
        )

        assert dataset.data_path == temp_dataset_dir
        assert dataset.config == sample_dataset_config
        assert dataset.rngs is not None
        assert hasattr(dataset, "data")

    def test_dataset_path_validation(self, sample_dataset_config, rngs):
        """Test dataset path validation."""
        non_existent_path = "/non/existent/path"

        with pytest.raises(FileNotFoundError):
            MockDataset(data_path=non_existent_path, config=sample_dataset_config, rngs=rngs)

    def test_dataset_data_loading(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test dataset data loading."""
        temp_dataset_dir.mkdir(exist_ok=True)

        # Update metadata
        sample_dataset_config.metadata.update({"n_samples": 500, "n_points": 256, "point_dim": 3})

        dataset = MockDataset(
            data_path=str(temp_dataset_dir), config=sample_dataset_config, rngs=rngs
        )

        assert "points" in dataset.data
        assert "labels" in dataset.data
        assert "metadata" in dataset.data

        # Check shapes
        assert dataset.data["points"].shape == (500, 256, 3)
        assert dataset.data["labels"].shape == (500,)

    def test_get_batch(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test batch retrieval."""
        temp_dataset_dir.mkdir(exist_ok=True)

        dataset = MockDataset(
            data_path=str(temp_dataset_dir), config=sample_dataset_config, rngs=rngs
        )

        # Test default batch size
        batch = dataset.get_batch()
        expected_batch_size = sample_dataset_config.metadata["batch_size"]

        assert "points" in batch
        assert "labels" in batch
        assert batch["points"].shape[0] == expected_batch_size
        assert batch["labels"].shape[0] == expected_batch_size

        # Test custom batch size
        custom_batch = dataset.get_batch(batch_size=16)
        assert custom_batch["points"].shape[0] == 16
        assert custom_batch["labels"].shape[0] == 16

    def test_get_dataset_info(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test dataset info retrieval."""
        temp_dataset_dir.mkdir(exist_ok=True)

        dataset = MockDataset(
            data_path=str(temp_dataset_dir), config=sample_dataset_config, rngs=rngs
        )

        info = dataset.get_dataset_info()

        assert isinstance(info, dict)
        assert "name" in info
        assert "modality" in info
        assert "n_samples" in info
        assert "data_shape" in info

        assert info["name"] == sample_dataset_config.dataset_name
        assert info["modality"] == sample_dataset_config.metadata["modality"]
        assert info["n_samples"] == 1000  # default value

    def test_dataset_rngs_handling(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test proper RNG handling in datasets."""
        temp_dataset_dir.mkdir(exist_ok=True)

        dataset = MockDataset(
            data_path=str(temp_dataset_dir), config=sample_dataset_config, rngs=rngs
        )

        # Test that batch generation works with rngs
        batch1 = dataset.get_batch(batch_size=10)
        batch2 = dataset.get_batch(batch_size=10)

        assert batch1["points"].shape == (10, 512, 3)  # default n_points=512, point_dim=3
        assert batch2["points"].shape == (10, 512, 3)

        # Test that data is finite
        assert jnp.isfinite(batch1["points"]).all()
        assert jnp.isfinite(batch2["points"]).all()


class TestDatasetRegistry:
    """Test DatasetRegistry functionality."""

    def test_dataset_registry_singleton(self):
        """Test dataset registry singleton pattern."""
        registry1 = DatasetRegistry()
        registry2 = DatasetRegistry()

        assert registry1 is registry2

    def test_register_dataset(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test dataset registration."""
        registry = DatasetRegistry()

        # Clear registry for clean test
        registry.datasets.clear()

        temp_dataset_dir.mkdir(exist_ok=True)
        dataset = MockDataset(
            data_path=str(temp_dataset_dir), config=sample_dataset_config, rngs=rngs
        )

        registry.register_dataset("test_dataset", dataset)

        assert "test_dataset" in registry.datasets
        assert registry.datasets["test_dataset"] is dataset

    def test_get_dataset(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test dataset retrieval."""
        registry = DatasetRegistry()
        registry.datasets.clear()

        temp_dataset_dir.mkdir(exist_ok=True)
        dataset = MockDataset(
            data_path=str(temp_dataset_dir), config=sample_dataset_config, rngs=rngs
        )

        registry.register_dataset("test_dataset", dataset)

        retrieved = registry.get_dataset("test_dataset")
        assert retrieved is dataset

        # Test non-existent dataset
        with pytest.raises(KeyError):
            registry.get_dataset("non_existent")

    def test_list_datasets(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test listing available datasets."""
        registry = DatasetRegistry()
        registry.datasets.clear()

        # No datasets initially
        available = registry.list_datasets()
        assert len(available) == 0

        # Register some datasets
        temp_dataset_dir.mkdir(exist_ok=True)
        for i in range(3):
            # Create a new config for each dataset
            config = DataConfig(
                name=f"dataset_{i}_config",
                dataset_name=f"dataset_{i}",
                data_dir=sample_dataset_config.data_dir,
                split=sample_dataset_config.split,
                num_workers=sample_dataset_config.num_workers,
                metadata=sample_dataset_config.metadata.copy(),
            )

            dataset = MockDataset(data_path=str(temp_dataset_dir), config=config, rngs=rngs)
            registry.register_dataset(f"dataset_{i}", dataset)

        available = registry.list_datasets()
        assert len(available) == 3
        assert "dataset_0" in available
        assert "dataset_1" in available
        assert "dataset_2" in available

    def test_dataset_metadata(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test dataset metadata retrieval."""
        registry = DatasetRegistry()
        registry.datasets.clear()

        temp_dataset_dir.mkdir(exist_ok=True)
        dataset = MockDataset(
            data_path=str(temp_dataset_dir), config=sample_dataset_config, rngs=rngs
        )

        registry.register_dataset("test_dataset", dataset)

        metadata = registry.get_dataset_metadata("test_dataset")

        assert isinstance(metadata, dict)
        assert "name" in metadata
        assert "modality" in metadata
        assert "n_samples" in metadata


class TestDatasetValidator:
    """Test DatasetValidator functionality."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = DatasetValidator()

        assert hasattr(validator, "validation_rules")
        assert isinstance(validator.validation_rules, dict)

    def test_validate_dataset_structure(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test dataset structure validation."""
        validator = DatasetValidator()

        temp_dataset_dir.mkdir(exist_ok=True)
        dataset = MockDataset(
            data_path=str(temp_dataset_dir), config=sample_dataset_config, rngs=rngs
        )

        # Valid dataset should pass
        is_valid, errors = validator.validate_structure(dataset)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_data_quality(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test data quality validation."""
        validator = DatasetValidator()

        temp_dataset_dir.mkdir(exist_ok=True)
        dataset = MockDataset(
            data_path=str(temp_dataset_dir), config=sample_dataset_config, rngs=rngs
        )

        # Get a sample batch for validation
        batch = dataset.get_batch(batch_size=10)

        is_valid, issues = validator.validate_data_quality(batch)
        assert is_valid is True
        assert len(issues) == 0

    def test_check_minimum_samples(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test minimum samples validation."""
        validator = DatasetValidator()

        temp_dataset_dir.mkdir(exist_ok=True)

        # Test with sufficient samples
        config_sufficient = DataConfig(
            name="sufficient_config",
            dataset_name="sufficient_dataset",
            data_dir=sample_dataset_config.data_dir,
            split=sample_dataset_config.split,
            metadata={**sample_dataset_config.metadata, "n_samples": 1000},
        )

        dataset = MockDataset(data_path=str(temp_dataset_dir), config=config_sufficient, rngs=rngs)

        min_samples = config_sufficient.metadata["validation"]["min_samples"]
        has_enough = validator.check_minimum_samples(dataset, min_samples)
        assert has_enough is True

        # Test with insufficient samples
        config_insufficient = DataConfig(
            name="insufficient_config",
            dataset_name="insufficient_dataset",
            data_dir=sample_dataset_config.data_dir,
            split=sample_dataset_config.split,
            metadata={**sample_dataset_config.metadata, "n_samples": 50},
        )

        dataset_small = MockDataset(
            data_path=str(temp_dataset_dir), config=config_insufficient, rngs=rngs
        )

        has_enough_small = validator.check_minimum_samples(dataset_small, min_samples)
        assert has_enough_small is False


class TestDatasetLoader:
    """Test DatasetLoader functionality."""

    def test_loader_initialization(self):
        """Test dataset loader initialization."""
        loader = DatasetLoader()

        assert hasattr(loader, "supported_formats")
        assert isinstance(loader.supported_formats, list)

    def test_load_from_config(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test loading dataset from configuration."""
        import dataclasses

        loader = DatasetLoader()

        # Register mock dataset type
        loader.register_dataset_type("mock", MockDataset)

        # Create new config with the temp_dataset_dir (frozen dataclass)
        config = dataclasses.replace(sample_dataset_config, data_dir=temp_dataset_dir)

        temp_dataset_dir.mkdir(exist_ok=True)

        dataset = loader.load_from_config(config, rngs=rngs)

        assert isinstance(dataset, MockDataset)
        assert dataset.config.dataset_name == config.dataset_name
        assert dataset.data_path == temp_dataset_dir

    def test_register_dataset_type(self):
        """Test dataset type registration."""
        loader = DatasetLoader()

        # Clear existing registrations
        loader.dataset_types = {}

        loader.register_dataset_type("mock", MockDataset)

        assert "mock" in loader.dataset_types
        assert loader.dataset_types["mock"] is MockDataset

    def test_list_supported_types(self):
        """Test listing supported dataset types."""
        loader = DatasetLoader()
        loader.dataset_types = {}

        # No types initially
        types = loader.list_supported_types()
        assert len(types) == 0

        # Register some types
        loader.register_dataset_type("mock", MockDataset)
        loader.register_dataset_type("test", MockDataset)

        types = loader.list_supported_types()
        assert len(types) == 2
        assert "mock" in types
        assert "test" in types


class TestDatasetIntegration:
    """Integration tests for dataset system."""

    def test_end_to_end_dataset_workflow(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test complete dataset workflow."""
        temp_dataset_dir.mkdir(exist_ok=True)

        # 1. Create dataset
        dataset = MockDataset(
            data_path=str(temp_dataset_dir), config=sample_dataset_config, rngs=rngs
        )

        # 2. Register dataset
        registry = DatasetRegistry()
        registry.datasets.clear()
        registry.register_dataset("test_dataset", dataset)

        # 3. Validate dataset
        validator = DatasetValidator()
        is_valid, errors = validator.validate_structure(dataset)
        assert is_valid is True

        # 4. Get batches and validate quality
        batch = dataset.get_batch(batch_size=16)
        is_quality_valid, issues = validator.validate_data_quality(batch)
        assert is_quality_valid is True

        # 5. Verify dataset info
        info = dataset.get_dataset_info()
        assert info["name"] == "test_dataset"
        assert info["modality"] == "geometric"

    def test_batch_consistency(self, temp_dataset_dir, sample_dataset_config, rngs):
        """Test batch consistency across multiple calls."""
        temp_dataset_dir.mkdir(exist_ok=True)

        dataset = MockDataset(
            data_path=str(temp_dataset_dir), config=sample_dataset_config, rngs=rngs
        )

        # Get multiple batches
        batches = []
        for _ in range(5):
            batch = dataset.get_batch(batch_size=8)
            batches.append(batch)

        # Check that all batches have correct structure and shape
        for batch in batches:
            assert "points" in batch
            assert "labels" in batch
            assert batch["points"].shape == (8, 512, 3)  # batch_size, n_points, point_dim
            assert batch["labels"].shape == (8,)
            assert jnp.isfinite(batch["points"]).all()
            assert jnp.isfinite(batch["labels"]).all()

    def test_config_driven_behavior(self, temp_dataset_dir, rngs):
        """Test configuration-driven dataset behavior."""
        temp_dataset_dir.mkdir(exist_ok=True)

        # Test different configurations
        config_params = [
            {"name": "small", "n_samples": 100, "n_points": 128, "point_dim": 3, "batch_size": 8},
            {
                "name": "large",
                "n_samples": 2000,
                "n_points": 1024,
                "point_dim": 6,
                "batch_size": 64,
            },
        ]

        for params in config_params:
            config = DataConfig(
                name=params["name"] + "_config",
                dataset_name=params["name"],
                data_dir=temp_dataset_dir,
                split="train",
                metadata={"modality": "geometric", "type": "mock", **params},
            )
            dataset = MockDataset(data_path=str(temp_dataset_dir), config=config, rngs=rngs)

            # Verify dataset respects configuration
            info = dataset.get_dataset_info()
            assert info["n_samples"] == params["n_samples"]

            batch = dataset.get_batch()
            expected_shape = (params["batch_size"], params["n_points"], params["point_dim"])
            assert batch["points"].shape == expected_shape
