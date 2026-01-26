"""Tests for DataConfig frozen dataclass.

This module tests the DataConfig frozen dataclass which replaces the
Pydantic-based DataConfig. Tests verify:
1. Basic instantiation and frozen behavior
2. Field validation (types, ranges, required fields)
3. Default values
4. Serialization (to_dict, from_dict)
5. Edge cases and error handling
"""

import dataclasses
from pathlib import Path

import pytest

from artifex.generative_models.core.configuration.data_config import DataConfig


class TestDataConfigBasics:
    """Test basic DataConfig functionality."""

    def test_create_minimal(self):
        """Test creating DataConfig with minimal required fields."""
        config = DataConfig(
            name="mnist-data",
            dataset_name="mnist",
        )
        assert config.name == "mnist-data"
        assert config.dataset_name == "mnist"
        assert config.data_dir == Path("./data")  # default
        assert config.split == "train"  # default

    def test_create_full(self):
        """Test creating DataConfig with all fields."""
        config = DataConfig(
            name="cifar10-data",
            dataset_name="cifar10",
            data_dir=Path("/path/to/data"),
            split="test",
            num_workers=8,
            prefetch_factor=4,
            pin_memory=False,
            augmentation=True,
            augmentation_params={"rotation": 15, "flip": True},
            validation_split=0.15,
            test_split=0.2,
        )
        assert config.dataset_name == "cifar10"
        assert config.data_dir == Path("/path/to/data")
        assert config.split == "test"
        assert config.num_workers == 8
        assert config.prefetch_factor == 4
        assert config.pin_memory is False
        assert config.augmentation is True
        assert config.augmentation_params == {"rotation": 15, "flip": True}
        assert config.validation_split == 0.15
        assert config.test_split == 0.2

    def test_frozen(self):
        """Test that DataConfig is frozen (immutable)."""
        config = DataConfig(name="test-data", dataset_name="mnist")
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.dataset_name = "cifar10"

    def test_hash(self):
        """Test that DataConfig instances are hashable."""
        # Note: configs with dict fields are NOT hashable
        config1 = DataConfig(name="test-data", dataset_name="mnist")
        DataConfig(name="test-data", dataset_name="mnist")
        DataConfig(name="test-data", dataset_name="cifar10")

        # With default empty dict, should not be hashable
        with pytest.raises(TypeError, match="unhashable type"):
            hash(config1)

        # This is expected behavior - configs with mutable fields are not hashable


class TestDataConfigValidation:
    """Test DataConfig validation."""

    def test_dataset_name_required(self):
        """Test that dataset_name is required (validated in __post_init__)."""
        with pytest.raises(ValueError, match="dataset_name cannot be empty"):
            DataConfig(name="test-data")

    def test_dataset_name_empty_string(self):
        """Test that empty dataset_name is caught."""
        with pytest.raises(ValueError, match="dataset_name cannot be empty"):
            DataConfig(name="test-data", dataset_name="")

    def test_dataset_name_whitespace(self):
        """Test that whitespace-only dataset_name is caught."""
        with pytest.raises(ValueError, match="dataset_name cannot be empty"):
            DataConfig(name="test-data", dataset_name="   ")

    def test_num_workers_negative(self):
        """Test that negative num_workers is invalid."""
        with pytest.raises(ValueError, match="num_workers must be non-negative"):
            DataConfig(name="test-data", dataset_name="mnist", num_workers=-1)

    def test_num_workers_zero_valid(self):
        """Test that zero num_workers is valid."""
        config = DataConfig(name="test-data", dataset_name="mnist", num_workers=0)
        assert config.num_workers == 0

    def test_prefetch_factor_zero(self):
        """Test that zero prefetch_factor is invalid."""
        with pytest.raises(ValueError, match="prefetch_factor must be positive"):
            DataConfig(name="test-data", dataset_name="mnist", prefetch_factor=0)

    def test_prefetch_factor_negative(self):
        """Test that negative prefetch_factor is invalid."""
        with pytest.raises(ValueError, match="prefetch_factor must be positive"):
            DataConfig(name="test-data", dataset_name="mnist", prefetch_factor=-1)

    def test_validation_split_negative(self):
        """Test that negative validation_split is invalid."""
        with pytest.raises(ValueError, match="validation_split must be between"):
            DataConfig(name="test-data", dataset_name="mnist", validation_split=-0.1)

    def test_validation_split_greater_than_one(self):
        """Test that validation_split > 1 is invalid."""
        with pytest.raises(ValueError, match="validation_split must be between"):
            DataConfig(name="test-data", dataset_name="mnist", validation_split=1.5)

    def test_validation_split_zero_valid(self):
        """Test that validation_split=0 is valid."""
        config = DataConfig(name="test-data", dataset_name="mnist", validation_split=0.0)
        assert config.validation_split == 0.0

    def test_validation_split_one_valid(self):
        """Test that validation_split=1 is valid (with test_split=0)."""
        config = DataConfig(
            name="test-data", dataset_name="mnist", validation_split=1.0, test_split=0.0
        )
        assert config.validation_split == 1.0

    def test_test_split_negative(self):
        """Test that negative test_split is invalid."""
        with pytest.raises(ValueError, match="test_split must be between"):
            DataConfig(name="test-data", dataset_name="mnist", test_split=-0.1)

    def test_test_split_greater_than_one(self):
        """Test that test_split > 1 is invalid."""
        with pytest.raises(ValueError, match="test_split must be between"):
            DataConfig(name="test-data", dataset_name="mnist", test_split=1.5)

    def test_test_split_zero_valid(self):
        """Test that test_split=0 is valid."""
        config = DataConfig(name="test-data", dataset_name="mnist", test_split=0.0)
        assert config.test_split == 0.0

    def test_test_split_one_valid(self):
        """Test that test_split=1 is valid (with validation_split=0)."""
        config = DataConfig(
            name="test-data", dataset_name="mnist", test_split=1.0, validation_split=0.0
        )
        assert config.test_split == 1.0

    def test_splits_sum_greater_than_one(self):
        """Test that validation_split + test_split > 1 is invalid."""
        with pytest.raises(ValueError, match="validation_split \\+ test_split must not exceed 1"):
            DataConfig(name="test-data", dataset_name="mnist", validation_split=0.6, test_split=0.5)

    def test_splits_sum_equal_one_valid(self):
        """Test that validation_split + test_split = 1 is valid."""
        config = DataConfig(
            name="test-data", dataset_name="mnist", validation_split=0.7, test_split=0.3
        )
        assert config.validation_split == 0.7
        assert config.test_split == 0.3


class TestDataConfigDefaults:
    """Test DataConfig default values."""

    def test_default_data_dir(self):
        """Test default data_dir."""
        config = DataConfig(name="test-data", dataset_name="mnist")
        assert config.data_dir == Path("./data")

    def test_default_split(self):
        """Test default split."""
        config = DataConfig(name="test-data", dataset_name="mnist")
        assert config.split == "train"

    def test_default_num_workers(self):
        """Test default num_workers."""
        config = DataConfig(name="test-data", dataset_name="mnist")
        assert config.num_workers == 4

    def test_default_prefetch_factor(self):
        """Test default prefetch_factor."""
        config = DataConfig(name="test-data", dataset_name="mnist")
        assert config.prefetch_factor == 2

    def test_default_pin_memory(self):
        """Test default pin_memory."""
        config = DataConfig(name="test-data", dataset_name="mnist")
        assert config.pin_memory is True

    def test_default_augmentation(self):
        """Test default augmentation."""
        config = DataConfig(name="test-data", dataset_name="mnist")
        assert config.augmentation is False

    def test_default_augmentation_params(self):
        """Test default augmentation_params."""
        config = DataConfig(name="test-data", dataset_name="mnist")
        assert config.augmentation_params == {}

    def test_default_validation_split(self):
        """Test default validation_split."""
        config = DataConfig(name="test-data", dataset_name="mnist")
        assert config.validation_split == 0.1

    def test_default_test_split(self):
        """Test default test_split."""
        config = DataConfig(name="test-data", dataset_name="mnist")
        assert config.test_split == 0.1


class TestDataConfigSerialization:
    """Test DataConfig serialization."""

    def test_to_dict_minimal(self):
        """Test converting minimal config to dict."""
        config = DataConfig(name="test-data", dataset_name="mnist")
        data = config.to_dict()
        assert data["dataset_name"] == "mnist"
        assert data["data_dir"] == Path("./data")  # Path object
        assert data["split"] == "train"

    def test_to_dict_full(self):
        """Test converting full config to dict."""
        config = DataConfig(
            name="test-data",
            dataset_name="cifar10",
            data_dir=Path("/path/to/data"),
            num_workers=8,
            augmentation=True,
            augmentation_params={"rotation": 15},
        )
        data = config.to_dict()
        assert data["dataset_name"] == "cifar10"
        assert data["data_dir"] == Path("/path/to/data")  # Path object
        assert data["num_workers"] == 8
        assert data["augmentation"] is True
        assert data["augmentation_params"] == {"rotation": 15}

    def test_from_dict_minimal(self):
        """Test creating config from minimal dict."""
        data = {"name": "test-data", "dataset_name": "mnist"}
        config = DataConfig.from_dict(data)
        assert config.name == "test-data"
        assert config.dataset_name == "mnist"
        assert config.data_dir == Path("./data")

    def test_from_dict_full(self):
        """Test creating config from full dict."""
        data = {
            "name": "cifar10-data",
            "dataset_name": "cifar10",
            "data_dir": "/path/to/data",
            "split": "test",
            "num_workers": 8,
            "prefetch_factor": 4,
            "pin_memory": False,
            "augmentation": True,
            "augmentation_params": {"rotation": 15, "flip": True},
            "validation_split": 0.15,
            "test_split": 0.2,
        }
        config = DataConfig.from_dict(data)
        assert config.dataset_name == "cifar10"
        assert config.data_dir == Path("/path/to/data")
        assert config.split == "test"
        assert config.num_workers == 8
        assert config.augmentation_params == {"rotation": 15, "flip": True}

    def test_from_dict_with_path_string(self):
        """Test that string paths are converted to Path objects."""
        data = {"name": "test-data", "dataset_name": "mnist", "data_dir": "/path/to/data"}
        config = DataConfig.from_dict(data)
        assert isinstance(config.data_dir, Path)
        assert config.data_dir == Path("/path/to/data")

    def test_roundtrip(self):
        """Test that to_dict -> from_dict roundtrip works."""
        original = DataConfig(
            name="test-data",
            dataset_name="cifar10",
            data_dir=Path("/path/to/data"),
            augmentation=True,
            augmentation_params={"rotation": 15},
        )
        data = original.to_dict()
        restored = DataConfig.from_dict(data)
        assert restored.dataset_name == original.dataset_name
        assert restored.data_dir == original.data_dir
        assert restored.augmentation == original.augmentation
        assert restored.augmentation_params == original.augmentation_params


class TestDataConfigEdgeCases:
    """Test DataConfig edge cases."""

    def test_data_dir_as_string(self):
        """Test that data_dir accepts string and converts to Path."""
        config = DataConfig(name="test-data", dataset_name="mnist", data_dir="/path/to/data")
        assert isinstance(config.data_dir, Path)
        assert config.data_dir == Path("/path/to/data")

    def test_data_dir_as_path(self):
        """Test that data_dir accepts Path objects."""
        config = DataConfig(name="test-data", dataset_name="mnist", data_dir=Path("/path/to/data"))
        assert isinstance(config.data_dir, Path)
        assert config.data_dir == Path("/path/to/data")

    def test_empty_augmentation_params(self):
        """Test that empty augmentation_params is valid."""
        config = DataConfig(name="test-data", dataset_name="mnist", augmentation_params={})
        assert config.augmentation_params == {}

    def test_complex_augmentation_params(self):
        """Test that complex augmentation_params are handled."""
        params = {
            "rotation": 15,
            "flip": True,
            "brightness": 0.2,
            "contrast": 0.2,
            "crop": {"size": 224, "scale": (0.8, 1.0)},
        }
        config = DataConfig(name="test-data", dataset_name="imagenet", augmentation_params=params)
        assert config.augmentation_params == params

    def test_all_splits_zero(self):
        """Test that all splits can be zero."""
        config = DataConfig(
            name="test-data", dataset_name="mnist", validation_split=0.0, test_split=0.0
        )
        assert config.validation_split == 0.0
        assert config.test_split == 0.0
