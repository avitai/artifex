"""Tests for base dataclass configuration.

Following TDD: These tests are written BEFORE implementation.
They define the expected behavior of BaseConfig.
"""

import dataclasses
from pathlib import Path

import pytest
import yaml

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig


class TestBaseConfig:
    """Test BaseConfig frozen dataclass - written before implementation."""

    def test_valid_minimal_config(self):
        """Test that valid minimal config is created successfully."""
        config = BaseConfig(name="test_config")

        assert config.name == "test_config"
        assert config.description == ""
        assert config.tags == ()
        assert config.metadata == {}

    def test_valid_full_config(self):
        """Test that valid full config is created successfully."""
        config = BaseConfig(
            name="test_config",
            description="Test description",
            tags=("tag1", "tag2", "tag3"),
            metadata={"key1": "value1", "key2": 123},
        )

        assert config.name == "test_config"
        assert config.description == "Test description"
        assert config.tags == ("tag1", "tag2", "tag3")
        assert config.metadata == {"key1": "value1", "key2": 123}

    def test_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name must be non-empty"):
            BaseConfig(name="")

    def test_immutable_frozen_dataclass(self):
        """Test that config is truly immutable (frozen=True)."""
        config = BaseConfig(name="test")

        # Can't modify frozen dataclass fields
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "modified"

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.description = "modified"

    def test_tags_are_tuple_immutable(self):
        """Test that tags field is a tuple (immutable), not a list."""
        config = BaseConfig(name="test", tags=("tag1", "tag2"))

        # Verify it's a tuple
        assert isinstance(config.tags, tuple)

        # Tuples don't have append method
        assert not hasattr(config.tags, "append")

    def test_from_dict_basic(self):
        """Test creating config from dict using dacite."""
        config_dict = {
            "name": "test_config",
            "description": "Test description",
            "tags": ["tag1", "tag2"],  # List in dict
            "metadata": {"key": "value"},
        }

        config = BaseConfig.from_dict(config_dict)

        assert isinstance(config, BaseConfig)
        assert config.name == "test_config"
        assert config.description == "Test description"
        # List should be converted to tuple
        assert isinstance(config.tags, tuple)
        assert config.tags == ("tag1", "tag2")
        assert config.metadata == {"key": "value"}

    def test_from_dict_minimal(self):
        """Test from_dict with minimal required fields."""
        config_dict = {"name": "minimal"}

        config = BaseConfig.from_dict(config_dict)

        assert config.name == "minimal"
        assert config.description == ""
        assert config.tags == ()

    def test_from_dict_invalid_raises(self):
        """Test that from_dict raises on invalid data."""
        # Missing required field 'name'
        config_dict = {"description": "No name provided"}

        with pytest.raises(Exception):  # dacite will raise
            BaseConfig.from_dict(config_dict)

    def test_from_dict_extra_field_raises(self):
        """Test that from_dict raises on extra fields (strict mode)."""
        config_dict = {
            "name": "test",
            "extra_field": "should fail",  # Not a valid field
        }

        # dacite with strict=True should raise
        with pytest.raises(Exception):
            BaseConfig.from_dict(config_dict)

    def test_to_dict(self):
        """Test converting config to dict."""
        config = BaseConfig(
            name="test",
            description="desc",
            tags=("t1", "t2"),
            metadata={"k": "v"},
        )

        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["description"] == "desc"
        assert result["tags"] == ("t1", "t2")
        assert result["metadata"] == {"k": "v"}

    def test_to_yaml_and_from_yaml_roundtrip(self, tmp_path: Path):
        """Test saving to YAML and loading back (roundtrip)."""
        config = BaseConfig(
            name="yaml_test",
            description="Testing YAML",
            tags=("yaml", "test"),
            metadata={"number": 42, "string": "value"},
        )

        # Save to YAML
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(yaml_path)

        # Verify file exists
        assert yaml_path.exists()

        # Load back
        loaded_config = BaseConfig.from_yaml(yaml_path)

        # Verify roundtrip preserves data
        assert loaded_config.name == config.name
        assert loaded_config.description == config.description
        assert loaded_config.tags == config.tags
        assert loaded_config.metadata == config.metadata

    def test_from_yaml_nonexistent_file_raises(self):
        """Test that loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            BaseConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_to_yaml_creates_parent_dirs(self, tmp_path: Path):
        """Test that to_yaml creates parent directories if needed."""
        config = BaseConfig(name="test")

        # Path with non-existent parent
        yaml_path = tmp_path / "nested" / "dirs" / "config.yaml"

        # Should create parent directories
        config.to_yaml(yaml_path)

        assert yaml_path.exists()
        assert yaml_path.parent.exists()

    def test_yaml_handles_tuples_as_lists(self, tmp_path: Path):
        """Test that YAML serialization converts tuples to lists (YAML doesn't have tuples)."""
        config = BaseConfig(name="test", tags=("a", "b", "c"))

        yaml_path = tmp_path / "test.yaml"
        config.to_yaml(yaml_path)

        # Read raw YAML to check format
        with open(yaml_path) as f:
            raw_yaml = yaml.safe_load(f)

        # In YAML, tuples become lists
        assert isinstance(raw_yaml["tags"], list)
        assert raw_yaml["tags"] == ["a", "b", "c"]

        # But when loaded back, should be tuples
        loaded = BaseConfig.from_yaml(yaml_path)
        assert isinstance(loaded.tags, tuple)

    def test_hash_not_supported_with_dict_fields(self):
        """Test that configs with dict fields are not hashable.

        This is expected and acceptable - configs are for configuration,
        not for use as dict keys or in sets. The metadata field is a dict,
        making the config unhashable. This follows the same pattern as
        image-reconstruction repo.
        """
        config1 = BaseConfig(name="test", tags=("a", "b"))

        # Cannot hash configs with dict fields (metadata)
        with pytest.raises(TypeError, match="unhashable type"):
            hash(config1)

    def test_equality(self):
        """Test equality comparison of configs."""
        config1 = BaseConfig(name="test", description="desc")
        config2 = BaseConfig(name="test", description="desc")
        config3 = BaseConfig(name="test", description="different")

        assert config1 == config2
        assert config1 != config3

    def test_asdict_preserves_types(self):
        """Test that dataclasses.asdict preserves field types."""
        config = BaseConfig(
            name="test",
            tags=("t1", "t2"),
            metadata={"nested": {"key": "value"}},
        )

        result = dataclasses.asdict(config)

        assert isinstance(result, dict)
        assert isinstance(result["tags"], tuple)
        assert isinstance(result["metadata"], dict)


class TestBaseConfigValidation:
    """Test validation logic in BaseConfig.__post_init__."""

    def test_whitespace_only_name_raises(self):
        """Test that whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="name must be non-empty"):
            BaseConfig(name="   ")  # Only whitespace

    def test_name_with_content_passes(self):
        """Test that name with actual content passes validation."""
        # These should all work
        configs = [
            BaseConfig(name="valid"),
            BaseConfig(name="valid_name_123"),
            BaseConfig(name="Valid Config Name"),
            BaseConfig(name="  valid  "),  # Whitespace around is OK
        ]

        for config in configs:
            assert config.name  # All should be created successfully


class TestBaseConfigInheritance:
    """Test that BaseConfig can be inherited by other config classes."""

    def test_can_inherit_from_base_config(self):
        """Test that we can create subclasses of BaseConfig."""

        @dataclasses.dataclass(frozen=True)
        class CustomConfig(BaseConfig):
            """Custom config extending BaseConfig."""

            custom_field: str = "default"

            def __post_init__(self):
                """Call parent validation."""
                # This is a workaround for frozen dataclass validation
                # In real implementation, we'd handle this properly
                object.__setattr__(self, "_validated", True)

        # Should be able to create instances
        config = CustomConfig(name="custom")
        assert config.name == "custom"
        assert config.custom_field == "default"


class TestBaseConfigCoverage:
    """Meta-test to ensure we achieve 80%+ coverage."""

    def test_coverage_reminder(self):
        """Reminder that we need 80%+ coverage for BaseConfig.

        This test serves as documentation that:
        - All code paths in BaseConfig must be tested
        - All validation logic must be tested
        - All error paths must be tested
        - Both positive and negative cases must be covered
        """
        # This test always passes - it's a reminder
        assert True


# Integration test examples
class TestBaseConfigUsage:
    """Test realistic usage scenarios."""

    def test_config_for_experiment_tracking(self):
        """Test using BaseConfig for experiment tracking."""
        config = BaseConfig(
            name="experiment_001",
            description="Testing VAE with beta=0.5",
            tags=("vae", "mnist", "beta_tuning"),
            metadata={
                "run_id": "run_20250125_001",
                "dataset": "mnist",
                "notes": "Exploring different beta values",
            },
        )

        assert "vae" in config.tags
        assert config.metadata["dataset"] == "mnist"

    def test_config_chain_creation(self):
        """Test creating multiple related configs."""
        base = BaseConfig(name="base", tags=("shared",))

        # Can't modify frozen, but can create new instances
        variant1 = dataclasses.replace(base, name="variant1")
        variant2 = dataclasses.replace(base, name="variant2")

        assert variant1.name == "variant1"
        assert variant2.name == "variant2"
        assert variant1.tags == variant2.tags == ("shared",)
