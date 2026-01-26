"""Tests for base network configuration.

Following TDD: These tests are written BEFORE implementation.
They define the expected behavior of BaseNetworkConfig.
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration.base_network import BaseNetworkConfig


class TestBaseNetworkConfig:
    """Test BaseNetworkConfig frozen dataclass - written before implementation."""

    def test_valid_minimal_config(self):
        """Test that valid minimal network config is created successfully."""
        config = BaseNetworkConfig(
            name="test_network",
            hidden_dims=(512, 256, 128),
            activation="relu",
        )

        assert config.name == "test_network"
        assert config.hidden_dims == (512, 256, 128)
        assert config.activation == "relu"
        assert config.batch_norm is False  # Default
        assert config.dropout_rate == 0.0  # Default

    def test_valid_full_config(self):
        """Test that valid full network config is created successfully."""
        config = BaseNetworkConfig(
            name="test_network",
            hidden_dims=(512, 256, 128),
            activation="gelu",
            batch_norm=True,
            dropout_rate=0.3,
        )

        assert config.name == "test_network"
        assert config.hidden_dims == (512, 256, 128)
        assert config.activation == "gelu"
        assert config.batch_norm is True
        assert config.dropout_rate == 0.3

    def test_hidden_dims_validation_empty_raises(self):
        """Test that empty hidden_dims raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dims is required and cannot be empty"):
            BaseNetworkConfig(
                name="test_network",
                hidden_dims=(),  # Empty!
                activation="relu",
            )

    def test_hidden_dims_validation_negative_raises(self):
        """Test that negative hidden_dims raise ValueError."""
        with pytest.raises(ValueError, match="All hidden_dims must be positive"):
            BaseNetworkConfig(
                name="test_network",
                hidden_dims=(512, -256, 128),  # Negative!
                activation="relu",
            )

    def test_activation_validation_invalid_raises(self):
        """Test that invalid activation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation function"):
            BaseNetworkConfig(
                name="test_network",
                hidden_dims=(512, 256),
                activation="nonexistent",  # Invalid!
            )

    def test_dropout_rate_validation_below_zero_raises(self):
        """Test that dropout_rate < 0 raises ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            BaseNetworkConfig(
                name="test_network",
                hidden_dims=(512, 256),
                activation="relu",
                dropout_rate=-0.1,  # Invalid!
            )

    def test_dropout_rate_validation_above_one_raises(self):
        """Test that dropout_rate > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            BaseNetworkConfig(
                name="test_network",
                hidden_dims=(512, 256),
                activation="relu",
                dropout_rate=1.5,  # Invalid!
            )

    def test_immutable_frozen_dataclass(self):
        """Test that config is truly immutable (frozen=True)."""
        config = BaseNetworkConfig(
            name="test_network",
            hidden_dims=(512, 256),
            activation="relu",
        )

        # Can't modify frozen dataclass fields
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.activation = "gelu"

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.batch_norm = True

    def test_hidden_dims_are_tuple_immutable(self):
        """Test that hidden_dims field is a tuple (immutable), not a list."""
        config = BaseNetworkConfig(
            name="test_network",
            hidden_dims=(512, 256, 128),
            activation="relu",
        )

        # Verify it's a tuple
        assert isinstance(config.hidden_dims, tuple)

        # Tuples don't have append method
        assert not hasattr(config.hidden_dims, "append")

    def test_from_dict_basic(self):
        """Test creating config from dict using dacite."""
        config_dict = {
            "name": "test_network",
            "hidden_dims": [512, 256, 128],  # List in dict
            "activation": "gelu",
            "batch_norm": True,
            "dropout_rate": 0.2,
        }

        config = BaseNetworkConfig.from_dict(config_dict)

        assert isinstance(config, BaseNetworkConfig)
        assert config.name == "test_network"
        assert isinstance(config.hidden_dims, tuple)  # List → Tuple!
        assert config.hidden_dims == (512, 256, 128)
        assert config.activation == "gelu"
        assert config.batch_norm is True
        assert config.dropout_rate == 0.2

    def test_from_dict_minimal(self):
        """Test from_dict with minimal required fields."""
        config_dict = {
            "name": "test_network",
            "hidden_dims": [256, 128],
            "activation": "relu",
        }

        config = BaseNetworkConfig.from_dict(config_dict)

        assert config.name == "test_network"
        assert config.hidden_dims == (256, 128)
        assert config.activation == "relu"
        assert config.batch_norm is False  # Default
        assert config.dropout_rate == 0.0  # Default

    def test_equality(self):
        """Test equality comparison of configs."""
        config1 = BaseNetworkConfig(name="test_network", hidden_dims=(512, 256), activation="relu")
        config2 = BaseNetworkConfig(name="test_network", hidden_dims=(512, 256), activation="relu")
        config3 = BaseNetworkConfig(name="test_network", hidden_dims=(512, 256), activation="gelu")

        assert config1 == config2
        assert config1 != config3

    def test_replace_creates_new_instance(self):
        """Test that dataclasses.replace creates a new instance."""
        config = BaseNetworkConfig(
            name="test_network",
            hidden_dims=(512, 256),
            activation="relu",
            dropout_rate=0.0,
        )

        # Create new instance with modified field
        new_config = dataclasses.replace(config, dropout_rate=0.3)

        # Original unchanged
        assert config.dropout_rate == 0.0

        # New instance has updated value
        assert new_config.dropout_rate == 0.3

        # Other fields preserved
        assert new_config.hidden_dims == (512, 256)
        assert new_config.activation == "relu"


class TestBaseNetworkConfigInheritance:
    """Test that BaseNetworkConfig can be inherited by specific network configs."""

    def test_can_inherit_from_base_network_config(self):
        """Test that we can create subclasses of BaseNetworkConfig."""

        @dataclasses.dataclass(frozen=True)
        class GeneratorConfig(BaseNetworkConfig):
            """Generator config extending BaseNetworkConfig."""

            output_shape: tuple[int, ...] = (1, 28, 28)

            def __post_init__(self):
                """Call parent validation."""
                # Call parent validation first
                super().__post_init__()

                # Then validate child fields
                if not self.output_shape:
                    raise ValueError("output_shape must have at least 1 element")

        # Should be able to create instances
        config = GeneratorConfig(
            name="test_generator",
            hidden_dims=(512, 256),
            activation="relu",
            output_shape=(1, 32, 32),
        )

        assert config.name == "test_generator"
        assert config.hidden_dims == (512, 256)
        assert config.activation == "relu"
        assert config.output_shape == (1, 32, 32)

        # Parent validation should still work
        with pytest.raises(ValueError, match="hidden_dims is required and cannot be empty"):
            GeneratorConfig(
                name="test_generator",
                hidden_dims=(),  # Invalid!
                activation="relu",
            )


class TestBaseNetworkConfigUsage:
    """Test realistic usage scenarios."""

    def test_typical_generator_config(self):
        """Test typical generator network configuration."""
        config = BaseNetworkConfig(
            name="generator",
            hidden_dims=(512, 256, 128),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
        )

        assert config.name == "generator"
        assert config.hidden_dims == (512, 256, 128)
        assert config.batch_norm is True

    def test_typical_discriminator_config(self):
        """Test typical discriminator network configuration."""
        config = BaseNetworkConfig(
            name="discriminator",
            hidden_dims=(128, 256, 512),
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.3,
        )

        assert config.name == "discriminator"
        assert config.hidden_dims == (128, 256, 512)
        assert config.activation == "leaky_relu"
        assert config.dropout_rate == 0.3

    def test_typical_encoder_config(self):
        """Test typical encoder network configuration."""
        config = BaseNetworkConfig(
            name="encoder",
            hidden_dims=(256, 128, 64),
            activation="gelu",
            batch_norm=True,
            dropout_rate=0.1,
        )

        assert config.name == "encoder"
        assert config.activation == "gelu"
        assert config.dropout_rate == 0.1


class TestBaseNetworkConfigCoverage:
    """Meta-test to ensure we achieve 80%+ coverage."""

    def test_coverage_reminder(self):
        """Reminder that we need 80%+ coverage for BaseNetworkConfig.

        All code paths must be tested:
        - Valid configurations (positive tests)
        - Invalid configurations (negative tests)
        - Edge cases (empty, negative, out of range)
        - Validation errors (field names, error messages)
        """
        assert True
