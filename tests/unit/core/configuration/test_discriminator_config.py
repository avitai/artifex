"""Tests for DiscriminatorConfig dataclass.

This test module follows TDD principles to define the DiscriminatorConfig API
before implementation. Tests are organized into:
1. TestBasics - Instantiation, frozen behavior, equality
2. TestInheritance - Verify inheritance from BaseNetworkConfig
3. TestValidation - Field validation rules
4. TestDefaults - Default value verification
5. TestSerialization - to_dict, from_dict, roundtrip
6. TestEdgeCases - Boundary values and special cases
"""

import copy

import pytest

from artifex.generative_models.core.configuration import DiscriminatorConfig


class TestDiscriminatorConfigBasics:
    """Test basic DiscriminatorConfig instantiation and properties."""

    def test_create_minimal(self) -> None:
        """Test creating DiscriminatorConfig with minimal required fields."""
        config = DiscriminatorConfig(
            name="test_disc",
            hidden_dims=(128, 64),
            activation="relu",
            input_shape=(1, 28, 28),
        )
        assert config.name == "test_disc"
        assert config.hidden_dims == (128, 64)
        assert config.activation == "relu"
        assert config.input_shape == (1, 28, 28)
        assert config.leaky_relu_slope == 0.2  # default
        assert config.use_spectral_norm is False  # default

    def test_create_with_all_fields(self) -> None:
        """Test creating DiscriminatorConfig with all fields specified."""
        config = DiscriminatorConfig(
            # From BaseConfig
            name="full_disc",
            description="Full discriminator config",
            tags=["image", "gan"],
            metadata={"author": "test"},
            # From BaseNetworkConfig
            hidden_dims=(256, 128, 64),
            activation="leaky_relu",
            batch_norm=True,
            dropout_rate=0.3,
            # DiscriminatorConfig-specific
            input_shape=(3, 64, 64),
            leaky_relu_slope=0.3,
            use_spectral_norm=True,
        )
        # Verify all fields
        assert config.name == "full_disc"
        assert config.description == "Full discriminator config"
        assert config.tags == ["image", "gan"]
        assert config.metadata == {"author": "test"}
        assert config.hidden_dims == (256, 128, 64)
        assert config.activation == "leaky_relu"
        assert config.batch_norm is True
        assert config.dropout_rate == 0.3
        assert config.input_shape == (3, 64, 64)
        assert config.leaky_relu_slope == 0.3
        assert config.use_spectral_norm is True

    def test_frozen_behavior(self) -> None:
        """Test that DiscriminatorConfig is immutable (frozen)."""
        config = DiscriminatorConfig(
            name="frozen_test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(1, 28, 28),
        )
        with pytest.raises((AttributeError, TypeError)):
            config.leaky_relu_slope = 0.5  # type: ignore

    def test_equality(self) -> None:
        """Test that two configs with same values are equal."""
        config1 = DiscriminatorConfig(
            name="test", hidden_dims=(128,), activation="relu", input_shape=(1, 28, 28)
        )
        config2 = DiscriminatorConfig(
            name="test", hidden_dims=(128,), activation="relu", input_shape=(1, 28, 28)
        )
        assert config1 == config2

    def test_inequality(self) -> None:
        """Test that two configs with different values are not equal."""
        config1 = DiscriminatorConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(1, 28, 28),
            leaky_relu_slope=0.2,
        )
        config2 = DiscriminatorConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(1, 28, 28),
            leaky_relu_slope=0.3,
        )
        assert config1 != config2

    def test_hash(self) -> None:
        """Test that DiscriminatorConfig is NOT hashable due to metadata dict field."""
        config = DiscriminatorConfig(
            name="test", hidden_dims=(128,), activation="relu", input_shape=(1, 28, 28)
        )
        # Cannot be hashed due to dict field (metadata)
        with pytest.raises(TypeError, match="unhashable type"):
            hash(config)


class TestDiscriminatorConfigInheritance:
    """Test that DiscriminatorConfig properly inherits from BaseNetworkConfig."""

    def test_inherits_base_config_fields(self) -> None:
        """Test that DiscriminatorConfig has BaseConfig fields."""
        config = DiscriminatorConfig(
            name="test",
            description="Test config",
            tags=["test"],
            metadata={"key": "value"},
            hidden_dims=(128,),
            activation="relu",
            input_shape=(1, 28, 28),
        )
        # BaseConfig fields
        assert config.name == "test"
        assert config.description == "Test config"
        assert config.tags == ["test"]
        assert config.metadata == {"key": "value"}

    def test_inherits_base_network_config_fields(self) -> None:
        """Test that DiscriminatorConfig has BaseNetworkConfig fields."""
        config = DiscriminatorConfig(
            name="test",
            hidden_dims=(256, 128),
            activation="leaky_relu",
            batch_norm=True,
            dropout_rate=0.3,
            input_shape=(3, 64, 64),
        )
        # BaseNetworkConfig fields
        assert config.hidden_dims == (256, 128)
        assert config.activation == "leaky_relu"
        assert config.batch_norm is True
        assert config.dropout_rate == 0.3


class TestDiscriminatorConfigValidation:
    """Test DiscriminatorConfig validation rules."""

    def test_validate_input_shape_required(self) -> None:
        """Test that input_shape cannot be empty."""
        with pytest.raises(ValueError, match="input_shape is required"):
            DiscriminatorConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                input_shape=(),  # Empty tuple
            )

    def test_validate_input_shape_positive(self) -> None:
        """Test that input_shape must contain positive integers."""
        with pytest.raises(ValueError, match="positive"):
            DiscriminatorConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                input_shape=(1, 0, 28),  # Contains zero
            )

        with pytest.raises(ValueError, match="positive"):
            DiscriminatorConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                input_shape=(1, -28, 28),  # Contains negative
            )

    def test_validate_leaky_relu_slope_positive(self) -> None:
        """Test that leaky_relu_slope must be positive."""
        with pytest.raises(ValueError, match="leaky_relu_slope must be positive"):
            DiscriminatorConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                input_shape=(1, 28, 28),
                leaky_relu_slope=0.0,  # Zero not allowed
            )

        with pytest.raises(ValueError, match="leaky_relu_slope must be positive"):
            DiscriminatorConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                input_shape=(1, 28, 28),
                leaky_relu_slope=-0.2,  # Negative not allowed
            )

    def test_validate_inherited_fields(self) -> None:
        """Test that inherited validation rules still apply."""
        # Test hidden_dims validation (from BaseNetworkConfig)
        with pytest.raises(ValueError, match="hidden_dims"):
            DiscriminatorConfig(
                name="test",
                hidden_dims=(),  # Empty tuple
                activation="relu",
                input_shape=(1, 28, 28),
            )

        # Test activation validation (from BaseNetworkConfig)
        with pytest.raises(ValueError, match="Unknown activation"):
            DiscriminatorConfig(
                name="test",
                hidden_dims=(128,),
                activation="invalid_activation",
                input_shape=(1, 28, 28),
            )

        # Test dropout_rate validation (from BaseNetworkConfig)
        with pytest.raises(ValueError, match="dropout_rate"):
            DiscriminatorConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                input_shape=(1, 28, 28),
                dropout_rate=1.5,  # > 1.0
            )


class TestDiscriminatorConfigDefaults:
    """Test DiscriminatorConfig default values."""

    def test_default_leaky_relu_slope(self) -> None:
        """Test default leaky_relu_slope value."""
        config = DiscriminatorConfig(
            name="test", hidden_dims=(128,), activation="relu", input_shape=(1, 28, 28)
        )
        assert config.leaky_relu_slope == 0.2

    def test_default_use_spectral_norm(self) -> None:
        """Test default use_spectral_norm value."""
        config = DiscriminatorConfig(
            name="test", hidden_dims=(128,), activation="relu", input_shape=(1, 28, 28)
        )
        assert config.use_spectral_norm is False

    def test_default_batch_norm(self) -> None:
        """Test default batch_norm value from BaseNetworkConfig."""
        config = DiscriminatorConfig(
            name="test", hidden_dims=(128,), activation="relu", input_shape=(1, 28, 28)
        )
        assert config.batch_norm is False

    def test_default_dropout_rate(self) -> None:
        """Test default dropout_rate value from BaseNetworkConfig."""
        config = DiscriminatorConfig(
            name="test", hidden_dims=(128,), activation="relu", input_shape=(1, 28, 28)
        )
        assert config.dropout_rate == 0.0


class TestDiscriminatorConfigSerialization:
    """Test DiscriminatorConfig serialization (to_dict, from_dict)."""

    def test_to_dict_minimal(self) -> None:
        """Test to_dict with minimal config."""
        config = DiscriminatorConfig(
            name="test", hidden_dims=(128, 64), activation="relu", input_shape=(1, 28, 28)
        )
        config_dict = config.to_dict()

        assert config_dict["name"] == "test"
        # Note: to_dict preserves tuples, doesn't convert to lists
        assert config_dict["hidden_dims"] == (128, 64)
        assert config_dict["activation"] == "relu"
        assert config_dict["input_shape"] == (1, 28, 28)
        assert config_dict["leaky_relu_slope"] == 0.2
        assert config_dict["use_spectral_norm"] is False

    def test_to_dict_full(self) -> None:
        """Test to_dict with all fields specified."""
        config = DiscriminatorConfig(
            name="full_disc",
            description="Full config",
            tags=("image",),  # Use tuple, not list
            metadata={"author": "test"},
            hidden_dims=(256, 128),
            activation="leaky_relu",
            batch_norm=True,
            dropout_rate=0.3,
            input_shape=(3, 64, 64),
            leaky_relu_slope=0.3,
            use_spectral_norm=True,
        )
        config_dict = config.to_dict()

        assert config_dict["name"] == "full_disc"
        assert config_dict["description"] == "Full config"
        assert config_dict["tags"] == ("image",)
        assert config_dict["metadata"] == {"author": "test"}
        # Note: to_dict preserves tuples
        assert config_dict["hidden_dims"] == (256, 128)
        assert config_dict["activation"] == "leaky_relu"
        assert config_dict["batch_norm"] is True
        assert config_dict["dropout_rate"] == 0.3
        assert config_dict["input_shape"] == (3, 64, 64)
        assert config_dict["leaky_relu_slope"] == 0.3
        assert config_dict["use_spectral_norm"] is True

    def test_from_dict_minimal(self) -> None:
        """Test from_dict with minimal config."""
        config_dict = {
            "name": "test",
            "hidden_dims": [128, 64],
            "activation": "relu",
            "input_shape": [1, 28, 28],
        }
        config = DiscriminatorConfig.from_dict(config_dict)

        assert config.name == "test"
        assert config.hidden_dims == (128, 64)  # list → tuple
        assert config.activation == "relu"
        assert config.input_shape == (1, 28, 28)  # list → tuple
        assert config.leaky_relu_slope == 0.2  # default
        assert config.use_spectral_norm is False  # default

    def test_from_dict_full(self) -> None:
        """Test from_dict with all fields."""
        config_dict = {
            "name": "full_disc",
            "description": "Full config",
            "tags": ["image"],
            "metadata": {"author": "test"},
            "hidden_dims": [256, 128],
            "activation": "leaky_relu",
            "batch_norm": True,
            "dropout_rate": 0.3,
            "input_shape": [3, 64, 64],
            "leaky_relu_slope": 0.3,
            "use_spectral_norm": True,
        }
        config = DiscriminatorConfig.from_dict(config_dict)

        assert config.name == "full_disc"
        assert config.description == "Full config"
        # Note: tags list gets converted to tuple during __post_init__
        assert config.tags == ("image",)
        assert config.metadata == {"author": "test"}
        # Note: from_dict converts lists to tuples
        assert config.hidden_dims == (256, 128)
        assert config.activation == "leaky_relu"
        assert config.batch_norm is True
        assert config.dropout_rate == 0.3
        assert config.input_shape == (3, 64, 64)
        assert config.leaky_relu_slope == 0.3
        assert config.use_spectral_norm is True

    def test_from_dict_with_path_conversion(self) -> None:
        """Test from_dict does NOT automatically convert string paths in metadata.

        Path conversion only happens for dedicated Path-type fields, not for
        arbitrary strings in metadata dicts.
        """
        config_dict = {
            "name": "test",
            "hidden_dims": [128],
            "activation": "relu",
            "input_shape": [1, 28, 28],
            "metadata": {"checkpoint_path": "/path/to/checkpoint"},
        }
        config = DiscriminatorConfig.from_dict(config_dict)
        # Metadata values remain as-is (strings are not auto-converted to Path)
        assert isinstance(config.metadata["checkpoint_path"], str)
        assert config.metadata["checkpoint_path"] == "/path/to/checkpoint"

    def test_roundtrip_serialization(self) -> None:
        """Test that to_dict → from_dict preserves config."""
        original = DiscriminatorConfig(
            name="roundtrip_test",
            description="Test roundtrip",
            tags=("test", "serialization"),  # Use tuple for direct instantiation
            metadata={"version": "1.0"},
            hidden_dims=(256, 128, 64),
            activation="leaky_relu",
            batch_norm=True,
            dropout_rate=0.2,
            input_shape=(3, 64, 64),
            leaky_relu_slope=0.25,
            use_spectral_norm=True,
        )

        config_dict = original.to_dict()
        restored = DiscriminatorConfig.from_dict(config_dict)

        # Full equality check
        assert restored == original

        # Verify key fields explicitly
        assert restored.name == original.name
        assert restored.tags == original.tags
        assert restored.hidden_dims == original.hidden_dims
        assert restored.activation == original.activation
        assert restored.input_shape == original.input_shape
        assert restored.leaky_relu_slope == original.leaky_relu_slope
        assert restored.use_spectral_norm == original.use_spectral_norm


class TestDiscriminatorConfigEdgeCases:
    """Test DiscriminatorConfig edge cases and boundary values."""

    def test_single_hidden_dim(self) -> None:
        """Test with single hidden dimension."""
        config = DiscriminatorConfig(
            name="test", hidden_dims=(128,), activation="relu", input_shape=(1, 28, 28)
        )
        assert config.hidden_dims == (128,)

    def test_many_hidden_dims(self) -> None:
        """Test with many hidden dimensions."""
        dims = (512, 256, 128, 64, 32, 16)
        config = DiscriminatorConfig(
            name="test", hidden_dims=dims, activation="relu", input_shape=(1, 28, 28)
        )
        assert config.hidden_dims == dims

    def test_1d_input_shape(self) -> None:
        """Test with 1D input shape."""
        config = DiscriminatorConfig(
            name="test", hidden_dims=(128,), activation="relu", input_shape=(784,)
        )
        assert config.input_shape == (784,)

    def test_3d_input_shape(self) -> None:
        """Test with 3D input shape (channels, height, width)."""
        config = DiscriminatorConfig(
            name="test", hidden_dims=(128,), activation="relu", input_shape=(3, 64, 64)
        )
        assert config.input_shape == (3, 64, 64)

    def test_4d_input_shape(self) -> None:
        """Test with 4D input shape (for video or batch dimensions)."""
        config = DiscriminatorConfig(
            name="test", hidden_dims=(128,), activation="relu", input_shape=(16, 3, 64, 64)
        )
        assert config.input_shape == (16, 3, 64, 64)

    def test_very_small_leaky_relu_slope(self) -> None:
        """Test with very small leaky_relu_slope."""
        config = DiscriminatorConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(1, 28, 28),
            leaky_relu_slope=0.001,
        )
        assert config.leaky_relu_slope == 0.001

    def test_large_leaky_relu_slope(self) -> None:
        """Test with large leaky_relu_slope."""
        config = DiscriminatorConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(1, 28, 28),
            leaky_relu_slope=0.9,
        )
        assert config.leaky_relu_slope == 0.9

    def test_dropout_rate_zero(self) -> None:
        """Test with dropout_rate of 0.0."""
        config = DiscriminatorConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(1, 28, 28),
            dropout_rate=0.0,
        )
        assert config.dropout_rate == 0.0

    def test_dropout_rate_one(self) -> None:
        """Test with dropout_rate of 1.0."""
        config = DiscriminatorConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(1, 28, 28),
            dropout_rate=1.0,
        )
        assert config.dropout_rate == 1.0

    def test_copy_works(self) -> None:
        """Test that configs can be copied."""
        original = DiscriminatorConfig(
            name="test", hidden_dims=(128,), activation="relu", input_shape=(1, 28, 28)
        )
        copied = copy.copy(original)
        assert copied == original
        assert copied is not original

    def test_deepcopy_works(self) -> None:
        """Test that configs can be deep copied."""
        original = DiscriminatorConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(1, 28, 28),
            metadata={"nested": {"key": "value"}},
        )
        copied = copy.deepcopy(original)
        assert copied == original
        assert copied is not original
        assert copied.metadata is not original.metadata
