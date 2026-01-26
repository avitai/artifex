"""Tests for EncoderConfig dataclass.

This test module follows TDD principles to define the EncoderConfig API
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

from artifex.generative_models.core.configuration import EncoderConfig


class TestEncoderConfigBasics:
    """Test basic EncoderConfig instantiation and properties."""

    def test_create_minimal(self) -> None:
        """Test creating EncoderConfig with minimal required fields."""
        config = EncoderConfig(
            name="test_encoder",
            hidden_dims=(256, 128),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=64,
        )
        assert config.name == "test_encoder"
        assert config.hidden_dims == (256, 128)
        assert config.activation == "relu"
        assert config.input_shape == (28, 28, 1)
        assert config.latent_dim == 64
        assert config.use_batch_norm is True  # default
        assert config.use_layer_norm is False  # default

    def test_create_with_all_fields(self) -> None:
        """Test creating EncoderConfig with all fields specified."""
        config = EncoderConfig(
            # From BaseConfig
            name="full_encoder",
            description="Full encoder config",
            tags=("vision", "vae"),
            metadata={"author": "test"},
            # From BaseNetworkConfig
            hidden_dims=(512, 256, 128),
            activation="elu",
            batch_norm=True,
            dropout_rate=0.2,
            # EncoderConfig-specific
            input_shape=(64, 64, 3),
            latent_dim=128,
            use_batch_norm=False,
            use_layer_norm=True,
        )
        # Verify all fields
        assert config.name == "full_encoder"
        assert config.description == "Full encoder config"
        assert config.tags == ("vision", "vae")
        assert config.metadata == {"author": "test"}
        assert config.hidden_dims == (512, 256, 128)
        assert config.activation == "elu"
        assert config.batch_norm is True
        assert config.dropout_rate == 0.2
        assert config.input_shape == (64, 64, 3)
        assert config.latent_dim == 128
        assert config.use_batch_norm is False
        assert config.use_layer_norm is True

    def test_frozen_behavior(self) -> None:
        """Test that EncoderConfig is immutable (frozen)."""
        config = EncoderConfig(
            name="frozen_test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
        )
        with pytest.raises((AttributeError, TypeError)):
            config.latent_dim = 64  # type: ignore

    def test_equality(self) -> None:
        """Test that two configs with same values are equal."""
        config1 = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
        )
        config2 = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
        )
        assert config1 == config2

    def test_inequality(self) -> None:
        """Test that two configs with different values are not equal."""
        config1 = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
        )
        config2 = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=64,
        )
        assert config1 != config2

    def test_hash(self) -> None:
        """Test that EncoderConfig is NOT hashable due to metadata dict field."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
        )
        # Cannot be hashed due to dict field (metadata)
        with pytest.raises(TypeError, match="unhashable type"):
            hash(config)


class TestEncoderConfigInheritance:
    """Test that EncoderConfig properly inherits from BaseNetworkConfig."""

    def test_inherits_base_config_fields(self) -> None:
        """Test that EncoderConfig has BaseConfig fields."""
        config = EncoderConfig(
            name="test",
            description="Test config",
            tags=("test",),
            metadata={"key": "value"},
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
        )
        # BaseConfig fields
        assert config.name == "test"
        assert config.description == "Test config"
        assert config.tags == ("test",)
        assert config.metadata == {"key": "value"}

    def test_inherits_base_network_config_fields(self) -> None:
        """Test that EncoderConfig has BaseNetworkConfig fields."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(256, 128),
            activation="elu",
            batch_norm=True,
            dropout_rate=0.3,
            input_shape=(64, 64, 3),
            latent_dim=128,
        )
        # BaseNetworkConfig fields
        assert config.hidden_dims == (256, 128)
        assert config.activation == "elu"
        assert config.batch_norm is True
        assert config.dropout_rate == 0.3


class TestEncoderConfigValidation:
    """Test EncoderConfig validation rules."""

    def test_validate_input_shape_required(self) -> None:
        """Test that input_shape cannot be empty."""
        with pytest.raises(ValueError, match="input_shape is required"):
            EncoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                input_shape=(),  # Empty tuple
                latent_dim=32,
            )

    def test_validate_input_shape_positive(self) -> None:
        """Test that input_shape must contain positive integers."""
        with pytest.raises(ValueError, match="positive"):
            EncoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                input_shape=(0, 28, 28),  # Contains zero
                latent_dim=32,
            )

        with pytest.raises(ValueError, match="positive"):
            EncoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                input_shape=(-1, 28, 28),  # Contains negative
                latent_dim=32,
            )

    def test_validate_latent_dim_positive(self) -> None:
        """Test that latent_dim must be positive."""
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            EncoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                input_shape=(28, 28, 1),
                latent_dim=0,
            )

        with pytest.raises(ValueError, match="latent_dim must be positive"):
            EncoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                input_shape=(28, 28, 1),
                latent_dim=-10,
            )

    def test_validate_inherited_fields(self) -> None:
        """Test that inherited validation rules still apply."""
        # Test hidden_dims validation (from BaseNetworkConfig)
        with pytest.raises(ValueError, match="hidden_dims"):
            EncoderConfig(
                name="test",
                hidden_dims=(),  # Empty tuple
                activation="relu",
                input_shape=(28, 28, 1),
                latent_dim=32,
            )

        # Test activation validation (from BaseNetworkConfig)
        with pytest.raises(ValueError, match="Unknown activation"):
            EncoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="invalid_activation",
                input_shape=(28, 28, 1),
                latent_dim=32,
            )

        # Test dropout_rate validation (from BaseNetworkConfig)
        with pytest.raises(ValueError, match="dropout_rate"):
            EncoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                input_shape=(28, 28, 1),
                latent_dim=32,
                dropout_rate=1.5,  # > 1.0
            )


class TestEncoderConfigDefaults:
    """Test EncoderConfig default values."""

    def test_default_use_batch_norm(self) -> None:
        """Test default use_batch_norm value."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
        )
        assert config.use_batch_norm is True

    def test_default_use_layer_norm(self) -> None:
        """Test default use_layer_norm value."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
        )
        assert config.use_layer_norm is False

    def test_default_batch_norm_from_base(self) -> None:
        """Test default batch_norm value from BaseNetworkConfig."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
        )
        assert config.batch_norm is False

    def test_default_dropout_rate(self) -> None:
        """Test default dropout_rate value from BaseNetworkConfig."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
        )
        assert config.dropout_rate == 0.0


class TestEncoderConfigSerialization:
    """Test EncoderConfig serialization (to_dict, from_dict)."""

    def test_to_dict_minimal(self) -> None:
        """Test to_dict with minimal config."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(256, 128),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=64,
        )
        data = config.to_dict()

        assert data["name"] == "test"
        assert data["hidden_dims"] == (256, 128)
        assert data["activation"] == "relu"
        assert data["input_shape"] == (28, 28, 1)
        assert data["latent_dim"] == 64
        assert data["use_batch_norm"] is True
        assert data["use_layer_norm"] is False

    def test_to_dict_full(self) -> None:
        """Test to_dict with all fields specified."""
        config = EncoderConfig(
            name="full_encoder",
            description="Full config",
            tags=("vision",),
            metadata={"author": "test"},
            hidden_dims=(512, 256),
            activation="elu",
            batch_norm=True,
            dropout_rate=0.2,
            input_shape=(64, 64, 3),
            latent_dim=128,
            use_batch_norm=False,
            use_layer_norm=True,
        )
        data = config.to_dict()

        assert data["name"] == "full_encoder"
        assert data["description"] == "Full config"
        assert data["tags"] == ("vision",)
        assert data["metadata"] == {"author": "test"}
        assert data["hidden_dims"] == (512, 256)
        assert data["activation"] == "elu"
        assert data["batch_norm"] is True
        assert data["dropout_rate"] == 0.2
        assert data["input_shape"] == (64, 64, 3)
        assert data["latent_dim"] == 128
        assert data["use_batch_norm"] is False
        assert data["use_layer_norm"] is True

    def test_from_dict_minimal(self) -> None:
        """Test from_dict with minimal config."""
        config_dict = {
            "name": "test",
            "hidden_dims": [256, 128],
            "activation": "relu",
            "input_shape": [28, 28, 1],
            "latent_dim": 64,
        }
        config = EncoderConfig.from_dict(config_dict)

        assert config.name == "test"
        assert config.hidden_dims == (256, 128)  # list → tuple
        assert config.activation == "relu"
        assert config.input_shape == (28, 28, 1)  # list → tuple
        assert config.latent_dim == 64
        assert config.use_batch_norm is True  # default
        assert config.use_layer_norm is False  # default

    def test_from_dict_full(self) -> None:
        """Test from_dict with all fields."""
        config_dict = {
            "name": "full_encoder",
            "description": "Full config",
            "tags": ["vision"],
            "metadata": {"author": "test"},
            "hidden_dims": [512, 256],
            "activation": "elu",
            "batch_norm": True,
            "dropout_rate": 0.2,
            "input_shape": [64, 64, 3],
            "latent_dim": 128,
            "use_batch_norm": False,
            "use_layer_norm": True,
        }
        config = EncoderConfig.from_dict(config_dict)

        assert config.name == "full_encoder"
        assert config.description == "Full config"
        assert config.tags == ("vision",)  # list → tuple
        assert config.metadata == {"author": "test"}
        assert config.hidden_dims == (512, 256)  # list → tuple
        assert config.activation == "elu"
        assert config.batch_norm is True
        assert config.dropout_rate == 0.2
        assert config.input_shape == (64, 64, 3)  # list → tuple
        assert config.latent_dim == 128
        assert config.use_batch_norm is False
        assert config.use_layer_norm is True

    def test_from_dict_does_not_convert_metadata_paths(self) -> None:
        """Test from_dict does NOT automatically convert string paths in metadata.

        Path conversion only happens for dedicated Path-type fields, not for
        arbitrary strings in metadata dicts.
        """
        config_dict = {
            "name": "test",
            "hidden_dims": [128],
            "activation": "relu",
            "input_shape": [28, 28, 1],
            "latent_dim": 32,
            "metadata": {"checkpoint_path": "/path/to/checkpoint"},
        }
        config = EncoderConfig.from_dict(config_dict)
        # Metadata values remain as-is (strings are not auto-converted to Path)
        assert isinstance(config.metadata["checkpoint_path"], str)
        assert config.metadata["checkpoint_path"] == "/path/to/checkpoint"

    def test_roundtrip_serialization(self) -> None:
        """Test that to_dict → from_dict preserves config."""
        original = EncoderConfig(
            name="roundtrip_test",
            description="Test roundtrip",
            tags=("vision", "vae"),
            metadata={"version": "1.0"},
            hidden_dims=(512, 256, 128),
            activation="elu",
            batch_norm=True,
            dropout_rate=0.15,
            input_shape=(64, 64, 3),
            latent_dim=128,
            use_batch_norm=False,
            use_layer_norm=True,
        )

        config_dict = original.to_dict()
        restored = EncoderConfig.from_dict(config_dict)

        # Full equality check
        assert restored == original

        # Verify key fields explicitly
        assert restored.name == original.name
        assert restored.tags == original.tags
        assert restored.hidden_dims == original.hidden_dims
        assert restored.activation == original.activation
        assert restored.input_shape == original.input_shape
        assert restored.latent_dim == original.latent_dim
        assert restored.use_batch_norm == original.use_batch_norm
        assert restored.use_layer_norm == original.use_layer_norm


class TestEncoderConfigEdgeCases:
    """Test EncoderConfig edge cases and boundary values."""

    def test_single_hidden_dim(self) -> None:
        """Test with single hidden dimension."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(512,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
        )
        assert config.hidden_dims == (512,)

    def test_many_hidden_dims(self) -> None:
        """Test with many hidden dimensions."""
        dims = (1024, 512, 256, 128, 64, 32)
        config = EncoderConfig(
            name="test",
            hidden_dims=dims,
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=16,
        )
        assert config.hidden_dims == dims

    def test_1d_input_shape(self) -> None:
        """Test with 1D input shape."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(256,),
            activation="relu",
            input_shape=(784,),
            latent_dim=32,
        )
        assert config.input_shape == (784,)

    def test_3d_input_shape(self) -> None:
        """Test with 3D input shape (channels, height, width)."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(512,),
            activation="relu",
            input_shape=(3, 128, 128),
            latent_dim=128,
        )
        assert config.input_shape == (3, 128, 128)

    def test_small_latent_dim(self) -> None:
        """Test with very small latent dimension."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=2,
        )
        assert config.latent_dim == 2

    def test_large_latent_dim(self) -> None:
        """Test with large latent dimension."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(1024,),
            activation="relu",
            input_shape=(128, 128, 3),
            latent_dim=2048,
        )
        assert config.latent_dim == 2048

    def test_dropout_rate_zero(self) -> None:
        """Test with dropout_rate of 0.0."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
            dropout_rate=0.0,
        )
        assert config.dropout_rate == 0.0

    def test_dropout_rate_one(self) -> None:
        """Test with dropout_rate of 1.0."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
            dropout_rate=1.0,
        )
        assert config.dropout_rate == 1.0

    def test_both_norms_enabled(self) -> None:
        """Test with both batch norm and layer norm enabled."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
            use_batch_norm=True,
            use_layer_norm=True,
        )
        assert config.use_batch_norm is True
        assert config.use_layer_norm is True

    def test_both_norms_disabled(self) -> None:
        """Test with both batch norm and layer norm disabled."""
        config = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
            use_batch_norm=False,
            use_layer_norm=False,
        )
        assert config.use_batch_norm is False
        assert config.use_layer_norm is False

    def test_copy_works(self) -> None:
        """Test that configs can be copied."""
        original = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
        )
        copied = copy.copy(original)
        assert copied == original
        assert copied is not original

    def test_deepcopy_works(self) -> None:
        """Test that configs can be deep copied."""
        original = EncoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            input_shape=(28, 28, 1),
            latent_dim=32,
            metadata={"nested": {"key": "value"}},
        )
        copied = copy.deepcopy(original)
        assert copied == original
        assert copied is not original
        assert copied.metadata is not original.metadata
