"""Tests for DecoderConfig dataclass.

This test module follows TDD principles to define the DecoderConfig API
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

from artifex.generative_models.core.configuration import DecoderConfig


class TestDecoderConfigBasics:
    """Test basic DecoderConfig instantiation and properties."""

    def test_create_minimal(self) -> None:
        """Test creating DecoderConfig with minimal required fields."""
        config = DecoderConfig(
            name="test_decoder",
            hidden_dims=(128, 256),
            activation="relu",
            latent_dim=64,
            output_shape=(28, 28, 1),
        )
        assert config.name == "test_decoder"
        assert config.hidden_dims == (128, 256)
        assert config.activation == "relu"
        assert config.latent_dim == 64
        assert config.output_shape == (28, 28, 1)
        assert config.output_activation == "sigmoid"  # default

    def test_create_with_all_fields(self) -> None:
        """Test creating DecoderConfig with all fields specified."""
        config = DecoderConfig(
            # From BaseConfig
            name="full_decoder",
            description="Full decoder config",
            tags=("vision", "vae"),
            metadata={"author": "test"},
            # From BaseNetworkConfig
            hidden_dims=(128, 256, 512),
            activation="elu",
            batch_norm=True,
            dropout_rate=0.2,
            # DecoderConfig-specific
            latent_dim=128,
            output_shape=(64, 64, 3),
            output_activation="tanh",
        )
        # Verify all fields
        assert config.name == "full_decoder"
        assert config.description == "Full decoder config"
        assert config.tags == ("vision", "vae")
        assert config.metadata == {"author": "test"}
        assert config.hidden_dims == (128, 256, 512)
        assert config.activation == "elu"
        assert config.batch_norm is True
        assert config.dropout_rate == 0.2
        assert config.latent_dim == 128
        assert config.output_shape == (64, 64, 3)
        assert config.output_activation == "tanh"

    def test_frozen_behavior(self) -> None:
        """Test that DecoderConfig is immutable (frozen)."""
        config = DecoderConfig(
            name="frozen_test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
        )
        with pytest.raises((AttributeError, TypeError)):
            config.latent_dim = 64  # type: ignore

    def test_equality(self) -> None:
        """Test that two configs with same values are equal."""
        config1 = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
        )
        config2 = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
        )
        assert config1 == config2

    def test_inequality(self) -> None:
        """Test that two configs with different values are not equal."""
        config1 = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
        )
        config2 = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=64,
            output_shape=(28, 28, 1),
        )
        assert config1 != config2

    def test_hash(self) -> None:
        """Test that DecoderConfig is NOT hashable due to metadata dict field."""
        config = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
        )
        # Cannot be hashed due to dict field (metadata)
        with pytest.raises(TypeError, match="unhashable type"):
            hash(config)


class TestDecoderConfigInheritance:
    """Test that DecoderConfig properly inherits from BaseNetworkConfig."""

    def test_inherits_base_config_fields(self) -> None:
        """Test that DecoderConfig has BaseConfig fields."""
        config = DecoderConfig(
            name="test",
            description="Test config",
            tags=("test",),
            metadata={"key": "value"},
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
        )
        # BaseConfig fields
        assert config.name == "test"
        assert config.description == "Test config"
        assert config.tags == ("test",)
        assert config.metadata == {"key": "value"}

    def test_inherits_base_network_config_fields(self) -> None:
        """Test that DecoderConfig has BaseNetworkConfig fields."""
        config = DecoderConfig(
            name="test",
            hidden_dims=(128, 256),
            activation="elu",
            batch_norm=True,
            dropout_rate=0.3,
            latent_dim=128,
            output_shape=(64, 64, 3),
        )
        # BaseNetworkConfig fields
        assert config.hidden_dims == (128, 256)
        assert config.activation == "elu"
        assert config.batch_norm is True
        assert config.dropout_rate == 0.3


class TestDecoderConfigValidation:
    """Test DecoderConfig validation rules."""

    def test_validate_latent_dim_positive(self) -> None:
        """Test that latent_dim must be positive."""
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            DecoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                latent_dim=0,
                output_shape=(28, 28, 1),
            )

        with pytest.raises(ValueError, match="latent_dim must be positive"):
            DecoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                latent_dim=-10,
                output_shape=(28, 28, 1),
            )

    def test_validate_output_shape_required(self) -> None:
        """Test that output_shape cannot be empty."""
        with pytest.raises(ValueError, match="output_shape"):
            DecoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                latent_dim=32,
                output_shape=(),  # Empty tuple
            )

    def test_validate_output_shape_positive(self) -> None:
        """Test that output_shape must contain positive integers."""
        with pytest.raises(ValueError, match="positive"):
            DecoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                latent_dim=32,
                output_shape=(0, 28, 28),  # Contains zero
            )

        with pytest.raises(ValueError, match="positive"):
            DecoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                latent_dim=32,
                output_shape=(-1, 28, 28),  # Contains negative
            )

    def test_validate_output_activation(self) -> None:
        """Test that output_activation is validated if provided."""
        # Valid activations should work
        config = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
            output_activation="tanh",
        )
        assert config.output_activation == "tanh"

        # Invalid activation should fail
        with pytest.raises(ValueError, match="Unknown activation"):
            DecoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                latent_dim=32,
                output_shape=(28, 28, 1),
                output_activation="invalid_activation",
            )

    def test_validate_inherited_fields(self) -> None:
        """Test that inherited validation rules still apply."""
        # Test hidden_dims validation (from BaseNetworkConfig)
        with pytest.raises(ValueError, match="hidden_dims"):
            DecoderConfig(
                name="test",
                hidden_dims=(),  # Empty tuple
                activation="relu",
                latent_dim=32,
                output_shape=(28, 28, 1),
            )

        # Test activation validation (from BaseNetworkConfig)
        with pytest.raises(ValueError, match="Unknown activation"):
            DecoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="invalid_activation",
                latent_dim=32,
                output_shape=(28, 28, 1),
            )

        # Test dropout_rate validation (from BaseNetworkConfig)
        with pytest.raises(ValueError, match="dropout_rate"):
            DecoderConfig(
                name="test",
                hidden_dims=(128,),
                activation="relu",
                latent_dim=32,
                output_shape=(28, 28, 1),
                dropout_rate=1.5,  # > 1.0
            )


class TestDecoderConfigDefaults:
    """Test DecoderConfig default values."""

    def test_default_output_activation(self) -> None:
        """Test default output_activation value."""
        config = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
        )
        assert config.output_activation == "sigmoid"

    def test_default_batch_norm_from_base(self) -> None:
        """Test default batch_norm value from BaseNetworkConfig."""
        config = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
        )
        assert config.batch_norm is False

    def test_default_dropout_rate(self) -> None:
        """Test default dropout_rate value from BaseNetworkConfig."""
        config = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
        )
        assert config.dropout_rate == 0.0


class TestDecoderConfigSerialization:
    """Test DecoderConfig serialization (to_dict, from_dict)."""

    def test_to_dict_minimal(self) -> None:
        """Test to_dict with minimal config."""
        config = DecoderConfig(
            name="test",
            hidden_dims=(128, 256),
            activation="relu",
            latent_dim=64,
            output_shape=(28, 28, 1),
        )
        data = config.to_dict()

        assert data["name"] == "test"
        assert data["hidden_dims"] == (128, 256)
        assert data["activation"] == "relu"
        assert data["latent_dim"] == 64
        assert data["output_shape"] == (28, 28, 1)
        assert data["output_activation"] == "sigmoid"

    def test_to_dict_full(self) -> None:
        """Test to_dict with all fields specified."""
        config = DecoderConfig(
            name="full_decoder",
            description="Full config",
            tags=("vision",),
            metadata={"author": "test"},
            hidden_dims=(128, 256),
            activation="elu",
            batch_norm=True,
            dropout_rate=0.2,
            latent_dim=128,
            output_shape=(64, 64, 3),
            output_activation="tanh",
        )
        data = config.to_dict()

        assert data["name"] == "full_decoder"
        assert data["description"] == "Full config"
        assert data["tags"] == ("vision",)
        assert data["metadata"] == {"author": "test"}
        assert data["hidden_dims"] == (128, 256)
        assert data["activation"] == "elu"
        assert data["batch_norm"] is True
        assert data["dropout_rate"] == 0.2
        assert data["latent_dim"] == 128
        assert data["output_shape"] == (64, 64, 3)
        assert data["output_activation"] == "tanh"

    def test_from_dict_minimal(self) -> None:
        """Test from_dict with minimal config."""
        config_dict = {
            "name": "test",
            "hidden_dims": [128, 256],
            "activation": "relu",
            "latent_dim": 64,
            "output_shape": [28, 28, 1],
        }
        config = DecoderConfig.from_dict(config_dict)

        assert config.name == "test"
        assert config.hidden_dims == (128, 256)  # list → tuple
        assert config.activation == "relu"
        assert config.latent_dim == 64
        assert config.output_shape == (28, 28, 1)  # list → tuple
        assert config.output_activation == "sigmoid"  # default

    def test_from_dict_full(self) -> None:
        """Test from_dict with all fields."""
        config_dict = {
            "name": "full_decoder",
            "description": "Full config",
            "tags": ["vision"],
            "metadata": {"author": "test"},
            "hidden_dims": [128, 256],
            "activation": "elu",
            "batch_norm": True,
            "dropout_rate": 0.2,
            "latent_dim": 128,
            "output_shape": [64, 64, 3],
            "output_activation": "tanh",
        }
        config = DecoderConfig.from_dict(config_dict)

        assert config.name == "full_decoder"
        assert config.description == "Full config"
        assert config.tags == ("vision",)  # list → tuple
        assert config.metadata == {"author": "test"}
        assert config.hidden_dims == (128, 256)  # list → tuple
        assert config.activation == "elu"
        assert config.batch_norm is True
        assert config.dropout_rate == 0.2
        assert config.latent_dim == 128
        assert config.output_shape == (64, 64, 3)  # list → tuple
        assert config.output_activation == "tanh"

    def test_roundtrip_serialization(self) -> None:
        """Test that to_dict → from_dict preserves config."""
        original = DecoderConfig(
            name="roundtrip_test",
            description="Test roundtrip",
            tags=("vision", "vae"),
            metadata={"version": "1.0"},
            hidden_dims=(128, 256, 512),
            activation="elu",
            batch_norm=True,
            dropout_rate=0.15,
            latent_dim=128,
            output_shape=(64, 64, 3),
            output_activation="tanh",
        )

        config_dict = original.to_dict()
        restored = DecoderConfig.from_dict(config_dict)

        # Full equality check
        assert restored == original

        # Verify key fields explicitly
        assert restored.name == original.name
        assert restored.tags == original.tags
        assert restored.hidden_dims == original.hidden_dims
        assert restored.activation == original.activation
        assert restored.latent_dim == original.latent_dim
        assert restored.output_shape == original.output_shape
        assert restored.output_activation == original.output_activation


class TestDecoderConfigEdgeCases:
    """Test DecoderConfig edge cases and boundary values."""

    def test_single_hidden_dim(self) -> None:
        """Test with single hidden dimension."""
        config = DecoderConfig(
            name="test",
            hidden_dims=(512,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
        )
        assert config.hidden_dims == (512,)

    def test_many_hidden_dims(self) -> None:
        """Test with many hidden dimensions."""
        dims = (32, 64, 128, 256, 512, 1024)
        config = DecoderConfig(
            name="test",
            hidden_dims=dims,
            activation="relu",
            latent_dim=16,
            output_shape=(28, 28, 1),
        )
        assert config.hidden_dims == dims

    def test_1d_output_shape(self) -> None:
        """Test with 1D output shape."""
        config = DecoderConfig(
            name="test",
            hidden_dims=(256,),
            activation="relu",
            latent_dim=32,
            output_shape=(784,),
        )
        assert config.output_shape == (784,)

    def test_3d_output_shape(self) -> None:
        """Test with 3D output shape (channels, height, width)."""
        config = DecoderConfig(
            name="test",
            hidden_dims=(512,),
            activation="relu",
            latent_dim=128,
            output_shape=(3, 128, 128),
        )
        assert config.output_shape == (3, 128, 128)

    def test_small_latent_dim(self) -> None:
        """Test with very small latent dimension."""
        config = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=2,
            output_shape=(28, 28, 1),
        )
        assert config.latent_dim == 2

    def test_large_latent_dim(self) -> None:
        """Test with large latent dimension."""
        config = DecoderConfig(
            name="test",
            hidden_dims=(1024,),
            activation="relu",
            latent_dim=2048,
            output_shape=(128, 128, 3),
        )
        assert config.latent_dim == 2048

    def test_output_activation_none(self) -> None:
        """Test with output_activation set to None."""
        config = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
            output_activation=None,
        )
        assert config.output_activation is None

    def test_dropout_rate_zero(self) -> None:
        """Test with dropout_rate of 0.0."""
        config = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
            dropout_rate=0.0,
        )
        assert config.dropout_rate == 0.0

    def test_dropout_rate_one(self) -> None:
        """Test with dropout_rate of 1.0."""
        config = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
            dropout_rate=1.0,
        )
        assert config.dropout_rate == 1.0

    def test_copy_works(self) -> None:
        """Test that configs can be copied."""
        original = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
        )
        copied = copy.copy(original)
        assert copied == original
        assert copied is not original

    def test_deepcopy_works(self) -> None:
        """Test that configs can be deep copied."""
        original = DecoderConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
            latent_dim=32,
            output_shape=(28, 28, 1),
            metadata={"nested": {"key": "value"}},
        )
        copied = copy.deepcopy(original)
        assert copied == original
        assert copied is not original
        assert copied.metadata is not original.metadata
