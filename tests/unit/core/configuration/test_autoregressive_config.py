"""Tests for Autoregressive configuration classes.

This module tests the autoregressive configuration classes using the TDD approach.
All tests should pass after implementing the autoregressive_config.py module.
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration.autoregressive_config import (
    AutoregressiveConfig,
    PixelCNNConfig,
    TransformerConfig,
    TransformerNetworkConfig,
    WaveNetConfig,
)
from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.base_network import BaseNetworkConfig


# =============================================================================
# TransformerNetworkConfig Tests
# =============================================================================
class TestTransformerNetworkConfigBasics:
    """Test basic functionality of TransformerNetworkConfig."""

    def test_create_with_required_fields(self):
        """Test creation with required fields."""
        config = TransformerNetworkConfig(
            name="transformer_net",
            hidden_dims=(512, 512),
            activation="gelu",
        )
        assert config.name == "transformer_net"
        assert config.hidden_dims == (512, 512)
        assert config.activation == "gelu"

    def test_frozen(self):
        """Test that config is frozen."""
        config = TransformerNetworkConfig(
            name="transformer_net",
            hidden_dims=(512,),
            activation="gelu",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"  # type: ignore

    def test_inherits_from_base_network_config(self):
        """Test inheritance from BaseNetworkConfig."""
        config = TransformerNetworkConfig(
            name="transformer_net",
            hidden_dims=(512,),
            activation="gelu",
        )
        assert isinstance(config, BaseNetworkConfig)


class TestTransformerNetworkConfigDefaults:
    """Test default values of TransformerNetworkConfig."""

    def test_default_embed_dim(self):
        """Test embed_dim defaults to 512."""
        config = TransformerNetworkConfig(
            name="transformer_net",
            hidden_dims=(512,),
            activation="gelu",
        )
        assert config.embed_dim == 512

    def test_default_num_heads(self):
        """Test num_heads defaults to 8."""
        config = TransformerNetworkConfig(
            name="transformer_net",
            hidden_dims=(512,),
            activation="gelu",
        )
        assert config.num_heads == 8

    def test_default_mlp_ratio(self):
        """Test mlp_ratio defaults to 4.0."""
        config = TransformerNetworkConfig(
            name="transformer_net",
            hidden_dims=(512,),
            activation="gelu",
        )
        assert config.mlp_ratio == 4.0

    def test_default_positional_encoding(self):
        """Test positional_encoding defaults to 'sinusoidal'."""
        config = TransformerNetworkConfig(
            name="transformer_net",
            hidden_dims=(512,),
            activation="gelu",
        )
        assert config.positional_encoding == "sinusoidal"

    def test_default_use_bias(self):
        """Test use_bias defaults to True."""
        config = TransformerNetworkConfig(
            name="transformer_net",
            hidden_dims=(512,),
            activation="gelu",
        )
        assert config.use_bias is True

    def test_default_attention_dropout_rate(self):
        """Test attention_dropout_rate defaults to 0.0."""
        config = TransformerNetworkConfig(
            name="transformer_net",
            hidden_dims=(512,),
            activation="gelu",
        )
        assert config.attention_dropout_rate == 0.0


class TestTransformerNetworkConfigValidation:
    """Test validation of TransformerNetworkConfig."""

    def test_invalid_embed_dim(self):
        """Test that non-positive embed_dim raises ValueError."""
        with pytest.raises(ValueError, match="embed_dim"):
            TransformerNetworkConfig(
                name="transformer_net",
                hidden_dims=(512,),
                activation="gelu",
                embed_dim=0,
            )

    def test_invalid_num_heads(self):
        """Test that non-positive num_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads"):
            TransformerNetworkConfig(
                name="transformer_net",
                hidden_dims=(512,),
                activation="gelu",
                num_heads=0,
            )

    def test_embed_dim_not_divisible_by_num_heads(self):
        """Test that embed_dim not divisible by num_heads raises ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            TransformerNetworkConfig(
                name="transformer_net",
                hidden_dims=(512,),
                activation="gelu",
                embed_dim=100,  # Not divisible by 8
                num_heads=8,
            )

    def test_invalid_mlp_ratio(self):
        """Test that non-positive mlp_ratio raises ValueError."""
        with pytest.raises(ValueError, match="mlp_ratio"):
            TransformerNetworkConfig(
                name="transformer_net",
                hidden_dims=(512,),
                activation="gelu",
                mlp_ratio=0.0,
            )

    def test_invalid_positional_encoding(self):
        """Test that invalid positional_encoding raises ValueError."""
        with pytest.raises(ValueError, match="positional_encoding"):
            TransformerNetworkConfig(
                name="transformer_net",
                hidden_dims=(512,),
                activation="gelu",
                positional_encoding="invalid",
            )

    def test_valid_positional_encodings(self):
        """Test that valid positional_encodings are accepted."""
        for pe_type in ["sinusoidal", "learned", "rotary", "none"]:
            config = TransformerNetworkConfig(
                name="transformer_net",
                hidden_dims=(512,),
                activation="gelu",
                positional_encoding=pe_type,
            )
            assert config.positional_encoding == pe_type

    def test_invalid_attention_dropout_rate(self):
        """Test that invalid attention_dropout_rate raises ValueError."""
        with pytest.raises(ValueError, match="dropout"):
            TransformerNetworkConfig(
                name="transformer_net",
                hidden_dims=(512,),
                activation="gelu",
                attention_dropout_rate=1.5,
            )


class TestTransformerNetworkConfigSerialization:
    """Test serialization of TransformerNetworkConfig."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = TransformerNetworkConfig(
            name="transformer_net",
            hidden_dims=(256, 512),
            activation="gelu",
            embed_dim=256,
            num_heads=4,
        )
        data = config.to_dict()
        assert data["name"] == "transformer_net"
        assert data["hidden_dims"] == (256, 512)
        assert data["embed_dim"] == 256
        assert data["num_heads"] == 4

    def test_from_dict(self):
        """Test from_dict creation."""
        data = {
            "name": "transformer_net",
            "hidden_dims": [256, 512],  # List should be converted to tuple
            "activation": "gelu",
            "embed_dim": 256,
            "num_heads": 4,
            "positional_encoding": "rotary",
        }
        config = TransformerNetworkConfig.from_dict(data)
        assert config.name == "transformer_net"
        assert config.hidden_dims == (256, 512)
        assert config.positional_encoding == "rotary"

    def test_roundtrip(self):
        """Test roundtrip serialization."""
        original = TransformerNetworkConfig(
            name="transformer_net",
            hidden_dims=(256, 512, 256),
            activation="relu",
            embed_dim=256,
            num_heads=4,
            mlp_ratio=2.0,
            positional_encoding="learned",
            attention_dropout_rate=0.1,
        )
        data = original.to_dict()
        restored = TransformerNetworkConfig.from_dict(data)
        assert original == restored


# =============================================================================
# AutoregressiveConfig Tests (Base Class)
# =============================================================================
class TestAutoregressiveConfigBasics:
    """Test basic functionality of AutoregressiveConfig."""

    def test_create_with_required_fields(self):
        """Test creation with required fields."""
        config = AutoregressiveConfig(
            name="test_ar",
            vocab_size=1000,
            sequence_length=512,
        )
        assert config.name == "test_ar"
        assert config.vocab_size == 1000
        assert config.sequence_length == 512

    def test_frozen(self):
        """Test that config is frozen."""
        config = AutoregressiveConfig(
            name="test_ar",
            vocab_size=1000,
            sequence_length=512,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"  # type: ignore

    def test_inherits_from_base_config(self):
        """Test inheritance from BaseConfig."""
        config = AutoregressiveConfig(
            name="test_ar",
            vocab_size=1000,
            sequence_length=512,
        )
        assert isinstance(config, BaseConfig)


class TestAutoregressiveConfigDefaults:
    """Test default values of AutoregressiveConfig."""

    def test_default_dropout_rate(self):
        """Test dropout_rate defaults to 0.0."""
        config = AutoregressiveConfig(
            name="test_ar",
            vocab_size=1000,
            sequence_length=512,
        )
        assert config.dropout_rate == 0.0


class TestAutoregressiveConfigValidation:
    """Test validation of AutoregressiveConfig."""

    def test_invalid_vocab_size(self):
        """Test that non-positive vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size"):
            AutoregressiveConfig(
                name="test_ar",
                vocab_size=0,
                sequence_length=512,
            )

    def test_invalid_sequence_length(self):
        """Test that non-positive sequence_length raises ValueError."""
        with pytest.raises(ValueError, match="sequence_length"):
            AutoregressiveConfig(
                name="test_ar",
                vocab_size=1000,
                sequence_length=0,
            )

    def test_invalid_dropout_rate(self):
        """Test that invalid dropout_rate raises ValueError."""
        with pytest.raises(ValueError, match="dropout"):
            AutoregressiveConfig(
                name="test_ar",
                vocab_size=1000,
                sequence_length=512,
                dropout_rate=1.5,
            )


class TestAutoregressiveConfigSerialization:
    """Test serialization of AutoregressiveConfig."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = AutoregressiveConfig(
            name="test_ar",
            vocab_size=1000,
            sequence_length=512,
            dropout_rate=0.1,
        )
        data = config.to_dict()
        assert data["name"] == "test_ar"
        assert data["vocab_size"] == 1000
        assert data["sequence_length"] == 512
        assert data["dropout_rate"] == 0.1

    def test_roundtrip(self):
        """Test roundtrip serialization."""
        original = AutoregressiveConfig(
            name="test_ar",
            vocab_size=50000,
            sequence_length=1024,
            dropout_rate=0.1,
        )
        data = original.to_dict()
        restored = AutoregressiveConfig.from_dict(data)
        assert original == restored


# =============================================================================
# TransformerConfig Tests
# =============================================================================
class TestTransformerConfigBasics:
    """Test basic functionality of TransformerConfig."""

    @pytest.fixture
    def transformer_network(self):
        """Create a test transformer network config."""
        return TransformerNetworkConfig(
            name="test_network",
            hidden_dims=(512, 512),
            activation="gelu",
        )

    def test_create_with_required_fields(self, transformer_network):
        """Test creation with required fields."""
        config = TransformerConfig(
            name="test_transformer",
            network=transformer_network,
            vocab_size=50000,
            sequence_length=512,
        )
        assert config.name == "test_transformer"
        assert config.network == transformer_network
        assert config.vocab_size == 50000
        assert config.sequence_length == 512

    def test_frozen(self, transformer_network):
        """Test that config is frozen."""
        config = TransformerConfig(
            name="test_transformer",
            network=transformer_network,
            vocab_size=50000,
            sequence_length=512,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"  # type: ignore

    def test_inherits_from_autoregressive_config(self, transformer_network):
        """Test inheritance from AutoregressiveConfig."""
        config = TransformerConfig(
            name="test_transformer",
            network=transformer_network,
            vocab_size=50000,
            sequence_length=512,
        )
        assert isinstance(config, AutoregressiveConfig)

    def test_missing_network_raises_error(self):
        """Test that missing network raises ValueError."""
        with pytest.raises(ValueError, match="network.*required"):
            TransformerConfig(
                name="test_transformer",
                vocab_size=50000,
                sequence_length=512,
            )

    def test_invalid_network_type_raises_error(self):
        """Test that wrong type for network raises TypeError."""
        with pytest.raises(TypeError, match="network must be TransformerNetworkConfig"):
            TransformerConfig(
                name="test_transformer",
                network={"hidden_dims": (512,)},  # type: ignore
                vocab_size=50000,
                sequence_length=512,
            )


class TestTransformerConfigDefaults:
    """Test default values of TransformerConfig."""

    @pytest.fixture
    def transformer_network(self):
        """Create a test transformer network config."""
        return TransformerNetworkConfig(
            name="test_network",
            hidden_dims=(512, 512),
            activation="gelu",
        )

    def test_default_num_layers(self, transformer_network):
        """Test num_layers defaults to 6."""
        config = TransformerConfig(
            name="test_transformer",
            network=transformer_network,
            vocab_size=50000,
            sequence_length=512,
        )
        assert config.num_layers == 6

    def test_default_use_cache(self, transformer_network):
        """Test use_cache defaults to True."""
        config = TransformerConfig(
            name="test_transformer",
            network=transformer_network,
            vocab_size=50000,
            sequence_length=512,
        )
        assert config.use_cache is True


class TestTransformerConfigValidation:
    """Test validation of TransformerConfig."""

    @pytest.fixture
    def transformer_network(self):
        """Create a test transformer network config."""
        return TransformerNetworkConfig(
            name="test_network",
            hidden_dims=(512, 512),
            activation="gelu",
        )

    def test_invalid_num_layers(self, transformer_network):
        """Test that non-positive num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers"):
            TransformerConfig(
                name="test_transformer",
                network=transformer_network,
                vocab_size=50000,
                sequence_length=512,
                num_layers=0,
            )


class TestTransformerConfigSerialization:
    """Test serialization of TransformerConfig."""

    @pytest.fixture
    def transformer_network(self):
        """Create a test transformer network config."""
        return TransformerNetworkConfig(
            name="test_network",
            hidden_dims=(512, 512),
            activation="gelu",
        )

    def test_to_dict(self, transformer_network):
        """Test to_dict conversion with nested configs."""
        config = TransformerConfig(
            name="test_transformer",
            network=transformer_network,
            vocab_size=50000,
            sequence_length=512,
            num_layers=12,
        )
        data = config.to_dict()
        assert data["name"] == "test_transformer"
        assert data["vocab_size"] == 50000
        assert data["num_layers"] == 12
        # Nested config should be converted to dict
        assert isinstance(data["network"], dict)
        assert data["network"]["name"] == "test_network"

    def test_from_dict(self):
        """Test from_dict handles nested configs."""
        data = {
            "name": "test_transformer",
            "network": {
                "name": "test_network",
                "hidden_dims": [512, 512],
                "activation": "gelu",
            },
            "vocab_size": 50000,
            "sequence_length": 512,
            "num_layers": 12,
        }
        config = TransformerConfig.from_dict(data)
        assert config.name == "test_transformer"
        assert isinstance(config.network, TransformerNetworkConfig)
        assert config.network.hidden_dims == (512, 512)
        assert config.num_layers == 12

    def test_roundtrip(self, transformer_network):
        """Test roundtrip serialization."""
        original = TransformerConfig(
            name="test_transformer",
            network=transformer_network,
            vocab_size=50000,
            sequence_length=1024,
            num_layers=24,
            use_cache=False,
        )
        data = original.to_dict()
        restored = TransformerConfig.from_dict(data)
        assert original == restored


# =============================================================================
# PixelCNNConfig Tests
# =============================================================================
class TestPixelCNNConfigBasics:
    """Test basic functionality of PixelCNNConfig."""

    def test_create_with_required_fields(self):
        """Test creation with required fields."""
        config = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(28, 28, 1),
        )
        assert config.name == "test_pixelcnn"
        assert config.image_shape == (28, 28, 1)

    def test_frozen(self):
        """Test that config is frozen."""
        config = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(28, 28, 1),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"  # type: ignore

    def test_inherits_from_base_config(self):
        """Test inheritance from BaseConfig (not AutoregressiveConfig directly)."""
        config = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(28, 28, 1),
        )
        assert isinstance(config, BaseConfig)

    def test_missing_image_shape_raises_error(self):
        """Test that missing image_shape raises ValueError."""
        with pytest.raises(ValueError, match="image_shape.*required"):
            PixelCNNConfig(name="test_pixelcnn")


class TestPixelCNNConfigDefaults:
    """Test default values of PixelCNNConfig."""

    def test_default_hidden_channels(self):
        """Test hidden_channels defaults to 128."""
        config = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(28, 28, 1),
        )
        assert config.hidden_channels == 128

    def test_default_num_layers(self):
        """Test num_layers defaults to 7."""
        config = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(28, 28, 1),
        )
        assert config.num_layers == 7

    def test_default_num_residual_blocks(self):
        """Test num_residual_blocks defaults to 5."""
        config = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(28, 28, 1),
        )
        assert config.num_residual_blocks == 5

    def test_default_kernel_size(self):
        """Test kernel_size defaults to 3."""
        config = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(28, 28, 1),
        )
        assert config.kernel_size == 3

    def test_default_dropout_rate(self):
        """Test dropout_rate defaults to 0.0."""
        config = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(28, 28, 1),
        )
        assert config.dropout_rate == 0.0


class TestPixelCNNConfigDerivedProperties:
    """Test derived properties of PixelCNNConfig."""

    def test_derived_vocab_size(self):
        """Test derived_vocab_size returns 256 (8-bit pixels)."""
        config = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(28, 28, 1),
        )
        assert config.derived_vocab_size == 256

    def test_derived_sequence_length(self):
        """Test derived_sequence_length returns H * W * C."""
        config = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(28, 28, 1),
        )
        assert config.derived_sequence_length == 28 * 28 * 1

    def test_derived_sequence_length_rgb(self):
        """Test derived_sequence_length for RGB images."""
        config = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(32, 32, 3),
        )
        assert config.derived_sequence_length == 32 * 32 * 3


class TestPixelCNNConfigValidation:
    """Test validation of PixelCNNConfig."""

    def test_invalid_image_shape_dimensions(self):
        """Test that wrong number of dimensions raises ValueError."""
        with pytest.raises(ValueError, match="3 dimensions"):
            PixelCNNConfig(
                name="test_pixelcnn",
                image_shape=(28, 28),  # type: ignore  # Missing channels
            )

    def test_invalid_image_shape_zero_height(self):
        """Test that zero height raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            PixelCNNConfig(
                name="test_pixelcnn",
                image_shape=(0, 28, 1),
            )

    def test_invalid_image_shape_negative_channels(self):
        """Test that negative channels raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            PixelCNNConfig(
                name="test_pixelcnn",
                image_shape=(28, 28, -1),
            )

    def test_invalid_hidden_channels(self):
        """Test that non-positive hidden_channels raises ValueError."""
        with pytest.raises(ValueError, match="hidden_channels"):
            PixelCNNConfig(
                name="test_pixelcnn",
                image_shape=(28, 28, 1),
                hidden_channels=0,
            )

    def test_invalid_num_layers(self):
        """Test that non-positive num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers"):
            PixelCNNConfig(
                name="test_pixelcnn",
                image_shape=(28, 28, 1),
                num_layers=0,
            )

    def test_invalid_num_residual_blocks(self):
        """Test that negative num_residual_blocks raises ValueError."""
        with pytest.raises(ValueError, match="num_residual_blocks"):
            PixelCNNConfig(
                name="test_pixelcnn",
                image_shape=(28, 28, 1),
                num_residual_blocks=-1,
            )

    def test_zero_residual_blocks_allowed(self):
        """Test that zero num_residual_blocks is allowed."""
        config = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(28, 28, 1),
            num_residual_blocks=0,
        )
        assert config.num_residual_blocks == 0

    def test_invalid_kernel_size(self):
        """Test that non-positive kernel_size raises ValueError."""
        with pytest.raises(ValueError, match="kernel_size"):
            PixelCNNConfig(
                name="test_pixelcnn",
                image_shape=(28, 28, 1),
                kernel_size=0,
            )


class TestPixelCNNConfigSerialization:
    """Test serialization of PixelCNNConfig."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(32, 32, 3),
            hidden_channels=256,
            num_layers=10,
        )
        data = config.to_dict()
        assert data["name"] == "test_pixelcnn"
        assert data["image_shape"] == (32, 32, 3)
        assert data["hidden_channels"] == 256

    def test_from_dict(self):
        """Test from_dict handles tuple conversion."""
        data = {
            "name": "test_pixelcnn",
            "image_shape": [32, 32, 3],  # List should be converted to tuple
            "hidden_channels": 256,
            "num_layers": 10,
        }
        config = PixelCNNConfig.from_dict(data)
        assert config.image_shape == (32, 32, 3)
        assert config.hidden_channels == 256

    def test_roundtrip(self):
        """Test roundtrip serialization."""
        original = PixelCNNConfig(
            name="test_pixelcnn",
            image_shape=(64, 64, 3),
            hidden_channels=256,
            num_layers=12,
            num_residual_blocks=8,
            kernel_size=5,
            dropout_rate=0.1,
        )
        data = original.to_dict()
        restored = PixelCNNConfig.from_dict(data)
        assert original == restored


# =============================================================================
# WaveNetConfig Tests
# =============================================================================
class TestWaveNetConfigBasics:
    """Test basic functionality of WaveNetConfig."""

    def test_create_with_required_fields(self):
        """Test creation with required fields."""
        config = WaveNetConfig(
            name="test_wavenet",
            vocab_size=256,
            sequence_length=16000,
        )
        assert config.name == "test_wavenet"
        assert config.vocab_size == 256
        assert config.sequence_length == 16000

    def test_frozen(self):
        """Test that config is frozen."""
        config = WaveNetConfig(
            name="test_wavenet",
            vocab_size=256,
            sequence_length=16000,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"  # type: ignore

    def test_inherits_from_autoregressive_config(self):
        """Test inheritance from AutoregressiveConfig."""
        config = WaveNetConfig(
            name="test_wavenet",
            vocab_size=256,
            sequence_length=16000,
        )
        assert isinstance(config, AutoregressiveConfig)


class TestWaveNetConfigDefaults:
    """Test default values of WaveNetConfig."""

    def test_default_residual_channels(self):
        """Test residual_channels defaults to 32."""
        config = WaveNetConfig(
            name="test_wavenet",
            vocab_size=256,
            sequence_length=16000,
        )
        assert config.residual_channels == 32

    def test_default_skip_channels(self):
        """Test skip_channels defaults to 256."""
        config = WaveNetConfig(
            name="test_wavenet",
            vocab_size=256,
            sequence_length=16000,
        )
        assert config.skip_channels == 256

    def test_default_num_blocks(self):
        """Test num_blocks defaults to 3."""
        config = WaveNetConfig(
            name="test_wavenet",
            vocab_size=256,
            sequence_length=16000,
        )
        assert config.num_blocks == 3

    def test_default_layers_per_block(self):
        """Test layers_per_block defaults to 10."""
        config = WaveNetConfig(
            name="test_wavenet",
            vocab_size=256,
            sequence_length=16000,
        )
        assert config.layers_per_block == 10

    def test_default_kernel_size(self):
        """Test kernel_size defaults to 2."""
        config = WaveNetConfig(
            name="test_wavenet",
            vocab_size=256,
            sequence_length=16000,
        )
        assert config.kernel_size == 2

    def test_default_dilation_base(self):
        """Test dilation_base defaults to 2."""
        config = WaveNetConfig(
            name="test_wavenet",
            vocab_size=256,
            sequence_length=16000,
        )
        assert config.dilation_base == 2

    def test_default_use_gated_activation(self):
        """Test use_gated_activation defaults to True."""
        config = WaveNetConfig(
            name="test_wavenet",
            vocab_size=256,
            sequence_length=16000,
        )
        assert config.use_gated_activation is True


class TestWaveNetConfigReceptiveField:
    """Test receptive field calculation of WaveNetConfig."""

    def test_receptive_field_calculation(self):
        """Test receptive_field property calculation."""
        config = WaveNetConfig(
            name="test_wavenet",
            vocab_size=256,
            sequence_length=16000,
            num_blocks=3,
            layers_per_block=10,
            kernel_size=2,
            dilation_base=2,
        )
        # For kernel_size=2, dilation_base=2, layers_per_block=10, num_blocks=3:
        # Each block: sum(2^i for i in range(10)) = 1 + 2 + 4 + ... + 512 = 1023
        # Total for 3 blocks: 1 + 3 * 1023 = 3070
        # Formula: 1 + num_blocks * (dilation_base^layers_per_block - 1) for kernel=2
        expected = 1 + 3 * (2**10 - 1)
        assert config.receptive_field == expected

    def test_receptive_field_single_block(self):
        """Test receptive field for single block."""
        config = WaveNetConfig(
            name="test_wavenet",
            vocab_size=256,
            sequence_length=16000,
            num_blocks=1,
            layers_per_block=5,
            kernel_size=2,
            dilation_base=2,
        )
        # 1 + 1*(1+2+4+8+16) = 1 + 31 = 32
        expected = 1 + (2**5 - 1)
        assert config.receptive_field == expected


class TestWaveNetConfigValidation:
    """Test validation of WaveNetConfig."""

    def test_invalid_residual_channels(self):
        """Test that non-positive residual_channels raises ValueError."""
        with pytest.raises(ValueError, match="residual_channels"):
            WaveNetConfig(
                name="test_wavenet",
                vocab_size=256,
                sequence_length=16000,
                residual_channels=0,
            )

    def test_invalid_skip_channels(self):
        """Test that non-positive skip_channels raises ValueError."""
        with pytest.raises(ValueError, match="skip_channels"):
            WaveNetConfig(
                name="test_wavenet",
                vocab_size=256,
                sequence_length=16000,
                skip_channels=0,
            )

    def test_invalid_num_blocks(self):
        """Test that non-positive num_blocks raises ValueError."""
        with pytest.raises(ValueError, match="num_blocks"):
            WaveNetConfig(
                name="test_wavenet",
                vocab_size=256,
                sequence_length=16000,
                num_blocks=0,
            )

    def test_invalid_layers_per_block(self):
        """Test that non-positive layers_per_block raises ValueError."""
        with pytest.raises(ValueError, match="layers_per_block"):
            WaveNetConfig(
                name="test_wavenet",
                vocab_size=256,
                sequence_length=16000,
                layers_per_block=0,
            )

    def test_invalid_kernel_size(self):
        """Test that non-positive kernel_size raises ValueError."""
        with pytest.raises(ValueError, match="kernel_size"):
            WaveNetConfig(
                name="test_wavenet",
                vocab_size=256,
                sequence_length=16000,
                kernel_size=0,
            )

    def test_invalid_dilation_base(self):
        """Test that dilation_base < 2 raises ValueError."""
        with pytest.raises(ValueError, match="dilation_base"):
            WaveNetConfig(
                name="test_wavenet",
                vocab_size=256,
                sequence_length=16000,
                dilation_base=1,
            )


class TestWaveNetConfigSerialization:
    """Test serialization of WaveNetConfig."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = WaveNetConfig(
            name="test_wavenet",
            vocab_size=256,
            sequence_length=16000,
            residual_channels=64,
            skip_channels=512,
            num_blocks=4,
        )
        data = config.to_dict()
        assert data["name"] == "test_wavenet"
        assert data["vocab_size"] == 256
        assert data["residual_channels"] == 64
        assert data["num_blocks"] == 4

    def test_from_dict(self):
        """Test from_dict creation."""
        data = {
            "name": "test_wavenet",
            "vocab_size": 256,
            "sequence_length": 16000,
            "residual_channels": 64,
            "skip_channels": 512,
            "num_blocks": 4,
        }
        config = WaveNetConfig.from_dict(data)
        assert config.residual_channels == 64
        assert config.num_blocks == 4

    def test_roundtrip(self):
        """Test roundtrip serialization."""
        original = WaveNetConfig(
            name="test_wavenet",
            vocab_size=65536,  # 16-bit audio
            sequence_length=32000,
            residual_channels=128,
            skip_channels=512,
            num_blocks=5,
            layers_per_block=12,
            kernel_size=3,
            dilation_base=3,
            use_gated_activation=False,
            dropout_rate=0.1,
        )
        data = original.to_dict()
        restored = WaveNetConfig.from_dict(data)
        assert original == restored
