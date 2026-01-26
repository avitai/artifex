"""Tests for Flow configuration classes.

This module tests the flow configuration classes using the TDD approach.
All tests should pass after implementing the flow_config.py module.
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration.base_network import BaseNetworkConfig
from artifex.generative_models.core.configuration.flow_config import (
    CouplingNetworkConfig,
    FlowConfig,
    GlowConfig,
    IAFConfig,
    MAFConfig,
    NeuralSplineConfig,
    RealNVPConfig,
)


# =============================================================================
# CouplingNetworkConfig Tests
# =============================================================================
class TestCouplingNetworkConfigBasics:
    """Test basic functionality of CouplingNetworkConfig."""

    def test_create_with_required_fields(self):
        """Test creation with required fields."""
        config = CouplingNetworkConfig(
            name="coupling_net",
            hidden_dims=(128, 128),
            activation="relu",
        )
        assert config.name == "coupling_net"
        assert config.hidden_dims == (128, 128)
        assert config.activation == "relu"

    def test_frozen(self):
        """Test that config is frozen."""
        config = CouplingNetworkConfig(
            name="coupling_net",
            hidden_dims=(128,),
            activation="relu",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"  # type: ignore

    def test_inherits_from_base_network_config(self):
        """Test inheritance from BaseNetworkConfig."""
        config = CouplingNetworkConfig(
            name="coupling_net",
            hidden_dims=(128,),
            activation="relu",
        )
        assert isinstance(config, BaseNetworkConfig)


class TestCouplingNetworkConfigDefaults:
    """Test default values of CouplingNetworkConfig."""

    def test_default_network_type(self):
        """Test network_type defaults to 'mlp'."""
        config = CouplingNetworkConfig(
            name="coupling_net",
            hidden_dims=(128,),
            activation="relu",
        )
        assert config.network_type == "mlp"

    def test_default_scale_activation(self):
        """Test default scale_activation."""
        config = CouplingNetworkConfig(
            name="coupling_net",
            hidden_dims=(128,),
            activation="relu",
        )
        assert config.scale_activation == "tanh"

    def test_default_num_residual_blocks(self):
        """Test num_residual_blocks defaults to 0."""
        config = CouplingNetworkConfig(
            name="coupling_net",
            hidden_dims=(128,),
            activation="relu",
        )
        assert config.num_residual_blocks == 0

    def test_default_num_attention_heads(self):
        """Test num_attention_heads defaults to 4."""
        config = CouplingNetworkConfig(
            name="coupling_net",
            hidden_dims=(128,),
            activation="relu",
        )
        assert config.num_attention_heads == 4

    def test_default_batch_norm(self):
        """Test batch_norm defaults to False."""
        config = CouplingNetworkConfig(
            name="coupling_net",
            hidden_dims=(128,),
            activation="relu",
        )
        assert config.batch_norm is False

    def test_default_dropout_rate(self):
        """Test dropout_rate defaults to 0.0."""
        config = CouplingNetworkConfig(
            name="coupling_net",
            hidden_dims=(128,),
            activation="relu",
        )
        assert config.dropout_rate == 0.0


class TestCouplingNetworkConfigValidation:
    """Test validation of CouplingNetworkConfig."""

    def test_invalid_network_type(self):
        """Test that invalid network_type raises ValueError."""
        with pytest.raises(ValueError, match="network_type"):
            CouplingNetworkConfig(
                name="coupling_net",
                hidden_dims=(128,),
                activation="relu",
                network_type="invalid",
            )

    def test_valid_network_types(self):
        """Test that valid network_types are accepted."""
        for net_type in ["mlp", "resnet", "attention", "cnn"]:
            config = CouplingNetworkConfig(
                name="coupling_net",
                hidden_dims=(128,),
                activation="relu",
                network_type=net_type,
            )
            assert config.network_type == net_type

    def test_invalid_scale_activation(self):
        """Test that invalid scale_activation raises ValueError."""
        with pytest.raises(ValueError, match="scale_activation"):
            CouplingNetworkConfig(
                name="coupling_net",
                hidden_dims=(128,),
                activation="relu",
                scale_activation="invalid",
            )

    def test_valid_scale_activations(self):
        """Test that valid scale_activations are accepted."""
        for scale_act in ["tanh", "sigmoid", "exp", "softplus"]:
            config = CouplingNetworkConfig(
                name="coupling_net",
                hidden_dims=(128,),
                activation="relu",
                scale_activation=scale_act,
            )
            assert config.scale_activation == scale_act

    def test_invalid_num_residual_blocks(self):
        """Test that negative num_residual_blocks raises ValueError."""
        with pytest.raises(ValueError, match="num_residual_blocks"):
            CouplingNetworkConfig(
                name="coupling_net",
                hidden_dims=(128,),
                activation="relu",
                num_residual_blocks=-1,
            )

    def test_invalid_num_attention_heads(self):
        """Test that non-positive num_attention_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_attention_heads"):
            CouplingNetworkConfig(
                name="coupling_net",
                hidden_dims=(128,),
                activation="relu",
                num_attention_heads=0,
            )


class TestCouplingNetworkConfigSerialization:
    """Test serialization of CouplingNetworkConfig."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = CouplingNetworkConfig(
            name="coupling_net",
            hidden_dims=(128, 256),
            activation="gelu",
            network_type="resnet",
            scale_activation="sigmoid",
            num_residual_blocks=3,
        )
        data = config.to_dict()
        assert data["name"] == "coupling_net"
        assert data["hidden_dims"] == (128, 256)
        assert data["activation"] == "gelu"
        assert data["network_type"] == "resnet"
        assert data["scale_activation"] == "sigmoid"
        assert data["num_residual_blocks"] == 3

    def test_from_dict(self):
        """Test from_dict creation."""
        data = {
            "name": "coupling_net",
            "hidden_dims": [128, 256],
            "activation": "gelu",
            "network_type": "attention",
            "scale_activation": "sigmoid",
            "num_attention_heads": 8,
        }
        config = CouplingNetworkConfig.from_dict(data)
        assert config.name == "coupling_net"
        assert config.hidden_dims == (128, 256)
        assert config.activation == "gelu"
        assert config.network_type == "attention"
        assert config.num_attention_heads == 8

    def test_roundtrip(self):
        """Test roundtrip serialization."""
        original = CouplingNetworkConfig(
            name="coupling_net",
            hidden_dims=(64, 128, 64),
            activation="relu",
            network_type="resnet",
            scale_activation="tanh",
            batch_norm=True,
            num_residual_blocks=2,
        )
        data = original.to_dict()
        restored = CouplingNetworkConfig.from_dict(data)
        assert original == restored


# =============================================================================
# FlowConfig Tests (Base Class)
# =============================================================================
class TestFlowConfigBasics:
    """Test basic functionality of FlowConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(64, 64),
            activation="relu",
        )

    def test_create_with_required_fields(self, coupling_network):
        """Test creation with required fields."""
        config = FlowConfig(
            name="test_flow",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.name == "test_flow"
        assert config.coupling_network == coupling_network
        assert config.input_dim == 16

    def test_frozen(self, coupling_network):
        """Test that config is frozen."""
        config = FlowConfig(
            name="test_flow",
            coupling_network=coupling_network,
            input_dim=16,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"  # type: ignore

    def test_missing_coupling_network_raises_error(self):
        """Test that missing coupling_network raises ValueError."""
        with pytest.raises(ValueError, match="coupling_network.*required"):
            FlowConfig(
                name="test_flow",
                input_dim=16,
            )

    def test_invalid_coupling_network_type_raises_error(self):
        """Test that wrong type for coupling_network raises TypeError."""
        with pytest.raises(TypeError, match="coupling_network must be CouplingNetworkConfig"):
            FlowConfig(
                name="test_flow",
                coupling_network={"hidden_dims": (64,)},  # type: ignore
                input_dim=16,
            )


class TestFlowConfigDefaults:
    """Test default values of FlowConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(64, 64),
            activation="relu",
        )

    def test_default_latent_dim(self, coupling_network):
        """Test latent_dim defaults to input_dim."""
        config = FlowConfig(
            name="test_flow",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.latent_dim == 16  # Defaults to input_dim

    def test_default_base_distribution(self, coupling_network):
        """Test base_distribution defaults to 'normal'."""
        config = FlowConfig(
            name="test_flow",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.base_distribution == "normal"

    def test_default_base_distribution_params(self, coupling_network):
        """Test base_distribution_params has sensible defaults."""
        config = FlowConfig(
            name="test_flow",
            coupling_network=coupling_network,
            input_dim=16,
        )
        # Default params for normal distribution
        assert "loc" in config.base_distribution_params
        assert "scale" in config.base_distribution_params


class TestFlowConfigValidation:
    """Test validation of FlowConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(64, 64),
            activation="relu",
        )

    def test_invalid_input_dim(self, coupling_network):
        """Test that non-positive input_dim raises ValueError."""
        with pytest.raises(ValueError, match="input_dim"):
            FlowConfig(
                name="test_flow",
                coupling_network=coupling_network,
                input_dim=0,
            )

    def test_invalid_latent_dim(self, coupling_network):
        """Test that non-positive latent_dim raises ValueError."""
        with pytest.raises(ValueError, match="latent_dim"):
            FlowConfig(
                name="test_flow",
                coupling_network=coupling_network,
                input_dim=16,
                latent_dim=-1,
            )

    def test_invalid_base_distribution(self, coupling_network):
        """Test that invalid base_distribution raises ValueError."""
        with pytest.raises(ValueError, match="base_distribution"):
            FlowConfig(
                name="test_flow",
                coupling_network=coupling_network,
                input_dim=16,
                base_distribution="invalid",
            )

    def test_valid_base_distributions(self, coupling_network):
        """Test that valid base_distributions are accepted."""
        for dist in ["normal", "uniform"]:
            config = FlowConfig(
                name="test_flow",
                coupling_network=coupling_network,
                input_dim=16,
                base_distribution=dist,
            )
            assert config.base_distribution == dist


class TestFlowConfigSerialization:
    """Test serialization of FlowConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(64, 64),
            activation="relu",
        )

    def test_to_dict(self, coupling_network):
        """Test to_dict conversion with nested configs."""
        config = FlowConfig(
            name="test_flow",
            coupling_network=coupling_network,
            input_dim=16,
            base_distribution="normal",
        )
        data = config.to_dict()
        assert data["name"] == "test_flow"
        assert data["input_dim"] == 16
        # Nested config should be converted to dict
        assert isinstance(data["coupling_network"], dict)
        assert data["coupling_network"]["name"] == "test_coupling"

    def test_from_dict(self, coupling_network):
        """Test from_dict handles nested configs."""
        data = {
            "name": "test_flow",
            "coupling_network": {
                "name": "test_coupling",
                "hidden_dims": [64, 64],
                "activation": "relu",
            },
            "input_dim": 16,
        }
        config = FlowConfig.from_dict(data)
        assert config.name == "test_flow"
        assert isinstance(config.coupling_network, CouplingNetworkConfig)
        assert config.coupling_network.hidden_dims == (64, 64)

    def test_roundtrip(self, coupling_network):
        """Test roundtrip serialization."""
        original = FlowConfig(
            name="test_flow",
            coupling_network=coupling_network,
            input_dim=32,
            latent_dim=32,
            base_distribution="uniform",
        )
        data = original.to_dict()
        restored = FlowConfig.from_dict(data)
        assert original == restored


# =============================================================================
# RealNVPConfig Tests
# =============================================================================
class TestRealNVPConfigBasics:
    """Test basic functionality of RealNVPConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(64, 64),
            activation="relu",
        )

    def test_create_with_required_fields(self, coupling_network):
        """Test creation with required fields."""
        config = RealNVPConfig(
            name="test_realnvp",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.name == "test_realnvp"
        assert config.input_dim == 16

    def test_inherits_from_flow_config(self, coupling_network):
        """Test inheritance from FlowConfig."""
        config = RealNVPConfig(
            name="test_realnvp",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert isinstance(config, FlowConfig)


class TestRealNVPConfigDefaults:
    """Test default values of RealNVPConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(64, 64),
            activation="relu",
        )

    def test_default_num_coupling_layers(self, coupling_network):
        """Test num_coupling_layers defaults to 8."""
        config = RealNVPConfig(
            name="test_realnvp",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.num_coupling_layers == 8

    def test_default_mask_type(self, coupling_network):
        """Test mask_type defaults to 'checkerboard'."""
        config = RealNVPConfig(
            name="test_realnvp",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.mask_type == "checkerboard"


class TestRealNVPConfigValidation:
    """Test validation of RealNVPConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(64, 64),
            activation="relu",
        )

    def test_invalid_num_coupling_layers(self, coupling_network):
        """Test that non-positive num_coupling_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_coupling_layers"):
            RealNVPConfig(
                name="test_realnvp",
                coupling_network=coupling_network,
                input_dim=16,
                num_coupling_layers=0,
            )

    def test_invalid_mask_type(self, coupling_network):
        """Test that invalid mask_type raises ValueError."""
        with pytest.raises(ValueError, match="mask_type"):
            RealNVPConfig(
                name="test_realnvp",
                coupling_network=coupling_network,
                input_dim=16,
                mask_type="invalid",
            )

    def test_valid_mask_types(self, coupling_network):
        """Test that valid mask_types are accepted."""
        for mask_type in ["checkerboard", "channel-wise"]:
            config = RealNVPConfig(
                name="test_realnvp",
                coupling_network=coupling_network,
                input_dim=16,
                mask_type=mask_type,
            )
            assert config.mask_type == mask_type


class TestRealNVPConfigSerialization:
    """Test serialization of RealNVPConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(64, 64),
            activation="relu",
        )

    def test_roundtrip(self, coupling_network):
        """Test roundtrip serialization."""
        original = RealNVPConfig(
            name="test_realnvp",
            coupling_network=coupling_network,
            input_dim=16,
            num_coupling_layers=12,
            mask_type="channel-wise",
        )
        data = original.to_dict()
        restored = RealNVPConfig.from_dict(data)
        assert original == restored


# =============================================================================
# GlowConfig Tests
# =============================================================================
class TestGlowConfigBasics:
    """Test basic functionality of GlowConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(512, 512),
            activation="relu",
        )

    def test_create_with_required_fields(self, coupling_network):
        """Test creation with required fields."""
        config = GlowConfig(
            name="test_glow",
            coupling_network=coupling_network,
            input_dim=3 * 32 * 32,  # Flattened image
            image_shape=(32, 32, 3),
        )
        assert config.name == "test_glow"
        assert config.image_shape == (32, 32, 3)

    def test_inherits_from_flow_config(self, coupling_network):
        """Test inheritance from FlowConfig."""
        config = GlowConfig(
            name="test_glow",
            coupling_network=coupling_network,
            input_dim=3 * 32 * 32,
            image_shape=(32, 32, 3),
        )
        assert isinstance(config, FlowConfig)


class TestGlowConfigDefaults:
    """Test default values of GlowConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(512, 512),
            activation="relu",
        )

    def test_default_num_scales(self, coupling_network):
        """Test num_scales defaults to 3."""
        config = GlowConfig(
            name="test_glow",
            coupling_network=coupling_network,
            input_dim=3 * 32 * 32,
            image_shape=(32, 32, 3),
        )
        assert config.num_scales == 3

    def test_default_blocks_per_scale(self, coupling_network):
        """Test blocks_per_scale defaults to 6."""
        config = GlowConfig(
            name="test_glow",
            coupling_network=coupling_network,
            input_dim=3 * 32 * 32,
            image_shape=(32, 32, 3),
        )
        assert config.blocks_per_scale == 6


class TestGlowConfigValidation:
    """Test validation of GlowConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(512, 512),
            activation="relu",
        )

    def test_missing_image_shape(self, coupling_network):
        """Test that missing image_shape raises ValueError."""
        with pytest.raises(ValueError, match="image_shape"):
            GlowConfig(
                name="test_glow",
                coupling_network=coupling_network,
                input_dim=3 * 32 * 32,
            )

    def test_invalid_num_scales(self, coupling_network):
        """Test that non-positive num_scales raises ValueError."""
        with pytest.raises(ValueError, match="num_scales"):
            GlowConfig(
                name="test_glow",
                coupling_network=coupling_network,
                input_dim=3 * 32 * 32,
                image_shape=(32, 32, 3),
                num_scales=0,
            )

    def test_invalid_blocks_per_scale(self, coupling_network):
        """Test that non-positive blocks_per_scale raises ValueError."""
        with pytest.raises(ValueError, match="blocks_per_scale"):
            GlowConfig(
                name="test_glow",
                coupling_network=coupling_network,
                input_dim=3 * 32 * 32,
                image_shape=(32, 32, 3),
                blocks_per_scale=0,
            )


class TestGlowConfigSerialization:
    """Test serialization of GlowConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(512, 512),
            activation="relu",
        )

    def test_roundtrip(self, coupling_network):
        """Test roundtrip serialization."""
        original = GlowConfig(
            name="test_glow",
            coupling_network=coupling_network,
            input_dim=3 * 64 * 64,
            image_shape=(64, 64, 3),
            num_scales=4,
            blocks_per_scale=8,
        )
        data = original.to_dict()
        restored = GlowConfig.from_dict(data)
        assert original == restored


# =============================================================================
# MAFConfig Tests
# =============================================================================
class TestMAFConfigBasics:
    """Test basic functionality of MAFConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(512,),
            activation="relu",
        )

    def test_create_with_required_fields(self, coupling_network):
        """Test creation with required fields."""
        config = MAFConfig(
            name="test_maf",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.name == "test_maf"
        assert config.input_dim == 16

    def test_inherits_from_flow_config(self, coupling_network):
        """Test inheritance from FlowConfig."""
        config = MAFConfig(
            name="test_maf",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert isinstance(config, FlowConfig)


class TestMAFConfigDefaults:
    """Test default values of MAFConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(512,),
            activation="relu",
        )

    def test_default_num_layers(self, coupling_network):
        """Test num_layers defaults to 5."""
        config = MAFConfig(
            name="test_maf",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.num_layers == 5

    def test_default_reverse_ordering(self, coupling_network):
        """Test reverse_ordering defaults to True."""
        config = MAFConfig(
            name="test_maf",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.reverse_ordering is True


class TestMAFConfigValidation:
    """Test validation of MAFConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(512,),
            activation="relu",
        )

    def test_invalid_num_layers(self, coupling_network):
        """Test that non-positive num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers"):
            MAFConfig(
                name="test_maf",
                coupling_network=coupling_network,
                input_dim=16,
                num_layers=0,
            )


class TestMAFConfigSerialization:
    """Test serialization of MAFConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(512,),
            activation="relu",
        )

    def test_roundtrip(self, coupling_network):
        """Test roundtrip serialization."""
        original = MAFConfig(
            name="test_maf",
            coupling_network=coupling_network,
            input_dim=32,
            num_layers=8,
            reverse_ordering=False,
        )
        data = original.to_dict()
        restored = MAFConfig.from_dict(data)
        assert original == restored


# =============================================================================
# IAFConfig Tests
# =============================================================================
class TestIAFConfigBasics:
    """Test basic functionality of IAFConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(512,),
            activation="relu",
        )

    def test_create_with_required_fields(self, coupling_network):
        """Test creation with required fields."""
        config = IAFConfig(
            name="test_iaf",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.name == "test_iaf"
        assert config.input_dim == 16

    def test_inherits_from_flow_config(self, coupling_network):
        """Test inheritance from FlowConfig."""
        config = IAFConfig(
            name="test_iaf",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert isinstance(config, FlowConfig)


class TestIAFConfigDefaults:
    """Test default values of IAFConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(512,),
            activation="relu",
        )

    def test_default_num_layers(self, coupling_network):
        """Test num_layers defaults to 5."""
        config = IAFConfig(
            name="test_iaf",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.num_layers == 5

    def test_default_reverse_ordering(self, coupling_network):
        """Test reverse_ordering defaults to True."""
        config = IAFConfig(
            name="test_iaf",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.reverse_ordering is True


class TestIAFConfigValidation:
    """Test validation of IAFConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(512,),
            activation="relu",
        )

    def test_invalid_num_layers(self, coupling_network):
        """Test that non-positive num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers"):
            IAFConfig(
                name="test_iaf",
                coupling_network=coupling_network,
                input_dim=16,
                num_layers=0,
            )


class TestIAFConfigSerialization:
    """Test serialization of IAFConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(512,),
            activation="relu",
        )

    def test_roundtrip(self, coupling_network):
        """Test roundtrip serialization."""
        original = IAFConfig(
            name="test_iaf",
            coupling_network=coupling_network,
            input_dim=32,
            num_layers=8,
            reverse_ordering=False,
        )
        data = original.to_dict()
        restored = IAFConfig.from_dict(data)
        assert original == restored


# =============================================================================
# NeuralSplineConfig Tests
# =============================================================================
class TestNeuralSplineConfigBasics:
    """Test basic functionality of NeuralSplineConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(128, 128),
            activation="relu",
        )

    def test_create_with_required_fields(self, coupling_network):
        """Test creation with required fields."""
        config = NeuralSplineConfig(
            name="test_nsf",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.name == "test_nsf"
        assert config.input_dim == 16

    def test_inherits_from_flow_config(self, coupling_network):
        """Test inheritance from FlowConfig."""
        config = NeuralSplineConfig(
            name="test_nsf",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert isinstance(config, FlowConfig)


class TestNeuralSplineConfigDefaults:
    """Test default values of NeuralSplineConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(128, 128),
            activation="relu",
        )

    def test_default_num_layers(self, coupling_network):
        """Test num_layers defaults to 8."""
        config = NeuralSplineConfig(
            name="test_nsf",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.num_layers == 8

    def test_default_num_bins(self, coupling_network):
        """Test num_bins defaults to 8."""
        config = NeuralSplineConfig(
            name="test_nsf",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.num_bins == 8

    def test_default_tail_bound(self, coupling_network):
        """Test tail_bound defaults to 3.0."""
        config = NeuralSplineConfig(
            name="test_nsf",
            coupling_network=coupling_network,
            input_dim=16,
        )
        assert config.tail_bound == 3.0


class TestNeuralSplineConfigValidation:
    """Test validation of NeuralSplineConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(128, 128),
            activation="relu",
        )

    def test_invalid_num_layers(self, coupling_network):
        """Test that non-positive num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers"):
            NeuralSplineConfig(
                name="test_nsf",
                coupling_network=coupling_network,
                input_dim=16,
                num_layers=0,
            )

    def test_invalid_num_bins(self, coupling_network):
        """Test that non-positive num_bins raises ValueError."""
        with pytest.raises(ValueError, match="num_bins"):
            NeuralSplineConfig(
                name="test_nsf",
                coupling_network=coupling_network,
                input_dim=16,
                num_bins=0,
            )

    def test_invalid_tail_bound(self, coupling_network):
        """Test that non-positive tail_bound raises ValueError."""
        with pytest.raises(ValueError, match="tail_bound"):
            NeuralSplineConfig(
                name="test_nsf",
                coupling_network=coupling_network,
                input_dim=16,
                tail_bound=0.0,
            )


class TestNeuralSplineConfigSerialization:
    """Test serialization of NeuralSplineConfig."""

    @pytest.fixture
    def coupling_network(self):
        """Create a test coupling network config."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(128, 128),
            activation="relu",
        )

    def test_roundtrip(self, coupling_network):
        """Test roundtrip serialization."""
        original = NeuralSplineConfig(
            name="test_nsf",
            coupling_network=coupling_network,
            input_dim=32,
            num_layers=12,
            num_bins=16,
            tail_bound=5.0,
        )
        data = original.to_dict()
        restored = NeuralSplineConfig.from_dict(data)
        assert original == restored
