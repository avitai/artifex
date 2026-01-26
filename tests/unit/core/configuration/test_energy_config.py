"""Tests for Energy-Based Model configuration classes.

This module tests the energy configuration classes using the TDD approach.
All tests should pass after implementing the energy_config.py module.
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.base_network import BaseNetworkConfig
from artifex.generative_models.core.configuration.energy_config import (
    DeepEBMConfig,
    EBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)


# =============================================================================
# EnergyNetworkConfig Tests
# =============================================================================
class TestEnergyNetworkConfigBasics:
    """Test basic functionality of EnergyNetworkConfig."""

    def test_create_with_required_fields(self):
        """Test creation with required fields."""
        config = EnergyNetworkConfig(
            name="energy_net",
            hidden_dims=(64, 128, 64),
            activation="silu",
        )
        assert config.name == "energy_net"
        assert config.hidden_dims == (64, 128, 64)
        assert config.activation == "silu"

    def test_frozen(self):
        """Test that config is frozen."""
        config = EnergyNetworkConfig(
            name="energy_net",
            hidden_dims=(64,),
            activation="gelu",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"  # type: ignore

    def test_inherits_from_base_network_config(self):
        """Test inheritance from BaseNetworkConfig."""
        config = EnergyNetworkConfig(
            name="energy_net",
            hidden_dims=(64,),
            activation="gelu",
        )
        assert isinstance(config, BaseNetworkConfig)


class TestEnergyNetworkConfigDefaults:
    """Test default values of EnergyNetworkConfig."""

    def test_default_network_type(self):
        """Test network_type defaults to 'mlp'."""
        config = EnergyNetworkConfig(
            name="energy_net",
            hidden_dims=(64,),
            activation="gelu",
        )
        assert config.network_type == "mlp"

    def test_default_use_bias(self):
        """Test use_bias defaults to True."""
        config = EnergyNetworkConfig(
            name="energy_net",
            hidden_dims=(64,),
            activation="gelu",
        )
        assert config.use_bias is True

    def test_default_use_spectral_norm(self):
        """Test use_spectral_norm defaults to False."""
        config = EnergyNetworkConfig(
            name="energy_net",
            hidden_dims=(64,),
            activation="gelu",
        )
        assert config.use_spectral_norm is False

    def test_default_use_residual(self):
        """Test use_residual defaults to False."""
        config = EnergyNetworkConfig(
            name="energy_net",
            hidden_dims=(64,),
            activation="gelu",
        )
        assert config.use_residual is False


class TestEnergyNetworkConfigValidation:
    """Test validation of EnergyNetworkConfig."""

    def test_invalid_network_type(self):
        """Test that invalid network_type raises ValueError."""
        with pytest.raises(ValueError, match="network_type"):
            EnergyNetworkConfig(
                name="energy_net",
                hidden_dims=(64,),
                activation="gelu",
                network_type="invalid",
            )

    def test_valid_network_types(self):
        """Test that valid network_types are accepted."""
        for net_type in ["mlp", "cnn"]:
            config = EnergyNetworkConfig(
                name="energy_net",
                hidden_dims=(64,),
                activation="gelu",
                network_type=net_type,
            )
            assert config.network_type == net_type


class TestEnergyNetworkConfigSerialization:
    """Test serialization of EnergyNetworkConfig."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = EnergyNetworkConfig(
            name="energy_net",
            hidden_dims=(64, 128),
            activation="silu",
            network_type="cnn",
            use_spectral_norm=True,
        )
        data = config.to_dict()
        assert data["name"] == "energy_net"
        assert data["hidden_dims"] == (64, 128)
        assert data["network_type"] == "cnn"
        assert data["use_spectral_norm"] is True

    def test_from_dict(self):
        """Test from_dict creation."""
        data = {
            "name": "energy_net",
            "hidden_dims": [64, 128],  # List should be converted to tuple
            "activation": "silu",
            "network_type": "cnn",
        }
        config = EnergyNetworkConfig.from_dict(data)
        assert config.hidden_dims == (64, 128)
        assert config.network_type == "cnn"

    def test_roundtrip(self):
        """Test roundtrip serialization."""
        original = EnergyNetworkConfig(
            name="energy_net",
            hidden_dims=(32, 64, 128),
            activation="relu",
            network_type="mlp",
            use_bias=False,
            use_spectral_norm=True,
            use_residual=True,
        )
        data = original.to_dict()
        restored = EnergyNetworkConfig.from_dict(data)
        assert original == restored


# =============================================================================
# MCMCConfig Tests
# =============================================================================
class TestMCMCConfigBasics:
    """Test basic functionality of MCMCConfig."""

    def test_create_with_required_fields(self):
        """Test creation with required fields."""
        config = MCMCConfig(name="mcmc_config")
        assert config.name == "mcmc_config"

    def test_frozen(self):
        """Test that config is frozen."""
        config = MCMCConfig(name="mcmc_config")
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"  # type: ignore

    def test_inherits_from_base_config(self):
        """Test inheritance from BaseConfig."""
        config = MCMCConfig(name="mcmc_config")
        assert isinstance(config, BaseConfig)


class TestMCMCConfigDefaults:
    """Test default values of MCMCConfig."""

    def test_default_n_steps(self):
        """Test n_steps defaults to 60."""
        config = MCMCConfig(name="mcmc_config")
        assert config.n_steps == 60

    def test_default_step_size(self):
        """Test step_size defaults to 0.01."""
        config = MCMCConfig(name="mcmc_config")
        assert config.step_size == 0.01

    def test_default_noise_scale(self):
        """Test noise_scale defaults to 0.005."""
        config = MCMCConfig(name="mcmc_config")
        assert config.noise_scale == 0.005

    def test_default_clip_value(self):
        """Test clip_value defaults to 1.0."""
        config = MCMCConfig(name="mcmc_config")
        assert config.clip_value == 1.0


class TestMCMCConfigValidation:
    """Test validation of MCMCConfig."""

    def test_invalid_n_steps(self):
        """Test that non-positive n_steps raises ValueError."""
        with pytest.raises(ValueError, match="n_steps"):
            MCMCConfig(name="mcmc_config", n_steps=0)

    def test_invalid_step_size(self):
        """Test that non-positive step_size raises ValueError."""
        with pytest.raises(ValueError, match="step_size"):
            MCMCConfig(name="mcmc_config", step_size=0.0)

    def test_invalid_noise_scale(self):
        """Test that negative noise_scale raises ValueError."""
        with pytest.raises(ValueError, match="noise_scale"):
            MCMCConfig(name="mcmc_config", noise_scale=-0.1)

    def test_zero_noise_scale_allowed(self):
        """Test that zero noise_scale is allowed (deterministic)."""
        config = MCMCConfig(name="mcmc_config", noise_scale=0.0)
        assert config.noise_scale == 0.0

    def test_invalid_clip_value(self):
        """Test that non-positive clip_value raises ValueError."""
        with pytest.raises(ValueError, match="clip_value"):
            MCMCConfig(name="mcmc_config", clip_value=0.0)


class TestMCMCConfigSerialization:
    """Test serialization of MCMCConfig."""

    def test_roundtrip(self):
        """Test roundtrip serialization."""
        original = MCMCConfig(
            name="mcmc_config",
            n_steps=100,
            step_size=0.005,
            noise_scale=0.001,
            clip_value=2.0,
        )
        data = original.to_dict()
        restored = MCMCConfig.from_dict(data)
        assert original == restored


# =============================================================================
# SampleBufferConfig Tests
# =============================================================================
class TestSampleBufferConfigBasics:
    """Test basic functionality of SampleBufferConfig."""

    def test_create_with_required_fields(self):
        """Test creation with required fields."""
        config = SampleBufferConfig(name="buffer_config")
        assert config.name == "buffer_config"

    def test_frozen(self):
        """Test that config is frozen."""
        config = SampleBufferConfig(name="buffer_config")
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"  # type: ignore


class TestSampleBufferConfigDefaults:
    """Test default values of SampleBufferConfig."""

    def test_default_capacity(self):
        """Test capacity defaults to 8192."""
        config = SampleBufferConfig(name="buffer_config")
        assert config.capacity == 8192

    def test_default_reinit_prob(self):
        """Test reinit_prob defaults to 0.05."""
        config = SampleBufferConfig(name="buffer_config")
        assert config.reinit_prob == 0.05


class TestSampleBufferConfigValidation:
    """Test validation of SampleBufferConfig."""

    def test_invalid_capacity(self):
        """Test that non-positive capacity raises ValueError."""
        with pytest.raises(ValueError, match="capacity"):
            SampleBufferConfig(name="buffer_config", capacity=0)

    def test_invalid_reinit_prob_negative(self):
        """Test that negative reinit_prob raises ValueError."""
        with pytest.raises(ValueError, match="reinit_prob"):
            SampleBufferConfig(name="buffer_config", reinit_prob=-0.1)

    def test_invalid_reinit_prob_over_one(self):
        """Test that reinit_prob > 1 raises ValueError."""
        with pytest.raises(ValueError, match="reinit_prob"):
            SampleBufferConfig(name="buffer_config", reinit_prob=1.5)

    def test_valid_reinit_prob_zero(self):
        """Test that zero reinit_prob is valid."""
        config = SampleBufferConfig(name="buffer_config", reinit_prob=0.0)
        assert config.reinit_prob == 0.0

    def test_valid_reinit_prob_one(self):
        """Test that reinit_prob=1.0 is valid."""
        config = SampleBufferConfig(name="buffer_config", reinit_prob=1.0)
        assert config.reinit_prob == 1.0


class TestSampleBufferConfigSerialization:
    """Test serialization of SampleBufferConfig."""

    def test_roundtrip(self):
        """Test roundtrip serialization."""
        original = SampleBufferConfig(
            name="buffer_config",
            capacity=4096,
            reinit_prob=0.1,
        )
        data = original.to_dict()
        restored = SampleBufferConfig.from_dict(data)
        assert original == restored


# =============================================================================
# EBMConfig Tests
# =============================================================================
class TestEBMConfigBasics:
    """Test basic functionality of EBMConfig."""

    @pytest.fixture
    def energy_network(self):
        """Create a test energy network config."""
        return EnergyNetworkConfig(
            name="test_energy_net",
            hidden_dims=(64, 128, 64),
            activation="silu",
        )

    @pytest.fixture
    def mcmc_config(self):
        """Create a test MCMC config."""
        return MCMCConfig(name="test_mcmc")

    @pytest.fixture
    def sample_buffer(self):
        """Create a test sample buffer config."""
        return SampleBufferConfig(name="test_buffer")

    def test_create_with_required_fields(self, energy_network, mcmc_config, sample_buffer):
        """Test creation with required fields."""
        config = EBMConfig(
            name="test_ebm",
            input_dim=784,
            energy_network=energy_network,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer,
        )
        assert config.name == "test_ebm"
        assert config.input_dim == 784
        assert config.energy_network == energy_network
        assert config.mcmc == mcmc_config
        assert config.sample_buffer == sample_buffer

    def test_frozen(self, energy_network, mcmc_config, sample_buffer):
        """Test that config is frozen."""
        config = EBMConfig(
            name="test_ebm",
            input_dim=784,
            energy_network=energy_network,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"  # type: ignore

    def test_inherits_from_base_config(self, energy_network, mcmc_config, sample_buffer):
        """Test inheritance from BaseConfig."""
        config = EBMConfig(
            name="test_ebm",
            input_dim=784,
            energy_network=energy_network,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer,
        )
        assert isinstance(config, BaseConfig)

    def test_missing_energy_network_raises_error(self, mcmc_config, sample_buffer):
        """Test that missing energy_network raises ValueError."""
        with pytest.raises(ValueError, match="energy_network.*required"):
            EBMConfig(
                name="test_ebm",
                input_dim=784,
                mcmc=mcmc_config,
                sample_buffer=sample_buffer,
            )

    def test_missing_mcmc_raises_error(self, energy_network, sample_buffer):
        """Test that missing mcmc raises ValueError."""
        with pytest.raises(ValueError, match="mcmc.*required"):
            EBMConfig(
                name="test_ebm",
                input_dim=784,
                energy_network=energy_network,
                sample_buffer=sample_buffer,
            )

    def test_missing_sample_buffer_raises_error(self, energy_network, mcmc_config):
        """Test that missing sample_buffer raises ValueError."""
        with pytest.raises(ValueError, match="sample_buffer.*required"):
            EBMConfig(
                name="test_ebm",
                input_dim=784,
                energy_network=energy_network,
                mcmc=mcmc_config,
            )


class TestEBMConfigDefaults:
    """Test default values of EBMConfig."""

    @pytest.fixture
    def energy_network(self):
        """Create a test energy network config."""
        return EnergyNetworkConfig(
            name="test_energy_net",
            hidden_dims=(64, 128, 64),
            activation="silu",
        )

    @pytest.fixture
    def mcmc_config(self):
        """Create a test MCMC config."""
        return MCMCConfig(name="test_mcmc")

    @pytest.fixture
    def sample_buffer(self):
        """Create a test sample buffer config."""
        return SampleBufferConfig(name="test_buffer")

    def test_default_alpha(self, energy_network, mcmc_config, sample_buffer):
        """Test alpha defaults to 0.01."""
        config = EBMConfig(
            name="test_ebm",
            input_dim=784,
            energy_network=energy_network,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer,
        )
        assert config.alpha == 0.01


class TestEBMConfigValidation:
    """Test validation of EBMConfig."""

    @pytest.fixture
    def energy_network(self):
        """Create a test energy network config."""
        return EnergyNetworkConfig(
            name="test_energy_net",
            hidden_dims=(64, 128, 64),
            activation="silu",
        )

    @pytest.fixture
    def mcmc_config(self):
        """Create a test MCMC config."""
        return MCMCConfig(name="test_mcmc")

    @pytest.fixture
    def sample_buffer(self):
        """Create a test sample buffer config."""
        return SampleBufferConfig(name="test_buffer")

    def test_invalid_input_dim(self, energy_network, mcmc_config, sample_buffer):
        """Test that non-positive input_dim raises ValueError."""
        with pytest.raises(ValueError, match="input_dim"):
            EBMConfig(
                name="test_ebm",
                input_dim=0,
                energy_network=energy_network,
                mcmc=mcmc_config,
                sample_buffer=sample_buffer,
            )

    def test_invalid_alpha_negative(self, energy_network, mcmc_config, sample_buffer):
        """Test that negative alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            EBMConfig(
                name="test_ebm",
                input_dim=784,
                energy_network=energy_network,
                mcmc=mcmc_config,
                sample_buffer=sample_buffer,
                alpha=-0.01,
            )

    def test_zero_alpha_allowed(self, energy_network, mcmc_config, sample_buffer):
        """Test that zero alpha is allowed (no regularization)."""
        config = EBMConfig(
            name="test_ebm",
            input_dim=784,
            energy_network=energy_network,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer,
            alpha=0.0,
        )
        assert config.alpha == 0.0

    def test_invalid_energy_network_type(self, mcmc_config, sample_buffer):
        """Test that wrong type for energy_network raises TypeError."""
        with pytest.raises(TypeError, match="energy_network must be EnergyNetworkConfig"):
            EBMConfig(
                name="test_ebm",
                input_dim=784,
                energy_network={"hidden_dims": (64,)},  # type: ignore
                mcmc=mcmc_config,
                sample_buffer=sample_buffer,
            )


class TestEBMConfigSerialization:
    """Test serialization of EBMConfig."""

    @pytest.fixture
    def energy_network(self):
        """Create a test energy network config."""
        return EnergyNetworkConfig(
            name="test_energy_net",
            hidden_dims=(64, 128, 64),
            activation="silu",
        )

    @pytest.fixture
    def mcmc_config(self):
        """Create a test MCMC config."""
        return MCMCConfig(name="test_mcmc")

    @pytest.fixture
    def sample_buffer(self):
        """Create a test sample buffer config."""
        return SampleBufferConfig(name="test_buffer")

    def test_to_dict(self, energy_network, mcmc_config, sample_buffer):
        """Test to_dict conversion with nested configs."""
        config = EBMConfig(
            name="test_ebm",
            input_dim=784,
            energy_network=energy_network,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer,
            alpha=0.001,
        )
        data = config.to_dict()
        assert data["name"] == "test_ebm"
        assert data["input_dim"] == 784
        assert data["alpha"] == 0.001
        # Nested configs should be converted to dicts
        assert isinstance(data["energy_network"], dict)
        assert isinstance(data["mcmc"], dict)
        assert isinstance(data["sample_buffer"], dict)

    def test_from_dict(self):
        """Test from_dict handles nested configs."""
        data = {
            "name": "test_ebm",
            "input_dim": 784,
            "energy_network": {
                "name": "energy_net",
                "hidden_dims": [64, 128],
                "activation": "silu",
            },
            "mcmc": {
                "name": "mcmc",
                "n_steps": 100,
            },
            "sample_buffer": {
                "name": "buffer",
                "capacity": 4096,
            },
        }
        config = EBMConfig.from_dict(data)
        assert config.name == "test_ebm"
        assert isinstance(config.energy_network, EnergyNetworkConfig)
        assert isinstance(config.mcmc, MCMCConfig)
        assert isinstance(config.sample_buffer, SampleBufferConfig)
        assert config.energy_network.hidden_dims == (64, 128)
        assert config.mcmc.n_steps == 100
        assert config.sample_buffer.capacity == 4096

    def test_roundtrip(self, energy_network, mcmc_config, sample_buffer):
        """Test roundtrip serialization."""
        original = EBMConfig(
            name="test_ebm",
            input_dim=784,
            energy_network=energy_network,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer,
            alpha=0.005,
        )
        data = original.to_dict()
        restored = EBMConfig.from_dict(data)
        assert original == restored


# =============================================================================
# DeepEBMConfig Tests
# =============================================================================
class TestDeepEBMConfigBasics:
    """Test basic functionality of DeepEBMConfig."""

    @pytest.fixture
    def energy_network(self):
        """Create a test energy network config for deep EBM."""
        return EnergyNetworkConfig(
            name="deep_energy_net",
            hidden_dims=(128, 256, 512),
            activation="silu",
            network_type="cnn",
            use_residual=True,
            use_spectral_norm=True,
        )

    @pytest.fixture
    def mcmc_config(self):
        """Create a test MCMC config for deep EBM."""
        return MCMCConfig(name="deep_mcmc", n_steps=100, step_size=0.005)

    @pytest.fixture
    def sample_buffer(self):
        """Create a test sample buffer config."""
        return SampleBufferConfig(name="test_buffer", capacity=8192)

    def test_create_with_required_fields(self, energy_network, mcmc_config, sample_buffer):
        """Test creation with required fields."""
        config = DeepEBMConfig(
            name="test_deep_ebm",
            input_shape=(32, 32, 3),
            energy_network=energy_network,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer,
        )
        assert config.name == "test_deep_ebm"
        assert config.input_shape == (32, 32, 3)

    def test_inherits_from_ebm_config(self, energy_network, mcmc_config, sample_buffer):
        """Test inheritance from EBMConfig."""
        config = DeepEBMConfig(
            name="test_deep_ebm",
            input_shape=(32, 32, 3),
            energy_network=energy_network,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer,
        )
        assert isinstance(config, EBMConfig)


class TestDeepEBMConfigDefaults:
    """Test default values of DeepEBMConfig."""

    @pytest.fixture
    def energy_network(self):
        """Create a test energy network config."""
        return EnergyNetworkConfig(
            name="deep_energy_net",
            hidden_dims=(128, 256),
            activation="silu",
            network_type="cnn",
        )

    @pytest.fixture
    def mcmc_config(self):
        """Create a test MCMC config."""
        return MCMCConfig(name="deep_mcmc")

    @pytest.fixture
    def sample_buffer(self):
        """Create a test sample buffer config."""
        return SampleBufferConfig(name="test_buffer")

    def test_default_alpha_for_deep(self, energy_network, mcmc_config, sample_buffer):
        """Test alpha defaults to 0.001 for DeepEBM (lower than standard EBM)."""
        config = DeepEBMConfig(
            name="test_deep_ebm",
            input_shape=(32, 32, 3),
            energy_network=energy_network,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer,
        )
        assert config.alpha == 0.001


class TestDeepEBMConfigValidation:
    """Test validation of DeepEBMConfig."""

    @pytest.fixture
    def energy_network(self):
        """Create a test energy network config."""
        return EnergyNetworkConfig(
            name="deep_energy_net",
            hidden_dims=(128, 256),
            activation="silu",
            network_type="cnn",
        )

    @pytest.fixture
    def mcmc_config(self):
        """Create a test MCMC config."""
        return MCMCConfig(name="deep_mcmc")

    @pytest.fixture
    def sample_buffer(self):
        """Create a test sample buffer config."""
        return SampleBufferConfig(name="test_buffer")

    def test_missing_input_shape_raises_error(self, energy_network, mcmc_config, sample_buffer):
        """Test that missing input_shape raises ValueError."""
        with pytest.raises(ValueError, match="input_shape.*required"):
            DeepEBMConfig(
                name="test_deep_ebm",
                energy_network=energy_network,
                mcmc=mcmc_config,
                sample_buffer=sample_buffer,
            )

    def test_invalid_input_shape_dimensions(self, energy_network, mcmc_config, sample_buffer):
        """Test that invalid input_shape raises ValueError."""
        with pytest.raises(ValueError, match="input_shape"):
            DeepEBMConfig(
                name="test_deep_ebm",
                input_shape=(32, 32),  # Missing channels
                energy_network=energy_network,
                mcmc=mcmc_config,
                sample_buffer=sample_buffer,
            )


class TestDeepEBMConfigDerivedProperties:
    """Test derived properties of DeepEBMConfig."""

    @pytest.fixture
    def energy_network(self):
        """Create a test energy network config."""
        return EnergyNetworkConfig(
            name="deep_energy_net",
            hidden_dims=(128, 256),
            activation="silu",
            network_type="cnn",
        )

    @pytest.fixture
    def mcmc_config(self):
        """Create a test MCMC config."""
        return MCMCConfig(name="deep_mcmc")

    @pytest.fixture
    def sample_buffer(self):
        """Create a test sample buffer config."""
        return SampleBufferConfig(name="test_buffer")

    def test_derived_input_dim(self, energy_network, mcmc_config, sample_buffer):
        """Test derived_input_dim returns flattened size."""
        config = DeepEBMConfig(
            name="test_deep_ebm",
            input_shape=(32, 32, 3),
            energy_network=energy_network,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer,
        )
        assert config.derived_input_dim == 32 * 32 * 3


class TestDeepEBMConfigSerialization:
    """Test serialization of DeepEBMConfig."""

    @pytest.fixture
    def energy_network(self):
        """Create a test energy network config."""
        return EnergyNetworkConfig(
            name="deep_energy_net",
            hidden_dims=(128, 256, 512),
            activation="silu",
            network_type="cnn",
            use_residual=True,
        )

    @pytest.fixture
    def mcmc_config(self):
        """Create a test MCMC config."""
        return MCMCConfig(name="deep_mcmc", n_steps=100)

    @pytest.fixture
    def sample_buffer(self):
        """Create a test sample buffer config."""
        return SampleBufferConfig(name="test_buffer", capacity=8192)

    def test_from_dict_with_input_shape(self):
        """Test from_dict handles input_shape tuple conversion."""
        data = {
            "name": "deep_ebm",
            "input_shape": [64, 64, 3],  # List should be converted to tuple
            "energy_network": {
                "name": "energy_net",
                "hidden_dims": [128, 256],
                "activation": "silu",
                "network_type": "cnn",
            },
            "mcmc": {"name": "mcmc"},
            "sample_buffer": {"name": "buffer"},
        }
        config = DeepEBMConfig.from_dict(data)
        assert config.input_shape == (64, 64, 3)

    def test_roundtrip(self, energy_network, mcmc_config, sample_buffer):
        """Test roundtrip serialization."""
        original = DeepEBMConfig(
            name="test_deep_ebm",
            input_shape=(64, 64, 3),
            energy_network=energy_network,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer,
            alpha=0.0005,
        )
        data = original.to_dict()
        restored = DeepEBMConfig.from_dict(data)
        assert original == restored
