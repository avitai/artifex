"""Tests for EBM model Principle #4 compliance.

These tests verify that EBM follows Principle #4:
"Methods Take Configs, NOT Individual Parameters"

The signature should be:
    EBM(config: EBMConfig, *, rngs: nnx.Rngs)

NOT:
    EBM(config: ModelConfig, *, rngs, energy_fn, ...)

Following TDD - these tests define the expected behavior.
"""

import inspect

import pytest
from flax import nnx

from artifex.generative_models.core.configuration.energy_config import (
    DeepEBMConfig,
    EBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def rngs():
    """Create nnx.Rngs for testing."""
    return nnx.Rngs(42)


@pytest.fixture
def energy_network_config():
    """Create an EnergyNetworkConfig for testing."""
    return EnergyNetworkConfig(
        name="test_energy_network",
        hidden_dims=(32, 64),
        activation="gelu",
        network_type="mlp",
        use_bias=True,
    )


@pytest.fixture
def mcmc_config():
    """Create an MCMCConfig for testing."""
    return MCMCConfig(
        name="test_mcmc",
        n_steps=10,  # Small for fast testing
        step_size=0.01,
        noise_scale=0.005,
    )


@pytest.fixture
def sample_buffer_config():
    """Create a SampleBufferConfig for testing."""
    return SampleBufferConfig(
        name="test_buffer",
        capacity=100,  # Small for testing
        reinit_prob=0.05,
    )


@pytest.fixture
def ebm_config(energy_network_config, mcmc_config, sample_buffer_config):
    """Create an EBMConfig for testing."""
    return EBMConfig(
        name="test_ebm",
        input_dim=10,
        energy_network=energy_network_config,
        mcmc=mcmc_config,
        sample_buffer=sample_buffer_config,
        alpha=0.01,
    )


@pytest.fixture
def deep_ebm_config(energy_network_config, mcmc_config, sample_buffer_config):
    """Create a DeepEBMConfig for testing."""
    # Use CNN for DeepEBM
    cnn_energy_config = EnergyNetworkConfig(
        name="test_cnn_energy",
        hidden_dims=(16, 32),
        activation="silu",
        network_type="cnn",
        use_bias=True,
    )
    return DeepEBMConfig(
        name="test_deep_ebm",
        input_shape=(8, 8, 3),  # Small for testing
        energy_network=cnn_energy_config,
        mcmc=mcmc_config,
        sample_buffer=sample_buffer_config,
    )


# =============================================================================
# EBM Signature Tests
# =============================================================================


class TestEBMSignature:
    """Test that EBM.__init__ has the correct signature."""

    def test_init_signature_has_only_config_and_rngs(self):
        """Test that __init__ signature is (config, *, rngs)."""
        from artifex.generative_models.models.energy.ebm import EBM

        sig = inspect.signature(EBM.__init__)
        params = list(sig.parameters.keys())

        # Should have: self, config, rngs
        assert "self" in params
        assert "config" in params
        assert "rngs" in params

        # Should NOT have energy_fn
        assert "energy_fn" not in params, "energy_fn violates Principle #4"

    def test_init_rngs_is_keyword_only(self):
        """Test that rngs is a keyword-only parameter."""
        from artifex.generative_models.models.energy.ebm import EBM

        sig = inspect.signature(EBM.__init__)
        rngs_param = sig.parameters.get("rngs")

        assert rngs_param is not None
        assert rngs_param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_init_config_type_is_ebm_config(self):
        """Test that config parameter accepts EBMConfig."""
        from artifex.generative_models.models.energy.ebm import EBM

        sig = inspect.signature(EBM.__init__)
        config_param = sig.parameters.get("config")

        assert config_param is not None
        # The annotation should be EBMConfig (not ModelConfig)
        annotation = config_param.annotation
        # Check that annotation contains EBMConfig
        assert "EBMConfig" in str(annotation) or annotation == EBMConfig


# =============================================================================
# EBM Model Creation Tests
# =============================================================================


class TestEBMModelCreation:
    """Test creating EBM model with EBMConfig."""

    def test_create_ebm_with_ebm_config(self, ebm_config, rngs):
        """Test that EBM can be created with EBMConfig."""
        from artifex.generative_models.models.energy.ebm import EBM

        model = EBM(config=ebm_config, rngs=rngs)

        assert model is not None

    def test_ebm_has_energy_fn_attribute(self, ebm_config, rngs):
        """Test that EBM has energy_fn attribute after creation."""
        from artifex.generative_models.models.energy.ebm import EBM

        model = EBM(config=ebm_config, rngs=rngs)

        assert hasattr(model, "energy_fn")
        assert model.energy_fn is not None

    def test_ebm_creates_mlp_energy_from_config(self, ebm_config, rngs):
        """Test that EBM creates MLPEnergyFunction from config."""
        from artifex.generative_models.models.energy.base import MLPEnergyFunction
        from artifex.generative_models.models.energy.ebm import EBM

        model = EBM(config=ebm_config, rngs=rngs)

        # Config has network_type="mlp", so should create MLPEnergyFunction
        assert isinstance(model.energy_fn, MLPEnergyFunction)

    def test_ebm_config_values_preserved(self, ebm_config, rngs):
        """Test that config values are preserved in model."""
        from artifex.generative_models.models.energy.ebm import EBM

        model = EBM(config=ebm_config, rngs=rngs)

        # Check MCMC parameters
        assert model.mcmc_steps == ebm_config.mcmc.n_steps
        assert model.alpha == ebm_config.alpha


# =============================================================================
# DeepEBM Signature Tests
# =============================================================================


class TestDeepEBMSignature:
    """Test that DeepEBM.__init__ has the correct signature."""

    def test_init_signature_has_only_config_and_rngs(self):
        """Test that __init__ signature is (config, *, rngs)."""
        from artifex.generative_models.models.energy.ebm import DeepEBM

        sig = inspect.signature(DeepEBM.__init__)
        params = list(sig.parameters.keys())

        # Should have: self, config, rngs
        assert "self" in params
        assert "config" in params
        assert "rngs" in params

        # Should NOT have energy_fn
        assert "energy_fn" not in params, "energy_fn violates Principle #4"


# =============================================================================
# DeepEBM Model Creation Tests
# =============================================================================


class TestDeepEBMModelCreation:
    """Test creating DeepEBM model with DeepEBMConfig."""

    def test_create_deep_ebm_with_config(self, deep_ebm_config, rngs):
        """Test that DeepEBM can be created with DeepEBMConfig."""
        from artifex.generative_models.models.energy.ebm import DeepEBM

        model = DeepEBM(config=deep_ebm_config, rngs=rngs)

        assert model is not None

    def test_deep_ebm_creates_cnn_energy_from_config(self, deep_ebm_config, rngs):
        """Test that DeepEBM creates DeepCNNEnergyFunction from config."""
        from artifex.generative_models.models.energy.ebm import DeepCNNEnergyFunction, DeepEBM

        model = DeepEBM(config=deep_ebm_config, rngs=rngs)

        # Config has network_type="cnn", DeepEBM uses DeepCNNEnergyFunction
        assert isinstance(model.energy_fn, DeepCNNEnergyFunction)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestEBMErrorHandling:
    """Test error handling in EBM model."""

    def test_reject_model_configuration(self, rngs):
        """Test that ModelConfig is rejected."""
        from artifex.generative_models.core.configuration import ModelConfig
        from artifex.generative_models.models.energy.ebm import EBM

        # Old-style Pydantic config should be rejected
        old_config = ModelConfig(
            name="test",
            model_class="test",
            input_dim=10,
            hidden_dims=[32, 64],
            output_dim=1,
        )

        with pytest.raises(TypeError):
            EBM(config=old_config, rngs=rngs)

    def test_reject_dict_config(self, rngs):
        """Test that dict config is rejected."""
        from artifex.generative_models.models.energy.ebm import EBM

        dict_config = {"input_dim": 10}

        with pytest.raises(TypeError):
            EBM(config=dict_config, rngs=rngs)
