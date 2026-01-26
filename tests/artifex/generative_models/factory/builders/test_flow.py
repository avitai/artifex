"""Tests for Flow builder using dataclass configs.

These tests verify the FlowBuilder functionality with the new dataclass-based
configuration system (RealNVPConfig, GlowConfig, etc.) following Principle #4.
"""

import jax
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.flow_config import (
    CouplingNetworkConfig,
    FlowConfig,
    GlowConfig,
    IAFConfig,
    MAFConfig,
    NeuralSplineConfig,
    RealNVPConfig,
)
from artifex.generative_models.factory.builders.flow import FlowBuilder


class TestFlowBuilder:
    """Test Flow builder functionality with dataclass configs."""

    @pytest.fixture
    def rngs(self):
        """Create RNGs for testing."""
        key = jax.random.PRNGKey(42)
        return nnx.Rngs(
            params=key,
            dropout=jax.random.fold_in(key, 1),
            sample=jax.random.fold_in(key, 2),
        )

    @pytest.fixture
    def coupling_network_config(self):
        """Create CouplingNetworkConfig for testing."""
        return CouplingNetworkConfig(
            name="test_coupling",
            hidden_dims=(128, 128),
            activation="relu",
        )

    def test_build_normalizing_flow(self, rngs, coupling_network_config):
        """Test building a base NormalizingFlow."""
        config = FlowConfig(
            name="test_flow",
            input_dim=8,
            coupling_network=coupling_network_config,
        )

        builder = FlowBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "inverse")

    def test_build_realnvp(self, rngs, coupling_network_config):
        """Test building a RealNVP flow."""
        config = RealNVPConfig(
            name="test_realnvp",
            input_dim=8,
            coupling_network=coupling_network_config,
            num_coupling_layers=4,
        )

        builder = FlowBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "inverse")

    def test_build_glow(self, rngs, coupling_network_config):
        """Test building a Glow flow."""
        config = GlowConfig(
            name="test_glow",
            input_dim=64,  # 8x8 = 64 flattened
            coupling_network=coupling_network_config,
            image_shape=(8, 8, 1),
            num_scales=2,
            blocks_per_scale=2,
        )

        builder = FlowBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "inverse")

    def test_build_maf(self, rngs, coupling_network_config):
        """Test building a MAF (Masked Autoregressive Flow)."""
        config = MAFConfig(
            name="test_maf",
            input_dim=8,
            coupling_network=coupling_network_config,
            num_layers=4,
        )

        builder = FlowBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "inverse")

    def test_build_iaf(self, rngs, coupling_network_config):
        """Test building an IAF (Inverse Autoregressive Flow)."""
        config = IAFConfig(
            name="test_iaf",
            input_dim=8,
            coupling_network=coupling_network_config,
            num_layers=4,
        )

        builder = FlowBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "inverse")

    def test_build_neural_spline(self, rngs, coupling_network_config):
        """Test building a Neural Spline Flow."""
        config = NeuralSplineConfig(
            name="test_nsf",
            input_dim=8,
            coupling_network=coupling_network_config,
            num_layers=4,
            num_bins=8,
            tail_bound=3.0,
        )

        builder = FlowBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "inverse")

    def test_build_with_none_config(self, rngs):
        """Test that None config raises TypeError."""
        builder = FlowBuilder()

        with pytest.raises(TypeError, match="config cannot be None"):
            builder.build(None, rngs=rngs)

    def test_build_with_dict_config(self, rngs):
        """Test that dict config raises TypeError."""
        builder = FlowBuilder()
        config = {"name": "test", "input_dim": 8}

        with pytest.raises(TypeError, match="config must be a dataclass"):
            builder.build(config, rngs=rngs)

    def test_build_with_invalid_config_type(self, rngs):
        """Test that unsupported config type raises TypeError."""
        builder = FlowBuilder()

        class FakeConfig:
            pass

        fake_config = FakeConfig()

        with pytest.raises(TypeError, match="Unsupported config type"):
            builder.build(fake_config, rngs=rngs)

    def test_config_validation(self):
        """Test that Flow configs properly validate."""
        coupling = CouplingNetworkConfig(
            name="valid_coupling",
            hidden_dims=(64, 64),
            activation="relu",
        )

        # Valid RealNVP config
        valid_realnvp = RealNVPConfig(
            name="valid_realnvp",
            input_dim=8,
            coupling_network=coupling,
            num_coupling_layers=4,
        )
        assert valid_realnvp.num_coupling_layers == 4

        # Valid Glow config
        valid_glow = GlowConfig(
            name="valid_glow",
            input_dim=64,
            coupling_network=coupling,
            image_shape=(8, 8, 1),
            num_scales=3,
            blocks_per_scale=4,
        )
        assert valid_glow.num_scales == 3

        # Valid Neural Spline config
        valid_nsf = NeuralSplineConfig(
            name="valid_nsf",
            input_dim=8,
            coupling_network=coupling,
            num_layers=4,
            num_bins=16,
            tail_bound=5.0,
        )
        assert valid_nsf.num_bins == 16
        assert valid_nsf.tail_bound == 5.0
