"""Tests for DiffusionModel Principle #4 compliance.

These tests verify that DiffusionModel follows Principle #4:
"Methods Take Configs, NOT Individual Parameters"

The signature should be:
    DiffusionModel(config: DiffusionConfig, *, rngs: nnx.Rngs)

NOT:
    DiffusionModel(config, backbone_fn, *, rngs)

Following TDD - these tests define the expected behavior.
"""

import inspect

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    DiffusionConfig,
    NoiseScheduleConfig,
)
from artifex.generative_models.core.configuration.backbone_config import (
    UNetBackboneConfig,
)
from artifex.generative_models.models.diffusion.base import DiffusionModel


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def unet_backbone_config():
    """Create a UNetBackboneConfig for testing."""
    return UNetBackboneConfig(
        name="test_unet",
        hidden_dims=(32, 64),
        activation="gelu",
        in_channels=3,
        out_channels=3,
        time_embedding_dim=64,
    )


@pytest.fixture
def noise_schedule_config():
    """Create a NoiseScheduleConfig for testing."""
    return NoiseScheduleConfig(
        name="test_schedule",
        schedule_type="linear",
        num_timesteps=10,  # Small for testing
        beta_start=1e-4,
        beta_end=2e-2,
    )


@pytest.fixture
def diffusion_config(unet_backbone_config, noise_schedule_config):
    """Create a DiffusionConfig for testing."""
    return DiffusionConfig(
        name="test_diffusion",
        input_shape=(8, 8, 3),  # Small for testing
        backbone=unet_backbone_config,
        noise_schedule=noise_schedule_config,
    )


@pytest.fixture
def rngs():
    """Create nnx.Rngs for testing."""
    return nnx.Rngs(0)


# =============================================================================
# Principle #4 Compliance Tests
# =============================================================================


class TestDiffusionModelSignature:
    """Test that DiffusionModel follows Principle #4 signature."""

    def test_init_signature_has_only_config_and_rngs(self):
        """Test that __init__ signature is (config, *, rngs)."""
        sig = inspect.signature(DiffusionModel.__init__)
        params = list(sig.parameters.keys())

        # Should have: self, config, rngs (keyword-only)
        # Should NOT have: backbone_fn or any other extra parameter
        assert "self" in params
        assert "config" in params
        assert "rngs" in params

        # backbone_fn should NOT be in the signature (Principle #4)
        assert "backbone_fn" not in params, (
            "backbone_fn parameter violates Principle #4: "
            "Methods should take configs, NOT individual parameters"
        )

    def test_init_rngs_is_keyword_only(self):
        """Test that rngs parameter is keyword-only."""
        sig = inspect.signature(DiffusionModel.__init__)
        rngs_param = sig.parameters.get("rngs")

        assert rngs_param is not None
        assert rngs_param.kind == inspect.Parameter.KEYWORD_ONLY, (
            "rngs should be a keyword-only parameter"
        )

    def test_create_model_with_config_only(self, diffusion_config, rngs):
        """Test creating DiffusionModel with just config and rngs."""
        # This should work - backbone is created from config.backbone
        model = DiffusionModel(diffusion_config, rngs=rngs)

        assert model is not None
        assert model.backbone is not None
        assert model.config == diffusion_config

    def test_backbone_created_from_config(self, diffusion_config, rngs):
        """Test that backbone is created from config.backbone."""
        model = DiffusionModel(diffusion_config, rngs=rngs)

        # Backbone should exist
        assert model.backbone is not None

        # Backbone should be a UNet (from UNetConfig)
        from artifex.generative_models.models.backbones.unet import UNet

        assert isinstance(model.backbone, UNet)

    def test_model_forward_pass_works(self, diffusion_config, rngs):
        """Test that model forward pass works with config-created backbone."""
        model = DiffusionModel(diffusion_config, rngs=rngs)

        # Create input data
        batch_size = 2
        input_shape = diffusion_config.input_shape
        x = jnp.ones((batch_size, *input_shape))
        timesteps = jnp.zeros((batch_size,), dtype=jnp.int32)

        # Forward pass should work
        output = model(x, timesteps)

        assert isinstance(output, dict)
        assert "predicted_noise" in output
        assert output["predicted_noise"].shape == x.shape


class TestDiffusionModelBackboneFromConfig:
    """Test that DiffusionModel creates correct backbone from config."""

    def test_unet_backbone_from_unet_backbone_config(self, noise_schedule_config, rngs):
        """Test UNet backbone is created from UNetBackboneConfig."""
        unet_backbone_config = UNetBackboneConfig(
            name="unet",
            hidden_dims=(32, 64),
            activation="gelu",
            in_channels=3,
            out_channels=3,
        )
        config = DiffusionConfig(
            name="diffusion",
            input_shape=(8, 8, 3),
            backbone=unet_backbone_config,
            noise_schedule=noise_schedule_config,
        )

        model = DiffusionModel(config, rngs=rngs)

        from artifex.generative_models.models.backbones.unet import UNet

        assert isinstance(model.backbone, UNet)

    def test_backbone_config_preserved(self, diffusion_config, rngs):
        """Test that backbone config is properly used."""
        _model = DiffusionModel(diffusion_config, rngs=rngs)  # noqa: F841

        # The backbone should match the config
        backbone_config = diffusion_config.backbone
        assert backbone_config.hidden_dims == (32, 64)
        assert backbone_config.activation == "gelu"
        assert backbone_config.in_channels == 3


class TestDiffusionModelGeneration:
    """Test that DiffusionModel generation works with config-based backbone."""

    def test_generate_works(self, diffusion_config, rngs):
        """Test that generate method works."""
        model = DiffusionModel(diffusion_config, rngs=rngs)

        samples = model.generate(n_samples=2)

        expected_shape = (2, *diffusion_config.input_shape)
        assert samples.shape == expected_shape
        assert jnp.all(jnp.isfinite(samples))

    def test_loss_fn_works(self, diffusion_config, rngs):
        """Test that loss_fn method works."""
        model = DiffusionModel(diffusion_config, rngs=rngs)

        # Create input data
        batch_size = 2
        x = jnp.ones((batch_size, *diffusion_config.input_shape))
        timesteps = jnp.zeros((batch_size,), dtype=jnp.int32)

        # Forward pass to get model outputs
        model_outputs = model(x, timesteps)

        # Compute loss
        result = model.loss_fn(x, model_outputs)

        assert isinstance(result, dict)
        assert "loss" in result
        assert jnp.isfinite(result["loss"])
