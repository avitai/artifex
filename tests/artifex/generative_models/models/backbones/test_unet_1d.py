"""Tests for UNet1D backbone for 1D signals."""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.backbone_config import (
    create_backbone,
    UNet1DBackboneConfig,
)


def create_unet1d_config(**kwargs) -> UNet1DBackboneConfig:
    """Create a UNet1D config with required fields."""
    defaults = {
        "name": "test_unet1d",
        "hidden_dims": (32, 64, 128),
        "activation": "gelu",
        "in_channels": 1,
        "time_embedding_dim": 64,
    }
    defaults.update(kwargs)
    return UNet1DBackboneConfig(**defaults)


class TestUNet1DBackbone:
    """Tests for UNet1D backbone."""

    @pytest.fixture
    def rngs(self):
        """Create RNG for tests."""
        return nnx.Rngs(42)

    def test_creation_via_factory(self, rngs):
        """Test UNet1D can be created via create_backbone factory."""
        config = create_unet1d_config()
        backbone = create_backbone(config, rngs=rngs)

        assert backbone is not None

    def test_forward_pass_shape(self, rngs):
        """Test forward pass produces correct output shape."""
        config = create_unet1d_config()
        backbone = create_backbone(config, rngs=rngs)

        batch_size = 2
        sequence_length = 64
        x = jnp.ones((batch_size, sequence_length))
        timesteps = jnp.array([10, 20])

        output = backbone(x, timesteps)

        # Output should match input shape
        assert output.shape == (batch_size, sequence_length)

    def test_forward_pass_with_channel_dim(self, rngs):
        """Test forward pass with explicit channel dimension."""
        config = create_unet1d_config()
        backbone = create_backbone(config, rngs=rngs)

        batch_size = 2
        sequence_length = 64
        x = jnp.ones((batch_size, sequence_length, 1))  # Explicit channel dim
        timesteps = jnp.array([10, 20])

        output = backbone(x, timesteps)

        # Output should match input shape
        assert output.shape == (batch_size, sequence_length, 1)

    def test_stereo_audio(self, rngs):
        """Test with 2-channel (stereo) audio."""
        config = create_unet1d_config(in_channels=2)
        backbone = create_backbone(config, rngs=rngs)

        batch_size = 2
        sequence_length = 64
        x = jnp.ones((batch_size, sequence_length, 2))
        timesteps = jnp.array([10, 20])

        output = backbone(x, timesteps)

        assert output.shape == (batch_size, sequence_length, 2)

    def test_deterministic_mode(self, rngs):
        """Test deterministic mode produces consistent outputs."""
        config = create_unet1d_config()
        backbone = create_backbone(config, rngs=rngs)

        x = jnp.ones((1, 64))
        timesteps = jnp.array([10])

        output1 = backbone(x, timesteps, deterministic=True)
        output2 = backbone(x, timesteps, deterministic=True)

        assert jnp.allclose(output1, output2)

    def test_conditioning_parameter_ignored(self, rngs):
        """Test that conditioning parameter is accepted but ignored."""
        config = create_unet1d_config()
        backbone = create_backbone(config, rngs=rngs)

        x = jnp.ones((1, 64))
        timesteps = jnp.array([10])

        # Should work with conditioning=None
        output1 = backbone(x, timesteps, conditioning=None)

        # Should work with conditioning provided (but ignored)
        conditioning = jnp.ones((1, 32))
        output2 = backbone(x, timesteps, conditioning=conditioning)

        # Both should produce finite outputs
        assert jnp.isfinite(output1).all()
        assert jnp.isfinite(output2).all()

    def test_output_is_finite(self, rngs):
        """Test output values are finite."""
        config = create_unet1d_config()
        backbone = create_backbone(config, rngs=rngs)

        x = jnp.ones((2, 128))
        timesteps = jnp.array([10, 50])

        output = backbone(x, timesteps)

        assert jnp.isfinite(output).all()

    def test_different_sequence_lengths(self, rngs):
        """Test with different sequence lengths."""
        config = create_unet1d_config()
        backbone = create_backbone(config, rngs=rngs)

        timesteps = jnp.array([10])

        for seq_len in [32, 64, 128, 256]:
            x = jnp.ones((1, seq_len))
            output = backbone(x, timesteps)

            assert output.shape == (1, seq_len)
            assert jnp.isfinite(output).all()

    def test_config_stored(self, rngs):
        """Test that config is stored on the model."""
        config = create_unet1d_config()
        backbone = create_backbone(config, rngs=rngs)

        assert backbone.config == config
