"""Tests for the VoxelModel."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.geometric_config import (
    VoxelConfig,
    VoxelNetworkConfig,
)
from artifex.generative_models.models.geometric import VoxelModel


@pytest.fixture
def voxel_config():
    """Create a configuration for testing VoxelModel."""
    network = VoxelNetworkConfig(
        name="test_voxel_network",
        hidden_dims=(128, 64, 32),
        base_channels=128,
        num_layers=4,
        kernel_size=3,
        use_residual=True,
        activation="relu",
    )
    return VoxelConfig(
        name="test_voxel",
        network=network,
        voxel_size=32,
        voxel_dim=1,
        use_sparse=False,
        loss_type="bce",
    )


@pytest.fixture
def voxel_model(voxel_config):
    """Create a VoxelModel instance for testing."""
    rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
    return VoxelModel(voxel_config, rngs=rngs)


class TestVoxelModel:
    """Tests for the VoxelModel class."""

    def test_init(self, voxel_model, voxel_config):
        """Test that a VoxelModel can be initialized."""
        assert isinstance(voxel_model, VoxelModel)
        assert voxel_model.resolution == voxel_config.voxel_size
        assert voxel_model.latent_dim == voxel_config.network.base_channels

    def test_call(self, voxel_model):
        """Test the forward pass of a VoxelModel."""
        batch_size = 2

        # Create input voxel grid [B, resolution, resolution, resolution, 1]
        voxel_grid = jnp.zeros(
            (
                batch_size,
                voxel_model.resolution,
                voxel_model.resolution,
                voxel_model.resolution,
                1,
            )
        )

        # Run forward pass
        output, aux_outputs = voxel_model(voxel_grid)

        # Check output has the same shape as input except for channel dimension
        assert output.shape == voxel_grid.shape[:-1]

        # Check output values are probabilities (between 0 and 1)
        assert jnp.all(output >= 0.0)
        assert jnp.all(output <= 1.0)

    def test_generation_shapes(self, voxel_model):
        """Test shapes of generation outputs."""
        batch_size = 3

        # Call model's generation method with new NNX-compatible API
        generated = voxel_model.generate(
            n_samples=batch_size, rngs=nnx.Rngs(params=jax.random.key(2))
        )

        # Check output shape
        expected_shape = (
            batch_size,
            voxel_model.resolution,
            voxel_model.resolution,
            voxel_model.resolution,
        )
        assert generated.shape == expected_shape

        # Verify values are valid probabilities or binary values
        assert jnp.all(generated >= 0.0)
        assert jnp.all(generated <= 1.0)

    def test_with_deterministic(self, voxel_model):
        """Test model with deterministic flag."""
        batch_size = 2

        # Create input voxel grid
        voxel_grid = jnp.zeros(
            (
                batch_size,
                voxel_model.resolution,
                voxel_model.resolution,
                voxel_model.resolution,
                1,
            )
        )

        # Run with deterministic=True (for inference)
        output_det, aux_det = voxel_model(voxel_grid, deterministic=True)

        # Check shape
        assert output_det.shape == voxel_grid.shape[:-1]

    def test_with_threshold(self, voxel_model):
        """Test model generation with threshold for binary output."""
        batch_size = 2

        # Generate with a threshold using new NNX-compatible API
        generated = voxel_model.generate(
            n_samples=batch_size,
            rngs=nnx.Rngs(params=jax.random.key(3)),
            threshold=0.5,  # Apply threshold for binary output
        )

        # Check that generated values are binary (0 or 1)
        unique_values = jnp.unique(generated)
        assert len(unique_values) <= 2
        # Check if values are close to 0 or 1
        is_zero = jnp.isclose(generated, 0.0)
        is_one = jnp.isclose(generated, 1.0)
        assert jnp.all(jnp.logical_or(is_zero, is_one))

    def test_with_smaller_resolution(self):
        """Test model with a smaller resolution."""
        # Create config with smaller resolution
        network = VoxelNetworkConfig(
            name="small_voxel_network",
            hidden_dims=(64, 32),
            base_channels=64,
            num_layers=3,
            kernel_size=3,
            use_residual=True,
            activation="relu",
        )
        config = VoxelConfig(
            name="test_voxel_small",
            network=network,
            voxel_size=16,
            voxel_dim=1,
            use_sparse=False,
            loss_type="bce",
        )

        # Create model with custom config
        rngs = nnx.Rngs(params=jax.random.key(4))
        small_model = VoxelModel(config, rngs=rngs)

        # Check resolution is as expected
        assert small_model.resolution == 16

        # Test generation with new NNX-compatible API
        batch_size = 2
        generated = small_model.generate(
            n_samples=batch_size,
            rngs=nnx.Rngs(params=jax.random.key(5)),
            threshold=0.5,  # Apply threshold for binary output
        )

        # Check shapes match smaller resolution
        expected_shape = (batch_size, 16, 16, 16)
        assert generated.shape == expected_shape

        # Check that generated values are binary (0 or 1)
        unique_values = jnp.unique(generated)
        assert len(unique_values) <= 2
        assert jnp.all(jnp.logical_or(jnp.isclose(generated, 0.0), jnp.isclose(generated, 1.0)))
