"""Tests for the PointCloudModel."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)
from artifex.generative_models.models.geometric import PointCloudModel


@pytest.fixture
def point_cloud_config():
    """Create a configuration for testing PointCloudModel."""
    network_config = PointCloudNetworkConfig(
        name="test_point_cloud_network",
        hidden_dims=(64, 64),
        activation="relu",
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
    )
    return PointCloudConfig(
        name="test_point_cloud",
        num_points=128,
        network=network_config,
    )


@pytest.fixture
def point_cloud_model(point_cloud_config):
    """Create a PointCloudModel instance for testing."""
    rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
    return PointCloudModel(point_cloud_config, rngs=rngs)


class TestPointCloudModel:
    """Tests for the PointCloudModel class."""

    def test_init(self, point_cloud_model, point_cloud_config):
        """Test that a PointCloudModel can be initialized."""
        assert isinstance(point_cloud_model, PointCloudModel)
        assert point_cloud_model.embed_dim == point_cloud_config.network.embed_dim
        assert point_cloud_model.num_points == point_cloud_config.num_points

    def test_call(self, point_cloud_model):
        """Test the forward pass of a PointCloudModel."""
        batch_size = 2
        input_dim = 3  # 3D coordinates

        # Create input point cloud
        x = jnp.ones((batch_size, point_cloud_model.num_points, input_dim))

        # Run forward pass
        outputs = point_cloud_model(x)

        # Check output shape matches input shape for coordinates
        assert outputs["positions"].shape == x.shape
        assert "embeddings" in outputs
        assert "extension_outputs" in outputs

    def test_generation_shapes(self, point_cloud_model):
        """Test shapes of generation outputs."""
        batch_size = 3

        # Call model's generation method
        generated = point_cloud_model.sample(
            n_samples=batch_size, rngs=nnx.Rngs(params=jax.random.key(2))
        )

        # Check shape
        assert generated.shape[0] == batch_size
        assert generated.shape[1] == point_cloud_model.num_points
        assert generated.shape[2] == 3  # 3D coordinates

    def test_with_deterministic(self, point_cloud_model):
        """Test model with deterministic flag."""
        batch_size = 2
        input_dim = 3

        # Create input point cloud
        x = jnp.ones((batch_size, point_cloud_model.num_points, input_dim))

        # Run with deterministic=True (for inference)
        outputs1 = point_cloud_model(x, deterministic=True)
        outputs2 = point_cloud_model(x, deterministic=True)

        # Check shape
        assert outputs1["positions"].shape == x.shape

        # Deterministic runs should be identical
        assert jnp.allclose(outputs1["positions"], outputs2["positions"])

    def test_with_rngs(self, point_cloud_model):
        """Test model with explicit RNGs."""
        batch_size = 2
        input_dim = 3

        # Create input point cloud
        x = jnp.ones((batch_size, point_cloud_model.num_points, input_dim))

        # Run without rngs parameter (removed in refactoring)
        outputs = point_cloud_model(x)

        # Check shape
        assert outputs["positions"].shape == x.shape
