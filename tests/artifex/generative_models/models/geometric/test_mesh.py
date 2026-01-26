"""Tests for the MeshModel."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.geometric_config import (
    MeshConfig,
    MeshNetworkConfig,
)
from artifex.generative_models.models.geometric import MeshModel


@pytest.fixture
def mesh_config():
    """Create a configuration for testing MeshModel."""
    network = MeshNetworkConfig(
        name="test_mesh_network",
        hidden_dims=(128, 64),
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        edge_features_dim=32,
        activation="gelu",
    )
    return MeshConfig(
        name="test_mesh",
        network=network,
        num_vertices=256,
        num_faces=512,
        vertex_dim=3,
    )


@pytest.fixture
def mesh_model(mesh_config):
    """Create a MeshModel instance for testing."""
    rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))
    return MeshModel(mesh_config, rngs=rngs)


class TestMeshModel:
    """Tests for the MeshModel class."""

    def test_init(self, mesh_model, mesh_config):
        """Test that a MeshModel can be initialized."""
        assert isinstance(mesh_model, MeshModel)
        assert mesh_model.embed_dim == mesh_config.network.embed_dim
        assert mesh_model.num_vertices == mesh_config.num_vertices

    def test_call(self, mesh_model):
        """Test the forward pass of a MeshModel."""
        batch_size = 2
        input_dim = 3  # 3D vertex coordinates

        # Create input mesh vertices
        vertices = jnp.ones((batch_size, mesh_model.num_vertices, input_dim))

        # Create a simple mesh representation
        mesh_input = vertices

        # Run forward pass
        outputs = mesh_model(mesh_input)

        # Check output shape matches input shape for vertex coordinates
        assert "vertices" in outputs
        assert outputs["vertices"].shape == vertices.shape
        assert "latent" in outputs
        assert "faces" in outputs

    def test_generation_shapes(self, mesh_model):
        """Test shapes of generation outputs."""
        batch_size = 3

        # Call model's generation method
        generated = mesh_model.sample(n_samples=batch_size, rngs=nnx.Rngs(params=jax.random.key(2)))

        # Check output shape
        assert generated.shape == (batch_size, mesh_model.num_vertices, 3)

    def test_with_deterministic(self, mesh_model):
        """Test model with deterministic flag."""
        batch_size = 2
        input_dim = 3

        # Create input mesh
        vertices = jnp.ones((batch_size, mesh_model.num_vertices, input_dim))

        # Run with deterministic=True (for inference)
        outputs1 = mesh_model(vertices, deterministic=True)
        outputs2 = mesh_model(vertices, deterministic=True)

        # Check output shape matches input shape
        assert outputs1["vertices"].shape == vertices.shape

        # Deterministic runs should be identical
        assert jnp.allclose(outputs1["vertices"], outputs2["vertices"])

    def test_with_custom_template(self):
        """Test model with a custom template mesh."""
        # Create a simple custom template
        num_vertices = 128  # Smaller than default

        # Create new config with custom template
        custom_network = MeshNetworkConfig(
            name="custom_mesh_network",
            hidden_dims=(128, 64),
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            edge_features_dim=32,
            activation="gelu",
        )
        custom_config = MeshConfig(
            name="test_mesh_custom",
            network=custom_network,
            num_vertices=num_vertices,
            num_faces=256,
            vertex_dim=3,
        )

        # Create model with custom template
        rngs = nnx.Rngs(params=jax.random.key(3))
        custom_model = MeshModel(custom_config, rngs=rngs)

        # Check model uses custom template
        assert custom_model.num_vertices == num_vertices

        # Test generation
        batch_size = 2
        generated = custom_model.sample(
            n_samples=batch_size, rngs=nnx.Rngs(params=jax.random.key(4))
        )

        # Check shapes match custom template
        assert generated.shape == (batch_size, num_vertices, 3)
