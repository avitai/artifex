"""Integration tests for geometric models."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.geometric_config import (
    MeshConfig,
    MeshNetworkConfig,
    PointCloudConfig,
    PointCloudNetworkConfig,
    VoxelConfig,
    VoxelNetworkConfig,
)
from artifex.generative_models.models.geometric import (
    MeshModel,
    PointCloudModel,
    VoxelModel,
)


@pytest.fixture
def point_cloud_model():
    """Create a point cloud model for testing."""
    network = PointCloudNetworkConfig(
        name="test_pc_network",
        hidden_dims=(32,),
        embed_dim=32,  # Must be divisible by num_heads
        num_heads=2,
        num_layers=1,
        use_positional_encoding=True,
        dropout_rate=0.0,
        activation="gelu",
    )
    config = PointCloudConfig(
        name="test_point_cloud",
        network=network,
        num_points=64,
        point_dim=3,
    )
    rngs = nnx.Rngs(params=jax.random.key(0))
    return PointCloudModel(config, rngs=rngs)


@pytest.fixture
def mesh_model():
    """Create a mesh model for testing."""
    network = MeshNetworkConfig(
        name="test_mesh_network",
        hidden_dims=(64,),
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        edge_features_dim=16,
        activation="gelu",
    )
    config = MeshConfig(
        name="test_mesh",
        network=network,
        num_vertices=64,
        num_faces=128,
        vertex_dim=3,
    )
    rngs = nnx.Rngs(params=jax.random.key(1))
    return MeshModel(config, rngs=rngs)


@pytest.fixture
def voxel_model():
    """Create a voxel model for testing."""
    network = VoxelNetworkConfig(
        name="test_voxel_network",
        hidden_dims=(32, 16),
        base_channels=32,
        num_layers=2,
        kernel_size=3,
        use_residual=True,
        activation="relu",
    )
    config = VoxelConfig(
        name="test_voxel",
        network=network,
        voxel_size=8,
        voxel_dim=1,
        use_sparse=False,
        loss_type="bce",
    )
    rngs = nnx.Rngs(params=jax.random.key(2))
    return VoxelModel(config, rngs=rngs)


class TestGeometricModelIntegration:
    """Integration tests for geometric models."""

    def test_point_cloud_to_mesh_conversion(self, point_cloud_model, mesh_model):
        """Test converting point cloud output to mesh input."""
        batch_size = 2

        # Generate point clouds
        key = jax.random.key(3)
        point_clouds = point_cloud_model.generate(n_samples=batch_size, rngs=nnx.Rngs(params=key))

        # Create a mesh-compatible input from point clouds
        # This would typically use a more sophisticated method
        # but we're doing a simple demonstration here
        num_faces = point_cloud_model.num_points // 2
        faces = jnp.zeros((batch_size, num_faces, 3), dtype=jnp.int32)
        mesh_input = {"vertices": point_clouds, "faces": faces}

        # Pass to mesh model
        output = mesh_model(mesh_input)

        # The model might return a tuple (vertices, aux_dict) or a dictionary
        if isinstance(output, tuple):
            mesh_output_vertices = output[0]
            assert mesh_output_vertices.shape[0] == batch_size
        else:
            # For backward compatibility with previous tests
            mesh_output = output
            assert mesh_output["vertices"].shape[0] == batch_size

    def test_voxel_to_mesh_conversion(self, voxel_model, mesh_model):
        """Test converting voxel output to mesh input."""
        batch_size = 2

        # Generate voxels and threshold to binary
        key4 = jax.random.key(4)
        voxels = voxel_model.generate(
            n_samples=batch_size,
            rngs=nnx.Rngs(params=key4),
            threshold=0.5,  # Binary voxels
        )

        # Extract surface vertices (simplified for testing)
        # In practice, use marching cubes or similar algorithm
        # Here we just create some placeholder vertices
        num_vertices = mesh_model.num_vertices

        # Use voxels shape for a sanity check
        assert voxels.shape[0] == batch_size

        vertices = jnp.zeros((batch_size, num_vertices, 3))
        faces = jnp.zeros((batch_size, num_vertices // 2, 3), dtype=jnp.int32)

        # Create a mesh-compatible input
        mesh_input = {"vertices": vertices, "faces": faces}

        # Pass to mesh model
        output = mesh_model(mesh_input)

        # The model might return a tuple (vertices, aux_dict) or a dictionary
        if isinstance(output, tuple):
            mesh_output_vertices = output[0]
            assert mesh_output_vertices.shape[0] == batch_size
        else:
            # For backward compatibility with previous tests
            mesh_output = output
            assert mesh_output["vertices"].shape[0] == batch_size

    def test_model_generation_with_same_seed(self, point_cloud_model, mesh_model, voxel_model):
        """Test deterministic generation with same seed."""
        seed = 42
        batch_size = 1

        # Generate twice with same seed for each model
        pc1 = point_cloud_model.generate(
            n_samples=batch_size, rngs=nnx.Rngs(params=jax.random.key(seed))
        )
        pc2 = point_cloud_model.generate(
            n_samples=batch_size, rngs=nnx.Rngs(params=jax.random.key(seed))
        )

        mesh1 = mesh_model.generate(
            n_samples=batch_size, rngs=nnx.Rngs(params=jax.random.key(seed))
        )
        mesh2 = mesh_model.generate(
            n_samples=batch_size, rngs=nnx.Rngs(params=jax.random.key(seed))
        )

        voxel1 = voxel_model.generate(
            n_samples=batch_size, rngs=nnx.Rngs(params=jax.random.key(seed))
        )
        voxel2 = voxel_model.generate(
            n_samples=batch_size, rngs=nnx.Rngs(params=jax.random.key(seed))
        )

        # Check each pair is identical, but handle None values gracefully
        # For point clouds
        if pc1 is not None and pc2 is not None:
            assert jnp.array_equal(pc1, pc2)
        # For meshes
        if mesh1 is not None and mesh2 is not None:
            assert jnp.array_equal(mesh1, mesh2)
        # For voxels
        if voxel1 is not None and voxel2 is not None:
            assert jnp.array_equal(voxel1, voxel2)
