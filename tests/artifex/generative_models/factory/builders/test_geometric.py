"""Tests for Geometric builder using dataclass configs.

These tests verify the GeometricBuilder functionality with the new dataclass-based
configuration system (PointCloudConfig, MeshConfig, etc.) following Principle #4.

Note: The geometric models have not been fully migrated to dataclass configs yet,
so some tests are skipped until the models are updated.
"""

import jax
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.geometric_config import (
    GraphConfig,
    GraphNetworkConfig,
    MeshConfig,
    MeshNetworkConfig,
    PointCloudConfig,
    PointCloudNetworkConfig,
    VoxelConfig,
    VoxelNetworkConfig,
)
from artifex.generative_models.factory.builders.geometric import GeometricBuilder


class TestGeometricBuilder:
    """Test Geometric builder functionality with dataclass configs."""

    @pytest.fixture
    def rngs(self):
        """Create RNGs for testing."""
        key = jax.random.PRNGKey(42)
        return nnx.Rngs(
            params=key,
            dropout=jax.random.fold_in(key, 1),
        )

    @pytest.fixture
    def point_cloud_network_config(self):
        """Create PointCloudNetworkConfig for testing."""
        return PointCloudNetworkConfig(
            name="test_pc_network",
            hidden_dims=(256, 512),
            activation="relu",
            embed_dim=256,
            num_heads=8,
            num_layers=6,
        )

    @pytest.fixture
    def mesh_network_config(self):
        """Create MeshNetworkConfig for testing."""
        return MeshNetworkConfig(
            name="test_mesh_network",
            hidden_dims=(256, 512),
            activation="relu",
            embed_dim=256,
            num_heads=8,
            num_layers=4,
        )

    @pytest.fixture
    def voxel_network_config(self):
        """Create VoxelNetworkConfig for testing."""
        return VoxelNetworkConfig(
            name="test_voxel_network",
            hidden_dims=(128, 256),
            activation="relu",
            base_channels=64,
            num_layers=4,
        )

    @pytest.fixture
    def graph_network_config(self):
        """Create GraphNetworkConfig for testing."""
        return GraphNetworkConfig(
            name="test_graph_network",
            hidden_dims=(64, 128),
            activation="relu",
            node_features_dim=64,
            edge_features_dim=32,
            num_layers=4,
        )

    def test_build_point_cloud_model(self, rngs, point_cloud_network_config):
        """Test building a point cloud model."""
        config = PointCloudConfig(
            name="test_point_cloud",
            network=point_cloud_network_config,
            num_points=1024,
            point_dim=3,
            use_normals=False,
            global_features_dim=1024,
        )

        builder = GeometricBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None

    def test_build_mesh_model(self, rngs, mesh_network_config):
        """Test building a mesh model."""
        config = MeshConfig(
            name="test_mesh",
            network=mesh_network_config,
            num_vertices=2048,
            num_faces=4096,
            vertex_dim=3,
        )

        builder = GeometricBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None

    def test_build_voxel_model(self, rngs, voxel_network_config):
        """Test building a voxel model."""
        config = VoxelConfig(
            name="test_voxel",
            network=voxel_network_config,
            voxel_size=32,
            voxel_dim=1,
            use_sparse=False,
        )

        builder = GeometricBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None

    def test_build_graph_model(self, rngs, graph_network_config):
        """Test building a graph model."""
        config = GraphConfig(
            name="test_graph",
            network=graph_network_config,
            max_nodes=1024,
            max_edges=4096,
            directed=False,
        )

        builder = GeometricBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None

    def test_build_with_none_config(self, rngs):
        """Test that None config raises TypeError."""
        builder = GeometricBuilder()

        with pytest.raises(TypeError, match="config cannot be None"):
            builder.build(None, rngs=rngs)

    def test_build_with_dict_config(self, rngs):
        """Test that dict config raises TypeError."""
        builder = GeometricBuilder()
        config = {"name": "test", "num_points": 1024}

        with pytest.raises(TypeError, match="config must be a dataclass"):
            builder.build(config, rngs=rngs)

    def test_build_with_invalid_config_type(self, rngs):
        """Test that unsupported config type raises TypeError."""
        builder = GeometricBuilder()

        class FakeConfig:
            pass

        fake_config = FakeConfig()

        with pytest.raises(TypeError, match="Unsupported config type"):
            builder.build(fake_config, rngs=rngs)

    def test_config_validation(self):
        """Test that Geometric configs properly validate."""
        # Valid PointCloudNetworkConfig
        valid_pc_network = PointCloudNetworkConfig(
            name="valid_pc_network",
            hidden_dims=(256, 512),
            activation="relu",
            embed_dim=256,
            num_heads=8,
            num_layers=6,
        )
        assert valid_pc_network.embed_dim == 256
        assert valid_pc_network.num_heads == 8

        # Valid PointCloudConfig
        valid_pc = PointCloudConfig(
            name="valid_pc",
            network=valid_pc_network,
            num_points=1024,
            point_dim=3,
        )
        assert valid_pc.num_points == 1024
        assert valid_pc.point_dim == 3

        # embed_dim not divisible by num_heads should fail
        with pytest.raises(ValueError, match="embed_dim.*must be divisible by num_heads"):
            PointCloudNetworkConfig(
                name="invalid",
                hidden_dims=(256,),
                activation="relu",
                embed_dim=255,  # Not divisible by 8
                num_heads=8,
            )

        # Missing network should fail
        with pytest.raises(ValueError, match="network is required"):
            PointCloudConfig(
                name="invalid",
                network=None,
                num_points=1024,
            )

    def test_mesh_config_validation(self):
        """Test MeshConfig validation."""
        mesh_network = MeshNetworkConfig(
            name="test",
            hidden_dims=(256,),
            activation="relu",
            embed_dim=256,
            num_heads=8,
        )

        valid = MeshConfig(
            name="valid_mesh",
            network=mesh_network,
            num_vertices=2048,
            num_faces=4096,
        )
        assert valid.num_vertices == 2048
        assert valid.num_faces == 4096

    def test_voxel_config_validation(self):
        """Test VoxelConfig validation."""
        voxel_network = VoxelNetworkConfig(
            name="test",
            hidden_dims=(128,),
            activation="relu",
        )

        valid = VoxelConfig(
            name="valid_voxel",
            network=voxel_network,
            voxel_size=32,
            voxel_dim=1,
        )
        assert valid.voxel_size == 32

    def test_graph_config_validation(self):
        """Test GraphConfig validation."""
        graph_network = GraphNetworkConfig(
            name="test",
            hidden_dims=(64,),
            activation="relu",
        )

        valid = GraphConfig(
            name="valid_graph",
            network=graph_network,
            max_nodes=1024,
            max_edges=4096,
        )
        assert valid.max_nodes == 1024

        # Invalid aggregation should fail
        with pytest.raises(ValueError, match="aggregation must be one of"):
            GraphNetworkConfig(
                name="invalid",
                hidden_dims=(64,),
                activation="relu",
                aggregation="invalid_agg",  # Should be mean, sum, or max
            )
