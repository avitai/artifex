"""Tests for DistributedConfig frozen dataclass configuration.

Tests cover DistributedBackend enum, DistributedConfig creation, defaults,
validation (field-level and cross-field), auto-mesh configuration,
helper methods, and serialization.
"""

import dataclasses

import pytest

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.distributed_config import (
    DistributedBackend,
    DistributedConfig,
)


# =============================================================================
# DistributedBackend Enum Tests
# =============================================================================
class TestDistributedBackend:
    """Test DistributedBackend enum."""

    def test_enum_values(self):
        """Test that enum values match expected strings."""
        assert DistributedBackend.NCCL.value == "nccl"
        assert DistributedBackend.GLOO.value == "gloo"
        assert DistributedBackend.MPI.value == "mpi"

    def test_str_enum_behavior(self):
        """Test that DistributedBackend works as a string."""
        assert str(DistributedBackend.NCCL) == "nccl"
        assert f"{DistributedBackend.GLOO}" == "gloo"

    def test_from_string(self):
        """Test creating enum from string value."""
        assert DistributedBackend("nccl") == DistributedBackend.NCCL
        assert DistributedBackend("gloo") == DistributedBackend.GLOO
        assert DistributedBackend("mpi") == DistributedBackend.MPI

    def test_invalid_backend(self):
        """Test that invalid backend string raises ValueError."""
        with pytest.raises(ValueError):
            DistributedBackend("invalid")


# =============================================================================
# DistributedConfig Basic Tests
# =============================================================================
class TestDistributedConfigBasics:
    """Test basic functionality of DistributedConfig."""

    def test_create_with_defaults(self):
        """Test creation with default values."""
        config = DistributedConfig(name="dist")
        assert config.name == "dist"
        assert config.enabled is False
        assert config.backend == DistributedBackend.NCCL

    def test_frozen(self):
        """Test that config is frozen."""
        config = DistributedConfig(name="dist")
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"  # type: ignore

    def test_inherits_from_base_config(self):
        """Test inheritance from BaseConfig."""
        config = DistributedConfig(name="dist")
        assert isinstance(config, BaseConfig)


class TestDistributedConfigDefaults:
    """Test default values of DistributedConfig."""

    def test_default_world_size(self):
        """Test world_size defaults to 1."""
        config = DistributedConfig(name="dist")
        assert config.world_size == 1

    def test_default_rank(self):
        """Test rank defaults to 0."""
        config = DistributedConfig(name="dist")
        assert config.rank == 0

    def test_default_local_rank(self):
        """Test local_rank defaults to 0."""
        config = DistributedConfig(name="dist")
        assert config.local_rank == 0

    def test_default_num_nodes(self):
        """Test num_nodes defaults to 1."""
        config = DistributedConfig(name="dist")
        assert config.num_nodes == 1

    def test_default_num_processes_per_node(self):
        """Test num_processes_per_node defaults to 1."""
        config = DistributedConfig(name="dist")
        assert config.num_processes_per_node == 1

    def test_default_master_addr(self):
        """Test master_addr defaults to localhost."""
        config = DistributedConfig(name="dist")
        assert config.master_addr == "localhost"

    def test_default_master_port(self):
        """Test master_port defaults to 29500."""
        config = DistributedConfig(name="dist")
        assert config.master_port == 29500

    def test_default_parallelism(self):
        """Test tensor/pipeline parallel default to 1."""
        config = DistributedConfig(name="dist")
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1

    def test_default_mesh_none(self):
        """Test mesh_shape/axis_names default to None when disabled."""
        config = DistributedConfig(name="dist", enabled=False)
        assert config.mesh_shape is None
        assert config.mesh_axis_names is None

    def test_default_communication_settings(self):
        """Test communication setting defaults."""
        config = DistributedConfig(name="dist")
        assert config.find_unused_parameters is False
        assert config.gradient_as_bucket_view is True
        assert config.broadcast_buffers is True

    def test_default_mixed_precision(self):
        """Test mixed_precision defaults to 'no'."""
        config = DistributedConfig(name="dist")
        assert config.mixed_precision == "no"


# =============================================================================
# DistributedConfig Validation Tests
# =============================================================================
class TestDistributedConfigFieldValidation:
    """Test field-level validation of DistributedConfig."""

    def test_invalid_world_size_zero(self):
        """Test that world_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="world_size"):
            DistributedConfig(name="dist", world_size=0)

    def test_invalid_world_size_negative(self):
        """Test that negative world_size raises ValueError."""
        with pytest.raises(ValueError, match="world_size"):
            DistributedConfig(name="dist", world_size=-1)

    def test_invalid_rank_negative(self):
        """Test that negative rank raises ValueError."""
        with pytest.raises(ValueError, match="rank"):
            DistributedConfig(name="dist", rank=-1)

    def test_invalid_local_rank_negative(self):
        """Test that negative local_rank raises ValueError."""
        with pytest.raises(ValueError, match="local_rank"):
            DistributedConfig(name="dist", local_rank=-1)

    def test_invalid_num_nodes_zero(self):
        """Test that num_nodes=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_nodes"):
            DistributedConfig(name="dist", num_nodes=0)

    def test_invalid_num_processes_per_node_zero(self):
        """Test that num_processes_per_node=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_processes_per_node"):
            DistributedConfig(name="dist", num_processes_per_node=0)

    def test_invalid_tensor_parallel_zero(self):
        """Test that tensor_parallel_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="tensor_parallel_size"):
            DistributedConfig(name="dist", tensor_parallel_size=0)

    def test_invalid_pipeline_parallel_zero(self):
        """Test that pipeline_parallel_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="pipeline_parallel_size"):
            DistributedConfig(name="dist", pipeline_parallel_size=0)

    def test_invalid_master_port_too_low(self):
        """Test that master_port below 1024 raises ValueError."""
        with pytest.raises(ValueError, match="master_port"):
            DistributedConfig(name="dist", master_port=80)

    def test_invalid_master_port_too_high(self):
        """Test that master_port above 65535 raises ValueError."""
        with pytest.raises(ValueError, match="master_port"):
            DistributedConfig(name="dist", master_port=70000)

    def test_valid_master_port_boundaries(self):
        """Test that boundary ports are valid."""
        config_low = DistributedConfig(name="dist", master_port=1024)
        assert config_low.master_port == 1024
        config_high = DistributedConfig(name="dist", master_port=65535)
        assert config_high.master_port == 65535


class TestDistributedConfigCrossFieldValidation:
    """Test cross-field validation of DistributedConfig."""

    def test_rank_exceeds_world_size(self):
        """Test that rank >= world_size raises ValueError."""
        with pytest.raises(ValueError, match="rank.*world_size"):
            DistributedConfig(
                name="dist", world_size=4, rank=4, num_nodes=4, num_processes_per_node=1
            )

    def test_local_rank_exceeds_processes_per_node(self):
        """Test that local_rank >= num_processes_per_node raises ValueError."""
        with pytest.raises(ValueError, match="local_rank.*num_processes_per_node"):
            DistributedConfig(name="dist", local_rank=2, num_processes_per_node=2)

    def test_world_size_mismatch(self):
        """Test that world_size != num_nodes * num_processes_per_node raises."""
        with pytest.raises(ValueError, match="world_size.*num_nodes"):
            DistributedConfig(name="dist", world_size=4, num_nodes=2, num_processes_per_node=1)

    def test_parallelism_exceeds_world_size(self):
        """Test that tensor*pipeline > world_size raises ValueError."""
        with pytest.raises(ValueError, match="parallel.*world_size"):
            DistributedConfig(
                name="dist",
                world_size=4,
                num_nodes=4,
                num_processes_per_node=1,
                tensor_parallel_size=2,
                pipeline_parallel_size=4,
            )

    def test_world_size_not_divisible_by_parallelism(self):
        """Test that world_size not divisible by parallelism product raises."""
        with pytest.raises(ValueError, match="divisible"):
            DistributedConfig(
                name="dist",
                world_size=6,
                num_nodes=6,
                num_processes_per_node=1,
                tensor_parallel_size=4,
            )

    def test_valid_multi_node_config(self):
        """Test a valid multi-node configuration."""
        config = DistributedConfig(
            name="dist",
            enabled=True,
            world_size=8,
            rank=3,
            local_rank=1,
            num_nodes=2,
            num_processes_per_node=4,
        )
        assert config.world_size == 8
        assert config.rank == 3
        assert config.local_rank == 1

    def test_valid_parallelism_config(self):
        """Test a valid parallelism configuration."""
        config = DistributedConfig(
            name="dist",
            enabled=True,
            world_size=8,
            rank=0,
            local_rank=0,
            num_nodes=2,
            num_processes_per_node=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        )
        assert config.tensor_parallel_size == 2
        assert config.pipeline_parallel_size == 2


# =============================================================================
# Mesh Validation Tests
# =============================================================================
class TestDistributedConfigMeshValidation:
    """Test mesh shape and axis name validation."""

    def test_mesh_shape_product_mismatch(self):
        """Test that mesh_shape product != world_size raises ValueError."""
        with pytest.raises(ValueError, match="mesh_shape.*world_size"):
            DistributedConfig(
                name="dist", world_size=4, num_nodes=4, num_processes_per_node=1, mesh_shape=(2, 3)
            )

    def test_mesh_axis_names_without_shape(self):
        """Test that mesh_axis_names without mesh_shape raises ValueError."""
        with pytest.raises(ValueError, match="mesh_axis_names.*mesh_shape"):
            DistributedConfig(name="dist", mesh_axis_names=("data",))

    def test_mesh_axis_names_length_mismatch(self):
        """Test that mismatched axis names/shape lengths raise ValueError."""
        with pytest.raises(ValueError, match="mesh_axis_names.*length"):
            DistributedConfig(name="dist", mesh_shape=(1,), mesh_axis_names=("data", "model"))

    def test_valid_explicit_mesh(self):
        """Test valid explicit mesh configuration."""
        config = DistributedConfig(
            name="dist",
            world_size=4,
            num_nodes=4,
            num_processes_per_node=1,
            mesh_shape=(2, 2),
            mesh_axis_names=("data", "model"),
        )
        assert config.mesh_shape == (2, 2)
        assert config.mesh_axis_names == ("data", "model")


# =============================================================================
# Auto-Configure Mesh Tests
# =============================================================================
class TestDistributedConfigAutoMesh:
    """Test auto-configuration of mesh shape and axis names."""

    def test_auto_mesh_single_device(self):
        """Test auto-mesh for single device (world_size=1)."""
        config = DistributedConfig(name="dist", enabled=True)
        assert config.mesh_shape == (1,)
        assert config.mesh_axis_names == ("data",)

    def test_auto_mesh_data_parallel_only(self):
        """Test auto-mesh for data parallelism only."""
        config = DistributedConfig(
            name="dist",
            enabled=True,
            world_size=4,
            num_nodes=4,
            num_processes_per_node=1,
        )
        assert config.mesh_shape == (4,)
        assert config.mesh_axis_names == ("data",)

    def test_auto_mesh_tensor_parallel(self):
        """Test auto-mesh with tensor parallelism."""
        config = DistributedConfig(
            name="dist",
            enabled=True,
            world_size=4,
            num_nodes=4,
            num_processes_per_node=1,
            tensor_parallel_size=2,
        )
        assert config.mesh_shape == (2, 2)
        assert config.mesh_axis_names == ("data", "model")

    def test_auto_mesh_both_parallel(self):
        """Test auto-mesh with both tensor and pipeline parallelism."""
        config = DistributedConfig(
            name="dist",
            enabled=True,
            world_size=8,
            num_nodes=8,
            num_processes_per_node=1,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        )
        assert config.mesh_shape == (2, 2, 2)
        assert config.mesh_axis_names == ("data", "model", "pipeline")

    def test_no_auto_mesh_when_disabled(self):
        """Test that mesh is not auto-configured when enabled=False."""
        config = DistributedConfig(name="dist", enabled=False)
        assert config.mesh_shape is None
        assert config.mesh_axis_names is None

    def test_no_auto_mesh_when_explicit(self):
        """Test that explicit mesh is not overridden."""
        config = DistributedConfig(
            name="dist",
            enabled=True,
            mesh_shape=(1,),
            mesh_axis_names=("data",),
        )
        assert config.mesh_shape == (1,)
        assert config.mesh_axis_names == ("data",)


# =============================================================================
# Helper Method Tests
# =============================================================================
class TestDistributedConfigHelpers:
    """Test helper methods of DistributedConfig."""

    def test_get_data_parallel_size_default(self):
        """Test data parallel size with no model parallelism."""
        config = DistributedConfig(
            name="dist",
            world_size=4,
            num_nodes=4,
            num_processes_per_node=1,
        )
        assert config.get_data_parallel_size() == 4

    def test_get_data_parallel_size_with_tensor(self):
        """Test data parallel size with tensor parallelism."""
        config = DistributedConfig(
            name="dist",
            enabled=True,
            world_size=8,
            num_nodes=8,
            num_processes_per_node=1,
            tensor_parallel_size=2,
        )
        assert config.get_data_parallel_size() == 4

    def test_get_data_parallel_size_with_both(self):
        """Test data parallel size with both parallelism types."""
        config = DistributedConfig(
            name="dist",
            enabled=True,
            world_size=8,
            num_nodes=8,
            num_processes_per_node=1,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        )
        assert config.get_data_parallel_size() == 2

    def test_get_mesh_config_when_configured(self):
        """Test get_mesh_config returns dict when mesh is configured."""
        config = DistributedConfig(name="dist", enabled=True)
        mesh = config.get_mesh_config()
        assert mesh is not None
        assert "mesh_shape" in mesh
        assert "axis_names" in mesh

    def test_get_mesh_config_when_not_configured(self):
        """Test get_mesh_config returns None when mesh is not set."""
        config = DistributedConfig(name="dist", enabled=False)
        assert config.get_mesh_config() is None

    def test_is_main_process_rank_zero(self):
        """Test is_main_process returns True for rank 0."""
        config = DistributedConfig(name="dist")
        assert config.is_main_process() is True

    def test_is_main_process_non_zero(self):
        """Test is_main_process returns False for non-zero rank."""
        config = DistributedConfig(
            name="dist",
            world_size=4,
            rank=1,
            num_nodes=4,
            num_processes_per_node=1,
        )
        assert config.is_main_process() is False

    def test_is_local_main_process_zero(self):
        """Test is_local_main_process for local_rank 0."""
        config = DistributedConfig(name="dist")
        assert config.is_local_main_process() is True

    def test_is_local_main_process_non_zero(self):
        """Test is_local_main_process for non-zero local_rank."""
        config = DistributedConfig(
            name="dist",
            world_size=4,
            rank=1,
            local_rank=1,
            num_nodes=2,
            num_processes_per_node=2,
        )
        assert config.is_local_main_process() is False


# =============================================================================
# Serialization Tests
# =============================================================================
class TestDistributedConfigSerialization:
    """Test serialization of DistributedConfig."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = DistributedConfig(
            name="dist",
            enabled=True,
            world_size=4,
            num_nodes=4,
            num_processes_per_node=1,
            backend=DistributedBackend.GLOO,
        )
        data = config.to_dict()
        assert data["name"] == "dist"
        assert data["enabled"] is True
        assert data["world_size"] == 4

    def test_roundtrip_simple(self):
        """Test roundtrip serialization for simple config."""
        original = DistributedConfig(name="dist")
        data = original.to_dict()
        restored = DistributedConfig.from_dict(data)
        assert original.name == restored.name
        assert original.enabled == restored.enabled
        assert original.world_size == restored.world_size

    def test_roundtrip_complex(self):
        """Test roundtrip serialization for complex config."""
        original = DistributedConfig(
            name="dist",
            enabled=True,
            world_size=8,
            rank=3,
            local_rank=1,
            num_nodes=2,
            num_processes_per_node=4,
            tensor_parallel_size=2,
            master_port=30000,
            mixed_precision="bf16",
        )
        data = original.to_dict()
        restored = DistributedConfig.from_dict(data)
        assert restored.world_size == 8
        assert restored.rank == 3
        assert restored.tensor_parallel_size == 2
        assert restored.mixed_precision == "bf16"

    def test_from_dict_casts_backend_string_to_enum(self):
        """YAML-style backend strings should materialize as DistributedBackend."""
        restored = DistributedConfig.from_dict(
            {
                "name": "dist",
                "enabled": True,
                "backend": "gloo",
                "world_size": 4,
                "num_nodes": 2,
                "num_processes_per_node": 2,
            }
        )

        assert restored.backend is DistributedBackend.GLOO
