"""Tests for distributed mesh module.

Tests for DeviceMeshManager - creating and managing JAX device meshes
for distributed training.
"""

from __future__ import annotations

import jax
import pytest

from artifex.generative_models.training.distributed.mesh import DeviceMeshManager


class TestDeviceMeshManagerCreation:
    """Tests for creating DeviceMeshManager instances."""

    def test_init_default(self):
        """Test initializing DeviceMeshManager with defaults."""
        manager = DeviceMeshManager()
        assert isinstance(manager, DeviceMeshManager)

    def test_init_with_devices(self):
        """Test initializing with explicit devices."""
        devices = jax.devices()
        manager = DeviceMeshManager(devices=devices)
        assert manager.devices == devices


class TestCreateDeviceMesh:
    """Tests for create_device_mesh method."""

    def test_create_mesh_with_dict_specification(self):
        """Test creating mesh with dict specification."""
        manager = DeviceMeshManager()
        mesh = manager.create_device_mesh({"data": 1})
        assert mesh.devices.shape == (1,)
        assert mesh.axis_names == ("data",)
        assert mesh.devices.size == 1

    def test_create_mesh_with_list_specification(self):
        """Test creating mesh with list of tuples specification."""
        manager = DeviceMeshManager()
        mesh = manager.create_device_mesh([("batch", 1)])
        assert mesh.devices.shape == (1,)
        assert mesh.axis_names == ("batch",)
        assert mesh.devices.size == 1

    def test_create_mesh_with_explicit_devices(self):
        """Test creating mesh with explicitly provided devices."""
        devices = jax.devices()
        manager = DeviceMeshManager()
        mesh = manager.create_device_mesh([("data", 1)], devices=devices)
        assert mesh.devices.shape == (1,)
        assert mesh.devices[0] == devices[0]

    def test_create_mesh_multiple_axes(self):
        """Test creating mesh with multiple axes (requires single device)."""
        manager = DeviceMeshManager()
        mesh = manager.create_device_mesh([("data", 1), ("model", 1)])
        assert mesh.devices.shape == (1, 1)
        assert mesh.axis_names == ("data", "model")
        assert mesh.devices.size == 1

    def test_create_mesh_insufficient_devices_raises_error(self):
        """Test that creating mesh with too few devices raises ValueError."""
        manager = DeviceMeshManager()
        with pytest.raises(ValueError, match="Not enough devices"):
            manager.create_device_mesh([("data", 100)])


class TestDataParallelMesh:
    """Tests for data-parallel mesh creation."""

    def test_create_data_parallel_mesh_default(self):
        """Test creating data-parallel mesh with all available devices."""
        manager = DeviceMeshManager()
        mesh = manager.create_data_parallel_mesh()
        available_devices = len(jax.devices())
        assert mesh.devices.shape == (available_devices,)
        assert mesh.axis_names == ("data",)

    def test_create_data_parallel_mesh_single_device(self):
        """Test creating data-parallel mesh with single device."""
        manager = DeviceMeshManager()
        mesh = manager.create_data_parallel_mesh(num_devices=1)
        assert mesh.devices.shape == (1,)
        assert mesh.axis_names == ("data",)

    def test_create_data_parallel_mesh_custom_axis_name(self):
        """Test creating data-parallel mesh with custom axis name."""
        manager = DeviceMeshManager()
        mesh = manager.create_data_parallel_mesh(num_devices=1, axis_name="batch")
        assert mesh.axis_names == ("batch",)


class TestModelParallelMesh:
    """Tests for model-parallel mesh creation."""

    def test_create_model_parallel_mesh_single_device(self):
        """Test creating model-parallel mesh with single device."""
        manager = DeviceMeshManager()
        mesh = manager.create_model_parallel_mesh(num_devices=1)
        assert mesh.devices.shape == (1,)
        assert mesh.axis_names == ("model",)

    def test_create_model_parallel_mesh_insufficient_devices_raises_error(self):
        """Test that model-parallel mesh with too many devices raises ValueError."""
        manager = DeviceMeshManager()
        with pytest.raises(ValueError, match="Not enough devices"):
            manager.create_model_parallel_mesh(num_devices=100)


class TestHybridMesh:
    """Tests for hybrid (data + model parallel) mesh creation."""

    def test_create_hybrid_mesh_single_device(self):
        """Test creating hybrid mesh with 1x1 configuration."""
        manager = DeviceMeshManager()
        mesh = manager.create_hybrid_mesh(data_parallel_size=1, model_parallel_size=1)
        assert mesh.devices.shape == (1, 1)
        assert mesh.axis_names == ("data", "model")
        assert mesh.devices.size == 1

    def test_create_hybrid_mesh_insufficient_devices_raises_error(self):
        """Test that hybrid mesh with too many devices raises ValueError."""
        manager = DeviceMeshManager()
        with pytest.raises(ValueError, match="Not enough devices"):
            manager.create_hybrid_mesh(data_parallel_size=10, model_parallel_size=10)


class TestMeshInfo:
    """Tests for getting mesh information."""

    def test_get_mesh_info_single_axis(self):
        """Test getting info from a single-axis mesh."""
        manager = DeviceMeshManager()
        mesh = manager.create_device_mesh([("data", 1)])
        info = manager.get_mesh_info(mesh)

        assert info["total_devices"] == 1
        assert "axes" in info
        assert info["axes"]["data"] == 1

    def test_get_mesh_info_multiple_axes(self):
        """Test getting info from a multi-axis mesh."""
        manager = DeviceMeshManager()
        mesh = manager.create_hybrid_mesh(data_parallel_size=1, model_parallel_size=1)
        info = manager.get_mesh_info(mesh)

        assert info["total_devices"] == 1
        assert info["axes"]["data"] == 1
        assert info["axes"]["model"] == 1
        assert len(info["axes"]) == 2


class TestMeshContextManager:
    """Tests for mesh context manager functionality."""

    def test_mesh_context_manager(self):
        """Test using mesh as context manager."""
        manager = DeviceMeshManager()
        mesh = manager.create_data_parallel_mesh(num_devices=1)

        # Mesh should work as context manager
        with mesh:
            # Inside mesh context
            pass  # Just verify it doesn't raise


@pytest.mark.usefixtures("skip_if_single_device")
class TestMultiDeviceMesh:
    """Tests that require multiple devices."""

    def test_create_mesh_with_two_devices(self):
        """Test creating a mesh with 2 devices."""
        manager = DeviceMeshManager()
        mesh = manager.create_device_mesh([("data", 2)])
        assert mesh.devices.shape == (2,)
        assert mesh.axis_names == ("data",)

    def test_create_data_parallel_mesh_two_devices(self):
        """Test creating data-parallel mesh with 2 devices."""
        manager = DeviceMeshManager()
        mesh = manager.create_data_parallel_mesh(num_devices=2)
        assert mesh.devices.shape == (2,)

    def test_mesh_info_two_devices(self):
        """Test mesh info with 2 devices."""
        manager = DeviceMeshManager()
        mesh = manager.create_device_mesh([("data", 2)])
        info = manager.get_mesh_info(mesh)
        assert info["total_devices"] == 2
        assert info["axes"]["data"] == 2
