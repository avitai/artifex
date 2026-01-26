"""Tests for distributed device placement module.

Tests for DevicePlacement - explicit device placement utilities for JAX.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from artifex.generative_models.training.distributed.device_placement import (
    BatchSizeRecommendation,
    DevicePlacement,
    distribute_batch,
    get_batch_size_recommendation,
    HardwareType,
    place_on_device,
)


class TestHardwareType:
    """Tests for HardwareType enum."""

    def test_hardware_types_exist(self):
        """Test that all expected hardware types exist."""
        assert HardwareType.TPU_V5E
        assert HardwareType.TPU_V5P
        assert HardwareType.TPU_V4
        assert HardwareType.H100
        assert HardwareType.A100
        assert HardwareType.V100
        assert HardwareType.CPU
        assert HardwareType.UNKNOWN

    def test_hardware_type_values(self):
        """Test hardware type string values."""
        assert HardwareType.H100.value == "h100"
        assert HardwareType.CPU.value == "cpu"


class TestBatchSizeRecommendation:
    """Tests for BatchSizeRecommendation dataclass."""

    def test_create_recommendation(self):
        """Test creating a batch size recommendation."""
        rec = BatchSizeRecommendation(
            min_batch_size=32,
            optimal_batch_size=256,
            critical_batch_size=240,
        )
        assert rec.min_batch_size == 32
        assert rec.optimal_batch_size == 256
        assert rec.critical_batch_size == 240

    def test_recommendation_with_optional_fields(self):
        """Test recommendation with optional fields."""
        rec = BatchSizeRecommendation(
            min_batch_size=32,
            optimal_batch_size=256,
            critical_batch_size=240,
            max_memory_batch_size=512,
            notes="Test note",
        )
        assert rec.max_memory_batch_size == 512
        assert rec.notes == "Test note"

    def test_recommendation_frozen(self):
        """Test that recommendation is immutable."""
        rec = BatchSizeRecommendation(
            min_batch_size=32,
            optimal_batch_size=256,
            critical_batch_size=240,
        )
        with pytest.raises(AttributeError):
            rec.min_batch_size = 64  # type: ignore


class TestDevicePlacementCreation:
    """Tests for creating DevicePlacement instances."""

    def test_init_default(self):
        """Test initializing with defaults."""
        placement = DevicePlacement()
        assert placement.default_device is not None
        assert placement.default_device == jax.devices()[0]

    def test_init_with_device(self):
        """Test initializing with specific device."""
        devices = jax.devices()
        placement = DevicePlacement(default_device=devices[0])
        assert placement.default_device == devices[0]

    def test_hardware_type_detected(self):
        """Test that hardware type is detected."""
        placement = DevicePlacement()
        assert isinstance(placement.hardware_type, HardwareType)


class TestPlaceOnDevice:
    """Tests for placing data on devices."""

    def test_place_array_on_device(self):
        """Test placing a JAX array on device."""
        placement = DevicePlacement()
        data = jnp.ones((4, 8))
        placed = placement.place_on_device(data)
        assert placed.shape == (4, 8)

    def test_place_dict_on_device(self):
        """Test placing a dict PyTree on device."""
        placement = DevicePlacement()
        data = {"a": jnp.ones((2, 2)), "b": jnp.zeros((3,))}
        placed = placement.place_on_device(data)
        assert placed["a"].shape == (2, 2)
        assert placed["b"].shape == (3,)

    def test_place_on_specific_device(self):
        """Test placing data on specific device."""
        devices = jax.devices()
        placement = DevicePlacement()
        data = jnp.ones((4,))
        placed = placement.place_on_device(data, device=devices[0])
        assert placed.shape == (4,)


class TestDistributeBatch:
    """Tests for distributing batches across devices."""

    def test_distribute_batch_with_sharding(self):
        """Test distributing batch with explicit sharding."""
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        placement = DevicePlacement()
        mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("data",))
        sharding = NamedSharding(mesh, PartitionSpec("data"))

        data = jnp.ones((4, 8))
        distributed = placement.distribute_batch(data, sharding)
        assert distributed.shape == (4, 8)


class TestReplicateAcrossDevices:
    """Tests for replicating data across devices."""

    def test_replicate_array(self):
        """Test replicating array across devices."""
        placement = DevicePlacement()
        data = jnp.ones((4, 4))
        replicated = placement.replicate_across_devices(data)
        assert replicated.shape == (4, 4)

    def test_replicate_with_explicit_devices(self):
        """Test replicating with explicit device list."""
        devices = jax.devices()[:1]  # Use single device for test
        placement = DevicePlacement()
        data = jnp.ones((2, 2))
        replicated = placement.replicate_across_devices(data, devices=devices)
        assert replicated.shape == (2, 2)


class TestShardBatchDim:
    """Tests for sharding along batch dimension."""

    def test_shard_batch_dim_default(self):
        """Test sharding along default batch dimension."""
        from jax.sharding import Mesh

        placement = DevicePlacement()
        mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("data",))

        data = jnp.ones((8, 16))
        sharded = placement.shard_batch_dim(data, mesh)
        assert sharded.shape == (8, 16)

    def test_shard_batch_dim_dict(self):
        """Test sharding dict with batch dimension."""
        from jax.sharding import Mesh

        placement = DevicePlacement()
        mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("data",))

        data = {"images": jnp.ones((4, 28, 28, 3)), "labels": jnp.zeros((4,))}
        sharded = placement.shard_batch_dim(data, mesh)
        assert sharded["images"].shape == (4, 28, 28, 3)
        assert sharded["labels"].shape == (4,)


class TestBatchSizeRecommendations:
    """Tests for batch size recommendations."""

    def test_get_recommendation_default(self):
        """Test getting default recommendation."""
        placement = DevicePlacement()
        rec = placement.get_batch_size_recommendation()
        assert isinstance(rec, BatchSizeRecommendation)
        assert rec.min_batch_size > 0
        assert rec.optimal_batch_size >= rec.min_batch_size
        assert rec.critical_batch_size > 0

    def test_get_recommendation_override_hardware(self):
        """Test getting recommendation with hardware override."""
        placement = DevicePlacement()
        rec = placement.get_batch_size_recommendation(hardware_type=HardwareType.H100)
        assert rec.critical_batch_size == 298  # H100 specific value

    def test_get_recommendation_cpu(self):
        """Test getting recommendation for CPU."""
        placement = DevicePlacement()
        rec = placement.get_batch_size_recommendation(hardware_type=HardwareType.CPU)
        assert rec.min_batch_size == 1
        assert rec.critical_batch_size == 16


class TestValidateBatchSize:
    """Tests for batch size validation."""

    def test_validate_batch_size_valid(self):
        """Test validating a valid batch size."""
        placement = DevicePlacement()
        is_valid, message = placement.validate_batch_size(256)
        assert is_valid is True

    def test_validate_batch_size_too_small(self):
        """Test validating batch size that's too small."""
        placement = DevicePlacement()
        # Use CPU recommendation (min=1) for predictable test
        placement._hardware_type = HardwareType.CPU
        is_valid, message = placement.validate_batch_size(0)
        assert is_valid is False
        assert "below minimum" in message

    def test_validate_batch_size_suboptimal(self):
        """Test validating suboptimal but valid batch size."""
        placement = DevicePlacement()
        placement._hardware_type = HardwareType.H100  # critical=298
        is_valid, msg = placement.validate_batch_size(100, warn_suboptimal=True)
        assert is_valid is True
        assert "below critical" in msg


class TestDeviceInfo:
    """Tests for device information."""

    def test_num_devices(self):
        """Test getting number of devices."""
        placement = DevicePlacement()
        assert placement.num_devices >= 1

    def test_get_device_info(self):
        """Test getting device information dict."""
        placement = DevicePlacement()
        info = placement.get_device_info()

        assert "num_devices" in info
        assert "hardware_type" in info
        assert "platforms" in info
        assert "devices" in info
        assert info["num_devices"] >= 1


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_place_on_device_function(self):
        """Test place_on_device convenience function."""
        data = jnp.ones((4,))
        placed = place_on_device(data)
        assert placed.shape == (4,)

    def test_distribute_batch_function(self):
        """Test distribute_batch convenience function."""
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("data",))
        sharding = NamedSharding(mesh, PartitionSpec("data"))

        data = jnp.ones((4,))
        distributed = distribute_batch(data, sharding)
        assert distributed.shape == (4,)

    def test_get_batch_size_recommendation_function(self):
        """Test get_batch_size_recommendation convenience function."""
        rec = get_batch_size_recommendation()
        assert isinstance(rec, BatchSizeRecommendation)

    def test_get_batch_size_recommendation_with_hardware(self):
        """Test get_batch_size_recommendation with hardware type."""
        rec = get_batch_size_recommendation(HardwareType.A100)
        assert rec.critical_batch_size == 240  # A100 specific


@pytest.mark.usefixtures("skip_if_single_device")
class TestMultiDevicePlacement:
    """Tests that require multiple devices."""

    def test_replicate_across_multiple_devices(self):
        """Test replicating across multiple devices."""
        placement = DevicePlacement()
        data = jnp.ones((4, 4))
        replicated = placement.replicate_across_devices(data, devices=jax.devices()[:2])
        assert replicated.shape == (4, 4)
