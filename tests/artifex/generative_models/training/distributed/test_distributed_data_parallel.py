"""Tests for distributed data parallel module.

Tests for DataParallel - data-parallel training utilities for JAX.
"""

from __future__ import annotations

from unittest import mock

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from artifex.generative_models.training.distributed.data_parallel import DataParallel
from artifex.generative_models.training.distributed.mesh import DeviceMeshManager


class TestDataParallelCreation:
    """Tests for creating DataParallel instances."""

    def test_init(self):
        """Test initializing DataParallel."""
        dp = DataParallel()
        assert isinstance(dp, nnx.Module)

    def test_data_parallel_is_nnx_module(self):
        """Test that DataParallel is an NNX Module."""
        dp = DataParallel()
        # NNX modules have specific attributes
        assert hasattr(dp, "__nnx_state_dict__") or isinstance(dp, nnx.Module)


class TestCreateSharding:
    """Tests for creating data-parallel sharding."""

    def test_create_data_parallel_sharding_single_device(self):
        """Test creating sharding with single device mesh."""
        manager = DeviceMeshManager()
        dp = DataParallel()
        mesh = manager.create_device_mesh([("data", 1)])
        sharding = dp.create_data_parallel_sharding(mesh)

        # Should create a NamedSharding with data axis
        assert sharding.spec == jax.sharding.PartitionSpec("data")

    def test_create_data_parallel_sharding_custom_axis(self):
        """Test creating sharding with custom axis name."""
        manager = DeviceMeshManager()
        dp = DataParallel()
        mesh = manager.create_device_mesh([("batch", 1)])
        sharding = dp.create_data_parallel_sharding(mesh, data_axis="batch")

        assert sharding.spec == jax.sharding.PartitionSpec("batch")

    def test_create_data_parallel_sharding_static(self):
        """Test static method for creating sharding."""
        manager = DeviceMeshManager()
        mesh = manager.create_device_mesh([("data", 1)])
        sharding = DataParallel.create_data_parallel_sharding_static(mesh)

        assert sharding.spec == jax.sharding.PartitionSpec("data")


class TestShardBatch:
    """Tests for sharding batches of data."""

    def test_shard_batch_single_array(self):
        """Test sharding a single array."""
        manager = DeviceMeshManager()
        dp = DataParallel()
        mesh = manager.create_device_mesh([("data", 1)])
        sharding = dp.create_data_parallel_sharding(mesh)

        batch = jnp.ones((4, 8))
        sharded = dp.shard_batch(batch, sharding)

        assert sharded.shape == (4, 8)

    def test_shard_batch_dict(self):
        """Test sharding a dict batch."""
        manager = DeviceMeshManager()
        dp = DataParallel()
        mesh = manager.create_device_mesh([("data", 1)])
        sharding = dp.create_data_parallel_sharding(mesh)

        batch = {"inputs": jnp.ones((4, 2)), "targets": jnp.zeros((4,))}
        sharded = dp.shard_batch(batch, sharding)

        assert sharded["inputs"].shape == (4, 2)
        assert sharded["targets"].shape == (4,)

    def test_shard_batch_static(self):
        """Test static sharding method."""
        manager = DeviceMeshManager()
        mesh = manager.create_device_mesh([("data", 1)])
        sharding = DataParallel.create_data_parallel_sharding_static(mesh)

        batch = {"x": jnp.ones((4, 8))}
        sharded = DataParallel.shard_batch_static(batch, sharding)

        assert sharded["x"].shape == (4, 8)


class TestShardModelState:
    """Tests for sharding model state."""

    def test_shard_model_state_replicate(self):
        """Test replicating model state across devices."""
        manager = DeviceMeshManager()
        dp = DataParallel()
        mesh = manager.create_device_mesh([("data", 1)])

        # Simple state dict
        state = {"params": {"layer1": jnp.ones((10, 10))}}
        sharded = dp.shard_model_state(state, mesh, param_sharding="replicate")

        assert sharded["params"]["layer1"].shape == (10, 10)

    def test_shard_model_state_default_replicates(self):
        """Test that default sharding replicates parameters."""
        manager = DeviceMeshManager()
        dp = DataParallel()
        mesh = manager.create_device_mesh([("data", 1)])

        state = {"weights": jnp.ones((5, 5))}
        sharded = dp.shard_model_state(state, mesh)

        assert sharded["weights"].shape == (5, 5)


class TestAllReduceGradients:
    """Tests for gradient aggregation."""

    def test_all_reduce_gradients_mean(self):
        """Test reducing gradients with mean."""
        dp = DataParallel()

        # Mock pmean to return expected value
        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            gradients = jnp.array(4.0)
            reduced = dp.all_reduce_gradients(gradients, reduce_type="mean")
            assert float(reduced) == 2.0

    def test_all_reduce_gradients_sum(self):
        """Test reducing gradients with sum."""
        dp = DataParallel()

        with mock.patch("jax.lax.psum", return_value=jnp.array(8.0)):
            gradients = jnp.array(4.0)
            reduced = dp.all_reduce_gradients(gradients, reduce_type="sum")
            assert float(reduced) == 8.0

    def test_all_reduce_gradients_invalid_type_raises(self):
        """Test that invalid reduce type raises ValueError."""
        dp = DataParallel()

        with pytest.raises(ValueError, match="Unsupported reduce_type"):
            dp.all_reduce_gradients(jnp.array(1.0), reduce_type="invalid")

    def test_all_reduce_gradients_static_mean(self):
        """Test static mean reduction."""
        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            gradients = jnp.array(4.0)
            reduced = DataParallel.all_reduce_gradients_static(gradients, "mean")
            assert float(reduced) == 2.0

    def test_all_reduce_gradients_static_sum(self):
        """Test static sum reduction."""
        with mock.patch("jax.lax.psum", return_value=jnp.array(8.0)):
            gradients = jnp.array(4.0)
            reduced = DataParallel.all_reduce_gradients_static(gradients, "sum")
            assert float(reduced) == 8.0


class TestReplicateParams:
    """Tests for parameter replication."""

    def test_replicate_params(self):
        """Test replicating parameters across devices."""
        manager = DeviceMeshManager()
        dp = DataParallel()
        mesh = manager.create_device_mesh([("data", 1)])

        params = {"w": jnp.ones((3, 3))}
        replicated = dp.replicate_params(params, mesh)

        assert replicated["w"].shape == (3, 3)


@pytest.mark.usefixtures("skip_if_single_device")
class TestMultiDeviceDataParallel:
    """Tests that require multiple devices."""

    def test_create_sharding_two_devices(self):
        """Test creating sharding with 2 devices."""
        manager = DeviceMeshManager()
        dp = DataParallel()
        mesh = manager.create_device_mesh([("data", 2)])
        sharding = dp.create_data_parallel_sharding(mesh)

        assert sharding.spec == jax.sharding.PartitionSpec("data")

    def test_shard_batch_two_devices(self):
        """Test sharding batch across 2 devices."""
        manager = DeviceMeshManager()
        dp = DataParallel()
        mesh = manager.create_device_mesh([("data", 2)])
        sharding = dp.create_data_parallel_sharding(mesh)

        batch = {"inputs": jnp.ones((4, 2)), "targets": jnp.zeros((4,))}
        sharded = dp.shard_batch(batch, sharding)

        assert sharded["inputs"].shape == (4, 2)
        assert sharded["targets"].shape == (4,)
