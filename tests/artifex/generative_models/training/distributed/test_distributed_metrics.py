"""Tests for distributed metrics module.

Tests for DistributedMetrics - metrics aggregation across devices.
"""

from __future__ import annotations

from unittest import mock

import flax.nnx as nnx
import jax.numpy as jnp

from artifex.generative_models.training.distributed.metrics import DistributedMetrics


class TestDistributedMetricsCreation:
    """Tests for creating DistributedMetrics instances."""

    def test_init(self):
        """Test initializing DistributedMetrics."""
        metrics = DistributedMetrics()
        assert isinstance(metrics, nnx.Module)


class TestReduceMean:
    """Tests for mean reduction."""

    def test_reduce_mean_jax_array(self):
        """Test reducing JAX array with mean."""
        dm = DistributedMetrics()

        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            metrics = {"loss": jnp.array(3.0)}
            reduced = dm.reduce_mean(metrics)
            assert float(reduced["loss"]) == 2.0

    def test_reduce_mean_preserves_non_arrays(self):
        """Test that non-array values are preserved."""
        dm = DistributedMetrics()

        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            metrics = {"loss": jnp.array(3.0), "accuracy": 0.5}
            reduced = dm.reduce_mean(metrics)
            assert reduced["accuracy"] == 0.5  # Unchanged

    def test_reduce_mean_static(self):
        """Test static mean reduction."""
        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            metrics = {"loss": jnp.array(3.0), "lr": 0.001}
            reduced = DistributedMetrics.reduce_mean_static(metrics)
            assert float(reduced["loss"]) == 2.0
            assert reduced["lr"] == 0.001


class TestReduceSum:
    """Tests for sum reduction."""

    def test_reduce_sum_jax_array(self):
        """Test reducing JAX array with sum."""
        dm = DistributedMetrics()

        with mock.patch("jax.lax.psum", return_value=jnp.array(6.0)):
            metrics = {"loss": jnp.array(3.0)}
            reduced = dm.reduce_sum(metrics)
            assert float(reduced["loss"]) == 6.0

    def test_reduce_sum_preserves_non_arrays(self):
        """Test that non-array values are preserved."""
        dm = DistributedMetrics()

        with mock.patch("jax.lax.psum", return_value=jnp.array(6.0)):
            metrics = {"loss": jnp.array(3.0), "epoch": 5}
            reduced = dm.reduce_sum(metrics)
            assert reduced["epoch"] == 5  # Unchanged

    def test_reduce_sum_static(self):
        """Test static sum reduction."""
        with mock.patch("jax.lax.psum", return_value=jnp.array(6.0)):
            metrics = {"count": jnp.array(3.0)}
            reduced = DistributedMetrics.reduce_sum_static(metrics)
            assert float(reduced["count"]) == 6.0


class TestReduceMax:
    """Tests for max reduction."""

    def test_reduce_max_jax_array(self):
        """Test reducing JAX array with max."""
        dm = DistributedMetrics()

        with mock.patch("jax.lax.pmax", return_value=jnp.array(5.0)):
            metrics = {"max_loss": jnp.array(3.0)}
            reduced = dm.reduce_max(metrics)
            assert float(reduced["max_loss"]) == 5.0

    def test_reduce_max_static(self):
        """Test static max reduction."""
        with mock.patch("jax.lax.pmax", return_value=jnp.array(5.0)):
            metrics = {"max_val": jnp.array(3.0)}
            reduced = DistributedMetrics.reduce_max_static(metrics)
            assert float(reduced["max_val"]) == 5.0


class TestReduceMin:
    """Tests for min reduction."""

    def test_reduce_min_jax_array(self):
        """Test reducing JAX array with min."""
        dm = DistributedMetrics()

        with mock.patch("jax.lax.pmin", return_value=jnp.array(1.0)):
            metrics = {"min_loss": jnp.array(3.0)}
            reduced = dm.reduce_min(metrics)
            assert float(reduced["min_loss"]) == 1.0

    def test_reduce_min_static(self):
        """Test static min reduction."""
        with mock.patch("jax.lax.pmin", return_value=jnp.array(1.0)):
            metrics = {"min_val": jnp.array(3.0)}
            reduced = DistributedMetrics.reduce_min_static(metrics)
            assert float(reduced["min_val"]) == 1.0


class TestReduceCustom:
    """Tests for custom reduction operations."""

    def test_reduce_custom_mixed_operations(self):
        """Test custom reduction with mixed operations."""
        dm = DistributedMetrics()

        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            with mock.patch("jax.lax.psum", return_value=jnp.array(6.0)):
                with mock.patch("jax.lax.pmax", return_value=jnp.array(4.0)):
                    metrics = {
                        "loss": jnp.array(3.0),
                        "accuracy": jnp.array(0.5),
                        "step": jnp.array(10),
                    }
                    reduce_fn = {
                        "loss": "mean",
                        "accuracy": "sum",
                        "step": "max",
                    }
                    reduced = dm.reduce_custom(metrics, reduce_fn=reduce_fn)

                    assert float(reduced["loss"]) == 2.0
                    assert float(reduced["accuracy"]) == 6.0
                    assert float(reduced["step"]) == 4.0

    def test_reduce_custom_default_to_mean(self):
        """Test that custom reduction defaults to mean when no reduce_fn."""
        dm = DistributedMetrics()

        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            metrics = {"loss": jnp.array(3.0)}
            reduced = dm.reduce_custom(metrics, reduce_fn=None)
            assert float(reduced["loss"]) == 2.0

    def test_reduce_custom_static(self):
        """Test static custom reduction."""
        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            with mock.patch("jax.lax.psum", return_value=jnp.array(6.0)):
                metrics = {
                    "loss": jnp.array(3.0),
                    "count": jnp.array(3.0),
                }
                reduce_fn = {"loss": "mean", "count": "sum"}
                reduced = DistributedMetrics.reduce_custom_static(metrics, reduce_fn=reduce_fn)

                assert float(reduced["loss"]) == 2.0
                assert float(reduced["count"]) == 6.0


class TestAllGather:
    """Tests for gathering metrics from all devices."""

    def test_all_gather_jax_array(self):
        """Test gathering JAX array from all devices."""
        dm = DistributedMetrics()

        with mock.patch("jax.lax.all_gather", return_value=jnp.array([1.0, 2.0])):
            metrics = {"loss": jnp.array(1.0)}
            gathered = dm.all_gather(metrics)
            assert gathered["loss"].tolist() == [1.0, 2.0]

    def test_all_gather_preserves_non_arrays(self):
        """Test that non-arrays are preserved during gather."""
        dm = DistributedMetrics()

        with mock.patch("jax.lax.all_gather", return_value=jnp.array([1.0, 2.0])):
            metrics = {"loss": jnp.array(1.0), "epoch": 5}
            gathered = dm.all_gather(metrics)
            assert gathered["epoch"] == 5

    def test_all_gather_static(self):
        """Test static all_gather."""
        with mock.patch("jax.lax.all_gather", return_value=jnp.array([1.0, 2.0])):
            metrics = {"loss": jnp.array(1.0), "scalar": 0.5}
            gathered = DistributedMetrics.all_gather_static(metrics)
            assert gathered["loss"].tolist() == [1.0, 2.0]
            assert gathered["scalar"] == 0.5


class TestCollectFromDevices:
    """Tests for collecting metrics from device arrays."""

    def test_collect_from_devices_multi_device_array(self):
        """Test collecting from multi-device array."""
        dm = DistributedMetrics()

        metrics = {
            "loss": jnp.array([1.0, 2.0, 3.0]),  # Simulating 3 devices
            "accuracy": 0.95,  # Scalar
        }
        collected = dm.collect_from_devices(metrics)

        assert len(collected["loss"]) == 3
        assert float(collected["loss"][0]) == 1.0
        assert float(collected["loss"][1]) == 2.0
        assert float(collected["loss"][2]) == 3.0
        assert collected["accuracy"] == 0.95

    def test_collect_from_devices_preserves_scalars(self):
        """Test that scalars are preserved."""
        dm = DistributedMetrics()

        metrics = {"scalar": 0.5, "array": jnp.array([1.0, 2.0])}
        collected = dm.collect_from_devices(metrics)

        assert collected["scalar"] == 0.5
        assert len(collected["array"]) == 2

    def test_collect_from_devices_static(self):
        """Test static collect_from_devices."""
        metrics = {
            "loss": jnp.array([1.0, 2.0]),
            "epoch": 10,
        }
        collected = DistributedMetrics.collect_from_devices_static(metrics)

        assert len(collected["loss"]) == 2
        assert collected["epoch"] == 10


class TestCustomAxisName:
    """Tests for custom axis name support."""

    def test_reduce_mean_custom_axis(self):
        """Test reduce_mean with custom axis name."""
        dm = DistributedMetrics()

        # Verify axis_name is passed correctly
        with mock.patch("jax.lax.pmean") as mock_pmean:
            mock_pmean.return_value = jnp.array(2.0)
            metrics = {"loss": jnp.array(3.0)}
            dm.reduce_mean(metrics, axis_name="custom_axis")
            mock_pmean.assert_called_once()
            # Check that axis_name was passed
            call_kwargs = mock_pmean.call_args[1]
            assert call_kwargs["axis_name"] == "custom_axis"

    def test_all_gather_custom_axis(self):
        """Test all_gather with custom axis name."""
        dm = DistributedMetrics()

        with mock.patch("jax.lax.all_gather") as mock_gather:
            mock_gather.return_value = jnp.array([1.0])
            metrics = {"loss": jnp.array(1.0)}
            dm.all_gather(metrics, axis_name="my_axis")
            mock_gather.assert_called_once()
            call_kwargs = mock_gather.call_args[1]
            assert call_kwargs["axis_name"] == "my_axis"
