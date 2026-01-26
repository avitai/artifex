"""Data parallel training utilities for JAX distributed training.

This module provides utilities for data-parallel training across multiple
devices, including batch sharding and gradient aggregation.
"""

from __future__ import annotations

from typing import Any, Literal

import flax.nnx as nnx
import jax
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec


class DataParallel(nnx.Module):
    """Data parallel training utilities for Artifex.

    This class provides methods for creating data-parallel shardings,
    distributing batches across devices, and aggregating gradients.

    Supports both static method usage (stateless) and instance method
    usage (stateful).

    Example:
        >>> # Static method usage
        >>> sharding = DataParallel.create_data_parallel_sharding_static(mesh)
        >>> sharded_batch = DataParallel.shard_batch_static(batch, sharding)
        >>>
        >>> # Instance method usage
        >>> dp = DataParallel()
        >>> sharding = dp.create_data_parallel_sharding(mesh)
        >>> sharded_batch = dp.shard_batch(batch, sharding)
    """

    def __init__(self) -> None:
        """Initialize DataParallel module."""
        super().__init__()

    def create_data_parallel_sharding(
        self,
        mesh: Mesh,
        data_axis: str = "data",
    ) -> NamedSharding:
        """Create a NamedSharding for data parallelism.

        Args:
            mesh: The device mesh to use for sharding.
            data_axis: Name of the data parallel axis in the mesh.

        Returns:
            A NamedSharding that distributes the first dimension across
            the data axis.

        Example:
            >>> dp = DataParallel()
            >>> mesh = DeviceMeshManager().create_device_mesh({"data": 2})
            >>> sharding = dp.create_data_parallel_sharding(mesh)
        """
        return NamedSharding(mesh, PartitionSpec(data_axis))

    @staticmethod
    def create_data_parallel_sharding_static(
        mesh: Mesh,
        data_axis: str = "data",
    ) -> NamedSharding:
        """Static version of create_data_parallel_sharding.

        Args:
            mesh: The device mesh to use for sharding.
            data_axis: Name of the data parallel axis in the mesh.

        Returns:
            A NamedSharding for data parallelism.
        """
        return NamedSharding(mesh, PartitionSpec(data_axis))

    def shard_batch(
        self,
        batch: Any,
        sharding: NamedSharding,
    ) -> Any:
        """Shard a batch of data across devices.

        Args:
            batch: PyTree of data to shard (dict, array, etc.).
            sharding: The sharding specification to apply.

        Returns:
            The sharded batch distributed across devices.

        Example:
            >>> dp = DataParallel()
            >>> sharded = dp.shard_batch({"inputs": x, "targets": y}, sharding)
        """
        return jax.tree.map(
            lambda x: jax.device_put(x, sharding),
            batch,
        )

    @staticmethod
    def shard_batch_static(
        batch: Any,
        sharding: NamedSharding,
    ) -> Any:
        """Static version of shard_batch.

        Args:
            batch: PyTree of data to shard.
            sharding: The sharding specification to apply.

        Returns:
            The sharded batch.
        """
        return jax.tree.map(
            lambda x: jax.device_put(x, sharding),
            batch,
        )

    def shard_model_state(
        self,
        state: Any,
        mesh: Mesh,
        param_sharding: Literal["replicate", "shard"] = "replicate",
    ) -> Any:
        """Shard model state across devices.

        Args:
            state: The model state (parameters, optimizer state, etc.).
            mesh: The device mesh to use.
            param_sharding: How to shard parameters:
                - "replicate": Copy parameters to all devices (default)
                - "shard": Shard parameters across devices

        Returns:
            The sharded model state.

        Example:
            >>> dp = DataParallel()
            >>> sharded_state = dp.shard_model_state(state, mesh)
        """
        if param_sharding == "replicate":
            # Replicate across all devices (no partitioning)
            sharding = NamedSharding(mesh, PartitionSpec())
        else:
            # Shard along first axis of mesh
            first_axis = mesh.axis_names[0] if mesh.axis_names else None
            sharding = NamedSharding(mesh, PartitionSpec(first_axis))

        return jax.tree.map(
            lambda x: jax.device_put(x, sharding),
            state,
        )

    def all_reduce_gradients(
        self,
        gradients: Any,
        reduce_type: Literal["mean", "sum"] = "mean",
        axis_name: str = "batch",
    ) -> Any:
        """Aggregate gradients across devices.

        This should be called inside a pmap/shard_map context.

        Args:
            gradients: PyTree of gradients to aggregate.
            reduce_type: Type of reduction ("mean" or "sum").
            axis_name: The axis name for the parallel reduction.

        Returns:
            The aggregated gradients.

        Raises:
            ValueError: If reduce_type is not "mean" or "sum".

        Example:
            >>> dp = DataParallel()
            >>> # Inside pmap:
            >>> grads = dp.all_reduce_gradients(grads, reduce_type="mean")
        """
        if reduce_type == "mean":
            return jax.tree.map(
                lambda g: lax.pmean(g, axis_name=axis_name),
                gradients,
            )
        elif reduce_type == "sum":
            return jax.tree.map(
                lambda g: lax.psum(g, axis_name=axis_name),
                gradients,
            )
        else:
            raise ValueError(f"Unsupported reduce_type: {reduce_type}. Use 'mean' or 'sum'.")

    @staticmethod
    def all_reduce_gradients_static(
        gradients: Any,
        reduce_type: Literal["mean", "sum"] = "mean",
        axis_name: str = "batch",
    ) -> Any:
        """Static version of all_reduce_gradients.

        Args:
            gradients: PyTree of gradients to aggregate.
            reduce_type: Type of reduction ("mean" or "sum").
            axis_name: The axis name for the parallel reduction.

        Returns:
            The aggregated gradients.

        Raises:
            ValueError: If reduce_type is not "mean" or "sum".
        """
        if reduce_type == "mean":
            return jax.tree.map(
                lambda g: lax.pmean(g, axis_name=axis_name),
                gradients,
            )
        elif reduce_type == "sum":
            return jax.tree.map(
                lambda g: lax.psum(g, axis_name=axis_name),
                gradients,
            )
        else:
            raise ValueError(f"Unsupported reduce_type: {reduce_type}. Use 'mean' or 'sum'.")

    def replicate_params(
        self,
        params: Any,
        mesh: Mesh,
    ) -> Any:
        """Replicate parameters across all devices.

        Args:
            params: PyTree of parameters to replicate.
            mesh: The device mesh to use.

        Returns:
            The replicated parameters.

        Example:
            >>> dp = DataParallel()
            >>> replicated = dp.replicate_params(params, mesh)
        """
        # Replicate = no partitioning
        sharding = NamedSharding(mesh, PartitionSpec())
        return jax.tree.map(
            lambda x: jax.device_put(x, sharding),
            params,
        )
