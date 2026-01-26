"""Device mesh management utilities for JAX distributed training.

This module provides a unified implementation for creating and managing
device meshes across multiple accelerators in distributed training settings.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax
import numpy as np
from jax.sharding import Mesh


class DeviceMeshManager:
    """Manager for creating and configuring JAX device meshes.

    This class provides methods for creating device meshes with various
    configurations for data parallelism, model parallelism, and hybrid
    parallelism strategies.

    Example:
        >>> manager = DeviceMeshManager()
        >>> # Create a simple data-parallel mesh
        >>> mesh = manager.create_device_mesh([("data", 1)])
        >>> # Or using dict specification
        >>> mesh = manager.create_device_mesh({"data": 1})
    """

    def __init__(self, devices: Sequence[Any] | None = None) -> None:
        """Initialize DeviceMeshManager.

        Args:
            devices: Optional list of devices to use. If None, uses all
                available devices from jax.devices().
        """
        if devices is not None:
            self._devices = list(devices)
        else:
            self._devices = list(jax.devices())

    def create_device_mesh(
        self,
        mesh_shape: dict[str, int] | list[tuple[str, int]],
        devices: Sequence[Any] | None = None,
    ) -> Mesh:
        """Create a device mesh with the specified shape.

        Args:
            mesh_shape: Shape specification as either:
                - dict mapping axis names to sizes, e.g., {"data": 2, "model": 1}
                - list of (axis_name, size) tuples, e.g., [("data", 2), ("model", 1)]
            devices: Optional list of devices to use. If None, uses devices
                from the manager.

        Returns:
            A JAX Mesh with the specified configuration.

        Raises:
            ValueError: If mesh requires more devices than available.

        Example:
            >>> manager = DeviceMeshManager()
            >>> mesh = manager.create_device_mesh({"data": 2})
            >>> mesh = manager.create_device_mesh([("data", 2), ("model", 1)])
        """
        # Use provided devices or fall back to manager's devices
        available_devices = list(devices) if devices is not None else self._devices

        # Normalize to list of tuples
        if isinstance(mesh_shape, dict):
            axis_specs = list(mesh_shape.items())
        else:
            axis_specs = mesh_shape

        # Calculate total devices needed
        total_devices = 1
        for _, size in axis_specs:
            total_devices *= size

        if total_devices > len(available_devices):
            raise ValueError(
                f"Not enough devices: mesh requires {total_devices} devices but only "
                f"{len(available_devices)} available."
            )

        # Extract axis names and shape
        axis_names = tuple(name for name, _ in axis_specs)
        shape = tuple(size for _, size in axis_specs)

        # Create device array with the specified shape
        device_array = np.array(available_devices[:total_devices]).reshape(shape)

        return Mesh(device_array, axis_names=axis_names)

    def create_data_parallel_mesh(
        self,
        num_devices: int | None = None,
        axis_name: str = "data",
    ) -> Mesh:
        """Create a mesh for data parallelism.

        Args:
            num_devices: Number of devices to use. If None, uses all available.
            axis_name: Name of the data parallel axis.

        Returns:
            A JAX Mesh configured for data parallelism.

        Raises:
            ValueError: If requested more devices than available.

        Example:
            >>> manager = DeviceMeshManager()
            >>> mesh = manager.create_data_parallel_mesh()
        """
        if num_devices is None:
            num_devices = len(self._devices)

        return self.create_device_mesh({axis_name: num_devices})

    def create_model_parallel_mesh(
        self,
        num_devices: int | None = None,
        axis_name: str = "model",
    ) -> Mesh:
        """Create a mesh for model parallelism.

        Args:
            num_devices: Number of devices to use. If None, uses all available.
            axis_name: Name of the model parallel axis.

        Returns:
            A JAX Mesh configured for model parallelism.

        Raises:
            ValueError: If requested more devices than available.

        Example:
            >>> manager = DeviceMeshManager()
            >>> mesh = manager.create_model_parallel_mesh()
        """
        if num_devices is None:
            num_devices = len(self._devices)

        return self.create_device_mesh({axis_name: num_devices})

    def create_hybrid_mesh(
        self,
        data_parallel_size: int = 1,
        model_parallel_size: int = 1,
        data_axis: str = "data",
        model_axis: str = "model",
    ) -> Mesh:
        """Create a mesh for hybrid data and model parallelism.

        Args:
            data_parallel_size: Number of devices for data parallelism.
            model_parallel_size: Number of devices for model parallelism.
            data_axis: Name of the data parallel axis.
            model_axis: Name of the model parallel axis.

        Returns:
            A JAX Mesh configured for hybrid parallelism.

        Raises:
            ValueError: If total devices exceed available devices.

        Example:
            >>> manager = DeviceMeshManager()
            >>> mesh = manager.create_hybrid_mesh(
            ...     data_parallel_size=2,
            ...     model_parallel_size=2,
            ... )
        """
        return self.create_device_mesh(
            [
                (data_axis, data_parallel_size),
                (model_axis, model_parallel_size),
            ]
        )

    def get_mesh_info(self, mesh: Mesh) -> dict[str, Any]:
        """Get information about a device mesh.

        Args:
            mesh: The mesh to get information about.

        Returns:
            Dictionary containing mesh information with keys:
                - total_devices: Total number of devices in the mesh
                - axes: Dict mapping axis names to their sizes

        Example:
            >>> manager = DeviceMeshManager()
            >>> mesh = manager.create_device_mesh({"data": 1})
            >>> info = manager.get_mesh_info(mesh)
            >>> print(info["total_devices"])
        """
        return {
            "total_devices": mesh.size,
            "axes": {name: mesh.shape[name] for name in mesh.axis_names},
        }

    @property
    def num_devices(self) -> int:
        """Get the number of available devices."""
        return len(self._devices)

    @property
    def devices(self) -> list[Any]:
        """Get the list of available devices."""
        return list(self._devices)
