"""Device mesh management utilities for scalable distributed training.

This module provides comprehensive device mesh management including:
- Device mesh creation and optimization
- Topology optimization for different workloads
- Validation and configuration utilities
- Hardware-aware mesh shape calculation

All implementations prioritize performance and follow JAX/Flax NNX patterns.
"""

import math

import jax
from jax.sharding import Mesh

from .sharding import ParallelismConfig


class DeviceMeshManager:
    """Device mesh management for distributed training optimization.

    Provides utilities for creating and optimizing device meshes
    based on available hardware and workload characteristics.
    """

    def __init__(self, mesh_shape: tuple[int, ...], axis_names: tuple[str, ...]) -> None:
        """Initialize device mesh manager."""
        self.mesh_shape = mesh_shape
        self.axis_names = axis_names

    def create_mesh(self, mesh_shape: tuple[int, ...], axis_names: tuple[str, ...]) -> Mesh:
        """Create device mesh with specified shape and axis names.

        Args:
            mesh_shape: Shape of the device mesh
            axis_names: Names for each mesh axis

        Returns:
            JAX device mesh
        """
        devices = jax.devices()
        total_devices = math.prod(mesh_shape)

        if len(devices) < total_devices:
            raise ValueError(f"Insufficient devices: need {total_devices}, have {len(devices)}")

        # Use first N devices for mesh
        mesh_devices = devices[:total_devices]
        reshaped_devices = [
            mesh_devices[i : i + mesh_shape[-1]] for i in range(0, total_devices, mesh_shape[-1])
        ]

        # Flatten and reshape according to mesh_shape
        flat_devices = [device for row in reshaped_devices for device in row]

        # Create properly shaped device array
        device_array = []
        devices_per_row = mesh_shape[-1] if len(mesh_shape) > 1 else total_devices

        for i in range(0, total_devices, devices_per_row):
            device_array.append(flat_devices[i : i + devices_per_row])

        return Mesh(device_array, axis_names)

    def create_mesh_from_config(self, config: ParallelismConfig) -> Mesh:
        """Create mesh from parallelism configuration.

        Args:
            config: Parallelism configuration

        Returns:
            JAX device mesh
        """
        return self.create_mesh(config.mesh_shape, config.mesh_axis_names)

    def get_optimal_mesh_shape(self, device_count: int, dimensions: int = 2) -> tuple[int, ...]:
        """Calculate optimal mesh shape for given device count.

        Args:
            device_count: Number of available devices
            dimensions: Number of mesh dimensions

        Returns:
            Optimal mesh shape tuple
        """
        if dimensions == 1:
            return (device_count,)
        elif dimensions == 2:
            # Find factors that balance the dimensions
            factors = self._get_factors(device_count)
            mid_point = len(factors) // 2

            # Try to balance dimensions
            if len(factors) >= 2:
                factor1 = factors[mid_point - 1] if mid_point > 0 else factors[0]
                factor2 = device_count // factor1
                return (factor1, factor2)
            else:
                return (1, device_count)
        else:
            # For 3+ dimensions, use iterative factorization
            factors = self._get_factors(device_count)

            if dimensions == 3:
                # Try to find three factors that multiply to device_count
                best_shape = (1, 1, device_count)
                best_variance = float("inf")

                for i in factors:
                    remaining = device_count // i
                    for j in self._get_factors(remaining):
                        k = remaining // j
                        if i * j * k == device_count:
                            # Calculate variance to prefer balanced shapes
                            variance = (
                                (i - device_count ** (1 / 3)) ** 2
                                + (j - device_count ** (1 / 3)) ** 2
                                + (k - device_count ** (1 / 3)) ** 2
                            )
                            if variance < best_variance:
                                best_variance = variance
                                best_shape = (i, j, k)

                return best_shape
            else:
                # For 4+ dimensions, recursively build shape
                # Start with largest factor and recurse
                largest_factor = max(factors)
                remaining_devices = device_count // largest_factor
                remaining_shape = self.get_optimal_mesh_shape(remaining_devices, dimensions - 1)
                return (largest_factor, *remaining_shape)

    def _get_factors(self, n: int) -> list[int]:
        """Get all factors of a number."""
        factors = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.append(i)
                if i != n // i:
                    factors.append(n // i)
        return sorted(factors)

    def optimize_for_transformer(
        self, device_count: int, model_size: str, sequence_length: int
    ) -> tuple[int, ...]:
        """Optimize mesh shape for transformer workloads.

        Args:
            device_count: Number of available devices
            model_size: Model size (e.g., '7B', '13B', '70B')
            sequence_length: Input sequence length

        Returns:
            Optimized mesh shape for transformer workloads
        """
        # Parse model size
        if model_size.endswith("B"):
            size_billions = float(model_size[:-1])
        elif model_size.endswith("M"):
            size_billions = float(model_size[:-1]) / 1000
        else:
            size_billions = 1.0  # Default

        # Optimization heuristics for transformers
        if size_billions < 1:  # < 1B parameters
            # Small models: prefer data parallelism
            return (device_count, 1) if device_count <= 8 else (8, device_count // 8)
        elif size_billions < 10:  # 1B - 10B parameters
            # Medium models: balanced data + tensor parallelism
            if device_count <= 4:
                return (device_count,)
            else:
                tensor_parallel = min(4, device_count // 2)
                data_parallel = device_count // tensor_parallel
                return (data_parallel, tensor_parallel)
        else:  # > 10B parameters
            # Large models: more tensor parallelism
            tensor_parallel = min(8, device_count // 2)
            data_parallel = device_count // tensor_parallel
            return (data_parallel, tensor_parallel)

    def validate_mesh_config(self, mesh_shape: tuple[int, ...], device_count: int) -> bool:
        """Validate mesh configuration.

        Args:
            mesh_shape: Proposed mesh shape
            device_count: Available device count

        Returns:
            True if configuration is valid
        """
        required_devices = math.prod(mesh_shape)
        return required_devices <= device_count and required_devices > 0


def create_device_mesh(mesh_shape: tuple[int, ...], axis_names: tuple[str, ...]) -> Mesh:
    """Create device mesh with specified configuration.

    Args:
        mesh_shape: Shape of the device mesh
        axis_names: Names for each mesh axis

    Returns:
        JAX device mesh
    """
    manager = DeviceMeshManager(mesh_shape=mesh_shape, axis_names=axis_names)
    return manager.create_mesh(mesh_shape, axis_names)


def get_optimal_mesh_shape(
    device_count: int, parallelism_config: ParallelismConfig
) -> tuple[int, ...]:
    """Get optimal mesh shape for given configuration.

    Args:
        device_count: Number of available devices
        parallelism_config: Parallelism configuration

    Returns:
        Optimal mesh shape
    """
    # Use the shape from the config if it's valid
    if parallelism_config.is_valid():
        return parallelism_config.mesh_shape

    # Otherwise, calculate optimal shape
    manager = DeviceMeshManager(
        mesh_shape=parallelism_config.mesh_shape,
        axis_names=parallelism_config.mesh_axis_names,
    )
    dimensions = len(parallelism_config.mesh_axis_names)
    return manager.get_optimal_mesh_shape(device_count, dimensions)
