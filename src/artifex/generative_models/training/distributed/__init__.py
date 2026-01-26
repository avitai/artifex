"""Distributed training utilities for generative models.

This module provides utilities for distributed training across multiple
devices, including device mesh management, data parallelism, device
placement, and distributed metrics collection.

Key Components:
    - DeviceMeshManager: Create and manage device meshes
    - DataParallel: Data-parallel training utilities
    - DevicePlacement: Explicit device placement utilities
    - DistributedMetrics: Metrics aggregation across devices

Example:
    >>> from artifex.generative_models.training.distributed import (
    ...     DeviceMeshManager,
    ...     DataParallel,
    ...     DevicePlacement,
    ...     DistributedMetrics,
    ... )
    >>>
    >>> # Create a device mesh
    >>> manager = DeviceMeshManager()
    >>> mesh = manager.create_device_mesh({"data": 1})
    >>>
    >>> # Create data parallel sharding
    >>> dp = DataParallel()
    >>> sharding = dp.create_data_parallel_sharding(mesh)
"""

from artifex.generative_models.training.distributed.data_parallel import (
    DataParallel,
)
from artifex.generative_models.training.distributed.device_placement import (
    BatchSizeRecommendation,
    DevicePlacement,
    distribute_batch,
    get_batch_size_recommendation,
    HardwareType,
    place_on_device,
)
from artifex.generative_models.training.distributed.mesh import DeviceMeshManager
from artifex.generative_models.training.distributed.metrics import DistributedMetrics


__all__ = [
    # Mesh management
    "DeviceMeshManager",
    # Data parallelism
    "DataParallel",
    # Device placement
    "DevicePlacement",
    "HardwareType",
    "BatchSizeRecommendation",
    "place_on_device",
    "distribute_batch",
    "get_batch_size_recommendation",
    # Metrics
    "DistributedMetrics",
]
