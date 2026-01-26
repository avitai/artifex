"""Sharding strategies and parallelism configuration for scalable training.

This module provides comprehensive sharding infrastructure including:
- Abstract base class for sharding strategies
- Concrete implementations for different parallelism types
- Multi-dimensional parallelism support
- Configuration management for complex sharding setups

All implementations prioritize performance and follow JAX/Flax NNX patterns.
"""

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec


@dataclass
class ShardingConfig:
    """Configuration for multi-dimensional parallelism setup.

    Defines the parallelism dimensions and FSDP settings for a model.
    """

    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    fsdp_enabled: bool = False
    fsdp_min_weight_size: int = 1024

    def get_total_device_count(self) -> int:
        """Calculate total devices needed for this configuration."""
        return self.data_parallel_size * self.tensor_parallel_size * self.pipeline_parallel_size

    @classmethod
    def from_device_count(cls, device_count: int) -> "ShardingConfig":
        """Create optimal sharding config for given device count.

        Uses heuristics to balance different parallelism dimensions.
        """
        # Simple heuristic: prioritize data parallel, then tensor parallel
        if device_count == 1:
            return cls()
        elif device_count <= 4:
            return cls(data_parallel_size=device_count)
        elif device_count == 8:
            return cls(data_parallel_size=2, tensor_parallel_size=4)
        else:
            # For larger counts, use balanced approach
            tensor_size = min(8, int(math.sqrt(device_count)))
            data_size = device_count // tensor_size
            return cls(data_parallel_size=data_size, tensor_parallel_size=tensor_size)


@dataclass
class ParallelismConfig:
    """Complete parallelism configuration including mesh topology.

    Combines sharding configuration with device mesh setup.
    """

    mesh_shape: tuple[int, ...]
    mesh_axis_names: tuple[str, ...]
    sharding_config: ShardingConfig

    def is_valid(self) -> bool:
        """Validate that mesh shape matches sharding configuration."""
        expected_devices = self.sharding_config.get_total_device_count()
        actual_devices = math.prod(self.mesh_shape)
        return expected_devices == actual_devices

    @classmethod
    def from_sharding_config(cls, config: ShardingConfig) -> "ParallelismConfig":
        """Create parallelism config from sharding configuration."""
        # Build mesh shape from sharding config
        mesh_shape = []
        axis_names = []

        if config.data_parallel_size > 1:
            mesh_shape.append(config.data_parallel_size)
            axis_names.append("data")

        if config.tensor_parallel_size > 1:
            mesh_shape.append(config.tensor_parallel_size)
            axis_names.append("model")

        if config.pipeline_parallel_size > 1:
            mesh_shape.append(config.pipeline_parallel_size)
            axis_names.append("pipeline")

        # Default to data parallel if no dimensions specified
        if not mesh_shape:
            mesh_shape = [1]
            axis_names = ["data"]

        return cls(
            mesh_shape=tuple(mesh_shape), mesh_axis_names=tuple(axis_names), sharding_config=config
        )


class ShardingStrategy(ABC):
    """Abstract base class for sharding strategies.

    Defines the interface that all sharding strategies must implement
    for consistent handling of different parallelism types.
    """

    def __init__(self, axis_name: str, mesh_axis: int) -> None:
        """Initialize sharding strategy.

        Args:
            axis_name: Name of the mesh axis for this strategy
            mesh_axis: Index of the mesh axis
        """
        self.axis_name = axis_name
        self.mesh_axis = mesh_axis

    @abstractmethod
    def get_partition_spec(self, tensor_shape: tuple[str, ...]) -> PartitionSpec:
        """Get partition specification for a tensor with given shape names.

        Args:
            tensor_shape: Tuple of dimension names for the tensor

        Returns:
            PartitionSpec defining how to shard the tensor
        """
        pass

    @abstractmethod
    def apply_sharding(self, array: Array, mesh: Mesh) -> Array:
        """Apply sharding to an array using the given mesh.

        Args:
            array: JAX array to shard
            mesh: Device mesh for sharding

        Returns:
            Sharded array
        """
        pass

    def get_sharding_constraints(self) -> dict[str, Any]:
        """Get sharding constraints for this strategy.

        Returns:
            Dictionary of sharding constraints
        """
        return {"axis_name": self.axis_name, "mesh_axis": self.mesh_axis}


class DataParallelStrategy(ShardingStrategy):
    """Data parallel sharding strategy.

    Shards the batch dimension across devices while replicating
    model parameters and computation.
    """

    def get_partition_spec(self, tensor_shape: tuple[str, ...]) -> PartitionSpec:
        """Get partition spec for data parallel sharding.

        Only shards the batch dimension, leaves others replicated.
        """
        specs: list[str | None] = []
        for dim_name in tensor_shape:
            if dim_name == "batch":
                specs.append(self.axis_name)
            else:
                specs.append(None)
        return PartitionSpec(*specs)

    def apply_sharding(self, array: Array, mesh: Mesh) -> Array:
        """Apply data parallel sharding to array."""
        # Create named sharding for data parallel
        none_specs = [None] * (array.ndim - 1)
        partition_spec = PartitionSpec(self.axis_name, *none_specs)
        sharding = NamedSharding(mesh, partition_spec)

        # Apply sharding
        return jax.device_put(array, sharding)


class FSDPStrategy(ShardingStrategy):
    """Fully Sharded Data Parallel strategy.

    Shards model parameters across devices to reduce memory usage
    while maintaining training efficiency.
    """

    def __init__(self, axis_name: str, mesh_axis: int, min_weight_size: int = 1024) -> None:
        """Initialize FSDP strategy.

        Args:
            axis_name: Name of the mesh axis
            mesh_axis: Index of the mesh axis
            min_weight_size: Minimum first dimension size to enable sharding
        """
        super().__init__(axis_name, mesh_axis)
        self.min_weight_size = min_weight_size

    def should_shard_weight(self, weight: Array) -> bool:
        """Determine if a weight should be sharded based on its size.

        Args:
            weight: Weight array to check

        Returns:
            True if weight should be sharded, False otherwise
        """
        # Check the first dimension size (what FSDP actually shards)
        return weight.shape[0] >= self.min_weight_size

    def get_partition_spec(self, tensor_shape: tuple[str, ...]) -> PartitionSpec:
        """Get partition spec for FSDP sharding.

        Shards along the first dimension of weight tensors.
        """
        specs: list[str | None] = []
        for i, dim_name in enumerate(tensor_shape):
            if i == 0 and dim_name in ["out_features", "features", "hidden"]:
                specs.append(self.axis_name)
            else:
                specs.append(None)
        return PartitionSpec(*specs)

    def get_gradient_partition_spec(self, tensor_shape: tuple[str, ...]) -> PartitionSpec:
        """Get partition spec for gradient sharding (same as weights)."""
        return self.get_partition_spec(tensor_shape)

    def apply_sharding(self, array: Array, mesh: Mesh) -> Array:
        """Apply FSDP sharding to array."""
        if not self.should_shard_weight(array):
            # Replicate small weights
            partition_spec = PartitionSpec(*([None] * array.ndim))
        else:
            # Shard along first dimension
            none_specs = [None] * (array.ndim - 1)
            partition_spec = PartitionSpec(self.axis_name, *none_specs)

        sharding = NamedSharding(mesh, partition_spec)
        return jax.device_put(array, sharding)


class TensorParallelStrategy(ShardingStrategy):
    """Tensor parallel sharding strategy.

    Shards model computation across devices by splitting tensors
    along specific dimensions (typically features).
    """

    def __init__(self, axis_name: str, mesh_axis: int, shard_dimension: str | None = None) -> None:
        """Initialize tensor parallel strategy.

        Args:
            axis_name: Name of the mesh axis
            mesh_axis: Index of the mesh axis
            shard_dimension: Preferred dimension to shard
                ('in_features' or 'out_features')
        """
        super().__init__(axis_name, mesh_axis)
        self.shard_dimension = shard_dimension

    def get_partition_spec(self, tensor_shape: tuple[str, ...]) -> PartitionSpec:
        """Get partition spec for tensor parallel sharding."""
        specs: list[str | None] = []
        for dim_name in tensor_shape:
            if self.shard_dimension and dim_name == self.shard_dimension:
                specs.append(self.axis_name)
            elif not self.shard_dimension and dim_name == "hidden":
                specs.append(self.axis_name)
            else:
                specs.append(None)
        return PartitionSpec(*specs)

    def get_linear_weight_spec(self) -> PartitionSpec:
        """Get partition spec for linear layer weights."""
        if self.shard_dimension == "in_features":
            # (out, in) -> shard in
            return PartitionSpec(None, self.axis_name)
        else:
            # (out, in) -> shard out
            return PartitionSpec(self.axis_name, None)

    def get_attention_qkv_spec(self) -> PartitionSpec:
        """Get partition spec for attention QKV projections."""
        return PartitionSpec(None, self.axis_name)  # Shard output features

    def get_attention_output_spec(self) -> PartitionSpec:
        """Get partition spec for attention output projection."""
        return PartitionSpec(self.axis_name, None)  # Shard input features

    def apply_sharding(self, array: Array, mesh: Mesh) -> Array:
        """Apply tensor parallel sharding to array."""
        # Default to sharding the last dimension if not specified
        if array.ndim == 2:  # Weight matrix
            if self.shard_dimension == "in_features":
                partition_spec = PartitionSpec(None, self.axis_name)
            else:
                partition_spec = PartitionSpec(self.axis_name, None)
        else:
            # For other tensors, shard the last dimension
            specs: list[str | None] = [None] * array.ndim
            specs[-1] = self.axis_name
            partition_spec = PartitionSpec(*specs)

        sharding = NamedSharding(mesh, partition_spec)
        return jax.device_put(array, sharding)


class PipelineParallelStrategy(ShardingStrategy):
    """Pipeline parallel sharding strategy.

    Distributes model layers across devices to enable pipeline parallelism
    for very large models that don't fit on single devices.
    """

    def __init__(self, axis_name: str, mesh_axis: int, num_stages: int) -> None:
        """Initialize pipeline parallel strategy.

        Args:
            axis_name: Name of the mesh axis
            mesh_axis: Index of the mesh axis
            num_stages: Number of pipeline stages
        """
        super().__init__(axis_name, mesh_axis)
        self.num_stages = num_stages

    def assign_layers_to_stages(self, num_layers: int) -> list[int]:
        """Assign layers to pipeline stages.

        Args:
            num_layers: Total number of layers in the model

        Returns:
            list of layer counts per stage
        """
        layers_per_stage = num_layers // self.num_stages
        remainder = num_layers % self.num_stages

        assignments = [layers_per_stage] * self.num_stages

        # Distribute remainder layers
        for i in range(remainder):
            assignments[i] += 1

        return assignments

    def get_partition_spec(self, tensor_shape: tuple[str, ...]) -> PartitionSpec:
        """Get partition spec for pipeline parallel sharding.

        Pipeline parallelism doesn't shard individual tensors,
        but rather assigns entire layers to different devices.
        """
        # No sharding of individual tensors in pipeline parallelism
        return PartitionSpec(*([None] * len(tensor_shape)))

    def get_forward_communication_pattern(self) -> list[tuple[int, int]]:
        """Get communication pattern for forward pass.

        Returns:
            list of (source_stage, dest_stage) pairs
        """
        return [(i, i + 1) for i in range(self.num_stages - 1)]

    def get_backward_communication_pattern(self) -> list[tuple[int, int]]:
        """Get communication pattern for backward pass.

        Returns:
            list of (source_stage, dest_stage) pairs
        """
        return [(i + 1, i) for i in range(self.num_stages - 1)]

    def apply_sharding(self, array: Array, mesh: Mesh) -> Array:
        """Apply pipeline parallel sharding to array.

        Pipeline parallelism handles layer assignment rather than
        tensor sharding.
        """
        # Replicate tensors within each pipeline stage
        partition_spec = PartitionSpec(*([None] * array.ndim))
        sharding = NamedSharding(mesh, partition_spec)
        return jax.device_put(array, sharding)


class MultiDimensionalStrategy:
    """Multi-dimensional parallelism strategy combining multiple approaches.

    Combines different sharding strategies (data, tensor, FSDP, pipeline)
    to achieve optimal performance for large-scale training.
    """

    def __init__(self, strategies: dict[str, ShardingStrategy], config: ParallelismConfig) -> None:
        """Initialize multi-dimensional strategy.

        Args:
            strategies: Dictionary mapping strategy names to strategy instances
            config: Sharding configuration for the multi-dimensional strategy
        """
        self.strategies = strategies
        self.config = config
        self._validate_strategies()

    def _validate_strategies(self) -> None:
        """Validate that strategies are compatible."""
        axis_names = [strategy.axis_name for strategy in self.strategies.values()]

        # Check for conflicting axis names
        axis_counts: dict[str, int] = {}
        for name in axis_names:
            axis_counts[name] = axis_counts.get(name, 0) + 1

        for axis_name, count in axis_counts.items():
            if count > 1:
                # Get strategies that use this axis
                strategies_with_axis = [
                    (strategy_name, type(s).__name__)
                    for strategy_name, s in self.strategies.items()
                    if s.axis_name == axis_name
                ]

                # Extract just the strategy type names
                strategy_types = [strategy_type for _, strategy_type in strategies_with_axis]

                # Allow data and FSDP to share axis (they're compatible)
                compatible_strategies = {"DataParallelStrategy", "FSDPStrategy"}

                # Check if all strategies sharing this axis are compatible
                if not (set(strategy_types) <= compatible_strategies):
                    raise ValueError(
                        f"Conflicting strategies for axis {axis_name}: {strategy_types}"
                    )

                # Also check for identical strategy types (not allowed)
                strategy_type_counts: dict[str, int] = {}
                for strategy_type in strategy_types:
                    strategy_type_counts[strategy_type] = (
                        strategy_type_counts.get(strategy_type, 0) + 1
                    )

                for strategy_type, type_count in strategy_type_counts.items():
                    if type_count > 1:
                        raise ValueError(
                            f"Conflicting strategies: multiple {strategy_type} instances "
                            f"using axis {axis_name}"
                        )

    def get_combined_partition_spec(
        self, tensor_name: str, tensor_shape: tuple[str, ...]
    ) -> PartitionSpec:
        """Get combined partition spec from all strategies.

        Args:
            tensor_name: Name/type of the tensor
            tensor_shape: Shape dimension names of the tensor

        Returns:
            Combined PartitionSpec
        """
        # Start with all None specs
        combined_specs: list[str | None] = [None] * len(tensor_shape)

        # Apply each strategy
        for strategy_name, strategy in self.strategies.items():
            strategy_spec = strategy.get_partition_spec(tensor_shape)

            # Merge non-None specs, handling conflicts
            for i, spec in enumerate(strategy_spec):
                if spec is not None:
                    if combined_specs[i] is None:
                        combined_specs[i] = spec
                    elif combined_specs[i] != spec:
                        # Conflict resolution needed
                        resolved = self._resolve_spec_conflict(
                            tensor_name, i, combined_specs[i], spec
                        )
                        combined_specs[i] = resolved

        return PartitionSpec(*combined_specs)

    def _resolve_spec_conflict(
        self, tensor_name: str, dim_index: int, existing_spec: str | None, new_spec: str
    ) -> str:
        """Resolve conflicts between partition specs.

        Args:
            tensor_name: Name of the tensor
            dim_index: Dimension index with conflict
            existing_spec: Current partition spec (can be None)
            new_spec: Conflicting partition spec

        Returns:
            Resolved partition spec
        """
        # If existing is None, use new spec
        if existing_spec is None:
            return new_spec

        # Priority rules for conflict resolution
        if "model" in [existing_spec, new_spec]:
            return "model"  # Tensor parallel takes priority
        elif "fsdp" in [existing_spec, new_spec]:
            return "fsdp"  # FSDP takes priority over data parallel
        else:
            return existing_spec  # Keep existing

    def resolve_sharding_conflicts(
        self, tensor_name: str, proposed_specs: dict[str, PartitionSpec]
    ) -> PartitionSpec:
        """Resolve conflicts between multiple proposed partition specs.

        Args:
            tensor_name: Name of the tensor
            proposed_specs: Dictionary of strategy names to proposed specs

        Returns:
            Resolved PartitionSpec
        """
        if not proposed_specs:
            return PartitionSpec()

        # Start with first spec
        first_spec = next(iter(proposed_specs.values()))
        result_specs = list(first_spec)

        # Merge other specs with conflict resolution
        for strategy_name, spec in proposed_specs.items():
            for i, spec_value in enumerate(spec):
                if spec_value is not None and result_specs[i] != spec_value:
                    if result_specs[i] is None:
                        result_specs[i] = spec_value
                    else:
                        # Apply resolution rules
                        result_specs[i] = self._resolve_spec_conflict(
                            tensor_name, i, result_specs[i], spec_value
                        )

        return PartitionSpec(*result_specs)

    def create_partition_spec(self, param_shape: tuple[int, ...], param_name: str) -> PartitionSpec:
        """Create PartitionSpec for pipeline parallel sharding."""
        # Simple stage assignment based on parameter name
        if "embedding" in param_name or "position" in param_name:
            stage = 0  # First stage
        elif "output" in param_name or "logits" in param_name:
            stage = self.config.pipeline_parallel_size - 1  # Last stage
        else:
            # Distribute transformer layers across stages
            layer_match = re.search(r"layer_(\d+)", param_name)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                layers_per_stage = 24 // self.config.pipeline_parallel_size  # Assume 24 layers
                stage = min(
                    layer_idx // layers_per_stage,
                    self.config.pipeline_parallel_size - 1,
                )
            else:
                stage = 0

        # Return PartitionSpec indicating pipeline stage assignment
        # (simplified - in practice this would be more complex)
        return PartitionSpec(f"stage_{stage}" if stage is not None else None)
