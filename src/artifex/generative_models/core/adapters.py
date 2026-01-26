"""Model adapter classes for different architectures.

This module provides standardized interfaces for scaling different model
architectures with consistent APIs and optimization strategies.

All implementations follow JAX/Flax NNX best practices and provide
hardware-aware optimization for different model types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx

from ..scaling.sharding import ParallelismConfig
from .performance import HardwareSpecs


@dataclass
class ModelSpecs:
    """Specifications for model architecture and scaling requirements."""

    model_type: str
    num_parameters: int
    memory_requirement_gb: float
    compute_intensity: str = "medium"
    preferred_parallelism: str = "data_parallel"


class ModelAdapter(ABC):
    """Abstract base class for model adapters.

    Provides standardized interface for scaling different model architectures
    with consistent optimization strategies and performance characteristics.
    """

    def __init__(
        self,
        model: nnx.Module,
        hardware_specs: HardwareSpecs,
        parallelism_config: ParallelismConfig | None = None,
    ) -> None:
        """Initialize model adapter.

        Args:
            model: Model to adapt for scaling
            hardware_specs: Hardware specifications
            parallelism_config: Optional parallelism configuration
        """
        self.model = model
        self.hardware_specs = hardware_specs
        self.parallelism_config = parallelism_config

    @abstractmethod
    def get_model_specs(self) -> ModelSpecs:
        """Get model specifications for scaling optimization."""
        pass

    @abstractmethod
    def estimate_memory_usage(self, batch_size: int) -> float:
        """Estimate memory usage for given batch size."""
        pass

    @abstractmethod
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for current hardware."""
        pass

    @abstractmethod
    def apply_sharding(self, sharding_config: Any) -> nnx.Module:
        """Apply sharding configuration to model."""
        pass

    def get_performance_characteristics(self) -> dict[str, Any]:
        """Get performance characteristics for optimization."""
        specs = self.get_model_specs()
        return {
            "model_type": specs.model_type,
            "compute_intensity": specs.compute_intensity,
            "memory_bound": specs.memory_requirement_gb > 10.0,
            "preferred_parallelism": specs.preferred_parallelism,
            "scalability_factor": self._estimate_scalability_factor(),
        }

    def _estimate_scalability_factor(self) -> float:
        """Estimate how well the model scales with additional devices."""
        specs = self.get_model_specs()

        # Larger models generally scale better
        if specs.num_parameters > 1e9:  # 1B+ parameters
            return 0.9
        elif specs.num_parameters > 1e8:  # 100M+ parameters
            return 0.7
        else:
            return 0.5


class TransformerAdapter(ModelAdapter):
    """Adapter for transformer-based models."""

    def __init__(
        self,
        model: nnx.Module,
        hardware_specs: HardwareSpecs,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        vocab_size: int | None = None,
        parallelism_config: ParallelismConfig | None = None,
    ) -> None:
        """Initialize transformer adapter.

        Args:
            model: Transformer model
            hardware_specs: Hardware specifications
            num_layers: Number of transformer layers
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            vocab_size: Vocabulary size (for language models)
            parallelism_config: Optional parallelism configuration
        """
        super().__init__(model, hardware_specs, parallelism_config)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.vocab_size = vocab_size

    def get_model_specs(self) -> ModelSpecs:
        """Get transformer model specifications."""
        # Estimate parameters: embedding + attention + FFN layers
        embedding_params = self.vocab_size * self.hidden_size if self.vocab_size else 0
        attention_params = self.num_layers * 4 * self.hidden_size * self.hidden_size
        ffn_params = self.num_layers * 8 * self.hidden_size * self.hidden_size

        total_params = embedding_params + attention_params + ffn_params
        memory_gb = total_params * 4 / (1024**3)  # Assuming float32

        return ModelSpecs(
            model_type="transformer",
            num_parameters=total_params,
            memory_requirement_gb=memory_gb,
            compute_intensity="high",
            preferred_parallelism="tensor_parallel",
        )

    def estimate_memory_usage(self, batch_size: int, sequence_length: int = 512) -> float:
        """Estimate memory usage for transformer with given batch size."""
        specs = self.get_model_specs()

        # Model parameters
        param_memory = specs.memory_requirement_gb

        # Activations (rough estimate)
        activation_memory = (
            batch_size * sequence_length * self.hidden_size * self.num_layers * 4 / (1024**3)
        )

        # Attention matrices
        attention_memory = (
            batch_size * self.num_heads * sequence_length * sequence_length * 4 / (1024**3)
        )

        return param_memory + activation_memory + attention_memory

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for transformer on current hardware."""
        # 80% utilization
        available_memory = self.hardware_specs.memory_gb * 0.8

        # Binary search for optimal batch size
        min_batch, max_batch = 1, 512

        while min_batch < max_batch:
            mid_batch = (min_batch + max_batch + 1) // 2
            estimated_memory = self.estimate_memory_usage(mid_batch)

            if estimated_memory <= available_memory:
                min_batch = mid_batch
            else:
                max_batch = mid_batch - 1

        return max(1, min_batch)

    def apply_sharding(self, sharding_config: Any) -> nnx.Module:
        """Apply tensor parallel sharding to transformer."""
        # Placeholder for tensor parallel implementation
        # In a real implementation, this would apply sharding annotations
        return self.model


class DiffusionAdapter(ModelAdapter):
    """Adapter for diffusion models (U-Net based)."""

    def __init__(
        self,
        model: nnx.Module,
        hardware_specs: HardwareSpecs,
        image_size: int,
        channels: int,
        model_channels: int,
        num_res_blocks: int,
        parallelism_config: ParallelismConfig | None = None,
    ) -> None:
        """Initialize diffusion adapter.

        Args:
            model: Diffusion model (U-Net)
            hardware_specs: Hardware specifications
            image_size: Input image size
            channels: Number of input channels
            model_channels: Base model channels
            num_res_blocks: Number of residual blocks
            parallelism_config: Optional parallelism configuration
        """
        super().__init__(model, hardware_specs, parallelism_config)
        self.image_size = image_size
        self.channels = channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks

    def get_model_specs(self) -> ModelSpecs:
        """Get diffusion model specifications."""
        # Rough parameter estimation for U-Net
        base_params = self.model_channels * self.model_channels * 9

        # Parameters scale with resolution levels and blocks
        resolution_levels = 4  # Typical U-Net downsampling levels
        total_params = (
            base_params * resolution_levels * self.num_res_blocks * 2
        )  # Encoder + decoder

        memory_gb = total_params * 4 / (1024**3)  # Assuming float32

        return ModelSpecs(
            model_type="diffusion",
            num_parameters=total_params,
            memory_requirement_gb=memory_gb,
            compute_intensity="medium",
            preferred_parallelism="data_parallel",
        )

    def estimate_memory_usage(self, batch_size: int) -> float:
        """Estimate memory usage for diffusion model."""
        specs = self.get_model_specs()

        # Model parameters
        param_memory = specs.memory_requirement_gb

        # Feature maps at different resolutions
        total_feature_memory = 0.0
        current_size = self.image_size
        current_channels = self.model_channels

        # Estimate memory for each resolution level
        for _ in range(4):  # Typical U-Net levels
            feature_memory = (
                batch_size * current_channels * current_size * current_size * 4 / (1024**3)
            )
            total_feature_memory += feature_memory

            current_size //= 2  # Downsample
            current_channels *= 2  # More channels at lower resolution

        return param_memory + total_feature_memory

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for diffusion model."""
        available_memory = self.hardware_specs.memory_gb * 0.8

        # Start with batch size 1 and increase until memory limit
        batch_size = 1
        while batch_size <= 64:  # Reasonable upper limit
            estimated_memory = self.estimate_memory_usage(batch_size)
            if estimated_memory > available_memory:
                break
            batch_size += 1

        return max(1, batch_size - 1)

    def apply_sharding(self, sharding_config: Any) -> nnx.Module:
        """Apply data parallel sharding to diffusion model."""
        # Diffusion models typically use data parallelism
        return self.model


class EnergyAdapter(ModelAdapter):
    """Adapter for energy-based models."""

    def __init__(
        self,
        model: nnx.Module,
        hardware_specs: HardwareSpecs,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        parallelism_config: ParallelismConfig | None = None,
    ) -> None:
        """Initialize energy model adapter.

        Args:
            model: Energy-based model
            hardware_specs: Hardware specifications
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            parallelism_config: Optional parallelism configuration
        """
        super().__init__(model, hardware_specs, parallelism_config)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

    def get_model_specs(self) -> ModelSpecs:
        """Get energy model specifications."""
        # Calculate parameters for fully connected layers
        layer_dims = (self.input_dim, *self.hidden_dims, 1)
        total_params = sum(
            layer_dims[i] * layer_dims[i + 1] + layer_dims[i + 1]
            for i in range(len(layer_dims) - 1)
        )

        memory_gb = total_params * 4 / (1024**3)

        return ModelSpecs(
            model_type="energy",
            num_parameters=total_params,
            memory_requirement_gb=memory_gb,
            compute_intensity="medium",
            preferred_parallelism="data_parallel",
        )

    def estimate_memory_usage(self, batch_size: int) -> float:
        """Estimate memory usage for energy model."""
        specs = self.get_model_specs()

        # Model parameters
        param_memory = specs.memory_requirement_gb

        # Activations for each layer
        activation_memory = 0.0
        for hidden_dim in self.hidden_dims:
            activation_memory += batch_size * hidden_dim * 4 / (1024**3)

        return param_memory + activation_memory

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for energy model."""
        available_memory = self.hardware_specs.memory_gb * 0.8

        # Energy models are typically less memory intensive
        batch_size = 1
        while batch_size <= 1024:  # Higher batch sizes possible
            estimated_memory = self.estimate_memory_usage(batch_size)
            if estimated_memory > available_memory:
                break
            batch_size *= 2

        return max(1, batch_size // 2)

    def apply_sharding(self, sharding_config: Any) -> nnx.Module:
        """Apply data parallel sharding to energy model."""
        return self.model


# Factory functions for creating adapters
def create_transformer_adapter(
    model: nnx.Module,
    hardware_specs: HardwareSpecs,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    vocab_size: int | None = None,
) -> TransformerAdapter:
    """Create transformer adapter with automatic configuration.

    Args:
        model: Transformer model
        hardware_specs: Hardware specifications
        num_layers: Number of layers
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        vocab_size: Optional vocabulary size

    Returns:
        TransformerAdapter instance
    """
    return TransformerAdapter(
        model=model,
        hardware_specs=hardware_specs,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        vocab_size=vocab_size,
    )


def create_diffusion_adapter(
    model: nnx.Module,
    hardware_specs: HardwareSpecs,
    image_size: int,
    channels: int,
    model_channels: int,
    num_res_blocks: int,
) -> DiffusionAdapter:
    """Create diffusion adapter with automatic configuration.

    Args:
        model: Diffusion model
        hardware_specs: Hardware specifications
        image_size: Image size
        channels: Input channels
        model_channels: Base model channels
        num_res_blocks: Number of residual blocks

    Returns:
        DiffusionAdapter instance
    """
    return DiffusionAdapter(
        model=model,
        hardware_specs=hardware_specs,
        image_size=image_size,
        channels=channels,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
    )


def create_energy_adapter(
    model: nnx.Module,
    hardware_specs: HardwareSpecs,
    input_dim: int,
    hidden_dims: tuple[int, ...],
) -> EnergyAdapter:
    """Create energy model adapter with automatic configuration.

    Args:
        model: Energy-based model
        hardware_specs: Hardware specifications
        input_dim: Input dimension
        hidden_dims: Hidden layer dimensions

    Returns:
        EnergyAdapter instance
    """
    return EnergyAdapter(
        model=model,
        hardware_specs=hardware_specs,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
    )
