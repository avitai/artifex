"""Performance infrastructure for roofline analysis and optimization.

This module provides core performance analysis capabilities including:
- Hardware detection and specification
- Roofline model analysis for performance estimation
- FLOP counting and arithmetic intensity calculation
- JAX function profiling and benchmarking

All implementations follow JAX/Flax NNX best practices and avoid numpy
usage within any performance-critical code paths.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

import jax


@dataclass
class HardwareSpecs:
    """Hardware specifications for performance optimization.

    Contains essential hardware information needed for performance modeling
    and optimization decisions.
    """

    platform: Literal["cpu", "gpu", "tpu"]
    device_count: int
    memory_gb: float
    compute_capability: str | None = None
    peak_flops_per_second: float | None = None
    memory_bandwidth_gb_per_second: float | None = None


@dataclass
class RooflineMetrics:
    """Roofline performance analysis metrics.

    Contains the results of roofline model analysis for understanding
    performance bottlenecks and optimization opportunities.
    """

    arithmetic_intensity: float
    achieved_flops_per_second: float
    achieved_bandwidth_gb_per_second: float
    efficiency_ratio: float
    bottleneck: Literal["compute", "memory"]


class HardwareDetector:
    """Automatic hardware detection and optimization recommendations.

    Detects available hardware and provides optimization recommendations
    based on detected capabilities. Follows performance-first design
    principles without backward compatibility concerns.
    """

    def __init__(self) -> None:
        """Initialize hardware detector."""
        self._hardware_specs: HardwareSpecs | None = None

    def detect_hardware(self) -> HardwareSpecs:
        """Detect current hardware configuration.

        Returns:
            HardwareSpecs containing detected hardware information.
        """
        devices = jax.devices()

        if not devices:
            # Fallback for CPU-only environments
            self._hardware_specs = HardwareSpecs(
                platform="cpu",
                device_count=1,
                memory_gb=8.0,  # Conservative default
            )
            return self._hardware_specs

        primary_device = devices[0]
        platform = primary_device.platform

        # Platform-specific detection
        if platform == "gpu":
            self._hardware_specs = self._detect_gpu_specs(devices)
        elif platform == "tpu":
            self._hardware_specs = self._detect_tpu_specs(devices)
        else:
            self._hardware_specs = self._detect_cpu_specs(devices)

        return self._hardware_specs

    def _detect_gpu_specs(self, devices: list[jax.Device]) -> HardwareSpecs:
        """Detect GPU specifications."""
        primary_device = devices[0]

        # Extract GPU information from device kind
        device_kind = getattr(primary_device, "device_kind", "Unknown GPU")

        # GPU-specific performance estimates based on common configurations
        memory_gb = 40.0  # A100 default
        peak_flops = 312e12  # A100 BF16 tensor performance
        bandwidth = 1555.0  # A100 memory bandwidth
        compute_capability = "8.6"

        # Adjust for different GPU types (basic heuristics)
        if "V100" in device_kind:
            memory_gb = 32.0
            peak_flops = 125e12
            bandwidth = 900.0
            compute_capability = "7.0"
        elif "RTX" in device_kind or "GTX" in device_kind:
            memory_gb = 24.0  # Conservative for RTX series
            peak_flops = 100e12
            bandwidth = 600.0
            compute_capability = "8.6"

        return HardwareSpecs(
            platform="gpu",
            device_count=len(devices),
            memory_gb=memory_gb,
            compute_capability=compute_capability,
            peak_flops_per_second=peak_flops,
            memory_bandwidth_gb_per_second=bandwidth,
        )

    def _detect_tpu_specs(self, devices: list[jax.Device]) -> HardwareSpecs:
        """Detect TPU specifications."""
        device_count = len(devices)

        # TPU v4 specifications (most common)
        memory_gb = 32.0  # Per chip
        peak_flops = 275e12  # BF16 matrix performance per chip
        bandwidth = 1200.0  # HBM bandwidth per chip

        return HardwareSpecs(
            platform="tpu",
            device_count=device_count,
            memory_gb=memory_gb * device_count,  # Total memory
            peak_flops_per_second=peak_flops * device_count,
            memory_bandwidth_gb_per_second=bandwidth * device_count,
        )

    def _detect_cpu_specs(self, devices: list[jax.Device]) -> HardwareSpecs:
        """Detect CPU specifications."""
        # Conservative CPU estimates
        memory_gb = 32.0
        peak_flops = 1e12  # 1 TFLOP conservative estimate
        bandwidth = 100.0  # DDR4 conservative estimate

        return HardwareSpecs(
            platform="cpu",
            device_count=len(devices),
            memory_gb=memory_gb,
            peak_flops_per_second=peak_flops,
            memory_bandwidth_gb_per_second=bandwidth,
        )

    def get_optimal_batch_size(self, memory_gb: float, model_memory_gb: float) -> int:
        """Recommend optimal batch size based on available memory."""
        if not self._hardware_specs:
            self.detect_hardware()

        available_memory = memory_gb * 0.8  # Reserve 20% for safety
        memory_per_sample = model_memory_gb / 32  # Rough estimate

        optimal_batch_size = int(available_memory / memory_per_sample)
        return max(1, min(optimal_batch_size, 512))  # Cap at reasonable range

    def estimate_memory_usage(
        self,
        batch_size: int,
        sequence_length: int,
        hidden_size: int,
        num_layers: int,
    ) -> float:
        """Estimate memory usage for transformer model configuration.

        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers

        Returns:
            Estimated memory usage in GB
        """
        # Parameters memory (model weights)
        # Rough estimate: 12 * hidden_size^2 * num_layers parameters
        params_memory = 12 * hidden_size * hidden_size * num_layers * 4 / (1024**3)

        # Activations memory
        activation_memory = batch_size * sequence_length * hidden_size * num_layers * 4 / (1024**3)

        # Gradients memory (same as parameters)
        gradients_memory = params_memory

        # Optimizer states (Adam: 2x parameters)
        optimizer_memory = params_memory * 2

        return params_memory + activation_memory + gradients_memory + optimizer_memory

    def get_critical_batch_size(self) -> int:
        """Get hardware-specific critical batch size for roofline optimization.

        Critical batch size is where arithmetic intensity crosses the
        roofline boundary between memory-bound and compute-bound.

        Returns:
            Optimal batch size for current hardware
        """
        if self._hardware_specs is None:
            self.detect_hardware()

        if self._hardware_specs is None:
            raise ValueError("Hardware specs not detected")

        platform = self._hardware_specs.platform

        if platform == "tpu":
            # TPU v5e critical batch size from JAX guide
            return 240
        elif platform == "gpu":
            # Detect specific GPU model
            device_kind = self._get_gpu_device_kind()

            gpu_critical_sizes = {
                "H100": 298,  # From JAX guide
                "A100": 256,  # Empirical
                "A6000": 256,  # Similar to A100
                "V100": 128,  # Older generation
                "T4": 64,  # Small GPU
                "RTX 4090": 192,  # Consumer GPU
                "RTX 3090": 128,  # Consumer GPU
            }

            for gpu_name, size in gpu_critical_sizes.items():
                if gpu_name in device_kind:
                    return size

            # Default for unknown GPUs
            return 128
        else:
            # CPU default
            return 32

    def is_batch_size_optimal(self, batch_size: int) -> bool:
        """Check if batch size is optimal for current hardware.

        Args:
            batch_size: Batch size to check

        Returns:
            True if batch size is >= 80% of critical size
        """
        critical = self.get_critical_batch_size()
        return batch_size >= int(critical * 0.8)

    def get_batch_size_recommendation(self, model_memory_gb: float) -> dict:
        """Get comprehensive batch size recommendations.

        Args:
            model_memory_gb: Estimated model memory usage

        Returns:
            Dictionary with recommendations
        """
        critical = self.get_critical_batch_size()

        if self._hardware_specs is None:
            raise ValueError("Hardware specs not detected")

        if self._hardware_specs.memory_gb is None:
            raise ValueError("Hardware specs memory_gb is not set")

        available_memory = self._hardware_specs.memory_gb

        # Calculate maximum batch size based on memory
        safety_factor = 0.75  # Leave 25% for gradients and activations
        max_batch_from_memory = int(
            (available_memory * safety_factor - model_memory_gb)
            / (model_memory_gb / 32)  # Assume linear scaling
        )

        recommended = min(critical, max_batch_from_memory)

        return {
            "critical_batch_size": critical,
            "memory_limited_batch_size": max_batch_from_memory,
            "recommended_batch_size": recommended,
            "is_memory_limited": max_batch_from_memory < critical,
            "efficiency_at_recommended": min(1.0, recommended / critical),
        }

    def _get_gpu_device_kind(self) -> str:
        """Get GPU device kind string."""
        try:
            devices = jax.devices()
            if devices and devices[0].platform == "gpu":
                return devices[0].device_kind
            return "Unknown GPU"
        except Exception as e:
            return f"Unknown GPU: {e}"


class PerformanceEstimator:
    """Roofline model analysis and performance estimation.

    Provides comprehensive performance analysis using the roofline model
    to understand compute vs memory bottlenecks and optimization opportunities.
    """

    def __init__(self) -> None:
        """Initialize performance estimator."""
        pass

    def estimate_flops_linear(self, batch_size: int, input_size: int, output_size: int) -> int:
        """Estimate FLOPs for linear layer computation.

        Args:
            batch_size: Batch size
            input_size: Input feature dimension
            output_size: Output feature dimension

        Returns:
            Total FLOPs for forward pass
        """
        # Matrix multiplication: batch_size * input_size * output_size
        # Multiply-accumulate counts as 2 operations
        return batch_size * input_size * output_size * 2

    def estimate_flops_attention(
        self,
        batch_size: int,
        sequence_length: int,
        hidden_size: int,
        num_heads: int,
    ) -> int:
        """Estimate FLOPs for attention mechanism.

        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            hidden_size: Hidden dimension
            num_heads: Number of attention heads

        Returns:
            Total FLOPs for attention computation
        """
        head_dim = hidden_size // num_heads

        # Q, K, V projections
        qkv_flops = 3 * self.estimate_flops_linear(
            batch_size * sequence_length, hidden_size, hidden_size
        )

        # Attention scores: Q @ K^T
        scores_flops = batch_size * num_heads * sequence_length * sequence_length * head_dim * 2

        # Attention output: scores @ V
        output_flops = batch_size * num_heads * sequence_length * sequence_length * head_dim * 2

        # Output projection
        proj_flops = self.estimate_flops_linear(
            batch_size * sequence_length, hidden_size, hidden_size
        )

        return qkv_flops + scores_flops + output_flops + proj_flops

    def calculate_arithmetic_intensity(self, total_flops: int, memory_bytes: int) -> float:
        """Calculate arithmetic intensity (FLOPs per byte).

        Args:
            total_flops: Total floating point operations
            memory_bytes: Total memory bytes accessed

        Returns:
            Arithmetic intensity (FLOPs/byte)
        """
        if memory_bytes == 0:
            return float("inf")
        return total_flops / memory_bytes

    def analyze_roofline(
        self,
        operation_flops: int,
        memory_bytes: int,
        hardware_specs: HardwareSpecs,
        execution_time_seconds: float,
    ) -> RooflineMetrics:
        """Perform roofline model analysis.

        Args:
            operation_flops: FLOPs in the operation
            memory_bytes: Memory bytes accessed
            hardware_specs: Hardware specifications
            execution_time_seconds: Measured execution time

        Returns:
            RooflineMetrics with performance analysis
        """
        arithmetic_intensity = self.calculate_arithmetic_intensity(operation_flops, memory_bytes)

        achieved_flops_per_second = operation_flops / execution_time_seconds
        achieved_bandwidth = memory_bytes / (execution_time_seconds * 1024**3)

        # Determine bottleneck using roofline model
        if hardware_specs.peak_flops_per_second is None:
            peak_flops = 1e12  # Default fallback
        else:
            peak_flops = hardware_specs.peak_flops_per_second

        if hardware_specs.memory_bandwidth_gb_per_second is None:
            peak_bandwidth = 100.0  # Default fallback
        else:
            peak_bandwidth = hardware_specs.memory_bandwidth_gb_per_second

        # Convert bandwidth from GB/s to bytes/s for consistent units
        peak_bandwidth_bytes_per_second = peak_bandwidth * 1024**3

        # Roofline intersection point (FLOPs per byte)
        ridge_point = peak_flops / peak_bandwidth_bytes_per_second

        if arithmetic_intensity < ridge_point:
            bottleneck: Literal["compute", "memory"] = "memory"
            theoretical_peak = peak_bandwidth_bytes_per_second * arithmetic_intensity
        else:
            bottleneck = "compute"
            theoretical_peak = peak_flops

        efficiency_ratio = achieved_flops_per_second / theoretical_peak

        return RooflineMetrics(
            arithmetic_intensity=arithmetic_intensity,
            achieved_flops_per_second=achieved_flops_per_second,
            achieved_bandwidth_gb_per_second=achieved_bandwidth,
            efficiency_ratio=efficiency_ratio,
            bottleneck=bottleneck,
        )

    def profile_jax_function(
        self,
        func: Callable,
        args: tuple[Any, ...],
        hardware_specs: HardwareSpecs,
    ) -> RooflineMetrics:
        """Profile a JAX function and analyze performance.

        Args:
            func: JAX function to profile
            args: Arguments to pass to function
            hardware_specs: Hardware specifications for analysis

        Returns:
            RooflineMetrics with performance analysis
        """
        # Warm up JIT compilation
        _ = func(*args)
        jax.block_until_ready(_)

        # Benchmark execution
        start_time = time.time()
        result = func(*args)
        jax.block_until_ready(result)
        execution_time = time.time() - start_time

        # Estimate FLOPs and memory access (basic heuristics)
        total_elements = sum(arg.size if hasattr(arg, "size") else 0 for arg in args)
        estimated_flops = total_elements * 2  # Conservative estimate
        memory_bytes = total_elements * 4  # Assuming float32

        return self.analyze_roofline(
            operation_flops=estimated_flops,
            memory_bytes=memory_bytes,
            hardware_specs=hardware_specs,
            execution_time_seconds=execution_time,
        )

    def estimate_transformer_layer_performance(
        self,
        batch_size: int,
        sequence_length: int,
        hidden_size: int,
        num_heads: int,
        feedforward_size: int,
        hardware_specs: HardwareSpecs,
    ) -> RooflineMetrics:
        """Estimate performance for a transformer layer.

        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            feedforward_size: Feedforward layer size
            hardware_specs: Hardware specifications

        Returns:
            RooflineMetrics with performance estimates
        """
        # Attention FLOPs
        attention_flops = self.estimate_flops_attention(
            batch_size, sequence_length, hidden_size, num_heads
        )

        # Feedforward FLOPs (2 linear layers)
        ff_flops = self.estimate_flops_linear(
            batch_size * sequence_length, hidden_size, feedforward_size
        ) + self.estimate_flops_linear(batch_size * sequence_length, feedforward_size, hidden_size)

        total_flops = attention_flops + ff_flops

        # Memory access estimate
        # Parameters + activations
        param_memory = (
            3 * hidden_size * hidden_size  # QKV projections
            + hidden_size * hidden_size  # Output projection
            + hidden_size * feedforward_size  # FF layer 1
            + feedforward_size * hidden_size  # FF layer 2
        ) * 4  # bytes per parameter

        activation_memory = (
            batch_size * sequence_length * hidden_size * 8  # Conservative
        ) * 4

        total_memory = param_memory + activation_memory

        # Estimate execution time based on hardware
        if hardware_specs.peak_flops_per_second:
            estimated_time = total_flops / hardware_specs.peak_flops_per_second
        else:
            estimated_time = total_flops / 1e12  # Default 1 TFLOP/s

        return self.analyze_roofline(
            operation_flops=total_flops,
            memory_bytes=total_memory,
            hardware_specs=hardware_specs,
            execution_time_seconds=estimated_time,
        )

    def benchmark_operation(
        self,
        operation: Callable[[], Any],
        num_iterations: int = 10,
        warmup_iterations: int = 3,
    ) -> float:
        """Benchmark a JAX operation.

        Args:
            operation: Function to benchmark (should include jit compilation)
            num_iterations: Number of timing iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Average execution time in seconds
        """
        # Warmup
        for _ in range(warmup_iterations):
            result = operation()
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            else:
                jax.block_until_ready(result)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            result = operation()
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            else:
                jax.block_until_ready(result)
            times.append(time.time() - start_time)

        return sum(times) / len(times)
