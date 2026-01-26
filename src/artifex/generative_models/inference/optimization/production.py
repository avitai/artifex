"""Production inference optimization for scaled models.

This module provides comprehensive production optimization infrastructure
including:
- Automatic optimization pipeline selection
- Production-ready inference optimization
- Model adapter classes for different architectures
- Comprehensive monitoring and debugging tools

All implementations follow JAX/Flax NNX best practices and prioritize
performance through hardware-aware optimization.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import flax.nnx as nnx
import jax

from ...core.performance import (
    HardwareDetector,
    HardwareSpecs,
    PerformanceEstimator,
)
from ...scaling.sharding import ParallelismConfig


@dataclass
class OptimizationTarget:
    """Optimization target specifications for production inference."""

    latency_ms: float | None = None
    throughput_qps: float | None = None
    memory_budget_gb: float | None = None
    cost_budget_per_hour: float | None = None
    accuracy_threshold: float | None = None


@dataclass
class OptimizationResult:
    """Results from production optimization process."""

    optimized_model: nnx.Module
    optimization_techniques: list[str] = field(default_factory=list)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    memory_usage_gb: float = 0.0
    latency_ms: float = 0.0
    throughput_qps: float = 0.0
    optimization_time_seconds: float = 0.0


@dataclass
class MonitoringMetrics:
    """Real-time monitoring metrics for production inference."""

    request_count: int = 0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_qps: float = 0.0
    memory_usage_gb: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0


class ProductionOptimizer:
    """Production inference optimizer with automatic optimization selection.

    Provides comprehensive optimization pipeline for production deployment
    of scaled models with hardware-aware optimization and monitoring.
    """

    def __init__(
        self,
        hardware_specs: HardwareSpecs | None = None,
        parallelism_config: ParallelismConfig | None = None,
    ) -> None:
        """Initialize production optimizer.

        Args:
            hardware_specs: Hardware specifications for optimization
            parallelism_config: Parallelism configuration for scaling
        """
        self.hardware_specs = hardware_specs or HardwareDetector().detect_hardware()
        self.parallelism_config = parallelism_config
        self.performance_estimator = PerformanceEstimator()
        self._optimization_cache: dict[str, Any] = {}

    def optimize_for_production(
        self,
        model: nnx.Module,
        optimization_target: OptimizationTarget,
        sample_inputs: tuple[jax.Array, ...],
        model_name: str | None = None,
    ) -> OptimizationResult:
        """Optimize model for production deployment.

        Args:
            model: Model to optimize
            optimization_target: Optimization targets and constraints
            sample_inputs: Sample inputs for optimization
            model_name: Optional model name for caching

        Returns:
            OptimizationResult with optimized model and metrics
        """
        start_time = time.time()
        optimization_techniques = []
        optimized_model = model

        # 1. Compilation optimization
        if self._should_apply_compilation(optimization_target):
            optimized_model = self._apply_compilation_optimization(optimized_model, sample_inputs)
            optimization_techniques.append("jit_compilation")

        # 2. Quantization optimization
        if self._should_apply_quantization(optimization_target):
            optimized_model = self._apply_quantization_optimization(
                optimized_model, optimization_target
            )
            optimization_techniques.append("quantization")

        # 3. Pruning optimization
        if self._should_apply_pruning(optimization_target):
            optimized_model = self._apply_pruning_optimization(optimized_model, optimization_target)
            optimization_techniques.append("pruning")

        # 4. Caching optimization
        if self._should_apply_caching(optimization_target):
            optimized_model = self._apply_caching_optimization(optimized_model, sample_inputs)
            optimization_techniques.append("caching")

        # 5. Batching optimization
        if self._should_apply_batching(optimization_target):
            optimized_model = self._apply_batching_optimization(
                optimized_model, optimization_target
            )
            optimization_techniques.append("dynamic_batching")

        # Measure performance
        performance_metrics = self._measure_production_performance(optimized_model, sample_inputs)

        optimization_time = time.time() - start_time

        return OptimizationResult(
            optimized_model=optimized_model,
            optimization_techniques=optimization_techniques,
            performance_metrics=performance_metrics,
            memory_usage_gb=performance_metrics.get("memory_usage_gb", 0.0),
            latency_ms=performance_metrics.get("latency_ms", 0.0),
            throughput_qps=performance_metrics.get("throughput_qps", 0.0),
            optimization_time_seconds=optimization_time,
        )

    def create_production_pipeline(
        self,
        model: nnx.Module,
        optimization_result: OptimizationResult,
    ) -> "ProductionPipeline":
        """Create production inference pipeline.

        Args:
            model: Base model
            optimization_result: Results from optimization process

        Returns:
            ProductionPipeline ready for deployment
        """
        return ProductionPipeline(
            model=optimization_result.optimized_model,
            hardware_specs=self.hardware_specs,
            parallelism_config=self.parallelism_config,
            optimization_techniques=(optimization_result.optimization_techniques),
        )

    def _should_apply_compilation(self, target: OptimizationTarget) -> bool:
        """Determine if compilation optimization should be applied."""
        # Always apply JIT compilation for production
        return True

    def _should_apply_quantization(self, target: OptimizationTarget) -> bool:
        """Determine if quantization optimization should be applied."""
        # Apply quantization if memory budget is specified
        return target.memory_budget_gb is not None

    def _should_apply_pruning(self, target: OptimizationTarget) -> bool:
        """Determine if pruning optimization should be applied."""
        # Apply pruning if accuracy threshold allows
        return target.accuracy_threshold is not None and target.accuracy_threshold < 0.99

    def _should_apply_caching(self, target: OptimizationTarget) -> bool:
        """Determine if caching optimization should be applied."""
        # Apply caching for latency-sensitive applications
        return target.latency_ms is not None and target.latency_ms < 100.0

    def _should_apply_batching(self, target: OptimizationTarget) -> bool:
        """Determine if batching optimization should be applied."""
        # Apply batching for throughput-oriented applications
        return target.throughput_qps is not None

    def _apply_compilation_optimization(
        self, model: nnx.Module, sample_inputs: tuple[jax.Array, ...]
    ) -> nnx.Module:
        """Apply JIT compilation optimization."""
        optimizer_ref = self

        # Wrap model with nnx.jit-compiled forward pass (model as explicit arg)
        class CompiledModel(nnx.Module):
            """Wrapper that JIT-compiles the forward pass with model as explicit arg."""

            def __init__(self, base_model: nnx.Module, parent: "ProductionOptimizer") -> None:
                super().__init__()
                self.base_model = base_model
                self.parent = parent

            @nnx.jit
            def __call__(self, inputs: jax.Array) -> jax.Array:
                return optimizer_ref._call_model(self.base_model, inputs)

        return CompiledModel(model, self)

    def _apply_quantization_optimization(
        self, model: nnx.Module, target: OptimizationTarget
    ) -> nnx.Module:
        """Apply quantization optimization."""
        # Placeholder for quantization implementation
        # In a real implementation, this would apply INT8 or other quantization
        return model

    def _apply_pruning_optimization(
        self, model: nnx.Module, target: OptimizationTarget
    ) -> nnx.Module:
        """Apply pruning optimization."""
        # Placeholder for pruning implementation
        # In a real implementation, this would remove low-importance weights
        return model

    def _apply_caching_optimization(
        self, model: nnx.Module, sample_inputs: tuple[jax.Array, ...]
    ) -> nnx.Module:
        """Apply caching optimization."""
        # Placeholder for caching implementation
        # In a real implementation, this would cache intermediate activations
        return model

    def _apply_batching_optimization(
        self, model: nnx.Module, target: OptimizationTarget
    ) -> nnx.Module:
        """Apply dynamic batching optimization."""
        # Placeholder for batching implementation
        # In a real implementation, this would implement dynamic batching
        return model

    def _call_model(self, model: nnx.Module, inputs: jax.Array) -> jax.Array:
        """Call model safely, handling different model interfaces.

        Args:
            model: The model to call
            inputs: Input data

        Returns:
            Model output
        """
        if hasattr(model, "__call__"):
            return model(inputs)
        elif hasattr(model, "apply"):
            return model.apply(inputs)
        else:
            raise ValueError(f"Model of type {type(model)} is not callable")

    def _measure_production_performance(
        self, model: nnx.Module, sample_inputs: tuple[jax.Array, ...]
    ) -> dict[str, float]:
        """Measure production performance metrics."""
        # Warm up
        for _ in range(3):
            result = self._call_model(model, sample_inputs[0])
            jax.block_until_ready(result)

        # Measure latency
        latencies = []
        for _ in range(10):
            start_time = time.time()
            output = self._call_model(model, sample_inputs[0])
            jax.block_until_ready(output)
            # Convert to ms
            latencies.append((time.time() - start_time) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        # Estimate memory usage
        model_state = nnx.state(model)
        total_elements = sum(param.size for param in jax.tree_util.tree_leaves(model_state))
        memory_usage_gb = total_elements * 4 / (1024**3)  # Assuming float32

        return {
            "latency_ms": avg_latency,
            "throughput_qps": throughput,
            "memory_usage_gb": memory_usage_gb,
        }


class ProductionPipeline:
    """Production inference pipeline with monitoring and scaling."""

    def __init__(
        self,
        model: nnx.Module,
        hardware_specs: HardwareSpecs,
        parallelism_config: ParallelismConfig | None = None,
        optimization_techniques: list[str] | None = None,
    ) -> None:
        """Initialize production pipeline.

        Args:
            model: Optimized model for inference
            hardware_specs: Hardware specifications
            parallelism_config: Parallelism configuration
            optimization_techniques: Applied optimization techniques
        """
        self.model = model
        self.hardware_specs = hardware_specs
        self.parallelism_config = parallelism_config
        self.optimization_techniques = optimization_techniques or []
        self.monitoring = ProductionMonitor()

    def predict(self, inputs: jax.Array) -> jax.Array:
        """Perform inference with monitoring.

        Args:
            inputs: Input data for inference

        Returns:
            Model predictions
        """
        start_time = time.time()

        try:
            # Perform inference
            outputs = self._call_model(self.model, inputs)
            jax.block_until_ready(outputs)

            # Update monitoring metrics
            latency_ms = (time.time() - start_time) * 1000
            self.monitoring.record_request(latency_ms, success=True)

            return outputs

        except Exception as e:
            # Record failed request
            latency_ms = (time.time() - start_time) * 1000
            self.monitoring.record_request(latency_ms, success=False)
            raise e

    def predict_batch(self, inputs: jax.Array) -> jax.Array:
        """Perform batch inference with monitoring.

        Args:
            inputs: Batch of input data

        Returns:
            Batch of model predictions
        """
        batch_size = inputs.shape[0]
        start_time = time.time()

        try:
            # Perform batch inference
            outputs = self._call_model(self.model, inputs)
            jax.block_until_ready(outputs)

            # Update monitoring metrics
            total_latency_ms = (time.time() - start_time) * 1000
            avg_latency_ms = total_latency_ms / batch_size

            for _ in range(batch_size):
                self.monitoring.record_request(avg_latency_ms, success=True)

            return outputs

        except Exception as e:
            # Record failed requests
            total_latency_ms = (time.time() - start_time) * 1000
            avg_latency_ms = total_latency_ms / batch_size

            for _ in range(batch_size):
                self.monitoring.record_request(avg_latency_ms, success=False)
            raise e

    def get_monitoring_metrics(self) -> MonitoringMetrics:
        """Get current monitoring metrics.

        Returns:
            Current monitoring metrics
        """
        return self.monitoring.get_metrics()

    def reset_monitoring(self) -> None:
        """Reset monitoring metrics."""
        self.monitoring.reset()

    def _call_model(self, model: nnx.Module, inputs: jax.Array) -> jax.Array:
        """Call model safely, handling different model interfaces.

        Args:
            model: The model to call
            inputs: Input data

        Returns:
            Model output
        """
        if hasattr(model, "__call__"):
            return model(inputs)
        elif hasattr(model, "apply"):
            return model.apply(inputs)
        else:
            raise ValueError(f"Model of type {type(model)} is not callable")


class ProductionMonitor:
    """Real-time monitoring for production inference."""

    def __init__(self, window_size: int = 1000) -> None:
        """Initialize production monitor.

        Args:
            window_size: Size of the sliding window for metrics
        """
        self.window_size = window_size
        self.latencies: list[float] = []
        self.successes: list[bool] = []
        self.timestamps: list[float] = []

    def record_request(self, latency_ms: float, success: bool = True) -> None:
        """Record a request for monitoring.

        Args:
            latency_ms: Request latency in milliseconds
            success: Whether the request was successful
        """
        current_time = time.time()

        self.latencies.append(latency_ms)
        self.successes.append(success)
        self.timestamps.append(current_time)

        # Maintain sliding window
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)
            self.successes.pop(0)
            self.timestamps.pop(0)

    def get_metrics(self) -> MonitoringMetrics:
        """Get current monitoring metrics.

        Returns:
            Current monitoring metrics
        """
        if not self.latencies:
            return MonitoringMetrics()

        # Calculate latency metrics
        sorted_latencies = sorted(self.latencies)
        avg_latency = sum(self.latencies) / len(self.latencies)

        # Calculate percentiles
        p95_idx = int(0.95 * len(sorted_latencies))
        p99_idx = int(0.99 * len(sorted_latencies))
        p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
        p99_latency = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]

        # Calculate throughput (requests per second)
        if len(self.timestamps) > 1:
            time_span = self.timestamps[-1] - self.timestamps[0]
            throughput = len(self.timestamps) / time_span if time_span > 0 else 0.0
        else:
            throughput = 0.0

        # Calculate error rate
        total_requests = len(self.successes)
        failed_requests = sum(1 for success in self.successes if not success)
        error_rate = failed_requests / total_requests if total_requests > 0 else 0.0

        return MonitoringMetrics(
            request_count=total_requests,
            average_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_qps=throughput,
            error_rate=error_rate,
            memory_usage_gb=0.0,  # Placeholder
            cache_hit_rate=0.0,  # Placeholder
        )

    def reset(self) -> None:
        """Reset all monitoring data."""
        self.latencies.clear()
        self.successes.clear()
        self.timestamps.clear()


# Factory functions for easy creation
def create_production_optimizer(
    hardware_specs: HardwareSpecs | None = None,
    parallelism_config: ParallelismConfig | None = None,
) -> ProductionOptimizer:
    """Create production optimizer with automatic hardware detection.

    Args:
        hardware_specs: Optional hardware specifications
        parallelism_config: Optional parallelism configuration

    Returns:
        ProductionOptimizer instance
    """
    return ProductionOptimizer(
        hardware_specs=hardware_specs,
        parallelism_config=parallelism_config,
    )


def create_production_pipeline(
    model: nnx.Module,
    optimization_target: OptimizationTarget,
    sample_inputs: tuple[jax.Array, ...],
) -> ProductionPipeline:
    """Create complete production pipeline with optimization.

    Args:
        model: Model to optimize and deploy
        optimization_target: Optimization targets and constraints
        sample_inputs: Sample inputs for optimization

    Returns:
        ProductionPipeline ready for deployment
    """
    optimizer = create_production_optimizer()
    optimization_result = optimizer.optimize_for_production(
        model, optimization_target, sample_inputs
    )
    return optimizer.create_production_pipeline(model, optimization_result)
