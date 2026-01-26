"""Tests for production optimization infrastructure.

This module tests production optimization capabilities including:
- Automatic optimization pipeline selection
- Production-ready inference optimization
- Model monitoring and debugging tools
- Hardware-aware optimization strategies

All tests follow TDD principles and use Flax NNX exclusively.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from artifex.generative_models.core.performance import HardwareSpecs
from artifex.generative_models.inference.optimization.production import (
    create_production_optimizer,
    create_production_pipeline,
    OptimizationResult,
    OptimizationTarget,
    ProductionMonitor,
    ProductionOptimizer,
    ProductionPipeline,
)


class SimpleModel(nnx.Module):
    """Simple test model for production optimization tests."""

    def __init__(self, features: int, rngs: nnx.Rngs):
        """Initialize simple model."""
        self.linear = nnx.Linear(features, features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass."""
        return self.linear(x)


@pytest.fixture
def hardware_specs():
    """Create test hardware specifications."""
    return HardwareSpecs(
        platform="gpu",
        device_count=1,
        memory_gb=16.0,
        compute_capability="8.6",
        peak_flops_per_second=100e12,
        memory_bandwidth_gb_per_second=500.0,
    )


@pytest.fixture
def simple_model():
    """Create simple test model."""
    rngs = nnx.Rngs(0)
    return SimpleModel(64, rngs)


@pytest.fixture
def sample_inputs():
    """Create sample inputs for testing."""
    return (jnp.ones((8, 64)),)


@pytest.fixture
def optimization_target():
    """Create optimization target for testing."""
    return OptimizationTarget(
        latency_ms=50.0,
        throughput_qps=100.0,
        memory_budget_gb=8.0,
        accuracy_threshold=0.95,
    )


class TestOptimizationTarget:
    """Test optimization target specification."""

    def test_optimization_target_creation(self):
        """Test creating optimization target with various configurations."""
        # Test with all parameters
        target = OptimizationTarget(
            latency_ms=100.0,
            throughput_qps=50.0,
            memory_budget_gb=16.0,
            cost_budget_per_hour=10.0,
            accuracy_threshold=0.9,
        )

        assert target.latency_ms == 100.0
        assert target.throughput_qps == 50.0
        assert target.memory_budget_gb == 16.0
        assert target.cost_budget_per_hour == 10.0
        assert target.accuracy_threshold == 0.9

    def test_optimization_target_defaults(self):
        """Test optimization target with default values."""
        target = OptimizationTarget()

        assert target.latency_ms is None
        assert target.throughput_qps is None
        assert target.memory_budget_gb is None
        assert target.cost_budget_per_hour is None
        assert target.accuracy_threshold is None


class TestOptimizationResult:
    """Test optimization result structure."""

    def test_optimization_result_creation(self, simple_model):
        """Test creating optimization result."""
        result = OptimizationResult(
            optimized_model=simple_model,
            optimization_techniques=["jit_compilation", "quantization"],
            performance_metrics={"latency_ms": 25.0, "throughput_qps": 200.0},
            memory_usage_gb=4.0,
            latency_ms=25.0,
            throughput_qps=200.0,
            optimization_time_seconds=10.5,
        )

        assert result.optimized_model == simple_model
        assert len(result.optimization_techniques) == 2
        assert "jit_compilation" in result.optimization_techniques
        assert result.latency_ms == 25.0
        assert result.optimization_time_seconds == 10.5

    def test_optimization_result_defaults(self, simple_model):
        """Test optimization result with default values."""
        result = OptimizationResult(optimized_model=simple_model)

        assert result.optimization_techniques == []
        assert result.performance_metrics == {}
        assert result.memory_usage_gb == 0.0
        assert result.latency_ms == 0.0
        assert result.throughput_qps == 0.0
        assert result.optimization_time_seconds == 0.0


class TestProductionOptimizer:
    """Test production optimizer functionality."""

    def test_optimizer_initialization(self, hardware_specs):
        """Test optimizer initialization with hardware specs."""
        optimizer = ProductionOptimizer(hardware_specs=hardware_specs)

        assert optimizer.hardware_specs == hardware_specs
        assert optimizer.parallelism_config is None
        assert optimizer.performance_estimator is not None
        assert isinstance(optimizer._optimization_cache, dict)

    def test_optimizer_auto_hardware_detection(self):
        """Test optimizer with automatic hardware detection."""
        optimizer = ProductionOptimizer()

        assert optimizer.hardware_specs is not None
        assert optimizer.hardware_specs.platform in ["cpu", "gpu", "tpu"]
        assert optimizer.hardware_specs.device_count > 0

    def test_optimize_for_production(
        self, simple_model, optimization_target, sample_inputs, hardware_specs
    ):
        """Test production optimization process."""
        optimizer = ProductionOptimizer(hardware_specs=hardware_specs)

        result = optimizer.optimize_for_production(
            model=simple_model,
            optimization_target=optimization_target,
            sample_inputs=sample_inputs,
        )

        assert isinstance(result, OptimizationResult)
        assert result.optimized_model is not None
        assert len(result.optimization_techniques) > 0
        assert "jit_compilation" in result.optimization_techniques
        assert result.optimization_time_seconds > 0

    def test_optimization_technique_selection(self, hardware_specs):
        """Test optimization technique selection logic."""
        optimizer = ProductionOptimizer(hardware_specs=hardware_specs)

        # Test compilation selection (should always be true)
        target = OptimizationTarget()
        assert optimizer._should_apply_compilation(target) is True

        # Test quantization selection
        target_with_memory = OptimizationTarget(memory_budget_gb=8.0)
        assert optimizer._should_apply_quantization(target_with_memory) is True

        target_without_memory = OptimizationTarget()
        assert optimizer._should_apply_quantization(target_without_memory) is False

        # Test pruning selection
        target_with_accuracy = OptimizationTarget(accuracy_threshold=0.95)
        assert optimizer._should_apply_pruning(target_with_accuracy) is True

        target_high_accuracy = OptimizationTarget(accuracy_threshold=0.99)
        assert optimizer._should_apply_pruning(target_high_accuracy) is False

    def test_create_production_pipeline(
        self, simple_model, optimization_target, sample_inputs, hardware_specs
    ):
        """Test production pipeline creation."""
        optimizer = ProductionOptimizer(hardware_specs=hardware_specs)

        result = optimizer.optimize_for_production(
            model=simple_model,
            optimization_target=optimization_target,
            sample_inputs=sample_inputs,
        )

        pipeline = optimizer.create_production_pipeline(simple_model, result)

        assert isinstance(pipeline, ProductionPipeline)
        assert pipeline.model is not None
        assert pipeline.hardware_specs == hardware_specs


class TestProductionPipeline:
    """Test production pipeline functionality."""

    def test_pipeline_initialization(self, simple_model, hardware_specs):
        """Test pipeline initialization."""
        pipeline = ProductionPipeline(
            model=simple_model,
            hardware_specs=hardware_specs,
            optimization_techniques=["jit_compilation"],
        )

        assert pipeline.model == simple_model
        assert pipeline.hardware_specs == hardware_specs
        assert pipeline.optimization_techniques == ["jit_compilation"]
        assert isinstance(pipeline.monitoring, ProductionMonitor)

    def test_single_prediction(self, simple_model, hardware_specs):
        """Test single prediction with monitoring."""
        pipeline = ProductionPipeline(
            model=simple_model,
            hardware_specs=hardware_specs,
        )

        inputs = jnp.ones((1, 64))
        outputs = pipeline.predict(inputs)

        assert outputs.shape == (1, 64)

        # Check monitoring recorded the request
        metrics = pipeline.get_monitoring_metrics()
        assert metrics.request_count == 1
        assert metrics.average_latency_ms > 0

    def test_batch_prediction(self, simple_model, hardware_specs):
        """Test batch prediction with monitoring."""
        pipeline = ProductionPipeline(
            model=simple_model,
            hardware_specs=hardware_specs,
        )

        inputs = jnp.ones((8, 64))
        outputs = pipeline.predict_batch(inputs)

        assert outputs.shape == (8, 64)

        # Check monitoring recorded all requests
        metrics = pipeline.get_monitoring_metrics()
        assert metrics.request_count == 8

    def test_monitoring_reset(self, simple_model, hardware_specs):
        """Test monitoring reset functionality."""
        pipeline = ProductionPipeline(
            model=simple_model,
            hardware_specs=hardware_specs,
        )

        # Make some predictions
        inputs = jnp.ones((1, 64))
        pipeline.predict(inputs)

        # Verify metrics exist
        metrics = pipeline.get_monitoring_metrics()
        assert metrics.request_count > 0

        # Reset and verify
        pipeline.reset_monitoring()
        metrics = pipeline.get_monitoring_metrics()
        assert metrics.request_count == 0


class TestProductionMonitor:
    """Test production monitoring functionality."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = ProductionMonitor(window_size=100)

        assert len(monitor.latencies) == 0
        assert len(monitor.successes) == 0
        assert len(monitor.timestamps) == 0
        assert monitor.window_size == 100

    def test_record_request(self):
        """Test recording requests."""
        monitor = ProductionMonitor()

        monitor.record_request(25.0, success=True)
        monitor.record_request(30.0, success=False)

        assert len(monitor.latencies) == 2
        assert len(monitor.successes) == 2
        assert monitor.latencies[0] == 25.0
        assert monitor.successes[0] is True
        assert monitor.successes[1] is False

    def test_sliding_window(self):
        """Test sliding window behavior."""
        monitor = ProductionMonitor(window_size=3)

        # Add more requests than window size
        for i in range(5):
            monitor.record_request(float(i), success=True)

        # Should only keep the last 3 requests
        assert len(monitor.latencies) == 3
        assert monitor.latencies == [2.0, 3.0, 4.0]

    def test_get_metrics(self):
        """Test metrics calculation."""
        monitor = ProductionMonitor()

        # Add some test data
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        for i, latency in enumerate(latencies):
            monitor.record_request(latency, success=i < 4)  # One failure

        metrics = monitor.get_metrics()

        assert metrics.request_count == 5
        assert metrics.average_latency_ms == 30.0  # Mean of latencies
        assert metrics.error_rate == 0.2  # 1 failure out of 5
        assert metrics.p95_latency_ms >= 40.0
        assert metrics.throughput_qps > 0

    def test_empty_metrics(self):
        """Test metrics when no requests recorded."""
        monitor = ProductionMonitor()
        metrics = monitor.get_metrics()

        assert metrics.request_count == 0
        assert metrics.average_latency_ms == 0
        assert metrics.error_rate == 0
        assert metrics.throughput_qps == 0

    def test_reset_monitoring(self):
        """Test resetting monitoring data."""
        monitor = ProductionMonitor()

        # Add some data
        monitor.record_request(25.0)
        monitor.record_request(30.0)

        assert len(monitor.latencies) == 2

        # Reset and verify
        monitor.reset()
        assert len(monitor.latencies) == 0
        assert len(monitor.successes) == 0
        assert len(monitor.timestamps) == 0


class TestFactoryFunctions:
    """Test factory functions for easy creation."""

    def test_create_production_optimizer(self, hardware_specs):
        """Test creating optimizer with factory function."""
        optimizer = create_production_optimizer(hardware_specs=hardware_specs)

        assert isinstance(optimizer, ProductionOptimizer)
        assert optimizer.hardware_specs == hardware_specs

    def test_create_production_optimizer_auto_hardware(self):
        """Test creating optimizer with automatic hardware detection."""
        optimizer = create_production_optimizer()

        assert isinstance(optimizer, ProductionOptimizer)
        assert optimizer.hardware_specs is not None

    def test_create_production_pipeline(self, simple_model, optimization_target, sample_inputs):
        """Test creating complete production pipeline."""
        pipeline = create_production_pipeline(
            model=simple_model,
            optimization_target=optimization_target,
            sample_inputs=sample_inputs,
        )

        assert isinstance(pipeline, ProductionPipeline)
        assert pipeline.model is not None
        assert pipeline.hardware_specs is not None


class TestIntegration:
    """Integration tests for production optimization."""

    def test_end_to_end_optimization(self, simple_model, sample_inputs, hardware_specs):
        """Test complete end-to-end optimization workflow."""
        # Create optimization target
        target = OptimizationTarget(
            latency_ms=100.0,
            memory_budget_gb=8.0,
            accuracy_threshold=0.9,
        )

        # Create optimizer
        optimizer = ProductionOptimizer(hardware_specs=hardware_specs)

        # Optimize model
        result = optimizer.optimize_for_production(
            model=simple_model,
            optimization_target=target,
            sample_inputs=sample_inputs,
        )

        # Create pipeline
        pipeline = optimizer.create_production_pipeline(simple_model, result)

        # Test inference
        test_input = jnp.ones((1, 64))
        output = pipeline.predict(test_input)

        assert output.shape == (1, 64)

        # Verify monitoring
        metrics = pipeline.get_monitoring_metrics()
        assert metrics.request_count == 1

    def test_hardware_aware_optimization(self):
        """Test hardware-aware optimization selection."""
        # Test with high-memory GPU
        gpu_specs = HardwareSpecs(
            platform="gpu",
            device_count=1,
            memory_gb=80.0,  # High memory
            peak_flops_per_second=300e12,
        )

        optimizer = ProductionOptimizer(hardware_specs=gpu_specs)

        # Should handle larger models with GPU specs
        assert optimizer.hardware_specs.memory_gb == 80.0
        assert optimizer.hardware_specs.platform == "gpu"

        # Test with CPU specs
        cpu_specs = HardwareSpecs(
            platform="cpu",
            device_count=8,
            memory_gb=128.0,
            peak_flops_per_second=10e12,
        )

        optimizer = ProductionOptimizer(hardware_specs=cpu_specs)
        assert optimizer.hardware_specs.platform == "cpu"

    def test_multi_target_optimization(self, simple_model, sample_inputs):
        """Test optimization with multiple conflicting targets."""
        # Create conflicting targets
        # Very aggressive
        latency_target = OptimizationTarget(latency_ms=10.0)
        # High throughput
        throughput_target = OptimizationTarget(throughput_qps=1000.0)
        # Limited memory
        memory_target = OptimizationTarget(memory_budget_gb=2.0)

        optimizer = create_production_optimizer()

        # Test each target
        for target in [latency_target, throughput_target, memory_target]:
            result = optimizer.optimize_for_production(
                model=simple_model,
                optimization_target=target,
                sample_inputs=sample_inputs,
            )

            assert isinstance(result, OptimizationResult)
            assert len(result.optimization_techniques) > 0
