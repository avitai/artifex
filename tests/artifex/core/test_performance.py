"""Test suite for performance infrastructure components.

This module tests the core performance analysis and hardware detection
functionality that enables roofline analysis and hardware-aware optimization.
"""

from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest


# Import the components we'll implement
try:
    from artifex.generative_models.core.performance import (
        HardwareDetector,
        HardwareSpecs,
        PerformanceEstimator,
        RooflineMetrics,
    )
except ImportError:
    # These will be implemented after tests are written
    pytest.skip("Performance infrastructure not yet implemented", allow_module_level=True)


class TestHardwareSpecs:
    """Test the HardwareSpecs dataclass for hardware configuration."""

    def test_hardware_specs_creation(self):
        """Test creating hardware specifications."""
        specs = HardwareSpecs(
            platform="gpu",
            device_count=4,
            memory_gb=40.0,
            compute_capability="8.6",
            peak_flops_per_second=312e12,
            memory_bandwidth_gb_per_second=1555.0,
        )

        assert specs.platform == "gpu"
        assert specs.device_count == 4
        assert specs.memory_gb == 40.0
        assert specs.compute_capability == "8.6"
        assert specs.peak_flops_per_second == 312e12
        assert specs.memory_bandwidth_gb_per_second == 1555.0

    def test_hardware_specs_with_optional_fields(self):
        """Test hardware specs with minimal required fields."""
        specs = HardwareSpecs(platform="cpu", device_count=1, memory_gb=32.0)

        assert specs.platform == "cpu"
        assert specs.device_count == 1
        assert specs.memory_gb == 32.0
        assert specs.compute_capability is None


class TestRooflineMetrics:
    """Test the RooflineMetrics dataclass for performance analysis."""

    def test_roofline_metrics_creation(self):
        """Test creating roofline performance metrics."""
        metrics = RooflineMetrics(
            arithmetic_intensity=2.5,
            achieved_flops_per_second=150e12,
            achieved_bandwidth_gb_per_second=800.0,
            efficiency_ratio=0.48,
            bottleneck="compute",
        )

        assert metrics.arithmetic_intensity == 2.5
        assert metrics.achieved_flops_per_second == 150e12
        assert metrics.achieved_bandwidth_gb_per_second == 800.0
        assert metrics.efficiency_ratio == 0.48
        assert metrics.bottleneck == "compute"

    def test_roofline_metrics_memory_bound(self):
        """Test roofline metrics for memory-bound operations."""
        metrics = RooflineMetrics(
            arithmetic_intensity=0.5,
            achieved_flops_per_second=50e12,
            achieved_bandwidth_gb_per_second=1200.0,
            efficiency_ratio=0.77,
            bottleneck="memory",
        )

        assert metrics.bottleneck == "memory"
        assert metrics.arithmetic_intensity < 1.0


class TestHardwareDetector:
    """Test the HardwareDetector for automatic platform detection."""

    def test_hardware_detector_initialization(self):
        """Test hardware detector can be initialized."""
        detector = HardwareDetector()
        assert detector is not None

    @patch("jax.devices")
    def test_detect_gpu_hardware(self, mock_devices):
        """Test detection of GPU hardware."""
        # Mock GPU device
        mock_device = Mock()
        mock_device.platform = "gpu"
        mock_device.device_kind = "NVIDIA A100-SXM4-40GB"
        mock_devices.return_value = [mock_device]

        detector = HardwareDetector()
        specs = detector.detect_hardware()

        assert specs.platform == "gpu"
        assert specs.device_count == 1

    @patch("jax.devices")
    def test_detect_tpu_hardware(self, mock_devices):
        """Test detection of TPU hardware."""
        # Mock TPU device
        mock_device = Mock()
        mock_device.platform = "tpu"
        mock_device.device_kind = "TPU v4"
        mock_devices.return_value = [mock_device] * 8

        detector = HardwareDetector()
        specs = detector.detect_hardware()

        assert specs.platform == "tpu"
        assert specs.device_count == 8

    @patch("jax.devices")
    def test_detect_cpu_hardware(self, mock_devices):
        """Test detection of CPU hardware."""
        # Mock CPU device
        mock_device = Mock()
        mock_device.platform = "cpu"
        mock_device.device_kind = "cpu"
        mock_devices.return_value = [mock_device]

        detector = HardwareDetector()
        specs = detector.detect_hardware()

        assert specs.platform == "cpu"
        assert specs.device_count == 1

    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        detector = HardwareDetector()

        # Test with different model sizes
        optimal_batch = detector.get_optimal_batch_size(memory_gb=32.0, model_memory_gb=2.0)

        assert isinstance(optimal_batch, int)
        assert optimal_batch > 0

    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        detector = HardwareDetector()

        usage = detector.estimate_memory_usage(
            batch_size=16, sequence_length=512, hidden_size=768, num_layers=12
        )

        assert isinstance(usage, float)
        assert usage > 0

    def test_critical_batch_size_detection(self):
        """Test hardware-specific critical batch size."""
        detector = HardwareDetector()
        critical_size = detector.get_critical_batch_size()

        backend = jax.default_backend()
        if backend == "tpu":
            assert critical_size == 240  # TPU v5e
        elif backend == "gpu":
            # Should detect GPU type and return appropriate size
            assert critical_size in [298, 256, 128, 192]  # H100, A100, V100, RTX 4090
        else:
            assert critical_size == 32  # CPU fallback

    def test_is_batch_size_optimal(self):
        """Test batch size optimality check."""
        detector = HardwareDetector()

        # Suboptimal sizes
        assert not detector.is_batch_size_optimal(32)
        assert not detector.is_batch_size_optimal(64)

        # Optimal size (depends on hardware)
        critical = detector.get_critical_batch_size()
        assert detector.is_batch_size_optimal(critical)


class TestPerformanceEstimator:
    """Test the PerformanceEstimator for roofline analysis."""

    def test_performance_estimator_initialization(self):
        """Test performance estimator can be initialized."""
        estimator = PerformanceEstimator()
        assert estimator is not None

    def test_estimate_flops_linear_layer(self):
        """Test FLOP estimation for linear layers."""
        estimator = PerformanceEstimator()

        flops = estimator.estimate_flops_linear(batch_size=32, input_size=768, output_size=3072)

        expected_flops = 32 * 768 * 3072 * 2  # Forward pass
        assert flops == expected_flops

    def test_estimate_flops_attention(self):
        """Test FLOP estimation for attention mechanisms."""
        estimator = PerformanceEstimator()

        flops = estimator.estimate_flops_attention(
            batch_size=16, sequence_length=512, hidden_size=768, num_heads=12
        )

        assert isinstance(flops, int)
        assert flops > 0

    def test_calculate_arithmetic_intensity(self):
        """Test arithmetic intensity calculation."""
        estimator = PerformanceEstimator()

        intensity = estimator.calculate_arithmetic_intensity(
            total_flops=1000000,
            memory_bytes=4000,  # 1000 float32 numbers
        )

        expected_intensity = 1000000 / 4000
        assert intensity == expected_intensity

    def test_analyze_roofline_compute_bound(self):
        """Test roofline analysis for compute-bound operations."""
        estimator = PerformanceEstimator()

        # Mock hardware specs
        hardware_specs = HardwareSpecs(
            platform="gpu",
            device_count=1,
            memory_gb=40.0,
            peak_flops_per_second=312e12,
            memory_bandwidth_gb_per_second=1555.0,
        )

        metrics = estimator.analyze_roofline(
            operation_flops=1000000,
            memory_bytes=1000,  # High arithmetic intensity
            hardware_specs=hardware_specs,
            execution_time_seconds=0.001,
        )

        assert isinstance(metrics, RooflineMetrics)
        assert metrics.arithmetic_intensity > 100  # High intensity
        assert metrics.bottleneck == "compute"

    def test_analyze_roofline_memory_bound(self):
        """Test roofline analysis for memory-bound operations."""
        estimator = PerformanceEstimator()

        # Mock hardware specs
        hardware_specs = HardwareSpecs(
            platform="gpu",
            device_count=1,
            memory_gb=40.0,
            peak_flops_per_second=312e12,
            memory_bandwidth_gb_per_second=1555.0,
        )

        metrics = estimator.analyze_roofline(
            operation_flops=1000,
            memory_bytes=100000,  # Low arithmetic intensity
            hardware_specs=hardware_specs,
            execution_time_seconds=0.001,
        )

        assert isinstance(metrics, RooflineMetrics)
        assert metrics.arithmetic_intensity < 1  # Low intensity
        assert metrics.bottleneck == "memory"

    def test_profile_jax_function(self):
        """Test profiling JAX functions."""
        estimator = PerformanceEstimator()

        # Simple JAX function to profile
        @jax.jit
        def simple_matmul(x, y):
            return jnp.dot(x, y)

        # Create test arrays
        x = jnp.ones((100, 100))
        y = jnp.ones((100, 100))

        metrics = estimator.profile_jax_function(
            func=simple_matmul,
            args=(x, y),
            hardware_specs=HardwareSpecs(
                platform="cpu",
                device_count=1,
                memory_gb=16.0,
                peak_flops_per_second=1e12,
                memory_bandwidth_gb_per_second=100.0,
            ),
        )

        assert isinstance(metrics, RooflineMetrics)
        assert metrics.achieved_flops_per_second > 0
        assert metrics.efficiency_ratio >= 0

    def test_estimate_transformer_layer_performance(self):
        """Test performance estimation for transformer layers."""
        estimator = PerformanceEstimator()

        metrics = estimator.estimate_transformer_layer_performance(
            batch_size=16,
            sequence_length=512,
            hidden_size=768,
            num_heads=12,
            feedforward_size=3072,
            hardware_specs=HardwareSpecs(
                platform="gpu",
                device_count=1,
                memory_gb=40.0,
                peak_flops_per_second=312e12,
                memory_bandwidth_gb_per_second=1555.0,
            ),
        )

        assert isinstance(metrics, RooflineMetrics)
        assert metrics.achieved_flops_per_second > 0

    def test_benchmark_operation(self):
        """Test benchmarking of JAX operations."""
        estimator = PerformanceEstimator()

        # Simple operation to benchmark
        def operation():
            x = jnp.ones((1000, 1000))
            return jnp.sum(x @ x)

        execution_time = estimator.benchmark_operation(
            operation=operation, num_iterations=5, warmup_iterations=2
        )

        assert isinstance(execution_time, float)
        assert execution_time > 0


class TestIntegration:
    """Integration tests for performance infrastructure."""

    def test_hardware_detection_and_performance_estimation(self):
        """Test integration between hardware detection and performance
        estimation."""
        detector = HardwareDetector()
        estimator = PerformanceEstimator()

        # Detect hardware
        hardware_specs = detector.detect_hardware()

        # Estimate performance for a simple operation
        metrics = estimator.analyze_roofline(
            operation_flops=1000000,
            memory_bytes=4000,
            hardware_specs=hardware_specs,
            execution_time_seconds=0.001,
        )

        assert isinstance(metrics, RooflineMetrics)
        assert metrics.efficiency_ratio >= 0

    def test_optimal_configuration_recommendation(self):
        """Test recommendation of optimal configurations based on hardware."""
        detector = HardwareDetector()

        hardware_specs = detector.detect_hardware()

        # Test batch size recommendation
        optimal_batch = detector.get_optimal_batch_size(
            memory_gb=hardware_specs.memory_gb, model_memory_gb=1.0
        )

        assert isinstance(optimal_batch, int)
        assert optimal_batch > 0

        # Should recommend larger batches for more powerful hardware
        if hardware_specs.memory_gb > 16:
            assert optimal_batch >= 8
