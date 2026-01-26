"""
Foundation-first device testing framework for Artifex.

This module provides comprehensive testing capabilities for device management,
following test-driven development principles and prioritizing strong design.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx

from .device_manager import DeviceManager


class TestSeverity(Enum):
    """Test severity levels."""

    CRITICAL = "critical"  # Must pass for basic functionality
    IMPORTANT = "important"  # Should pass for optimal performance
    OPTIONAL = "optional"  # Nice to have, may fail on some systems


@dataclass
class TestResult:
    """Immutable test result."""

    test_name: str
    passed: bool
    severity: TestSeverity
    execution_time: float
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status_icon(self) -> str:
        """Get status icon for display."""
        return "✅" if self.passed else "❌"

    @property
    def severity_icon(self) -> str:
        """Get severity icon."""
        icons = {
            TestSeverity.CRITICAL: "🔴",
            TestSeverity.IMPORTANT: "🟡",
            TestSeverity.OPTIONAL: "🟢",
        }
        return icons[self.severity]


@dataclass
class TestSuite:
    """Test suite results."""

    name: str
    results: list[TestResult] = field(default_factory=list)

    @property
    def total_tests(self) -> int:
        """Get total number of tests."""
        return len(self.results)

    @property
    def passed_tests(self) -> int:
        """Get number of passed tests."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_tests(self) -> int:
        """Get number of failed tests."""
        return self.total_tests - self.passed_tests

    @property
    def critical_failures(self) -> list[TestResult]:
        """Get critical test failures."""
        return [r for r in self.results if not r.passed and r.severity == TestSeverity.CRITICAL]

    @property
    def success_rate(self) -> float:
        """Get test success rate."""
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0

    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy (no critical failures)."""
        return len(self.critical_failures) == 0


class DeviceTest(ABC):
    """Abstract base class for device tests."""

    def __init__(self, name: str, severity: TestSeverity = TestSeverity.IMPORTANT):
        """Initialize test."""
        self.name = name
        self.severity = severity

    @abstractmethod
    def run(self, device_manager: DeviceManager) -> TestResult:
        """Run the test and return result."""
        pass

    def _execute_with_timing(
        self, test_func: Callable[[], tuple[bool, str | None, dict[str, Any]]]
    ) -> TestResult:
        """Execute test function with timing and error handling."""
        start_time = time.time()

        try:
            passed, error_message, metadata = test_func()
            execution_time = time.time() - start_time

            return TestResult(
                test_name=self.name,
                passed=passed,
                severity=self.severity,
                execution_time=execution_time,
                error_message=error_message,
                metadata=metadata,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=self.name,
                passed=False,
                severity=self.severity,
                execution_time=execution_time,
                error_message=str(e),
                metadata={"exception_type": type(e).__name__},
            )


class BasicComputationTest(DeviceTest):
    """Test basic JAX computation capabilities."""

    def __init__(self):
        """Initialize the basic computation test."""
        super().__init__("Basic Computation", TestSeverity.CRITICAL)

    def run(self, device_manager: DeviceManager) -> TestResult:
        """Test basic JAX operations."""

        def test_func():
            # Test basic array operations
            x = jnp.array([1.0, 2.0, 3.0, 4.0])
            y = jnp.sum(x**2)
            result = float(y)

            expected = 30.0  # 1 + 4 + 9 + 16
            passed = abs(result - expected) < 1e-6

            metadata = {
                "input_shape": x.shape,
                "result": result,
                "expected": expected,
                "device": str(jax.default_backend()),
            }

            error_message = None if passed else f"Expected {expected}, got {result}"

            return passed, error_message, metadata

        return self._execute_with_timing(test_func)


class MatrixMultiplicationTest(DeviceTest):
    """Test matrix multiplication with different sizes."""

    def __init__(self, size: int = 1000):
        """Initialize the matrix multiplication test.

        Args:
            size: Size of the matrices to multiply
        """
        super().__init__(f"Matrix Multiplication ({size}x{size})", TestSeverity.IMPORTANT)
        self.size = size

    def run(self, device_manager: DeviceManager) -> TestResult:
        """Test matrix multiplication."""

        def test_func():
            key = jax.random.key(42)

            # Create test matrices
            a = jax.random.normal(key, (self.size, self.size))
            b = jax.random.normal(key, (self.size, self.size))

            # Perform multiplication
            result = jnp.dot(a, b)
            result.block_until_ready()  # Force computation

            # Verify result shape and basic properties
            passed = result.shape == (self.size, self.size) and jnp.isfinite(result).all()

            metadata = {
                "matrix_size": self.size,
                "result_shape": result.shape,
                "result_finite": bool(jnp.isfinite(result).all()),
                "result_mean": float(jnp.mean(result)),
                "result_std": float(jnp.std(result)),
            }

            error_message = None
            if not passed:
                if result.shape != (self.size, self.size):
                    error_message = (
                        f"Wrong shape: expected {(self.size, self.size)}, got {result.shape}"
                    )
                elif not jnp.isfinite(result).all():
                    error_message = "Result contains non-finite values"

            return passed, error_message, metadata

        return self._execute_with_timing(test_func)


class NeuralNetworkTest(DeviceTest):
    """Test neural network operations using Flax NNX."""

    def __init__(self):
        """Initialize the neural network test."""
        super().__init__("Neural Network Operations", TestSeverity.CRITICAL)

    def run(self, device_manager: DeviceManager) -> TestResult:
        """Test neural network forward pass and gradient computation."""

        def test_func():
            key = jax.random.key(123)
            key, init_key = jax.random.split(key)

            # Create a test MLP
            class TestMLP(nnx.Module):
                def __init__(self, *, rngs: nnx.Rngs):
                    self.linear1 = nnx.Linear(784, 256, rngs=rngs)
                    self.linear2 = nnx.Linear(256, 128, rngs=rngs)
                    self.linear3 = nnx.Linear(128, 10, rngs=rngs)

                def __call__(self, x):
                    x = nnx.relu(self.linear1(x))
                    x = nnx.relu(self.linear2(x))
                    return self.linear3(x)

            # Initialize model
            rngs = nnx.Rngs(init_key)
            model = TestMLP(rngs=rngs)

            # Test forward pass
            batch_size = 32
            x = jax.random.normal(key, (batch_size, 784))
            output = model(x)

            # Test gradient computation
            def loss_fn(model, x):
                return jnp.mean(model(x) ** 2)

            loss, grads = nnx.value_and_grad(loss_fn)(model, x)

            # Verify results
            forward_passed = output.shape == (batch_size, 10) and jnp.isfinite(output).all()

            grad_passed = (
                jnp.isfinite(loss) and loss > 0  # Loss should be positive for squared outputs
            )

            passed = forward_passed and grad_passed

            metadata = {
                "batch_size": batch_size,
                "output_shape": output.shape,
                "loss_value": float(loss),
                "output_finite": bool(jnp.isfinite(output).all()),
                "loss_finite": bool(jnp.isfinite(loss)),
                "output_mean": float(jnp.mean(output)),
                "output_std": float(jnp.std(output)),
            }

            error_message = None
            if not forward_passed:
                if output.shape != (batch_size, 10):
                    error_message = (
                        f"Wrong output shape: expected {(batch_size, 10)}, got {output.shape}"
                    )
                elif not jnp.isfinite(output).all():
                    error_message = "Output contains non-finite values"
            elif not grad_passed:
                if not jnp.isfinite(loss):
                    error_message = "Loss is not finite"
                elif loss <= 0:
                    error_message = f"Loss should be positive, got {loss}"

            return passed, error_message, metadata

        return self._execute_with_timing(test_func)


class GenerativeModelTest(DeviceTest):
    """Test generative model operations."""

    def __init__(self):
        """Initialize the generative model test."""
        super().__init__("Generative Model Operations", TestSeverity.IMPORTANT)

    def run(self, device_manager: DeviceManager) -> TestResult:
        """Test generative model specific operations."""

        def test_func():
            key = jax.random.key(456)

            # Test attention-like operations
            batch_size, seq_len, hidden_dim = 8, 64, 256
            x = jax.random.normal(key, (batch_size, seq_len, hidden_dim))

            @jax.jit
            def attention_like_op(x):
                # Simplified attention computation
                q = jnp.matmul(x, x.transpose(0, 2, 1))
                attention_weights = nnx.softmax(q / jnp.sqrt(hidden_dim))
                return jnp.matmul(attention_weights, x)

            attention_output = attention_like_op(x)

            # Test noise operations (common in diffusion models)
            noise = jax.random.normal(key, x.shape)

            @jax.jit
            def add_noise(x, noise, t):
                alpha = jnp.cos(t * jnp.pi / 2) ** 2
                return jnp.sqrt(alpha) * x + jnp.sqrt(1 - alpha) * noise

            t = jnp.array(0.5)
            noisy_x = add_noise(x, noise, t)

            # Verify results
            attention_passed = (
                attention_output.shape == x.shape and jnp.isfinite(attention_output).all()
            )

            noise_passed = noisy_x.shape == x.shape and jnp.isfinite(noisy_x).all()

            passed = attention_passed and noise_passed

            metadata = {
                "input_shape": x.shape,
                "attention_output_shape": attention_output.shape,
                "noisy_output_shape": noisy_x.shape,
                "attention_finite": bool(jnp.isfinite(attention_output).all()),
                "noise_finite": bool(jnp.isfinite(noisy_x).all()),
                "attention_mean": float(jnp.mean(attention_output)),
                "noise_mean": float(jnp.mean(noisy_x)),
            }

            error_message = None
            if not attention_passed:
                if attention_output.shape != x.shape:
                    error_message = (
                        f"Attention output wrong shape: expected {x.shape}, "
                        f"got {attention_output.shape}"
                    )
                elif not jnp.isfinite(attention_output).all():
                    error_message = "Attention output contains non-finite values"
            elif not noise_passed:
                if noisy_x.shape != x.shape:
                    error_message = (
                        f"Noise output wrong shape: expected {x.shape}, got {noisy_x.shape}"
                    )
                elif not jnp.isfinite(noisy_x).all():
                    error_message = "Noise output contains non-finite values"

            return passed, error_message, metadata

        return self._execute_with_timing(test_func)


class MemoryStressTest(DeviceTest):
    """Test memory allocation and management."""

    def __init__(self, size_mb: int = 100):
        """Initialize the memory stress test.

        Args:
            size_mb: Size of memory to allocate in MB
        """
        super().__init__(f"Memory Stress Test ({size_mb}MB)", TestSeverity.IMPORTANT)
        self.size_mb = size_mb

    def run(self, device_manager: DeviceManager) -> TestResult:
        """Test memory allocation and deallocation."""

        def test_func():
            # Calculate array size for target memory usage
            # Assuming float32 (4 bytes per element)
            target_elements = (self.size_mb * 1024 * 1024) // 4
            array_size = int(target_elements**0.5)  # Square array

            key = jax.random.key(789)

            try:
                # Allocate large array
                large_array = jax.random.normal(key, (array_size, array_size))

                # Perform some operations
                result = jnp.sum(large_array**2)
                result.block_until_ready()

                # Clean up
                del large_array

                passed = jnp.isfinite(result)

                metadata = {
                    "target_mb": self.size_mb,
                    "array_shape": (array_size, array_size),
                    "actual_mb": (array_size * array_size * 4) / (1024 * 1024),
                    "result": float(result),
                    "result_finite": bool(jnp.isfinite(result)),
                }

                error_message = None if passed else "Result is not finite"

                return passed, error_message, metadata

            except Exception as e:
                # Memory allocation failed
                metadata = {
                    "target_mb": self.size_mb,
                    "array_shape": (array_size, array_size),
                    "error_type": type(e).__name__,
                }

                return False, str(e), metadata

        return self._execute_with_timing(test_func)


class DeviceTestRunner:
    """Comprehensive device test runner."""

    def __init__(self, device_manager: DeviceManager):
        """Initialize test runner."""
        self.device_manager = device_manager
        self.tests: list[DeviceTest] = []
        self._setup_default_tests()

    def _setup_default_tests(self) -> None:
        """Setup default test suite."""
        self.tests = [
            BasicComputationTest(),
            MatrixMultiplicationTest(1000),
            NeuralNetworkTest(),
            GenerativeModelTest(),
            MatrixMultiplicationTest(5000),  # Larger test
            MemoryStressTest(50),
            MemoryStressTest(200),
        ]

    def add_test(self, test: DeviceTest) -> None:
        """Add a custom test."""
        self.tests.append(test)

    def run_all_tests(self) -> TestSuite:
        """Run all tests and return results."""
        suite = TestSuite("Device Comprehensive Test Suite")

        for test in self.tests:
            result = test.run(self.device_manager)
            suite.results.append(result)

            # Stop on critical failures for safety
            if not result.passed and result.severity == TestSeverity.CRITICAL:
                print(f"🔴 Critical test failed: {test.name}")
                print(f"   Error: {result.error_message}")
                print("   Stopping test execution for safety.")
                break

        return suite

    def run_critical_tests_only(self) -> TestSuite:
        """Run only critical tests."""
        suite = TestSuite("Critical Tests Only")

        critical_tests = [t for t in self.tests if t.severity == TestSeverity.CRITICAL]

        for test in critical_tests:
            result = test.run(self.device_manager)
            suite.results.append(result)

        return suite


def print_test_results(suite: TestSuite) -> None:
    """Print comprehensive test results."""
    print(f"\n🧪 {suite.name}")
    print("=" * 60)
    print(f"Tests: {suite.passed_tests}/{suite.total_tests} passed ({suite.success_rate:.1%})")
    print(f"Health: {'✅ Healthy' if suite.is_healthy else '❌ Critical Issues'}")

    print("\n📊 Test Results:")
    for result in suite.results:
        print(f"{result.status_icon} {result.severity_icon} {result.test_name}")
        print(f"   Time: {result.execution_time:.3f}s")
        if result.error_message:
            print(f"   Error: {result.error_message}")
        if result.metadata:
            for key, value in list(result.metadata.items())[:3]:  # Show first 3 metadata items
                print(f"   {key}: {value}")

    if suite.critical_failures:
        print("\n🔴 Critical Failures:")
        for failure in suite.critical_failures:
            print(f"   • {failure.test_name}: {failure.error_message}")

    print(f"\n⏱️  Total execution time: {sum(r.execution_time for r in suite.results):.3f}s")


def run_device_tests(
    device_manager: DeviceManager | None = None, critical_only: bool = False
) -> TestSuite:
    """Run device tests."""
    if device_manager is None:
        from .device_manager import get_device_manager

        device_manager = get_device_manager()

    runner = DeviceTestRunner(device_manager)

    if critical_only:
        suite = runner.run_critical_tests_only()
    else:
        suite = runner.run_all_tests()

    return suite


if __name__ == "__main__":
    # Run tests when executed directly
    from .device_manager import print_device_info

    print_device_info()

    print("\n🚀 Running comprehensive device tests...")
    suite = run_device_tests()
    print_test_results(suite)
