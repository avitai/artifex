"""Runtime-only device diagnostics for Artifex."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, TYPE_CHECKING


if TYPE_CHECKING:
    from .device_manager import DeviceManager


logger = logging.getLogger(__name__)


class TestSeverity(Enum):
    """Diagnostic severity levels."""

    CRITICAL = "critical"
    IMPORTANT = "important"
    OPTIONAL = "optional"


@dataclass(frozen=True, slots=True, kw_only=True)
class TestResult:
    """Immutable result for a single runtime diagnostic."""

    test_name: str
    passed: bool
    severity: TestSeverity
    execution_time: float
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze metadata so the result object is fully immutable."""
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def status_icon(self) -> str:
        """Return a display icon for pass/fail state."""
        return "PASS" if self.passed else "FAIL"


@dataclass(frozen=True, slots=True, kw_only=True)
class TestSuite:
    """Immutable collection of device diagnostic results."""

    name: str
    results: tuple[TestResult, ...] = ()

    @property
    def total_tests(self) -> int:
        """Return the number of executed diagnostics."""
        return len(self.results)

    @property
    def passed_tests(self) -> int:
        """Return the number of successful diagnostics."""
        return sum(1 for result in self.results if result.passed)

    @property
    def failed_tests(self) -> int:
        """Return the number of failed diagnostics."""
        return self.total_tests - self.passed_tests

    @property
    def critical_failures(self) -> tuple[TestResult, ...]:
        """Return failed critical diagnostics."""
        return tuple(
            result
            for result in self.results
            if not result.passed and result.severity == TestSeverity.CRITICAL
        )

    @property
    def success_rate(self) -> float:
        """Return the percentage of passing diagnostics."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests

    @property
    def is_healthy(self) -> bool:
        """Return whether the runtime passed all critical diagnostics."""
        return not self.critical_failures


_CheckRunner = Callable[["DeviceManager"], tuple[bool, str | None, dict[str, Any]]]


class _CheckLike(Protocol):
    """Protocol for injected diagnostic checks used by tests."""

    name: str
    severity: TestSeverity

    def run(self, device_manager: DeviceManager) -> tuple[bool, str | None, dict[str, Any]]:
        """Execute the diagnostic."""
        ...


@dataclass(frozen=True, slots=True)
class _DiagnosticCheck:
    """Private description of one runtime diagnostic."""

    name: str
    severity: TestSeverity
    runner: _CheckRunner

    def execute(self, device_manager: DeviceManager) -> TestResult:
        """Run this diagnostic with timing and exception capture."""
        start = time.perf_counter()

        try:
            passed, error_message, metadata = self.runner(device_manager)
        except Exception as exc:  # pragma: no cover - defensive path
            return TestResult(
                test_name=self.name,
                passed=False,
                severity=self.severity,
                execution_time=time.perf_counter() - start,
                error_message=str(exc),
                metadata={"exception_type": type(exc).__name__},
            )

        return TestResult(
            test_name=self.name,
            passed=passed,
            severity=self.severity,
            execution_time=time.perf_counter() - start,
            error_message=error_message,
            metadata=metadata,
        )


def _execute_check(
    check: _DiagnosticCheck | _CheckLike,
    device_manager: DeviceManager,
) -> TestResult:
    """Execute one check using the internal spec or a test double."""
    if isinstance(check, _DiagnosticCheck):
        return check.execute(device_manager)

    name = getattr(check, "name")
    severity = getattr(check, "severity")
    runner = getattr(check, "run")
    return _DiagnosticCheck(name=name, severity=severity, runner=runner).execute(device_manager)


def _run_basic_computation(
    device_manager: DeviceManager,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Verify basic JAX array math on the active runtime."""
    del device_manager
    import jax.numpy as jnp

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = float(jnp.sum(x**2))
    expected = 30.0
    passed = abs(result - expected) < 1e-6
    error = None if passed else f"Expected {expected}, got {result}"
    metadata = {"result": result, "expected": expected, "input_shape": tuple(x.shape)}
    return passed, error, metadata


def _matrix_multiplication_check(size: int) -> _CheckRunner:
    """Build a matrix multiplication diagnostic for one matrix size."""

    def run(device_manager: DeviceManager) -> tuple[bool, str | None, dict[str, Any]]:
        del device_manager
        import jax
        import jax.numpy as jnp

        key_a, key_b = jax.random.split(jax.random.key(size))
        a = jax.random.normal(key_a, (size, size))
        b = jax.random.normal(key_b, (size, size))
        result = jnp.matmul(a, b)
        result.block_until_ready()

        is_finite = bool(jnp.isfinite(result).all())
        passed = result.shape == (size, size) and is_finite
        error = None if passed else "Matrix multiplication produced invalid output"
        metadata = {
            "matrix_size": size,
            "result_shape": tuple(result.shape),
            "finite": is_finite,
        }
        return passed, error, metadata

    return run


def _run_nnx_forward_and_grad(
    device_manager: DeviceManager,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Verify a small Flax NNX forward-and-grad path."""
    del device_manager
    import jax
    import jax.numpy as jnp
    from flax import nnx

    class TestMLP(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs) -> None:
            self.linear1 = nnx.Linear(128, 64, rngs=rngs)
            self.linear2 = nnx.Linear(64, 8, rngs=rngs)

        def __call__(self, x: Any) -> Any:
            return self.linear2(nnx.relu(self.linear1(x)))

    rngs = nnx.Rngs(jax.random.key(123))
    model = TestMLP(rngs=rngs)
    inputs = jax.random.normal(jax.random.key(456), (16, 128))

    def loss_fn(model: TestMLP, x: Any) -> Any:
        return jnp.mean(model(x) ** 2)

    output = model(inputs)
    loss, grads = nnx.value_and_grad(loss_fn)(model, inputs)

    output_ok = output.shape == (16, 8) and bool(jnp.isfinite(output).all())
    loss_ok = bool(jnp.isfinite(loss)) and float(loss) >= 0.0
    grads_ok = grads is not None
    passed = output_ok and loss_ok and grads_ok
    error = None if passed else "NNX forward/grad diagnostic failed"
    metadata = {
        "output_shape": tuple(output.shape),
        "loss": float(loss),
        "output_finite": bool(jnp.isfinite(output).all()),
    }
    return passed, error, metadata


def _run_attention_and_noise_ops(
    device_manager: DeviceManager,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Verify attention-like and diffusion-style JAX operations."""
    del device_manager
    import jax
    import jax.numpy as jnp
    from flax import nnx

    key_input, key_noise = jax.random.split(jax.random.key(789))
    batch_size, seq_len, hidden_dim = 4, 32, 64
    x = jax.random.normal(key_input, (batch_size, seq_len, hidden_dim))
    noise = jax.random.normal(key_noise, x.shape)

    @jax.jit
    def attention_like_op(values: Any) -> Any:
        logits = jnp.matmul(values, values.transpose(0, 2, 1))
        weights = nnx.softmax(logits / jnp.sqrt(hidden_dim))
        return jnp.matmul(weights, values)

    @jax.jit
    def add_noise(values: Any, sample_noise: Any, timestep: Any) -> Any:
        alpha = jnp.cos(timestep * jnp.pi / 2) ** 2
        return jnp.sqrt(alpha) * values + jnp.sqrt(1 - alpha) * sample_noise

    attention_output = attention_like_op(x)
    noisy_output = add_noise(x, noise, jnp.array(0.5))

    attention_ok = attention_output.shape == x.shape and bool(jnp.isfinite(attention_output).all())
    noise_ok = noisy_output.shape == x.shape and bool(jnp.isfinite(noisy_output).all())
    passed = attention_ok and noise_ok
    error = None if passed else "Generative-model diagnostic produced invalid output"
    metadata = {
        "input_shape": tuple(x.shape),
        "attention_shape": tuple(attention_output.shape),
        "noise_shape": tuple(noisy_output.shape),
    }
    return passed, error, metadata


def _memory_allocation_check(size_mb: int) -> _CheckRunner:
    """Build a memory-allocation diagnostic for the given size."""

    def run(device_manager: DeviceManager) -> tuple[bool, str | None, dict[str, Any]]:
        del device_manager
        import jax
        import jax.numpy as jnp

        target_elements = (size_mb * 1024 * 1024) // 4
        side = max(1, int(target_elements**0.5))
        array = jax.random.normal(jax.random.key(size_mb), (side, side))
        value = jnp.sum(array**2)
        value.block_until_ready()
        passed = bool(jnp.isfinite(value))
        error = None if passed else "Memory allocation produced a non-finite result"
        metadata = {
            "target_mb": size_mb,
            "allocated_shape": (side, side),
            "result": float(value),
        }
        return passed, error, metadata

    return run


def _default_checks() -> tuple[_DiagnosticCheck, ...]:
    """Return the default runtime diagnostics in execution order."""
    return (
        _DiagnosticCheck(
            name="Basic Computation",
            severity=TestSeverity.CRITICAL,
            runner=_run_basic_computation,
        ),
        _DiagnosticCheck(
            name="Neural Network Operations",
            severity=TestSeverity.CRITICAL,
            runner=_run_nnx_forward_and_grad,
        ),
        _DiagnosticCheck(
            name="Matrix Multiplication (1024x1024)",
            severity=TestSeverity.IMPORTANT,
            runner=_matrix_multiplication_check(1024),
        ),
        _DiagnosticCheck(
            name="Generative Model Operations",
            severity=TestSeverity.IMPORTANT,
            runner=_run_attention_and_noise_ops,
        ),
        _DiagnosticCheck(
            name="Memory Allocation (128MB)",
            severity=TestSeverity.OPTIONAL,
            runner=_memory_allocation_check(128),
        ),
    )


def run_device_tests(
    device_manager: DeviceManager | None = None,
    *,
    critical_only: bool = False,
    checks: Sequence[_DiagnosticCheck | _CheckLike] | None = None,
) -> TestSuite:
    """Run runtime diagnostics for the active JAX backend."""
    if device_manager is None:
        from .device_manager import get_device_manager

        device_manager = get_device_manager()

    active_checks = tuple(checks if checks is not None else _default_checks())
    if critical_only:
        active_checks = tuple(
            check for check in active_checks if check.severity == TestSeverity.CRITICAL
        )

    results: list[TestResult] = []
    for check in active_checks:
        result = _execute_check(check, device_manager)
        results.append(result)
        if not result.passed and result.severity == TestSeverity.CRITICAL:
            logger.error("Critical device diagnostic failed: %s", result.test_name)
            break

    suite_name = "Critical Device Diagnostics" if critical_only else "Device Diagnostics"
    return TestSuite(name=suite_name, results=tuple(results))


def print_test_results(suite: TestSuite) -> None:
    """Log the runtime diagnostic summary."""
    logger.info("%s", suite.name)
    logger.info("=" * 60)
    logger.info(
        "Tests: %d/%d passed (%.1f%%)",
        suite.passed_tests,
        suite.total_tests,
        suite.success_rate * 100,
    )
    logger.info("Health: %s", "Healthy" if suite.is_healthy else "Critical Issues")

    for result in suite.results:
        logger.info(
            "%s [%s] %s (%.3fs)",
            result.status_icon,
            result.severity.value,
            result.test_name,
            result.execution_time,
        )
        if result.error_message:
            logger.error("  Error: %s", result.error_message)
        if result.metadata:
            for key, value in list(result.metadata.items())[:3]:
                logger.info("  %s: %s", key, value)

    if suite.critical_failures:
        logger.error("Critical Failures:")
        for failure in suite.critical_failures:
            logger.error("  %s: %s", failure.test_name, failure.error_message)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from .device_manager import print_device_info

    print_device_info()
    print_test_results(run_device_tests())
