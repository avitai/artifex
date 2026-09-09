#!/usr/bin/env python3
"""Repo-local GPU diagnostics over the substrax device inventory.

Two views of the runtime (``--detailed`` adds the per-device list and the JAX
backend) and two diagnostic runs: ``--test`` runs every check, ``--test-critical``
the two a training loop cannot do without.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger("gpu_utils")

CheckOutcome = tuple[bool, str | None, dict[str, Any]]


@dataclass(frozen=True, slots=True, kw_only=True)
class Check:
    """One runtime diagnostic."""

    name: str
    critical: bool
    run: Callable[[], CheckOutcome]


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckResult:
    """What one diagnostic reported."""

    name: str
    critical: bool
    passed: bool
    error: str | None
    metadata: dict[str, Any]
    seconds: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--detailed", action="store_true", help="List every device and the backend")
    group.add_argument("--test", action="store_true", help="Run the full diagnostic suite")
    group.add_argument("--test-critical", action="store_true", help="Run the critical checks only")
    return parser.parse_args(argv)


def _status_lines(*, detailed: bool) -> list[str]:
    """Render the device inventory substrax detects."""
    import jax
    from substrax.devices import detect_devices

    info = detect_devices()
    lines = [
        "Artifex GPU Status",
        "==================",
        f"Platform: {info.platform}",
        f"Device kind: {info.kind.value}",
        f"Visible devices: {info.count}",
    ]
    if detailed:
        lines.append(f"Default backend: {jax.default_backend()}")
        lines.append("Devices:")
        lines.extend(f"  - {device}" for device in jax.devices())
    elif info.device_kinds:
        lines.append("Device kinds: " + ", ".join(sorted(set(info.device_kinds))))
    return lines


def _basic_computation() -> CheckOutcome:
    """Verify basic JAX array math on the active runtime."""
    import jax.numpy as jnp

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = float(jnp.sum(x**2))
    expected = 30.0
    passed = abs(result - expected) < 1e-6
    return passed, None if passed else f"Expected {expected}, got {result}", {"result": result}


def _nnx_forward_and_grad() -> CheckOutcome:
    """Verify a small Flax NNX forward-and-grad path."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    class Probe(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs) -> None:
            self.linear1 = nnx.Linear(128, 64, rngs=rngs)
            self.linear2 = nnx.Linear(64, 8, rngs=rngs)

        def __call__(self, x: Any) -> Any:
            return self.linear2(nnx.relu(self.linear1(x)))

    model = Probe(rngs=nnx.Rngs(123))
    inputs = jax.random.normal(jax.random.key(456), (16, 128))
    loss, grads = nnx.value_and_grad(lambda m, x: jnp.mean(m(x) ** 2))(model, inputs)
    output = model(inputs)
    passed = (
        output.shape == (16, 8)
        and bool(jnp.isfinite(output).all())
        and bool(jnp.isfinite(loss))
        and grads is not None
    )
    error = None if passed else "NNX forward/grad diagnostic failed"
    return passed, error, {"output_shape": tuple(output.shape), "loss": float(loss)}


def _matrix_multiplication(size: int) -> Callable[[], CheckOutcome]:
    """Build a matrix multiplication diagnostic for one matrix size."""

    def run() -> CheckOutcome:
        import jax
        import jax.numpy as jnp

        key_a, key_b = jax.random.split(jax.random.key(size))
        result = jnp.matmul(
            jax.random.normal(key_a, (size, size)), jax.random.normal(key_b, (size, size))
        )
        result.block_until_ready()
        finite = bool(jnp.isfinite(result).all())
        passed = result.shape == (size, size) and finite
        error = None if passed else "Matrix multiplication produced invalid output"
        return passed, error, {"matrix_size": size, "finite": finite}

    return run


def _attention_and_noise_ops() -> CheckOutcome:
    """Verify attention-like and diffusion-style JAX operations."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    key_input, key_noise = jax.random.split(jax.random.key(789))
    hidden_dim = 64
    x = jax.random.normal(key_input, (4, 32, hidden_dim))
    noise = jax.random.normal(key_noise, x.shape)

    @jax.jit
    def attention_like(values: Any) -> Any:
        weights = nnx.softmax(jnp.matmul(values, values.transpose(0, 2, 1)) / jnp.sqrt(hidden_dim))
        return jnp.matmul(weights, values)

    @jax.jit
    def add_noise(values: Any, sample_noise: Any, timestep: Any) -> Any:
        alpha = jnp.cos(timestep * jnp.pi / 2) ** 2
        return jnp.sqrt(alpha) * values + jnp.sqrt(1 - alpha) * sample_noise

    attention = attention_like(x)
    noisy = add_noise(x, noise, jnp.array(0.5))
    passed = all(
        out.shape == x.shape and bool(jnp.isfinite(out).all()) for out in (attention, noisy)
    )
    error = None if passed else "Generative-model diagnostic produced invalid output"
    return passed, error, {"input_shape": tuple(x.shape)}


def _memory_allocation(size_mb: int) -> Callable[[], CheckOutcome]:
    """Build a memory-allocation diagnostic for the given size."""

    def run() -> CheckOutcome:
        import jax
        import jax.numpy as jnp

        side = max(1, int(((size_mb * 1024 * 1024) // 4) ** 0.5))
        value = jnp.sum(jax.random.normal(jax.random.key(size_mb), (side, side)) ** 2)
        value.block_until_ready()
        passed = bool(jnp.isfinite(value))
        error = None if passed else "Memory allocation produced a non-finite result"
        return passed, error, {"target_mb": size_mb, "allocated_shape": (side, side)}

    return run


CHECKS: tuple[Check, ...] = (
    Check(name="Basic Computation", critical=True, run=_basic_computation),
    Check(name="Neural Network Operations", critical=True, run=_nnx_forward_and_grad),
    Check(
        name="Matrix Multiplication (1024x1024)", critical=False, run=_matrix_multiplication(1024)
    ),
    Check(name="Generative Model Operations", critical=False, run=_attention_and_noise_ops),
    Check(name="Memory Allocation (128MB)", critical=False, run=_memory_allocation(128)),
)


def _execute(check: Check) -> CheckResult:
    """Run one check, turning an exception into a failed result."""
    started = time.perf_counter()
    try:
        passed, error, metadata = check.run()
    except (RuntimeError, ValueError, TypeError, MemoryError) as exc:
        passed, error, metadata = False, f"{type(exc).__name__}: {exc}", {}
    return CheckResult(
        name=check.name,
        critical=check.critical,
        passed=passed,
        error=error,
        metadata=metadata,
        seconds=time.perf_counter() - started,
    )


def run_checks(*, critical_only: bool) -> list[CheckResult]:
    """Run the diagnostics in order; a failed critical check ends the run."""
    results: list[CheckResult] = []
    for check in CHECKS:
        if critical_only and not check.critical:
            continue
        result = _execute(check)
        results.append(result)
        if result.critical and not result.passed:
            break
    return results


def _report(results: Sequence[CheckResult]) -> bool:
    """Log the results and return whether every critical check passed."""
    passed = sum(result.passed for result in results)
    logger.info("Device diagnostics: %d/%d passed", passed, len(results))
    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        level = "critical" if result.critical else "optional"
        logger.info("%s [%s] %s (%.3fs)", marker, level, result.name, result.seconds)
        if result.error:
            logger.error("  %s", result.error)
        for key, value in result.metadata.items():
            logger.info("  %s: %s", key, value)
    return all(result.passed for result in results if result.critical)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the GPU diagnostics CLI."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.test or args.test_critical:
        return 0 if _report(run_checks(critical_only=args.test_critical)) else 1

    print("\n".join(_status_lines(detailed=args.detailed)))  # noqa: T201 CLI entry point
    return 0


if __name__ == "__main__":
    sys.exit(main())
