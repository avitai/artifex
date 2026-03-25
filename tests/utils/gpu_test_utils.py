"""Utilities for GPU-aware testing."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import os
import sys
from collections.abc import Callable
from typing import Any, TypeVar

import pytest


F = TypeVar("F", bound=Callable[..., Any])


@dataclasses.dataclass(frozen=True)
class JAXRuntimeSummary:
    """Structured summary of the active JAX runtime."""

    gpu_available: bool
    default_backend: str | None
    visible_devices: tuple[str, ...]
    error: str | None = None


@contextlib.contextmanager
def suppress_process_stderr():
    """Temporarily redirect process stderr to avoid noisy plugin-init logs."""
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, "w", encoding="utf-8") as null_stream:
        saved_stderr_fd = os.dup(stderr_fd)
        try:
            os.dup2(null_stream.fileno(), stderr_fd)
            yield
        finally:
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)


def is_gpu_available() -> bool:
    """Return True only when the active JAX runtime exposes a GPU backend."""
    return get_jax_runtime_summary().gpu_available


def get_jax_runtime_summary() -> JAXRuntimeSummary:
    """Return a structured summary of the active JAX backend and devices."""
    try:
        with suppress_process_stderr():
            import jax
    except (ImportError, OSError, RuntimeError) as exc:
        return JAXRuntimeSummary(
            gpu_available=False,
            default_backend=None,
            visible_devices=(),
            error=str(exc),
        )

    try:
        with suppress_process_stderr():
            devices = tuple(
                f"{device.platform}:{getattr(device, 'device_kind', str(device))}"
                for device in jax.devices()
            )
            default_backend = jax.default_backend()
    except RuntimeError as exc:
        return JAXRuntimeSummary(
            gpu_available=False,
            default_backend=None,
            visible_devices=(),
            error=str(exc),
        )

    return JAXRuntimeSummary(
        gpu_available=any(device.startswith(("gpu:", "cuda:")) for device in devices),
        default_backend=default_backend,
        visible_devices=devices,
        error=None,
    )


def requires_gpu(func: F) -> F:
    """Decorator that skips the test at runtime when JAX cannot see a GPU."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        if not is_gpu_available():
            pytest.skip("Test requires GPU but no JAX GPU backend is available")
        return func(*args, **kwargs)

    return pytest.mark.gpu(wrapper)  # type: ignore[return-value]


def skip_on_gpu(func: F) -> F:
    """Decorator that skips the test when JAX is actively using a GPU backend."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        if is_gpu_available():
            pytest.skip("Test is known to cause issues on GPU, skipping")
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def gpu_test_fixture() -> None:
    """Helper used by tests that want fixture-style GPU enforcement."""
    if not is_gpu_available():
        pytest.skip("Test requires GPU but no JAX GPU backend is available")
