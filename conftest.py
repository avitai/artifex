"""Root pytest configuration for Artifex."""

from __future__ import annotations

import gc
import warnings

import pytest


def pytest_configure() -> None:
    """Register warning filters without forcing JAX initialization."""
    warnings.filterwarnings("ignore", category=UserWarning, module="jax._src.xla_bridge")
    warnings.filterwarnings("ignore", message=".*cuSPARSE.*")
    warnings.filterwarnings("ignore", message=".*CUDA-enabled jaxlib.*")


@pytest.fixture
def device():
    """Provide a device fixture that prefers GPU and falls back to CPU."""
    import jax

    try:
        gpu_devices = [
            candidate for candidate in jax.devices() if candidate.platform in {"gpu", "cuda"}
        ]
        if gpu_devices:
            return gpu_devices[0]
    except RuntimeError:
        pass

    return jax.devices("cpu")[0]


@pytest.fixture
def rngs():
    """Provide RNG fixture for tests."""
    import flax.nnx as nnx

    return nnx.Rngs(0)


def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:  # noqa: ARG001
    """Clear Python-side garbage after accelerator tests to reduce memory pressure."""
    if item.get_closest_marker("accelerator") is not None:
        gc.collect()
