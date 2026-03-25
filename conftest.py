"""Root pytest configuration for Artifex."""

from __future__ import annotations

import gc
import warnings

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register warning filters and metadata without forcing JAX initialization."""
    warnings.filterwarnings("ignore", category=UserWarning, module="jax._src.xla_bridge")
    warnings.filterwarnings("ignore", message=".*cuSPARSE.*")
    warnings.filterwarnings("ignore", message=".*CUDA-enabled jaxlib.*")
    config.addinivalue_line(
        "markers",
        "gpu_available: GPU availability is determined lazily from the active JAX runtime",
    )


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


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip GPU-marked tests only when JAX cannot see a GPU backend."""
    if (
        "gpu" not in item.keywords
        and "cuda" not in item.keywords
        and "requires_gpu" not in item.keywords
    ):
        return

    from tests.utils.gpu_test_utils import is_gpu_available

    if not is_gpu_available():
        pytest.skip("GPU test skipped: no JAX GPU backend is available")


def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:  # noqa: ARG001
    """Clear Python-side garbage after GPU tests to reduce memory pressure."""
    if "gpu" in item.keywords or "cuda" in item.keywords or "requires_gpu" in item.keywords:
        gc.collect()
