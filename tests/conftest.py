"""
Global pytest configuration and fixtures.

This file provides the main testing configuration for the artifex package,
including the JAX test environment and the shared fixture infrastructure.

The test infrastructure follows a three-tier architecture:
1. Base fixtures: Common patterns, RNG management, standard configurations
2. Data fixtures: Fast, synthetic, and cached data generation strategies
3. Model fixtures: Standardized model configurations and utilities
"""

import os
from pathlib import Path

import pytest

# jax reads JAX_PLATFORMS and JAX_NUM_CPU_DEVICES when it is imported, so the backend and the
# emulated CPU devices are chosen here, before the plugins below import it.
from tests.jax_test_environment import has_cuda_plugin, resolve_test_environment


os.environ.update(resolve_test_environment(os.environ, cuda_plugin_available=has_cuda_plugin()))

# Register the substrax JAX test plugin and the shared fixtures and hooks.
pytest_plugins = [
    "substrax.testing.pytest_plugin",
    "tests.utils.pytest_hooks",
    "tests.artifex.fixtures.base",
]


def _load_data_generators():
    """Load the heavyweight data-generator module only when a fixture needs it."""
    from tests.artifex.fixtures.data_generators import (
        CachedDataManager,
        FastDataGenerator,
        SyntheticDataGenerator,
    )

    return CachedDataManager, FastDataGenerator, SyntheticDataGenerator


def pytest_addoption(parser):
    """Add custom options to pytest.

    Args:
        parser: Pytest argument parser
    """
    parser.addini(
        "artifact_dir",
        default="test_artifacts",
        help="Directory for test artifacts",
    )
    parser.addoption(
        "--artifex-probe-jax-runtime",
        action="store_true",
        default=False,
        help="Probe the live JAX runtime for pytest header metadata.",
    )


def pytest_configure(config):
    """Configure pytest before test collection.

    This function is called once at the beginning of a test run and sets up
    the testing environment with proper warnings, markers, and metadata.

    Args:
        config: Pytest configuration object
    """
    # Add filters for common warnings that clutter test output
    config.addinivalue_line(
        "filterwarnings",
        "ignore:jax.interpreters.xla.pytype_aval_mappings is deprecated:DeprecationWarning",
    )
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:pkg_resources.*")
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning:jax.*",
    )

    # Set the artifact directory for test outputs
    if not hasattr(config, "workon"):
        config.workon = {}
    config.workon["artifact_dir"] = config.getini("artifact_dir")

    # Register custom markers; the device markers come from the substrax plugin
    config.addinivalue_line("markers", "blackjax: marks tests that use BlackJAX integration")

    if hasattr(config, "_metadata"):
        config._metadata["Artifex backend"] = os.environ.get("ARTIFEX_BACKEND", "unset")
        config._metadata["JAX runtime probe"] = (
            "enabled" if config.getoption("--artifex-probe-jax-runtime") else "deferred"
        )


def pytest_report_header(config):  # noqa: ARG001
    """Add lightweight backend information to the pytest header.

    Args:
        config: Pytest configuration object

    Returns:
        list[str]: The backend and, when probed, the visible devices
    """
    header_lines = [
        f"Artifex backend: {os.environ.get('ARTIFEX_BACKEND', 'unset')}",
    ]

    if not config.getoption("--artifex-probe-jax-runtime"):
        if hasattr(config, "_metadata"):
            config._metadata["Artifex backend"] = os.environ.get("ARTIFEX_BACKEND", "unset")
            config._metadata["JAX runtime probe"] = "deferred"
        header_lines.append(
            "JAX runtime probe: deferred (pass --artifex-probe-jax-runtime to inspect live devices)"
        )
        return header_lines

    from substrax import devices

    info = devices.detect_devices()
    header_lines.extend(
        [
            f"JAX default backend: {info.platform}",
            f"JAX visible devices: {info.count} ({', '.join(info.device_kinds)})",
            f"Accelerator available for testing: {info.has_accelerator}",
        ]
    )
    if hasattr(config, "_metadata"):
        config._metadata["Accelerator available for testing"] = str(info.has_accelerator)
        config._metadata["JAX default backend"] = info.platform
        config._metadata["Artifex backend"] = os.environ.get("ARTIFEX_BACKEND", "unset")
        config._metadata["JAX runtime probe"] = "enabled"

    return header_lines


# =====================================================================================
# DATA FIXTURES - Three-tier data generation strategy
# =====================================================================================


@pytest.fixture(scope="session")
def data_cache_manager():
    """Session-scoped data cache manager for complex test data."""
    CachedDataManager, _, _ = _load_data_generators()
    return CachedDataManager()


@pytest.fixture(params=["fast", "realistic"])
def test_data_strategy(request):
    """Parameterized fixture for different data generation strategies.

    For most tests, we use 'fast' and 'realistic' to balance speed and coverage.
    The 'cached' strategy is available for specific performance tests.
    """
    return request.param


@pytest.fixture
def diffusion_test_data(test_data_strategy, standard_shapes):
    """Generate diffusion test data based on strategy.

    This fixture provides the appropriate test data for diffusion models
    based on the current test strategy (fast/realistic/cached).
    """
    CachedDataManager, FastDataGenerator, SyntheticDataGenerator = _load_data_generators()
    if test_data_strategy == "fast":
        return FastDataGenerator.random_image_batch(standard_shapes["image_2d"], batch_size=8)
    elif test_data_strategy == "realistic":
        return SyntheticDataGenerator.synthetic_images(
            "mnist_like", standard_shapes["image_2d"], batch_size=8
        )
    elif test_data_strategy == "cached":
        spec = {
            "type": "diffusion_sequence",
            "shape": standard_shapes["image_2d"],
            "num_timesteps": 100,
            "batch_size": 8,
            "seed": 42,
        }
        return CachedDataManager.get_cached_data(spec)
    else:
        raise ValueError(f"Unknown test data strategy: {test_data_strategy}")


@pytest.fixture
def geometric_test_data(test_data_strategy):
    """Generate geometric test data based on strategy."""
    CachedDataManager, FastDataGenerator, SyntheticDataGenerator = _load_data_generators()
    if test_data_strategy == "fast":
        return FastDataGenerator.random_point_cloud(num_points=1024, batch_size=8)
    elif test_data_strategy == "realistic":
        return SyntheticDataGenerator.synthetic_point_clouds(
            "sphere", num_points=1024, batch_size=8
        )
    elif test_data_strategy == "cached":
        spec = {
            "type": "large_point_cloud",
            "cloud_type": "sphere",
            "num_points": 10000,
            "batch_size": 8,
            "seed": 42,
        }
        return CachedDataManager.get_cached_data(spec)
    else:
        raise ValueError(f"Unknown test data strategy: {test_data_strategy}")


@pytest.fixture
def vae_test_data(diffusion_test_data):
    """VAE test data (same as diffusion for consistency)."""
    return diffusion_test_data


@pytest.fixture
def fast_image_data(standard_shapes):
    """Fast random image data for smoke tests."""
    _, FastDataGenerator, _ = _load_data_generators()
    return FastDataGenerator.random_image_batch(standard_shapes["image_2d"], batch_size=4)


@pytest.fixture
def fast_point_cloud_data():
    """Fast random point cloud data for smoke tests."""
    _, FastDataGenerator, _ = _load_data_generators()
    return FastDataGenerator.random_point_cloud(num_points=512, batch_size=4)


@pytest.fixture
def realistic_image_data(standard_shapes):
    """Realistic synthetic image data for integration tests."""
    _, _, SyntheticDataGenerator = _load_data_generators()
    return SyntheticDataGenerator.synthetic_images(
        "mnist_like", standard_shapes["image_2d"], batch_size=4
    )


@pytest.fixture
def realistic_point_cloud_data():
    """Realistic synthetic point cloud data for integration tests."""
    _, _, SyntheticDataGenerator = _load_data_generators()
    return SyntheticDataGenerator.synthetic_point_clouds("sphere", num_points=512, batch_size=4)


@pytest.fixture
def test_timesteps():
    """Generate test timesteps for diffusion models."""
    _, FastDataGenerator, _ = _load_data_generators()
    return FastDataGenerator.random_timesteps(max_timesteps=100, batch_size=8)


@pytest.fixture
def test_noise(standard_shapes):
    """Generate test noise arrays."""
    _, FastDataGenerator, _ = _load_data_generators()
    return FastDataGenerator.random_noise(standard_shapes["image_2d"], batch_size=8)


@pytest.fixture
def test_labels():
    """Generate test class labels."""
    _, FastDataGenerator, _ = _load_data_generators()
    return FastDataGenerator.random_labels(num_classes=10, batch_size=8)


# =====================================================================================
# PYTEST CLEANUP - Clear caches after test sessions
# =====================================================================================


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_caches():
    """Clean up test caches at the end of test session."""
    yield
    # Clear data caches
    CachedDataManager, _, _ = _load_data_generators()
    CachedDataManager.clear_cache(disk_cache_only=False)

    # Clean up test artifacts directory if it exists
    artifacts_dir = Path("test_artifacts")
    if artifacts_dir.exists():
        import shutil

        shutil.rmtree(artifacts_dir, ignore_errors=True)
