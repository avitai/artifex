"""
Global pytest configuration and fixtures.

This file provides the main testing configuration for the artifex package,
including GPU-aware test selection and the shared fixture infrastructure.

The test infrastructure follows a three-tier architecture:
1. Base fixtures: Common patterns, RNG management, standard configurations
2. Data fixtures: Fast, synthetic, and cached data generation strategies
3. Model fixtures: Standardized model configurations and utilities
"""

import os
from pathlib import Path

import pytest


# Register shared fixtures and hooks as pytest plugins.
pytest_plugins = ["tests.utils.pytest_hooks", "tests.artifex.fixtures.base"]


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
    # CRITICAL: Set deterministic mode BEFORE any JAX imports
    # This must be done at the very start, before JAX initializes
    if os.environ.get("ARTIFEX_DETERMINISTIC", "0") == "1":
        # Set XLA flags for deterministic GPU operations
        existing_flags = os.environ.get("XLA_FLAGS", "")
        new_flags = "--xla_gpu_deterministic_ops=true"

        if existing_flags:
            os.environ["XLA_FLAGS"] = f"{existing_flags} {new_flags}"
        else:
            os.environ["XLA_FLAGS"] = new_flags

        # Set cuDNN deterministic mode
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
        os.environ["TF_DETERMINISTIC_OPS"] = "1"

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

    # Register custom markers for test categorization
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring GPU to run")
    config.addinivalue_line("markers", "skip_on_gpu: mark test to be skipped when GPU is available")
    config.addinivalue_line("markers", "blackjax: marks tests that use BlackJAX integration")

    if hasattr(config, "_metadata"):
        config._metadata["Artifex backend"] = os.environ.get("ARTIFEX_BACKEND", "unset")
        config._metadata["Deterministic test mode"] = os.environ.get("ARTIFEX_DETERMINISTIC", "0")
        config._metadata["JAX runtime probe"] = (
            "enabled" if config.getoption("--artifex-probe-jax-runtime") else "deferred"
        )


@pytest.fixture
def gpu_test_fixture():
    """Fixture to skip tests that require GPU if none is available.

    This fixture ensures that tests marked as requiring GPU are automatically
    skipped when no GPU is detected in the testing environment.

    Raises:
        pytest.skip: If no GPU is available for testing
    """
    from tests.utils.gpu_test_utils import is_gpu_available

    if not is_gpu_available():
        pytest.skip("Test requires GPU but none is available")


def pytest_report_header(config):  # noqa: ARG001
    """Add lightweight backend information to the pytest header.

    Args:
        config: Pytest configuration object

    Returns:
        str: Header information about GPU availability
    """
    header_lines = [
        f"Artifex backend: {os.environ.get('ARTIFEX_BACKEND', 'unset')}",
        f"Deterministic test mode: {os.environ.get('ARTIFEX_DETERMINISTIC', '0')}",
    ]

    if not config.getoption("--artifex-probe-jax-runtime"):
        if hasattr(config, "_metadata"):
            config._metadata["Artifex backend"] = os.environ.get("ARTIFEX_BACKEND", "unset")
            config._metadata["JAX runtime probe"] = "deferred"
        header_lines.append(
            "JAX runtime probe: deferred (pass --artifex-probe-jax-runtime to inspect live devices)"
        )
        return header_lines

    from tests.utils.gpu_test_utils import get_jax_runtime_summary

    summary = get_jax_runtime_summary()
    visible_devices = ", ".join(summary.visible_devices) if summary.visible_devices else "none"
    header_lines.extend(
        [
            f"JAX default backend: {summary.default_backend or 'unavailable'}",
            f"JAX visible devices: {visible_devices}",
            f"GPU available for testing: {summary.gpu_available}",
        ]
    )
    if hasattr(config, "_metadata"):
        config._metadata["GPU available for testing"] = str(summary.gpu_available)
        config._metadata["JAX default backend"] = summary.default_backend or "unavailable"
        config._metadata["Artifex backend"] = os.environ.get("ARTIFEX_BACKEND", "unset")
        config._metadata["JAX runtime probe"] = "enabled"

    if summary.error:
        header_lines.append(f"JAX runtime probe error: {summary.error}")

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
