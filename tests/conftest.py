"""
Global pytest configuration and fixtures.

This file provides the main testing configuration for the artifex package,
including GPU testing support, nnxX compatibility checks, BlackJAX
integration controls, and comprehensive test infrastructure.

The test infrastructure follows a three-tier architecture:
1. Base fixtures: Common patterns, RNG management, standard configurations
2. Data fixtures: Fast, synthetic, and cached data generation strategies
3. Model fixtures: Standardized model configurations and utilities
"""

import os
from pathlib import Path

import pytest

# Import all base fixtures and data generators
from tests.artifex.fixtures.base import *  # noqa: F403, F401
from tests.artifex.fixtures.data_generators import (
    CachedDataManager,
    FastDataGenerator,
    SyntheticDataGenerator,
)


# Note: is_gpu_available is imported lazily in pytest_configure to avoid
# importing JAX before CUDA environment is set up by root conftest.py


# Import our custom hooks
pytest_plugins = ["tests.utils.pytest_hooks"]


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

        print("✓ Deterministic mode enabled for all tests (XLA flags set before JAX import)")

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

    # Register GPU availability with pytest metadata
    # Lazy import to avoid importing JAX before CUDA environment is set up
    from tests.utils.gpu_test_utils import is_gpu_available

    is_gpu = is_gpu_available()

    # Store metadata about GPU availability
    if hasattr(config, "_metadata"):
        config._metadata["GPU available for testing"] = str(is_gpu)
    elif hasattr(config, "stash"):
        if "metadata" not in config.stash:
            config.stash["metadata"] = {}
        config.stash["metadata"]["GPU available for testing"] = str(is_gpu)

    # Log GPU availability for visibility
    print(f"GPU available for testing: {is_gpu}")


def nnxx_compatible():
    """Check if JAX and NNX are compatible.

    This function verifies that both JAX and Flax NNX can be imported
    and are compatible with each other for testing purposes.

    Returns:
        bool: True if JAX and NNX versions are compatible, False otherwise.
    """
    try:
        import jax
        from flax import nnx

        # Return True if both imports work
        return bool(jax) and bool(nnx)
    except ImportError:
        return False


# Skip marker for tests requiring JAX and NNX compatibility
skip_if_incompatible = pytest.mark.skipif(
    not nnxx_compatible(),
    reason="JAX and NNX versions are incompatible (required dependencies not available)",
)


def should_skip_blackjax_tests():
    """Determine if BlackJAX tests should be skipped.

    Checks both environment variables and nnxX version compatibility
    to determine if BlackJAX-related tests should run.

    Returns:
        bool: True if BlackJAX tests should be skipped, False otherwise.
    """
    # Skip if explicitly disabled via environment variable
    if os.environ.get("ENABLE_BLACKJAX_TESTS", "0") != "1":
        return True

    # Skip if nnxX versions are incompatible
    return not nnxx_compatible()


# Skip marker for tests that require BlackJAX and compatible nnxX versions
skip_blackjax_tests = pytest.mark.skipif(
    should_skip_blackjax_tests(),
    reason="BlackJAX tests disabled or nnxX versions incompatible",
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
    """Add GPU availability information to pytest header.

    Args:
        config: Pytest configuration object

    Returns:
        str: Header information about GPU availability
    """
    from tests.utils.gpu_test_utils import is_gpu_available

    return f"GPU available for testing: {is_gpu_available()}"


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Modify test items after collection.

    This allows us to automatically skip tests marked with @pytest.mark.gpu
    when no GPU is available, and apply other test modifications.

    Args:
        config: Pytest configuration object
        items: List of collected test items
    """
    from tests.utils.gpu_test_utils import is_gpu_available

    if not is_gpu_available():
        skip_gpu = pytest.mark.skip(reason="Test requires GPU but none is available")
        for item in items:
            if "gpu" in item.keywords or "requires_gpu" in item.keywords:
                item.add_marker(skip_gpu)


# =====================================================================================
# DATA FIXTURES - Three-tier data generation strategy
# =====================================================================================


@pytest.fixture(scope="session")
def data_cache_manager():
    """Session-scoped data cache manager for complex test data."""
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
    return FastDataGenerator.random_image_batch(standard_shapes["image_2d"], batch_size=4)


@pytest.fixture
def fast_point_cloud_data():
    """Fast random point cloud data for smoke tests."""
    return FastDataGenerator.random_point_cloud(num_points=512, batch_size=4)


@pytest.fixture
def realistic_image_data(standard_shapes):
    """Realistic synthetic image data for integration tests."""
    return SyntheticDataGenerator.synthetic_images(
        "mnist_like", standard_shapes["image_2d"], batch_size=4
    )


@pytest.fixture
def realistic_point_cloud_data():
    """Realistic synthetic point cloud data for integration tests."""
    return SyntheticDataGenerator.synthetic_point_clouds("sphere", num_points=512, batch_size=4)


@pytest.fixture
def test_timesteps():
    """Generate test timesteps for diffusion models."""
    return FastDataGenerator.random_timesteps(max_timesteps=100, batch_size=8)


@pytest.fixture
def test_noise(standard_shapes):
    """Generate test noise arrays."""
    return FastDataGenerator.random_noise(standard_shapes["image_2d"], batch_size=8)


@pytest.fixture
def test_labels():
    """Generate test class labels."""
    return FastDataGenerator.random_labels(num_classes=10, batch_size=8)


# =====================================================================================
# PYTEST CLEANUP - Clear caches after test sessions
# =====================================================================================


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_caches():
    """Clean up test caches at the end of test session."""
    yield
    # Clear data caches
    CachedDataManager.clear_cache(disk_cache_only=False)

    # Clean up test artifacts directory if it exists
    artifacts_dir = Path("test_artifacts")
    if artifacts_dir.exists():
        import shutil

        shutil.rmtree(artifacts_dir, ignore_errors=True)
