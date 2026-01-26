"""Base fixtures providing common testing patterns and utilities.

This module provides the foundational fixtures that are shared across all test modules,
including RNG management, standard data shapes, device configuration, and common
testing utilities.
"""

import jax
import numpy as np
import pytest
from flax import nnx


@pytest.fixture(scope="session")
def base_rngs():
    """Shared RNG for deterministic testing across all test sessions.

    Returns:
        nnx.Rngs: Deterministic RNG state with seed 42
    """
    return nnx.Rngs(42)


@pytest.fixture
def batch_sizes():
    """Return standard batch sizes for testing.

    Provides a range of batch sizes used for testing model functionality
    at different scales.
    """
    return [1, 2, 4, 8, 16, 32]


@pytest.fixture
def small_batch_sizes():
    """Small batch sizes for quick testing.

    Returns:
        list: List of small batch sizes [1, 4, 8]
    """
    return [1, 4, 8]


@pytest.fixture
def standard_shapes():
    """Return common shapes for testing.

    Provides standardized input shapes for testing model functionality
    across different dimensions.
    """
    return {
        "1d": (16,),
        "2d": (28, 28),
        "3d": (32, 32, 3),
        "4d": (16, 16, 16, 1),
        "image_2d": (28, 28, 1),  # For diffusion models
    }


@pytest.fixture
def test_device():
    """CPU device for consistent testing.

    Returns:
        jax.Device: CPU device for testing
    """
    return jax.devices("cpu")[0]


@pytest.fixture
def standard_timesteps():
    """Return standard timestep values for diffusion models.

    Provides timestep values at different granularities for testing
    diffusion model functionality.
    """
    return [10, 50, 100, 1000]


@pytest.fixture
def noise_schedules():
    """Return standard noise schedules for diffusion models.

    Provides different noise schedule configurations for testing
    diffusion model behavior.
    """
    return ["linear", "cosine", "sigmoid"]


@pytest.fixture
def model_configs():
    """Return basic model configurations for testing.

    Provides a set of minimal but functional model configurations
    for testing different model architectures.
    """
    return {
        "vae": {
            "latent_dim": 16,
            "hidden_dims": [32, 64, 128],
            "activation": "relu",
        },
        "diffusion": {
            "timesteps": 100,
            "noise_schedule": "linear",
            "hidden_dims": [32, 64, 128],
        },
        "flow": {
            "num_layers": 4,
            "hidden_dims": [32, 64],
            "activation": "tanh",
        },
        "gan": {
            "latent_dim": 64,
            "generator_dims": [128, 256],
            "discriminator_dims": [256, 128],
        },
        "transformer": {
            "num_layers": 4,
            "num_heads": 4,
            "d_model": 128,
            "d_ff": 512,
        },
    }


@pytest.fixture
def tolerance_levels():
    """Return standard tolerance levels for numerical testing.

    Provides different precision thresholds for testing numerical
    operations across different computation types.
    """
    return {
        "float32": {
            "atol": 1e-5,
            "rtol": 1e-5,
        },
        "float64": {
            "atol": 1e-10,
            "rtol": 1e-10,
        },
        "bfloat16": {
            "atol": 1e-2,
            "rtol": 1e-2,
        },
        "float16": {
            "atol": 5e-3,
            "rtol": 5e-3,
        },
        # For testing with very low precision or where numerical stability is challenging
        "loose": {
            "atol": 1e-2,
            "rtol": 1e-2,
        },
        # For testing with very high precision
        "strict": {
            "atol": 1e-12,
            "rtol": 1e-12,
        },
        # For testing where only approximate equality is needed
        "approximate": {
            "atol": 0.1,
            "rtol": 0.1,
        },
    }


@pytest.fixture
def test_markers():
    """Test markers for categorizing tests.

    Returns:
        dict: Dictionary with test marker configurations
    """
    return {
        "gpu": pytest.mark.gpu,
        "slow": pytest.mark.slow,
        "integration": pytest.mark.integration,
        "benchmark": pytest.mark.benchmark,
    }


@pytest.fixture(scope="session")
def temp_artifact_dir(tmp_path_factory):
    """Temporary directory for test artifacts.

    Args:
        tmp_path_factory: pytest temporary path factory

    Returns:
        pathlib.Path: Path to temporary artifacts directory
    """
    return tmp_path_factory.mktemp("test_artifacts")


@pytest.fixture
def performance_thresholds():
    """Define performance thresholds for benchmark tests.

    Sets baseline performance expectations for different operations
    to catch performance regressions.
    """
    return {
        "inference": {
            "small_model": {
                "max_time_ms": 50,
                "max_memory_mb": 200,
            },
            "medium_model": {
                "max_time_ms": 200,
                "max_memory_mb": 500,
            },
            "large_model": {
                "max_time_ms": 1000,
                "max_memory_mb": 2000,
            },
        },
        "training": {
            "small_model": {
                "max_time_per_step_ms": 100,
                "max_memory_mb": 500,
            },
            "medium_model": {
                "max_time_per_step_ms": 500,
                "max_memory_mb": 2000,
            },
            "large_model": {
                "max_time_per_step_ms": 2000,
                "max_memory_mb": 8000,
            },
        },
        "data_loading": {
            "small_batch": {
                "max_time_ms": 10,
            },
            "medium_batch": {
                "max_time_ms": 50,
            },
            "large_batch": {
                "max_time_ms": 200,
            },
        },
    }


@pytest.fixture
def memory_limits():
    """Memory usage limits for different test scenarios.

    Returns:
        dict: Dictionary with memory limits in MB
    """
    return {
        "small_model": 100,  # Small models should use < 100MB
        "medium_model": 500,  # Medium models should use < 500MB
        "large_model": 2000,  # Large models should use < 2GB
        "test_data": 50,  # Test data should use < 50MB
    }


@pytest.fixture
def jax_config():
    """JAX configuration for testing.

    Returns:
        dict: Dictionary with JAX configuration settings
    """
    return {
        "jax_enable_x64": False,
        "jax_platforms": "cpu",
        "xla_python_client_mem_fraction": 0.75,
        "xla_python_client_preallocate": False,
    }


@pytest.fixture(autouse=True)
def setup_jax_environment(jax_config):
    """Automatically setup JAX environment for each test.

    Args:
        jax_config: JAX configuration fixture
    """
    # Set JAX configuration
    import os

    for key, value in jax_config.items():
        env_var = key.upper()
        os.environ[env_var] = str(value)

    # Clear JAX cache to ensure clean state
    jax.clear_caches()


@pytest.fixture
def assert_helpers():
    """Provide helper functions for common test assertions.

    Creates a set of utility functions to simplify common assertion
    patterns in tests.
    """

    def assert_array_properties(
        arr, shape=None, dtype=None, finite=True, bounds=None, positive=False
    ):
        """Check common array properties in tests.

        Args:
            arr: The array to check
            shape: Expected shape tuple
            dtype: Expected dtype
            finite: Whether all values should be finite
            bounds: Tuple of (min, max) expected values
            positive: Whether all values should be positive
        """

        if shape is not None:
            assert arr.shape == shape, f"Expected shape {shape}, got {arr.shape}"

        if dtype is not None:
            assert arr.dtype == dtype, f"Expected dtype {dtype}, got {arr.dtype}"

        if finite:
            assert np.all(np.isfinite(arr)), "Array contains non-finite values"

        if bounds is not None:
            min_val, max_val = bounds
            assert np.min(arr) >= min_val, f"Min value {np.min(arr)} below bound {min_val}"
            assert np.max(arr) <= max_val, f"Max value {np.max(arr)} above bound {max_val}"

        if positive:
            assert np.all(arr >= 0), "Array contains negative values"

    return {
        "assert_array_properties": assert_array_properties,
    }


@pytest.fixture
def timing_context():
    """Context manager for timing operations.

    Returns:
        function: Context manager function for timing
    """
    import time
    from contextlib import contextmanager

    @contextmanager
    def timer():
        start_time = time.time()
        times = {"start": start_time}
        yield times
        times["end"] = time.time()
        times["duration"] = times["end"] - times["start"]

    return timer


@pytest.fixture
def memory_monitor():
    """Memory monitoring utilities.

    Returns:
        dict: Dictionary with memory monitoring functions
    """
    import os

    try:
        import psutil

        has_psutil = True
    except ImportError:
        has_psutil = False

    def get_memory_usage():
        """Get current memory usage in MB."""
        if not has_psutil:
            return 0.0
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def memory_context():
        """Context manager for monitoring memory usage."""
        from contextlib import contextmanager

        @contextmanager
        def monitor():
            start_memory = get_memory_usage()
            memory_info = {"start": start_memory}
            yield memory_info
            memory_info["end"] = get_memory_usage()
            memory_info["delta"] = memory_info["end"] - memory_info["start"]

        return monitor()

    return {"get_memory_usage": get_memory_usage, "memory_context": memory_context}
