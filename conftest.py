# conftest.py - Pytest configuration for Artifex

import os
import platform
import warnings
from pathlib import Path


# Detect platform BEFORE any other imports
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# Configure JAX for deterministic testing with float32 precision
# This must be set BEFORE any JAX import to ensure consistent behavior
# The artifex jax_config.py uses bfloat16 by default for performance,
# but tests require float32 for tight numerical tolerance checks
os.environ["ARTIFEX_MATMUL_PRECISION"] = "float32"

# Set TensorFlow environment variables BEFORE any TF import
# This prevents TensorFlow from hanging on macOS ARM64 during device detection
# See: https://github.com/tensorflow/tensorflow/issues/52138
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

if IS_MACOS:
    # macOS-specific settings to prevent TensorFlow hang on ARM64
    # These must be set BEFORE any TensorFlow import
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_METAL_DEVICE_SELECTOR"] = ""
    os.environ["TF_DISABLE_MLC_BRIDGE"] = "1"

import pytest


def setup_cuda_environment():
    """Set up CUDA environment variables for JAX.

    On macOS, CUDA is not available so this function only sets up
    CPU-compatible JAX settings.
    """
    # Skip CUDA setup on macOS - CUDA not available
    if IS_MACOS:
        # Ensure JAX uses CPU on macOS (Metal handled separately via JAX_PLATFORMS)
        if "JAX_PLATFORMS" not in os.environ:
            os.environ["JAX_PLATFORMS"] = "cpu"
        return

    # Set CUDA library path (Linux only)
    cuda_lib_path = "/usr/local/cuda/lib64"
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    if Path(cuda_lib_path).exists() and cuda_lib_path not in current_ld_path:
        if current_ld_path:
            new_ld_path = f"{cuda_lib_path}:{current_ld_path}"
        else:
            new_ld_path = cuda_lib_path
        os.environ["LD_LIBRARY_PATH"] = new_ld_path

    # Set additional CUDA environment variables
    os.environ["CUDA_ROOT"] = "/usr/local/cuda"
    os.environ["CUDA_HOME"] = "/usr/local/cuda"

    # JAX CUDA configuration - respect existing JAX_PLATFORMS setting
    if "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cuda,cpu"

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

    # Disable CUDA plugin validation to bypass cuSPARSE check
    os.environ["JAX_CUDA_PLUGIN_VERIFY"] = "false"
    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"


def pytest_configure(config):
    """Configure pytest environment with proper JAX/CUDA handling."""
    # Setup CUDA environment first, before any JAX imports
    setup_cuda_environment()

    # Suppress CUDA warnings and errors in test output
    warnings.filterwarnings("ignore", category=UserWarning, module="jax._src.xla_bridge")
    warnings.filterwarnings("ignore", message=".*cuSPARSE.*")
    warnings.filterwarnings("ignore", message=".*CUDA-enabled jaxlib.*")

    # Import JAX and configure after environment setup
    try:
        import jax

        # Force JAX to initialize with current environment
        # Respect the JAX_PLATFORMS environment variable
        current_platforms = os.environ.get("JAX_PLATFORMS", "cuda,cpu")
        jax.config.update("jax_platforms", current_platforms)

        # Check if CUDA is available
        try:
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.platform == "gpu"]
            cpu_devices = [d for d in devices if d.platform == "cpu"]

            config.addinivalue_line(
                "markers", f"gpu_available: GPU devices available: {len(gpu_devices) > 0}"
            )
            config.addinivalue_line("markers", f"devices: Available devices: {devices}")

            print(f"\nGPU available for testing: {len(gpu_devices) > 0}")
            if gpu_devices:
                print(f"GPU devices: {gpu_devices}")
            print(f"CPU devices: {cpu_devices}")

        except Exception as e:
            print(f"\nDevice detection failed: {e}")
            config.addinivalue_line("markers", "gpu_available: GPU devices available: False")

    except ImportError as e:
        print(f"\nJAX import failed: {e}")


@pytest.fixture
def device():
    """Provide a device fixture that works with both CPU and GPU."""
    import jax

    # Try to get GPU device first, fall back to CPU
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        if gpu_devices:
            return gpu_devices[0]
        else:
            return jax.devices("cpu")[0]
    except Exception:
        return jax.devices("cpu")[0]


@pytest.fixture
def rngs():
    """Provide RNG fixture for tests."""
    import flax.nnx as nnx

    return nnx.Rngs(0)


def pytest_runtest_setup(item):
    """Set up for each test run - ensure clean JAX state."""
    # Clear any JAX compilation cache to avoid issues
    try:
        import jax

        jax.clear_caches()

        # If this is a GPU test, ensure GPU memory is available
        if hasattr(item, "get_closest_marker"):
            gpu_marker = item.get_closest_marker("gpu")
            cuda_marker = item.get_closest_marker("cuda")

            if gpu_marker or cuda_marker:
                # Force garbage collection before GPU tests
                import gc

                gc.collect()

                # Verify GPU is available for GPU-marked tests
                try:
                    gpu_devices = jax.devices("gpu")
                    if not gpu_devices:
                        pytest.skip("GPU test skipped: No GPU devices available")
                except Exception:
                    pytest.skip("GPU test skipped: GPU not accessible")

    except Exception:
        pass


def pytest_runtest_teardown(item, nextitem):
    """Teardown after each test run - clean up GPU memory."""
    try:
        import jax

        # Clear JAX caches after each test
        jax.clear_caches()

        # Force garbage collection to free GPU memory
        import gc

        gc.collect()

        # If this was a GPU test, ensure memory cleanup
        if hasattr(item, "get_closest_marker"):
            gpu_marker = item.get_closest_marker("gpu")
            cuda_marker = item.get_closest_marker("cuda")

            if gpu_marker or cuda_marker:
                try:
                    # Additional GPU memory cleanup
                    import jax.numpy as jnp

                    # Force any pending operations to complete
                    dummy = jnp.array([1.0])
                    dummy.block_until_ready()
                    del dummy
                except Exception:
                    pass

    except Exception:
        pass
