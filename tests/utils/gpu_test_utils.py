"""Utilities for GPU-aware testing."""

import os

import jax
import pytest


def is_gpu_available():
    """Check if a GPU is available for testing.

    This function checks multiple ways to determine if a GPU is available:
    1. Checks if GPU is explicitly disabled via JAX_PLATFORMS environment variable
    2. Tries to get GPU devices via JAX
    3. Checks for CUDA availability via JAX
    4. Falls back to hardware detection if JAX fails

    Note: For some CUDA setups, you may need to set JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
    in your environment before running tests.

    Returns:
        bool: True if GPU is available and enabled, False otherwise
    """
    # Check if GPU is explicitly disabled via environment variable
    if os.environ.get("JAX_PLATFORMS", "") == "cpu":
        return False

    # Method 1: Try to get GPU devices via JAX
    try:
        gpu_devices = jax.devices("gpu")
        if len(gpu_devices) > 0:
            return True
    except Exception:
        pass

    # Method 2: Check if CUDA is available through JAX
    try:
        if getattr(jax.lib, "have_cuda", False):
            return True
    except Exception:
        pass

    # Method 3: Check if JAX default backend is GPU
    try:
        if jax.default_backend() in ["gpu", "cuda"]:
            return True
    except Exception:
        pass

    # Method 4: Fall back to hardware detection using nvidia-smi
    # This helps detect when hardware is available but JAX config has issues
    try:
        import subprocess

        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and "GPU" in result.stdout:
            # Hardware is available but JAX might need configuration
            # Check if CUDA constraints bypass might help
            skip_check = os.environ.get("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "0")
            if skip_check == "1":
                # If bypass is already set and we still can't access via JAX,
                # the GPU is likely having other issues
                return False
            else:
                # Hardware detected but JAX access failed - might need bypass
                # Return True but tests should handle the environment setup
                return True
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # GPU not detected by any method
    return False


def requires_gpu(f):
    """Decorator to mark tests that require a GPU.

    This decorator will skip the test if no GPU is available.

    Args:
        f: Test function to decorate

    Returns:
        Decorated function that will be skipped if no GPU is available
    """
    return pytest.mark.skipif(
        not is_gpu_available(),
        reason="Test requires GPU but none is available or GPU is disabled via JAX_PLATFORMS=cpu",
    )(f)


def skip_on_gpu(f):
    """Decorator to mark tests that should be skipped when running on GPU.

    This is useful for tests that are known to cause issues on GPU.

    Args:
        f: Test function to decorate

    Returns:
        Decorated function that will be skipped if GPU is available and enabled
    """
    return pytest.mark.skipif(
        is_gpu_available(), reason="Test is known to cause issues on GPU, skipping"
    )(f)


def gpu_test_fixture():
    """Pytest fixture that ensures GPU is available.

    Used to skip entire test classes that require GPU.

    Example:
        class TestGPUFeatures:
            pytestmark = pytest.mark.usefixtures("gpu_test_fixture")
    """
    if not is_gpu_available():
        pytest.skip("Test requires GPU but none is available")
