"""Pytest configuration for distributed training tests.

This conftest.py handles multi-device CPU emulation for testing distributed
training functionality. Since XLA_FLAGS must be set BEFORE JAX initializes,
we use a subprocess approach on CPU to ensure proper device emulation.

Key behaviors:
- GPU: Multi-device tests skip (only 1 GPU available typically)
- CPU: Tests are re-run in subprocess with XLA_FLAGS for multi-device emulation

The subprocess approach is necessary because:
1. XLA_FLAGS must be set before JAX initializes
2. Pytest's global conftest.py imports JAX before this conftest.py loads
3. Running tests in a subprocess allows us to set XLA_FLAGS first

References:
- https://github.com/jax-ml/jax/discussions/21670
- https://github.com/jax-ml/jax/discussions/13576
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


# CRITICAL: Set XLA_FLAGS early in subprocess BEFORE JAX initializes
# This must happen at module load time, before any functions are called
_IN_MULTIDEVICE_SUBPROCESS = os.environ.get("_ARTIFEX_MULTIDEVICE_SUBPROCESS") == "1"

if _IN_MULTIDEVICE_SUBPROCESS:
    _existing_flags = os.environ.get("XLA_FLAGS", "")
    _cpu_count = os.environ.get("JAX_CPU_DEVICE_COUNT", "4")
    _new_flag = f"--xla_force_host_platform_device_count={_cpu_count}"
    if _new_flag not in _existing_flags:
        os.environ["XLA_FLAGS"] = f"{_existing_flags} {_new_flag}".strip()


def _get_jax_backend_and_device_count() -> tuple[str, int]:
    """Get JAX backend and device count.

    Uses lazy import to avoid initializing JAX too early.
    """
    import jax

    return jax.default_backend(), jax.device_count()


@pytest.fixture
def skip_if_single_device():
    """Skip test if only one device is available.

    This fixture is used by multi-device tests via @pytest.mark.usefixtures.
    On GPU with single device, tests are skipped.
    On CPU with subprocess emulation, tests run with multiple devices.
    """
    import jax

    if jax.device_count() < 2:
        pytest.skip("Test requires at least 2 devices")


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest for distributed tests.

    On CPU without multi-device emulation, schedules a subprocess run.
    """
    if _IN_MULTIDEVICE_SUBPROCESS:
        # We're in subprocess with multi-device emulation - run normally
        return

    # Check if we need subprocess (CPU with single device)
    backend, device_count = _get_jax_backend_and_device_count()

    if backend == "cpu" and device_count == 1:
        # Store flag to trigger subprocess run in pytest_sessionfinish
        config._needs_multidevice_subprocess = True  # type: ignore[attr-defined]
    else:
        config._needs_multidevice_subprocess = False  # type: ignore[attr-defined]


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Modify test collection for distributed tests.

    If we're on CPU with single device and NOT in subprocess,
    mark tests to skip (they will run in subprocess later).
    """
    if _IN_MULTIDEVICE_SUBPROCESS:
        # In subprocess - let tests run normally
        return

    if not getattr(config, "_needs_multidevice_subprocess", False):
        # On GPU or already have multiple devices - run normally
        return

    # On CPU with single device - mark for subprocess run
    skip_marker = pytest.mark.skip(reason="Will run in subprocess with multi-device CPU emulation")

    distributed_tests = []
    for item in items:
        # Check if this is a distributed test (in this directory)
        if "/distributed/" in str(item.fspath):
            item.add_marker(skip_marker)
            distributed_tests.append(item)

    if distributed_tests:
        # Store unique test file paths for subprocess run
        config._distributed_test_paths = list(  # type: ignore[attr-defined]
            {str(item.fspath) for item in distributed_tests}
        )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Run distributed tests in subprocess after main session.

    This hook is called after all tests complete. If we're on CPU with single
    device, runs distributed tests in a subprocess with multi-device emulation.

    Args:
        session: The pytest session object
        exitstatus: The exit status from the main test run (unused but required)
    """
    del exitstatus  # Unused, but required by pytest hook signature

    config = session.config

    if _IN_MULTIDEVICE_SUBPROCESS:
        return

    if not getattr(config, "_needs_multidevice_subprocess", False):
        return

    test_paths = getattr(config, "_distributed_test_paths", [])
    if not test_paths:
        return

    print("\n" + "=" * 70)
    print("Running distributed tests with multi-device CPU emulation...")
    print("=" * 70 + "\n")

    # Set up environment for subprocess
    env = os.environ.copy()
    cpu_device_count = env.get("JAX_CPU_DEVICE_COUNT", "4")

    env["_ARTIFEX_MULTIDEVICE_SUBPROCESS"] = "1"
    env["JAX_PLATFORMS"] = "cpu"  # Force CPU backend

    # Build Python script that:
    # 1. Sets XLA_FLAGS before any imports
    # 2. Imports JAX to initialize it with multi-device config
    # 3. THEN imports pytest (which loads global conftest that also imports JAX)
    # This order is critical because JAX is already initialized when pytest loads
    test_paths_str = ", ".join(f'"{p}"' for p in test_paths)
    python_script = f"""
import os

# Step 1: Set XLA_FLAGS before any imports
existing_flags = os.environ.get("XLA_FLAGS", "")
new_flag = "--xla_force_host_platform_device_count={cpu_device_count}"
if new_flag not in existing_flags:
    os.environ["XLA_FLAGS"] = f"{{existing_flags}} {{new_flag}}".strip()

# Step 2: Import JAX first to initialize with multi-device emulation
# This MUST happen before pytest imports global conftest which imports JAX
import jax
_ = jax.device_count()  # Ensure JAX is fully initialized

# Step 3: Now import and run pytest
import sys
sys.exit(__import__("pytest").main([{test_paths_str}, "-v", "--tb=short", "--no-cov"]))
"""

    # Run subprocess with inline Python script
    cmd = [sys.executable, "-c", python_script]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # Print captured output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"\nDistributed tests failed with exit code {result.returncode}")
        session.exitstatus = result.returncode
