"""Runtime-oriented device helpers for generative-model workflows."""

from __future__ import annotations

from ...core.device_manager import (
    DeviceCapabilities,
    DeviceManager,
    DeviceType,
    get_default_device,
    get_device_manager,
    has_gpu,
    print_device_info,
)
from ...core.device_testing import (
    print_test_results,
    run_device_tests,
    TestResult,
    TestSeverity,
    TestSuite,
)


__all__ = [
    "DeviceCapabilities",
    "DeviceManager",
    "DeviceType",
    "get_default_device",
    "get_device_manager",
    "has_gpu",
    "print_device_info",
    "TestResult",
    "TestSeverity",
    "TestSuite",
    "print_test_results",
    "run_device_tests",
    "verify_device_setup",
    "get_recommended_batch_size",
]


def verify_device_setup(critical_only: bool = False) -> bool:
    """Run the device diagnostics suite and return its health verdict."""
    suite = run_device_tests(critical_only=critical_only)
    return suite.is_healthy


def get_recommended_batch_size(model_params: int, base_batch_size: int = 32) -> int:
    """Return a simple runtime-aware batch-size heuristic."""
    manager = get_device_manager()
    multiplier = 1.0

    if not manager.has_gpu:
        multiplier *= 0.25

    if model_params > 1e8:
        multiplier *= 0.5
    elif model_params < 1e6:
        multiplier *= 2.0

    return max(1, int(base_batch_size * multiplier))
