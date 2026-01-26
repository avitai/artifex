"""
Device utilities for artifex generative models.

This module provides a clean interface to the comprehensive device management system.
All functionality has been moved to the core device management architecture.
"""

# Import the comprehensive device management system
from ...core.device_manager import (
    configure_for_generative_models,
    DeviceCapabilities,
    DeviceConfiguration,
    DeviceManager,
    DeviceType,
    get_default_device,
    get_device_manager,
    has_gpu,
    MemoryStrategy,
    print_device_info,
)

# Import testing framework
from ...core.device_testing import (
    DeviceTestRunner,
    print_test_results,
    run_comprehensive_device_tests,
    TestResult,
    TestSeverity,
    TestSuite,
)


__all__ = [
    # Core device management
    "DeviceCapabilities",
    "DeviceConfiguration",
    "DeviceManager",
    "DeviceType",
    "MemoryStrategy",
    "configure_for_generative_models",
    "get_default_device",
    "get_device_manager",
    "has_gpu",
    "print_device_info",
    # Testing framework
    "DeviceTestRunner",
    "TestResult",
    "TestSeverity",
    "TestSuite",
    "print_test_results",
    "run_comprehensive_device_tests",
]


# Convenience functions for backward compatibility
def setup_device_for_training(memory_strategy: str = "balanced") -> DeviceManager:
    """Setup device configuration optimized for training."""
    strategy_map = {
        "conservative": MemoryStrategy.CONSERVATIVE,
        "balanced": MemoryStrategy.BALANCED,
        "aggressive": MemoryStrategy.AGGRESSIVE,
    }

    strategy = strategy_map.get(memory_strategy, MemoryStrategy.BALANCED)
    return configure_for_generative_models(memory_strategy=strategy)


def verify_device_setup(critical_only: bool = False) -> bool:
    """Verify device setup is working correctly."""
    suite = run_comprehensive_device_tests(critical_only=critical_only)
    return suite.is_healthy


def get_recommended_batch_size(model_params: int, base_batch_size: int = 32) -> int:
    """Get recommended batch size based on model size and available memory."""
    manager = get_device_manager()

    if not manager.has_gpu:
        return max(1, base_batch_size // 4)  # Smaller batches for CPU

    # Adjust based on memory strategy
    if manager.config.memory_strategy == MemoryStrategy.CONSERVATIVE:
        multiplier = 0.5
    elif manager.config.memory_strategy == MemoryStrategy.AGGRESSIVE:
        multiplier = 1.5
    else:  # BALANCED
        multiplier = 1.0

    # Adjust based on model size
    if model_params > 1e8:  # > 100M params
        multiplier *= 0.5
    elif model_params < 1e6:  # < 1M params
        multiplier *= 2.0

    return max(1, int(base_batch_size * multiplier))
