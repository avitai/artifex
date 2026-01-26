#!/usr/bin/env python
"""
Modern GPU utilities for Artifex - Foundation-first approach.

This script provides a clean interface to our comprehensive device management system,
completely replacing the old scattered GPU utilities with a unified architecture.
"""

import sys
from pathlib import Path


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from artifex.generative_models.core.device_manager import (
    configure_for_generative_models,
    get_device_manager,
    MemoryStrategy,
    print_device_info,
)
from artifex.generative_models.core.device_testing import (
    print_test_results,
    run_device_tests,
)


def main():
    """Main entry point for GPU utilities."""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "--detailed":
            print_comprehensive_info()
        elif command == "--test":
            run_tests()
        elif command == "--test-critical":
            run_critical_tests()
        elif command == "--configure-generative":
            configure_generative()
        elif command == "--help":
            print_help()
        else:
            print(f"Unknown command: {command}")
            print_help()
    else:
        print_quick_status()


def print_quick_status():
    """Print quick device status."""
    manager = get_device_manager()

    print("🔍 Artifex GPU Status")
    print("=" * 30)
    print(f"GPU Available: {manager.has_gpu}")
    print(f"Device Count: {manager.device_count}")
    print(f"GPU Count: {manager.gpu_count}")
    print(f"Memory Strategy: {manager.config.memory_strategy.value}")
    print(f"Memory Fraction: {manager.config.memory_fraction}")

    if manager.capabilities.cuda_version:
        print(f"CUDA Version: {manager.capabilities.cuda_version}")

    print("\nRun with --detailed for detailed information")
    print("Run with --test for comprehensive testing")


def print_comprehensive_info():
    """Print comprehensive device information."""
    print_device_info()

    # Additional configuration details
    manager = get_device_manager()
    info = manager.get_device_info()

    print("\n📋 Configuration Details:")
    print(f"Platform Priority: {manager.config.platform_priority}")
    print(f"Enable X64: {manager.config.enable_x64}")
    print(f"Enable JIT: {manager.config.enable_jit}")

    print("\n🎯 Available Devices:")
    for device in info["jax_devices"]:
        print(f"  • {device}")

    if manager.capabilities.supports_distributed:
        print(f"\n🔗 Distributed Training: Supported ({manager.gpu_count} GPUs)")
    else:
        print("\n🔗 Distributed Training: Not available")


def run_tests():
    """Run comprehensive device tests."""
    print("🧪 Running comprehensive device tests...")
    suite = run_device_tests()
    print_test_results(suite)

    # Exit with appropriate code
    sys.exit(0 if suite.is_healthy else 1)


def run_critical_tests():
    """Run only critical device tests."""
    print("🔴 Running critical device tests only...")
    suite = run_device_tests(critical_only=True)
    print_test_results(suite)

    # Exit with appropriate code
    sys.exit(0 if suite.is_healthy else 1)


def configure_generative():
    """Configure device manager for generative models."""
    print("🎨 Configuring for generative models...")

    configure_for_generative_models(
        memory_strategy=MemoryStrategy.BALANCED, enable_mixed_precision=True
    )

    print("✅ Configuration complete!")
    print_device_info()


def print_help():
    """Print help information."""
    print("Artifex GPU Utilities - Foundation-first Device Management")
    print("=" * 60)
    print()
    print("Usage: python scripts/gpu_utils.py [command]")
    print()
    print("Commands:")
    print("  (no args)              Quick device status")
    print("  --detailed             Detailed device information")
    print("  --test                 Run comprehensive device tests")
    print("  --test-critical        Run critical tests only")
    print("  --configure-generative Configure for generative models")
    print("  --help                 Show this help message")
    print()
    print("Examples:")
    print("  python scripts/gpu_utils.py")
    print("  python scripts/gpu_utils.py --detailed")
    print("  python scripts/gpu_utils.py --test")
    print()
    print("For advanced configuration, use the device manager directly:")
    print("  from artifex.generative_models.core.device_manager import *")


if __name__ == "__main__":
    main()
