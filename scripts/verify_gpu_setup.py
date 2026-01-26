#!/usr/bin/env python
"""
GPU Verification and Diagnostics Tool for Artifex
===================================================

PURPOSE:
    Comprehensive GPU setup verification and diagnostics using Artifex's
    unified device management framework. Provides detailed information about
    GPU availability, CUDA configuration, and JAX compatibility.

USAGE:
    python scripts/verify_gpu_setup.py [OPTIONS]

    Options:
        --critical-only     Run only critical tests for quick validation
        --configure-first   Configure device manager before verification
        --help             Show help message

FEATURES:
    - Automatic GPU/CPU detection
    - JAX device configuration verification
    - Memory management testing
    - CUDA library path validation
    - Performance characteristic analysis
    - Detailed diagnostic output with recommendations

OUTPUT SECTIONS:
    1. Device Information - Hardware capabilities and configuration
    2. Comprehensive Testing - Full test suite execution
    3. Recommendations - Actionable steps for fixing issues

EXIT CODES:
    0 - System is healthy and GPU (if available) is properly configured
    1 - Critical issues detected that need resolution

DEPENDENCIES:
    - artifex.generative_models.core.device_manager
    - artifex.generative_models.core.device_testing
    - JAX and CUDA libraries (if GPU mode)

ENVIRONMENT:
    Respects JAX_PLATFORMS and CUDA-related environment variables
    from .env file.

Author: Artifex Team
License: MIT
"""

import sys
from pathlib import Path


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from artifex.generative_models.core.device_manager import (
        configure_for_generative_models,
        get_device_manager,
        MemoryStrategy,
        print_device_info,
    )
    from artifex.generative_models.core.device_testing import (
        print_test_results,
        run_comprehensive_device_tests,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root and have installed dependencies.")
    sys.exit(1)


def main():
    """Main verification entry point."""
    print("🔍 Artifex GPU Verification Suite")
    print("=" * 50)
    print("Foundation-first device testing and validation")
    print()

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--critical-only":
            run_critical_verification()
        elif sys.argv[1] == "--configure-first":
            configure_and_verify()
        elif sys.argv[1] == "--help":
            print_help()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print_help()
            sys.exit(1)
    else:
        run_full_verification()


def run_full_verification():
    """Run complete verification suite."""
    print("🚀 Running complete verification suite...")

    # Step 1: Show device information
    print("\n📊 Step 1: Device Information")
    print("-" * 30)
    print_device_info()

    # Step 2: Run comprehensive tests
    print("\n🧪 Step 2: Comprehensive Testing")
    print("-" * 30)
    suite = run_comprehensive_device_tests()
    print_test_results(suite)

    # Step 3: Provide recommendations
    print("\n💡 Step 3: Recommendations")
    print("-" * 30)
    provide_recommendations(suite)

    # Exit with appropriate code
    if suite.is_healthy:
        print("\n✅ Verification complete: System is healthy!")
        sys.exit(0)
    else:
        print("\n❌ Verification failed: Critical issues detected!")
        sys.exit(1)


def run_critical_verification():
    """Run only critical tests for quick validation."""
    print("🔴 Running critical tests only...")

    print_device_info()

    suite = run_comprehensive_device_tests(critical_only=True)
    print_test_results(suite)

    if suite.is_healthy:
        print("\n✅ Critical tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Critical tests failed!")
        sys.exit(1)


def configure_and_verify():
    """Configure for generative models and verify."""
    print("🎨 Configuring for generative models and verifying...")

    # Configure for generative models
    configure_for_generative_models(
        memory_strategy=MemoryStrategy.BALANCED, enable_mixed_precision=True
    )

    print("✅ Configuration applied!")
    print_device_info()

    # Run verification
    suite = run_comprehensive_device_tests()
    print_test_results(suite)

    sys.exit(0 if suite.is_healthy else 1)


def provide_recommendations(suite):
    """Provide recommendations based on test results."""
    manager = get_device_manager()

    if suite.is_healthy:
        print("🎉 Your system is optimally configured!")
        print()
        print("✅ All critical tests passed")
        print("✅ Device management is working correctly")
        print("✅ Neural network operations are functional")

        if manager.has_gpu:
            print(f"✅ GPU acceleration available ({manager.gpu_count} GPUs)")
            if manager.capabilities.supports_distributed:
                print("✅ Multi-GPU training supported")

        print()
        print("🚀 Ready for Artifex development!")
        print("   • Run generative model training")
        print("   • Use multi-GPU distributed training")
        print("   • Develop new models with confidence")

    else:
        print("⚠️  Issues detected that need attention:")
        print()

        for failure in suite.critical_failures:
            print(f"🔴 {failure.test_name}")
            print(f"   Error: {failure.error_message}")
            print()

        print("🔧 Recommended fixes:")

        if not manager.has_gpu:
            print("   • Install NVIDIA drivers and CUDA toolkit")
            print("   • Verify GPU is detected: nvidia-smi")
            print("   • Reinstall JAX with CUDA support")

        if any("computation" in f.test_name.lower() for f in suite.critical_failures):
            print("   • Check JAX installation:")
            print("     uv pip install 'jax[cuda12_local]==0.6.1' jaxlib==0.6.1")
            print("   • Verify environment variables are set correctly")

        if any("neural" in f.test_name.lower() for f in suite.critical_failures):
            print("   • Check Flax NNX installation:")
            print("     uv pip install flax==0.10.6")
            print("   • Verify model initialization patterns")

        print()
        print("💡 After fixing issues, re-run verification:")
        print("   python scripts/verify_gpu_setup.py")


def print_help():
    """Print help information."""
    print("Artifex GPU Verification - Foundation-first Testing")
    print("=" * 55)
    print()
    print("Usage: python scripts/verify_gpu_setup.py [options]")
    print()
    print("Options:")
    print("  (no args)           Complete verification suite")
    print("  --critical-only     Run critical tests only")
    print("  --configure-first   Configure for generative models first")
    print("  --help              Show this help message")
    print()
    print("Examples:")
    print("  python scripts/verify_gpu_setup.py")
    print("  python scripts/verify_gpu_setup.py --critical-only")
    print("  python scripts/verify_gpu_setup.py --configure-first")
    print()
    print("Test Categories:")
    print("  🔴 Critical    - Must pass for basic functionality")
    print("  🟡 Important  - Should pass for optimal performance")
    print("  🟢 Optional   - Nice to have, may fail on some systems")
    print()
    print("For detailed device information:")
    print("  python scripts/gpu_utils.py --comprehensive")


if __name__ == "__main__":
    main()
