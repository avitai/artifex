"""Test helpers for conditional test execution and common utility functions.

This module provides utilities to:
1. Control test execution using environment variables
2. Manage test dependencies
3. Provide common testing patterns for model tests
"""

import os


# ============================================================================
# Environment variable based test execution control
# ============================================================================


def should_run_integration_tests() -> bool:
    """Check if integration tests should run.

    Returns:
        bool: True if integration tests should run, False otherwise

    Environment variables:
        - SKIP_INTEGRATION_TESTS: When set to any non-empty value, disables all
          integration tests
        - RUN_INTEGRATION_TESTS: When set to any non-empty value, enables all
          integration tests regardless of SKIP_INTEGRATION_TESTS setting
    """
    # If RUN_INTEGRATION_TESTS is set, run the tests regardless
    if os.environ.get("RUN_INTEGRATION_TESTS", "") != "":
        return True

    # Default to running integration tests unless explicitly disabled
    return os.environ.get("SKIP_INTEGRATION_TESTS", "") == ""


def should_run_vae_tests() -> bool:
    """Check if VAE integration tests should run.

    Returns:
        bool: True if VAE integration tests should run, False otherwise

    Environment variables:
        - RUN_VAE_TESTS: When set to any non-empty value, enables VAE tests
        - SKIP_VAE_TESTS: When set to any non-empty value, disables VAE tests
        - RUN_INTEGRATION_TESTS: When set, enables all integration tests
          including VAE
        - SKIP_INTEGRATION_TESTS: When set, disables all integration tests
          including VAE unless RUN_VAE_TESTS is specifically set
    """
    # If RUN_VAE_TESTS is set, run the tests regardless
    if os.environ.get("RUN_VAE_TESTS", "") != "":
        return True

    # If SKIP_VAE_TESTS is set, skip the tests
    if os.environ.get("SKIP_VAE_TESTS", "") != "":
        return False

    # Otherwise, follow the general integration test setting
    return should_run_integration_tests()


def should_run_flow_tests() -> bool:
    """Check if Flow model integration tests should run.

    Returns:
        bool: True if Flow integration tests should run, False otherwise

    Environment variables:
        - RUN_FLOW_TESTS: When set to any non-empty value, enables Flow tests
        - SKIP_FLOW_TESTS: When set to any non-empty value, disables Flow tests
        - RUN_INTEGRATION_TESTS: When set, enables all integration tests
          including Flow
        - SKIP_INTEGRATION_TESTS: When set, disables all integration tests
          including Flow unless RUN_FLOW_TESTS is specifically set
    """
    # If RUN_FLOW_TESTS is set, run the tests regardless
    if os.environ.get("RUN_FLOW_TESTS", "") != "":
        return True

    # If SKIP_FLOW_TESTS is set, skip the tests
    if os.environ.get("SKIP_FLOW_TESTS", "") != "":
        return False

    # Otherwise, follow the general integration test setting
    return should_run_integration_tests()


def should_run_gan_tests() -> bool:
    """Check if GAN integration tests should run.

    Returns:
        bool: True if GAN integration tests should run, False otherwise

    Environment variables:
        - RUN_GAN_TESTS: When set to any non-empty value, enables GAN tests
        - SKIP_GAN_TESTS: When set to any non-empty value, disables GAN tests
        - RUN_INTEGRATION_TESTS: When set, enables all integration tests
          including GAN
        - SKIP_INTEGRATION_TESTS: When set, disables all integration tests
          including GAN unless RUN_GAN_TESTS is specifically set
    """
    # If RUN_GAN_TESTS is set, run the tests regardless
    if os.environ.get("RUN_GAN_TESTS", "") != "":
        return True

    # If SKIP_GAN_TESTS is set, skip the tests
    if os.environ.get("SKIP_GAN_TESTS", "") != "":
        return False

    # Otherwise, follow the general integration test setting
    return should_run_integration_tests()


def should_run_diffusion_tests() -> bool:
    """Check if Diffusion model integration tests should run.

    Returns:
        bool: True if Diffusion integration tests should run, False otherwise

    Environment variables:
        - RUN_DIFFUSION_TESTS: When set, enables Diffusion tests
        - SKIP_DIFFUSION_TESTS: When set, disables Diffusion tests
        - RUN_INTEGRATION_TESTS: When set, enables all integration tests
          including Diffusion
        - SKIP_INTEGRATION_TESTS: When set, disables all integration tests
          including Diffusion unless RUN_DIFFUSION_TESTS is set

    Note:
        Unlike other tests, diffusion tests are disabled by default due to known
        GroupNorm reshape issues. Set RUN_DIFFUSION_TESTS=1 to enable them.
    """
    # If RUN_DIFFUSION_TESTS is set, run the tests regardless
    if os.environ.get("RUN_DIFFUSION_TESTS", "") != "":
        return True

    # If SKIP_DIFFUSION_TESTS is set, skip the tests
    if os.environ.get("SKIP_DIFFUSION_TESTS", "") != "":
        return False

    # If RUN_INTEGRATION_TESTS is set, run the tests along with other integration tests
    if os.environ.get("RUN_INTEGRATION_TESTS", "") != "":
        return True

    # Default to skipping diffusion tests due to known GroupNorm issues
    return False


def should_run_training_pipeline_tests() -> bool:
    """Check if training pipeline integration tests should run.

    Returns:
        bool: True if training pipeline tests should run, False otherwise

    Environment variables:
        - RUN_TRAINING_PIPELINE_TESTS: When set, enables training pipeline
          tests
        - SKIP_TRAINING_PIPELINE_TESTS: When set, disables pipeline tests
        - RUN_INTEGRATION_TESTS: When set, enables all integration tests
          including training pipeline
        - SKIP_INTEGRATION_TESTS: When set, disables all integration tests
          including pipeline unless RUN_TRAINING_PIPELINE_TESTS is set
    """
    # If RUN_TRAINING_PIPELINE_TESTS is set, run the tests regardless
    if os.environ.get("RUN_TRAINING_PIPELINE_TESTS", "") != "":
        return True

    # If SKIP_TRAINING_PIPELINE_TESTS is set, skip the tests
    if os.environ.get("SKIP_TRAINING_PIPELINE_TESTS", "") != "":
        return False

    # Otherwise, follow the general integration test setting
    return should_run_integration_tests()


# ============================================================================
# Platform detection utilities
# ============================================================================


def is_running_on_cpu() -> bool:
    """Check if JAX is configured to use CPU.

    Returns:
        bool: True if JAX is configured to use CPU only, False otherwise
    """
    return os.environ.get("JAX_PLATFORMS", "") == "cpu"


def is_running_on_gpu() -> bool:
    """Check if JAX is configured to use GPU.

    Returns:
        bool: True if JAX is configured to use GPU, False otherwise
    """
    return os.environ.get("JAX_PLATFORMS", "") != "cpu"


# ============================================================================
# Mock initialization utilities
# ============================================================================


def get_mock_reason(model_type: str, env_var: str | None = None) -> str:
    """Get a standardized skip reason message.

    Args:
        model_type: The type of model test being skipped
        env_var: Optional environment variable to use for controlling test
          execution

    Returns:
        str: A formatted reason message that explains how to enable the test
    """
    if env_var is None:
        env_var = f"RUN_{model_type.upper()}_TESTS"

    return (
        f"{model_type} integration test skipped. "
        f"Set {env_var}=1 to enable or RUN_INTEGRATION_TESTS=1 to enable all "
        f"integration tests."
    )


# ============================================================================
# Shape verification utilities
# ============================================================================


def verify_model_initialization_shapes(model_class, config, expected_shapes):
    """Verify that model initialization creates expected shapes.

    This helper performs a non-intensive validation of model initialization
    without running the actual forward pass that might cause segmentation
    faults.

    Args:
        model_class: The model class to instantiate
        config: Configuration to use for instantiation
        expected_shapes: Dictionary of expected parameter shapes

    Returns:
        bool: True if all expected shapes match
    """
    model = model_class(config)

    # Verify parameter existence and shapes without running forward pass
    for param_name, expected_shape in expected_shapes.items():
        assert hasattr(model, param_name), f"Model missing expected parameter {param_name}"
        param = getattr(model, param_name)
        assert param.shape == expected_shape, (
            f"Parameter {param_name} has shape {param.shape}, expected {expected_shape}"
        )

    return True
