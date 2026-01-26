"""Core functionality for generative models.

This package provides the fundamental building blocks and utilities for
implementing various types of generative models, including distributions,
layers, metrics, and sampling methods.
"""

# Core submodules
from artifex.generative_models.core import (
    cli,
    configuration,
    distributions,
    evaluation,
    layers,
    losses,
    protocols,
    sampling,
)

# Direct imports for convenience
from artifex.generative_models.core.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    setup_checkpoint_manager,
)
from artifex.generative_models.core.device_testing import (
    DeviceTestRunner,
    print_test_results,
    run_device_tests,
)
from artifex.generative_models.core.gradient_checkpointing import (
    apply_remat,
    CHECKPOINT_POLICIES,
    resolve_checkpoint_policy,
)


__all__ = [
    # Modules
    "cli",
    "configuration",
    "distributions",
    "evaluation",
    "layers",
    "losses",
    "protocols",
    "sampling",
    # Functions
    "load_checkpoint",
    "save_checkpoint",
    "setup_checkpoint_manager",
    "CHECKPOINT_POLICIES",
    "apply_remat",
    "resolve_checkpoint_policy",
    "DeviceTestRunner",
    "print_test_results",
    "run_device_tests",
]
