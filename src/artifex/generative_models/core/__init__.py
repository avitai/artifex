"""Core functionality for generative models."""

from importlib import import_module
from typing import Any


_LAZY_EXPORTS: dict[str, str | tuple[str, str]] = {
    "cli": "artifex.generative_models.core.cli",
    "configuration": "artifex.generative_models.core.configuration",
    "distributions": "artifex.generative_models.core.distributions",
    "evaluation": "artifex.generative_models.core.evaluation",
    "layers": "artifex.generative_models.core.layers",
    "losses": "artifex.generative_models.core.losses",
    "protocols": "artifex.generative_models.core.protocols",
    "sampling": "artifex.generative_models.core.sampling",
    "load_checkpoint": (
        "artifex.generative_models.core.checkpointing",
        "load_checkpoint",
    ),
    "save_checkpoint": (
        "artifex.generative_models.core.checkpointing",
        "save_checkpoint",
    ),
    "setup_checkpoint_manager": (
        "artifex.generative_models.core.checkpointing",
        "setup_checkpoint_manager",
    ),
    "CHECKPOINT_POLICIES": (
        "artifex.generative_models.core.gradient_checkpointing",
        "CHECKPOINT_POLICIES",
    ),
    "apply_remat": (
        "artifex.generative_models.core.gradient_checkpointing",
        "apply_remat",
    ),
    "resolve_checkpoint_policy": (
        "artifex.generative_models.core.gradient_checkpointing",
        "resolve_checkpoint_policy",
    ),
    "DeviceManager": (
        "artifex.generative_models.core.device_manager",
        "DeviceManager",
    ),
    "print_test_results": (
        "artifex.generative_models.core.device_testing",
        "print_test_results",
    ),
    "run_device_tests": (
        "artifex.generative_models.core.device_testing",
        "run_device_tests",
    ),
}


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
    "DeviceManager",
    "print_test_results",
    "run_device_tests",
]


def __getattr__(name: str) -> Any:
    """Load exported modules and symbols lazily on first attribute access."""
    try:
        export = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    if isinstance(export, tuple):
        module_path, attr_name = export
        return getattr(import_module(module_path), attr_name)

    return import_module(export)


def __dir__() -> list[str]:
    """Keep introspection aligned with the documented export surface."""
    return sorted(__all__)
