"""Canonical public factory surface for model creation."""

from importlib import import_module
from typing import Any


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "create_model": ("artifex.generative_models.factory.core", "create_model"),
    "create_model_with_extensions": (
        "artifex.generative_models.factory.core",
        "create_model_with_extensions",
    ),
    "ModelFactory": ("artifex.generative_models.factory.core", "ModelFactory"),
}

__all__ = ["ModelFactory", "create_model", "create_model_with_extensions"]


def __getattr__(name: str) -> Any:
    """Load supported factory exports lazily on first access."""
    try:
        module_path, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    return getattr(import_module(module_path), attr_name)


def __dir__() -> list[str]:
    """Keep introspection aligned with the supported factory surface."""
    return sorted(__all__)
