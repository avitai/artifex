"""Top-level Artifex package."""

# pyright: reportUnsupportedDunderAll=false

from importlib import import_module
from types import ModuleType


__all__ = ["generative_models"]


def __getattr__(name: str) -> ModuleType:
    """Load exported package modules lazily on first access."""
    if name == "generative_models":
        return import_module("artifex.generative_models")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Keep introspection aligned with the documented export surface."""
    return sorted(__all__)
