"""artifex generative models package."""

# pyright: reportUnsupportedDunderAll=false

from importlib import import_module
from types import ModuleType


_LAZY_EXPORTS = {
    "core": "artifex.generative_models.core",
    "extensions": "artifex.generative_models.extensions",
    "models": "artifex.generative_models.models",
    "scaling": "artifex.generative_models.scaling",
    "utils": "artifex.generative_models.utils",
    "jax_config": "artifex.generative_models.core.jax_config",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> ModuleType:
    """Load exported subpackages lazily on first attribute access."""
    try:
        module_path = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return import_module(module_path)


def __dir__() -> list[str]:
    """Keep introspection aligned with the documented export surface."""
    return sorted(__all__)
