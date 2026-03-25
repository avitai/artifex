"""Model implementations for generative models."""

from importlib import import_module
from typing import Any


_LAZY_EXPORTS: dict[str, str | tuple[str, str]] = {
    "diffusion": "artifex.generative_models.models.diffusion",
    "geometric": "artifex.generative_models.models.geometric",
    "vae": "artifex.generative_models.models.vae",
}


__all__ = [
    # Submodules
    "diffusion",
    "geometric",
    "vae",
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
