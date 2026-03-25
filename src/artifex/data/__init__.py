"""Narrow top-level data namespace for retained Artifex datasets.

General data loading and pipeline composition live in Datarax and in
modality-local dataset helpers. The only retained concrete top-level data
subpackage is `artifex.data.protein`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from . import protein


__all__ = ["protein"]


def __getattr__(name: str) -> Any:
    """Resolve the retained data subpackages lazily."""
    if name == "protein":
        return import_module(f"{__name__}.protein")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Keep interactive discovery aligned with the curated public surface."""
    return sorted(set(globals()) | set(__all__))
