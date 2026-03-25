"""Shared extension system with curated convenience exports.

The top-level `artifex.generative_models.extensions` barrel intentionally keeps
curated convenience exports for the registry/base types plus protein helpers.
Other shipped registry-backed family subpackages live under `.chemical`,
`.vision`, `.audio_processing`, and `.nlp`.
"""

from importlib import import_module
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from artifex.generative_models.extensions.base import (
        ConstraintExtension,
        ExtensionDict,
        ModelExtension,
    )
    from artifex.generative_models.extensions.protein import (
        BondAngleExtension,
        BondLengthExtension,
        create_protein_extensions,
        ProteinBackboneConstraint,
        ProteinDihedralConstraint,
        ProteinMixinExtension,
    )
    from artifex.generative_models.extensions.registry import (
        ExtensionsRegistry,
        ExtensionType,
        get_extensions_registry,
    )


_LAZY_EXPORTS: dict[str, str | tuple[str, str]] = {
    "ExtensionDict": ("artifex.generative_models.extensions.base", "ExtensionDict"),
    "ModelExtension": ("artifex.generative_models.extensions.base", "ModelExtension"),
    "ConstraintExtension": (
        "artifex.generative_models.extensions.base",
        "ConstraintExtension",
    ),
    "ExtensionsRegistry": (
        "artifex.generative_models.extensions.registry",
        "ExtensionsRegistry",
    ),
    "ExtensionType": (
        "artifex.generative_models.extensions.registry",
        "ExtensionType",
    ),
    "get_extensions_registry": (
        "artifex.generative_models.extensions.registry",
        "get_extensions_registry",
    ),
    "BondAngleExtension": (
        "artifex.generative_models.extensions.protein",
        "BondAngleExtension",
    ),
    "BondLengthExtension": (
        "artifex.generative_models.extensions.protein",
        "BondLengthExtension",
    ),
    "ProteinBackboneConstraint": (
        "artifex.generative_models.extensions.protein",
        "ProteinBackboneConstraint",
    ),
    "ProteinDihedralConstraint": (
        "artifex.generative_models.extensions.protein",
        "ProteinDihedralConstraint",
    ),
    "ProteinMixinExtension": (
        "artifex.generative_models.extensions.protein",
        "ProteinMixinExtension",
    ),
    "create_protein_extensions": (
        "artifex.generative_models.extensions.protein",
        "create_protein_extensions",
    ),
}


__all__ = [
    # Base extensions
    "ExtensionDict",
    "ModelExtension",
    "ConstraintExtension",
    "ExtensionsRegistry",
    "ExtensionType",
    "get_extensions_registry",
    # Protein extensions
    "BondAngleExtension",
    "BondLengthExtension",
    "ProteinBackboneConstraint",
    "ProteinDihedralConstraint",
    "ProteinMixinExtension",
    "create_protein_extensions",
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
