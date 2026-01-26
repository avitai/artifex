"""Extension system for generative models."""

from artifex.generative_models.extensions.base import (
    ConstraintExtension,
    ExtensionDict,
    ModelExtension,
)

# Import protein extensions
from artifex.generative_models.extensions.protein import (
    BondAngleExtension,
    BondLengthExtension,
    create_protein_extensions,
    ProteinMixinExtension,
)


__all__ = [
    # Base extensions
    "ExtensionDict",
    "ModelExtension",
    "ConstraintExtension",
    # Protein extensions
    "BondAngleExtension",
    "BondLengthExtension",
    "ProteinMixinExtension",
    "create_protein_extensions",
]
