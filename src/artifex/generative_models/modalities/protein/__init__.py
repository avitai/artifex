"""Protein modality components for generative models.

This package provides protein-specific components that adapt generic model
architectures to work with protein structure data.
"""

from artifex.generative_models.modalities.protein.adapters import (
    ProteinDiffusionAdapter,
    ProteinGeometricAdapter,
    ProteinModelAdapter,
)
from artifex.generative_models.modalities.protein.config import (
    create_default_protein_config,
    register_protein_modality,
)
from artifex.generative_models.modalities.protein.modality import (
    ProteinModality,
)
from artifex.generative_models.modalities.protein.utils import (
    get_protein_adapter,
)


__all__ = [
    # Modality
    "ProteinModality",
    # Adapters
    "ProteinModelAdapter",
    "ProteinGeometricAdapter",
    "ProteinDiffusionAdapter",
    "get_protein_adapter",
    # Configuration
    "create_default_protein_config",
    "register_protein_modality",
]

# We'll rely on explicit registration in tests to avoid
# registering multiple times during test runs.
# Do not register automatically when the module is imported.
# The following line is commented out to prevent test failures.
# register_protein_modality()
