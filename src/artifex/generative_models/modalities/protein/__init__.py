"""Protein modality helpers for generative models.

This package exposes the retained protein modality adapter and extension
helpers around the shared factory/runtime surface. `modality="protein"`
keeps the generic model family selected by the typed config and serves as the
typed protein extension bundle boundary; it does not swap in
`ProteinPointCloudModel` or `ProteinGraphModel` automatically.
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
