"""Configuration for protein modality.

This module provides configuration utilities for the protein modality.
"""

from typing import Any

from artifex.generative_models.modalities.protein.modality import (
    ProteinModality,
)
from artifex.generative_models.modalities.registry import list_modalities, register_modality


def register_protein_modality(force_register: bool = False) -> None:
    """Register the protein modality with the modality registry.

    Args:
        force_register: If True, will register even if already registered,
            which is useful for tests that need a clean registry state.
    """
    # Check if already registered
    modalities = list_modalities()
    if "protein" in modalities and not force_register:
        # Already registered and not forcing, nothing to do
        return
    elif "protein" in modalities and force_register:
        # Need to clear existing registration before re-registering
        from artifex.generative_models.modalities.registry import _MODALITY_REGISTRY

        if "protein" in _MODALITY_REGISTRY:
            del _MODALITY_REGISTRY["protein"]

    # Now register
    register_modality("protein", ProteinModality)


def create_default_protein_config() -> dict[str, Any]:
    """Create a default configuration for protein models.

    Returns:
        A dictionary containing default configuration values for protein
        models.
    """
    return {
        "extensions": {
            "use_backbone_constraints": True,
            "bond_length_weight": 1.0,
            "bond_angle_weight": 0.5,
            "use_protein_mixin": True,
        },
        "backbone": {
            "use_protein_constraints": True,
        },
    }


# Auto-registration removed to prevent test failures
# register_protein_modality()
