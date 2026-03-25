"""Configuration helpers for the protein modality."""

from artifex.generative_models.core.configuration import (
    ProteinExtensionConfig,
    ProteinExtensionsConfig,
    ProteinMixinConfig,
)
from artifex.generative_models.modalities.protein.modality import ProteinModality
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


def create_default_protein_config() -> ProteinExtensionsConfig:
    """Create the canonical default protein extension bundle.

    Returns:
        Typed default bundle for protein extensions.
    """
    return ProteinExtensionsConfig(
        name="default_protein_extensions",
        bond_length=ProteinExtensionConfig(
            name="bond_length",
            weight=1.0,
            bond_length_weight=1.0,
            ideal_bond_lengths={
                "N-CA": 1.45,
                "CA-C": 1.52,
                "C-N": 1.33,
            },
        ),
        bond_angle=ProteinExtensionConfig(
            name="bond_angle",
            weight=0.5,
            bond_angle_weight=0.5,
            ideal_bond_angles={
                "CA-C-N": 2.025,
                "C-N-CA": 2.11,
                "N-CA-C": 1.94,
            },
        ),
        backbone=ProteinExtensionConfig(
            name="backbone",
            weight=1.0,
            bond_length_weight=1.0,
            bond_angle_weight=0.5,
        ),
        mixin=ProteinMixinConfig(
            name="protein_mixin",
            weight=1.0,
            embedding_dim=16,
            num_aa_types=21,
        ),
    )


# Auto-registration removed to prevent test failures
# register_protein_modality()
