"""Tests for protein modality implementation.

This module tests the protein modality class that provides protein-specific
functionality to generative models.
"""

import importlib
from collections.abc import Mapping

import pytest

from artifex.generative_models.core.configuration import (
    ProteinExtensionConfig,
    ProteinExtensionsConfig,
    ProteinMixinConfig,
)

# Import registry functionality and modality classes
from artifex.generative_models.modalities.protein import ProteinModality
from artifex.generative_models.modalities.protein.config import (
    create_default_protein_config,
    register_protein_modality,
)
from artifex.generative_models.modalities.registry import _MODALITY_REGISTRY


@pytest.fixture(autouse=True)
def save_registry():
    """Isolate protein-modality registry setup per test."""
    saved = dict(_MODALITY_REGISTRY)
    _MODALITY_REGISTRY.clear()
    register_protein_modality(force_register=True)
    try:
        yield
    finally:
        _MODALITY_REGISTRY.clear()
        _MODALITY_REGISTRY.update(saved)


def test_protein_modality_init():
    """Test initializing a ProteinModality."""
    modality = ProteinModality()

    # Check basic properties
    assert modality.name == "protein"
    assert hasattr(modality, "get_adapter")
    assert hasattr(modality, "get_extensions")


def test_protein_package_surface_omits_phantom_helpers():
    """The protein package should not advertise helper exports that do not exist."""
    package = importlib.import_module("artifex.generative_models.modalities.protein")

    assert "ProteinEvaluator" not in package.__all__
    assert "ProteinRepresentation" not in package.__all__
    assert not hasattr(package, "ProteinEvaluator")
    assert not hasattr(package, "ProteinRepresentation")


def test_protein_modality_get_adapter():
    """Test getting a protein adapter."""
    modality = ProteinModality()

    # Get default adapter
    adapter = modality.get_adapter()
    assert adapter.modality == "protein"
    assert adapter.name == "protein_model"

    # Get geometric adapter
    geometric_adapter = modality.get_adapter("geometric")
    assert geometric_adapter.modality == "protein"
    assert geometric_adapter.name == "protein_geometric"

    # Get diffusion adapter
    diffusion_adapter = modality.get_adapter("diffusion")
    assert diffusion_adapter.modality == "protein"
    assert diffusion_adapter.name == "protein_diffusion"


def test_protein_modality_get_adapter_unknown():
    """Test getting an unknown adapter from protein modality."""
    modality = ProteinModality()

    with pytest.raises(ValueError):
        modality.get_adapter("unknown_adapter")


def test_protein_modality_get_extensions():
    """Test getting extensions from protein modality."""
    modality = ProteinModality()

    # Get extensions with default config
    extensions = modality.get_extensions(ProteinExtensionsConfig(name="empty"))
    assert isinstance(extensions, Mapping)  # nnx.Dict is a Mapping, not dict
    assert len(extensions) == 0  # Empty config = no extensions

    # Get extensions with backbone constraints
    config = ProteinExtensionsConfig(
        name="protein_extensions",
        bond_length=ProteinExtensionConfig(
            name="bond_length",
            weight=2.0,
            bond_length_weight=2.0,
        ),
        bond_angle=ProteinExtensionConfig(
            name="bond_angle",
            weight=1.5,
            bond_angle_weight=1.5,
        ),
    )
    extensions = modality.get_extensions(config)
    assert isinstance(extensions, Mapping)
    assert "bond_length" in extensions
    assert "bond_angle" in extensions
    assert extensions["bond_length"].weight == 2.0
    assert extensions["bond_angle"].weight == 1.5

    # Get extensions with protein mixin
    config = ProteinExtensionsConfig(
        name="protein_extensions",
        mixin=ProteinMixinConfig(
            name="protein_mixin",
            embedding_dim=32,
        ),
    )
    extensions = modality.get_extensions(config)
    assert isinstance(extensions, Mapping)
    assert "protein_mixin" in extensions
    assert extensions["protein_mixin"].embedding_dim == 32


def test_protein_modality_get_extensions_all():
    """Test getting all extensions from protein modality."""
    modality = ProteinModality()

    # Get all extensions
    config = ProteinExtensionsConfig(
        name="protein_extensions",
        bond_length=ProteinExtensionConfig(name="bond_length", weight=2.0, bond_length_weight=2.0),
        bond_angle=ProteinExtensionConfig(name="bond_angle", weight=1.5, bond_angle_weight=1.5),
        mixin=ProteinMixinConfig(name="protein_mixin", embedding_dim=32),
    )
    extensions = modality.get_extensions(config)

    # Verify extensions
    assert isinstance(extensions, Mapping)  # nnx.Dict is a Mapping, not dict
    assert "bond_length" in extensions
    assert "bond_angle" in extensions
    assert "protein_mixin" in extensions


def test_protein_modality_rejects_legacy_dict_contract():
    """Test the modality refuses the old dict-shaped extension config."""
    modality = ProteinModality()

    with pytest.raises(TypeError, match="ProteinExtensionsConfig"):
        modality.get_extensions({})


def test_create_default_protein_config_returns_typed_bundle():
    """The protein modality default config should be the canonical typed extension bundle."""
    config = create_default_protein_config()

    assert isinstance(config, ProteinExtensionsConfig)
    assert config.bond_length is not None
    assert config.bond_angle is not None
    assert config.backbone is not None
    assert config.mixin is not None
