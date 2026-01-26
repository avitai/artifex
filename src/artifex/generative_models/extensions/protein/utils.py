"""Utility functions for protein extensions.

This module provides utility functions for creating and managing protein
extensions in a composable way.
"""

from typing import Any

from flax import nnx

from artifex.generative_models.core.configuration import (
    ExtensionConfig,
    ProteinMixinConfig,
)
from artifex.generative_models.extensions.base import ExtensionDict, ModelExtension
from artifex.generative_models.extensions.protein.backbone import (
    BondAngleExtension,
    BondLengthExtension,
)
from artifex.generative_models.extensions.protein.mixin import (
    ProteinMixinExtension,
)


def create_protein_extensions(
    config: dict[str, Any],
    *,
    rngs: nnx.Rngs,
) -> ExtensionDict:
    """Create protein extensions from a configuration.

    Args:
        config: Configuration dictionary with extension parameters.
            May include keys like:
            - use_backbone_constraints: Whether to use backbone constraints
            - bond_length_weight: Weight for bond length constraints
            - bond_angle_weight: Weight for bond angle constraints
            - use_protein_mixin: Whether to use the protein mixin extension
            - aa_embedding_dim: Dimension for amino acid embeddings (for mixin)
            - aa_one_hot: Whether to use one-hot encoding (for mixin)
            - num_aa_types: Number of amino acid types (for mixin)
        rngs: Random number generator keys.

    Returns:
        ExtensionDict mapping extension names to extension instances.
        Uses ExtensionDict (subclass of nnx.Dict) for proper __contains__ support.
    """
    extensions_dict: dict[str, ModelExtension] = {}

    # Add backbone constraints if requested
    if config.get("use_backbone_constraints", False):
        # Bond length extension
        bond_length_config = ExtensionConfig(
            name="bond_length",
            weight=config.get("bond_length_weight", 1.0),
            enabled=True,
        )
        extensions_dict["bond_length"] = BondLengthExtension(
            bond_length_config,
            rngs=rngs,
        )

        # Bond angle extension
        bond_angle_config = ExtensionConfig(
            name="bond_angle",
            weight=config.get("bond_angle_weight", 0.5),
            enabled=True,
        )
        extensions_dict["bond_angle"] = BondAngleExtension(
            bond_angle_config,
            rngs=rngs,
        )

    # Add protein mixin if requested
    if config.get("use_protein_mixin", False):
        # Build ProteinMixinConfig with parameters from dict
        mixin_kwargs: dict[str, Any] = {
            "name": "protein_mixin",
            "weight": 1.0,
            "enabled": True,
        }
        # Map dict keys to config fields
        if "aa_embedding_dim" in config:
            mixin_kwargs["embedding_dim"] = config["aa_embedding_dim"]
        if "aa_one_hot" in config:
            mixin_kwargs["use_one_hot"] = config["aa_one_hot"]
        if "num_aa_types" in config:
            mixin_kwargs["num_aa_types"] = config["num_aa_types"]

        mixin_config = ProteinMixinConfig(**mixin_kwargs)
        extensions_dict["protein_mixin"] = ProteinMixinExtension(
            mixin_config,
            rngs=rngs,
        )

    # Wrap in ExtensionDict for proper __contains__ support
    # ExtensionDict is a subclass of nnx.Dict that fixes the 'in' operator
    return ExtensionDict(extensions_dict)
