"""Configuration schema for model extensions.

This module defines configuration schemas for various extension types
that can be attached to generative models.
"""

from pydantic import Field

from artifex.configs.schema.base import BaseConfig


class ExtensionConfig(BaseConfig):
    """Base configuration for model extensions."""

    enabled: bool = Field(True, description="Whether the extension is enabled")
    weight: float = Field(1.0, description="Weight for the extension's contribution")


class BondLengthConfig(ExtensionConfig):
    """Configuration for protein backbone bond length constraints."""

    ideal_lengths: list[float] | None = Field(
        None, description="list of ideal bond lengths [N-CA, CA-C, C-N+1] in Angstroms"
    )


class BondAngleConfig(ExtensionConfig):
    """Configuration for protein backbone bond angle constraints."""

    ideal_angles: list[float] | None = Field(
        None, description="list of ideal bond angles [N-CA-C, CA-C-O] in radians"
    )


class ProteinMixinConfig(ExtensionConfig):
    """Configuration for protein mixin extension."""

    embedding_dim: int = Field(16, description="Dimension for amino acid type embeddings")
    aa_one_hot: bool = Field(True, description="Whether to use one-hot encoding for amino acids")
    num_aa_types: int = Field(
        21, description="Number of amino acid types including unknown/padding"
    )


class ProteinExtensionsConfig(BaseConfig):
    """Configuration for protein-specific extensions."""

    use_backbone_constraints: bool = Field(False, description="Whether to use backbone constraints")
    bond_length: BondLengthConfig = Field(
        BondLengthConfig(), description="Bond length constraint configuration"
    )
    bond_angle: BondAngleConfig = Field(
        BondAngleConfig(), description="Bond angle constraint configuration"
    )
    use_protein_mixin: bool = Field(False, description="Whether to use protein mixin extension")
    protein_mixin: ProteinMixinConfig = Field(
        ProteinMixinConfig(), description="Protein mixin configuration"
    )


class ExtensionsConfig(BaseConfig):
    """Configuration for model extensions."""

    protein: ProteinExtensionsConfig | None = Field(
        None, description="Protein-specific extension configuration"
    )
