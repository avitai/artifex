"""Schema for extension configurations.

This module defines configuration schemas for model extensions.
"""

from pydantic import Field

# Import BaseConfig inline to avoid circular imports
# local imports
from artifex.configs.schema.base import BaseConfig


class ExtensionConfig(BaseConfig):
    """Base configuration for model extensions."""

    weight: float = Field(1.0, description="Weight for the extension's contribution")
    enabled: bool = Field(True, description="Whether the extension is enabled")


class ConstraintExtensionConfig(ExtensionConfig):
    """Configuration for constraint-based extensions."""

    constraint_type: str = Field(..., description="Type of constraint to apply")
    target_loss: float = Field(0.0, description="Target value for constraint loss")
    tolerance: float = Field(0.01, description="Tolerance for constraint satisfaction")


class ExtensionsConfig(BaseConfig):
    """Configuration for a collection of extensions."""

    extensions: dict[str, ExtensionConfig] = Field(
        {}, description="Dictionary of extension configurations by name"
    )
    enabled: bool = Field(True, description="Whether all extensions are enabled")


class ProteinBackboneConstraintConfig(ConstraintExtensionConfig):
    """Configuration for protein backbone constraints."""

    constraint_type: str = Field("backbone", description="Type of protein constraint")
    backbone_atoms: list[str] = Field(["N", "CA", "C"], description="Backbone atoms to constrain")
    bond_length_weight: float = Field(1.0, description="Weight for bond length constraints")
    bond_angle_weight: float = Field(0.5, description="Weight for bond angle constraints")


class ProteinMixinConfig(ExtensionConfig):
    """Configuration for protein mixin extension."""

    embedding_dim: int = Field(16, description="Dimension for amino acid embeddings")
    aa_one_hot: bool = Field(True, description="Whether to use one-hot encoding for amino acids")
    num_aa_types: int = Field(
        21, description="Number of amino acid types (including unknown/padding)"
    )


class ProteinExtensionConfig(ExtensionsConfig):
    """Configuration for protein-specific extensions."""

    use_backbone_constraints: bool = Field(True, description="Whether to use backbone constraints")
    use_protein_mixin: bool = Field(True, description="Whether to use protein mixin extension")
