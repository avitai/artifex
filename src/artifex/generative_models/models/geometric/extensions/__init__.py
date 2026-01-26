"""Extensions for geometric models."""

from artifex.generative_models.core.configuration import (
    ExtensionConfig,
    ProteinExtensionConfig,
    ProteinMixinConfig,
)
from artifex.generative_models.extensions.base import (
    ConstraintExtension,
    ModelExtension,
)
from artifex.generative_models.extensions.protein.mixin import ProteinMixinExtension


class GeometricModelExtension(ModelExtension):
    """Base class for geometric model extensions."""

    def __init__(self, **kwargs):
        """Initialize the extension."""
        super().__init__(**kwargs)


class PointCloudExtension(GeometricModelExtension):
    """Extension for point cloud models."""

    def __init__(self, **kwargs):
        """Initialize the extension."""
        super().__init__(**kwargs)


class MeshExtension(GeometricModelExtension):
    """Extension for mesh models."""

    def __init__(self, **kwargs):
        """Initialize the extension."""
        super().__init__(**kwargs)


class VoxelExtension(GeometricModelExtension):
    """Extension for voxel models."""

    def __init__(self, **kwargs):
        """Initialize the extension."""
        super().__init__(**kwargs)


class BondAngleExtension(ConstraintExtension):
    """Extension for enforcing bond angle constraints in structures."""

    def __init__(self, config: ProteinExtensionConfig | ExtensionConfig, **kwargs):
        """Initialize the extension.

        Args:
            config: ProteinExtensionConfig or ExtensionConfig with bond angle parameters.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(config, **kwargs)

        # Extract parameters from config using frozen dataclass fields
        if isinstance(config, ProteinExtensionConfig):
            self.ideal_angles = list(config.ideal_bond_angles.values()) or [2.025, 2.11, 1.94]
            self.tolerance = config.tolerance
            self.angle_targets = config.ideal_bond_angles
        else:
            # Default values for base ExtensionConfig
            self.ideal_angles = [2.025, 2.11, 1.94]  # CA-N-C, N-CA-C, CA-C-N in radians
            self.tolerance = 0.1
            self.angle_targets = {}

    def __call__(self, inputs, outputs, **kwargs):
        """Apply the bond angle constraints.

        Args:
            inputs: Input data.
            outputs: Model outputs.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary with constraint metrics.
        """
        # Extract coordinates from outputs (not used in this simplified version)
        if isinstance(outputs, dict) and "coordinates" in outputs:
            _ = outputs["coordinates"]
        elif isinstance(outputs, dict) and "positions" in outputs:
            _ = outputs["positions"]
        else:
            _ = outputs

        # Calculate angle metrics (simplified)
        return {
            "extension_type": "bond_angle",
            "status": "applied",
            "mean_deviation": 0.0,  # Placeholder
        }

    def loss_fn(self, batch, model_outputs, **kwargs):
        """Calculate the bond angle constraint loss.

        Args:
            batch: Input batch.
            model_outputs: Model outputs.
            **kwargs: Additional keyword arguments.

        Returns:
            Bond angle constraint loss as a JAX array.
        """
        # Simplified placeholder implementation
        import jax.numpy as jnp

        return jnp.array(0.0)  # Return as JAX array


class BondLengthExtension(ConstraintExtension):
    """Extension for enforcing bond length constraints in structures."""

    def __init__(self, config: ProteinExtensionConfig | ExtensionConfig, **kwargs):
        """Initialize the extension.

        Args:
            config: ProteinExtensionConfig or ExtensionConfig with bond length parameters.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(config, **kwargs)

        # Extract parameters from config using frozen dataclass fields
        if isinstance(config, ProteinExtensionConfig):
            self.ideal_lengths = list(config.ideal_bond_lengths.values()) or [1.45, 1.52, 1.33]
            self.tolerance = config.tolerance
        else:
            # Default values for base ExtensionConfig
            self.ideal_lengths = [1.45, 1.52, 1.33]  # N-CA, CA-C, C-N in Angstroms
            self.tolerance = 0.1

    def __call__(self, inputs, outputs, **kwargs):
        """Apply the bond length constraints.

        Args:
            inputs: Input data.
            outputs: Model outputs.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary with constraint metrics.
        """
        # Extract coordinates from outputs (not used in this simplified version)
        if isinstance(outputs, dict) and "coordinates" in outputs:
            _ = outputs["coordinates"]
        elif isinstance(outputs, dict) and "positions" in outputs:
            _ = outputs["positions"]
        else:
            _ = outputs

        # Calculate bond metrics (simplified)
        return {
            "extension_type": "bond_length",
            "status": "applied",
            "mean_deviation": 0.0,  # Placeholder
        }

    def loss_fn(self, batch, model_outputs, **kwargs):
        """Calculate the bond length constraint loss.

        Args:
            batch: Input batch.
            model_outputs: Model outputs.
            **kwargs: Additional keyword arguments.

        Returns:
            Bond length constraint loss as a JAX array.
        """
        # Simplified placeholder implementation
        import jax.numpy as jnp

        return jnp.array(0.0)  # Return as JAX array


# ProteinMixinExtension is imported from artifex.generative_models.extensions.protein.mixin


def create_protein_extensions(config, rngs=None):
    """Create protein-specific extensions based on config.

    Args:
        config: Configuration dictionary for protein extensions.
        rngs: Random number generators.

    Returns:
        Dictionary of protein extensions.
    """
    extensions = {}

    # Add bond length extension if requested
    if config.get("use_backbone_constraints", True):
        bond_length_config = ProteinExtensionConfig(
            name="bond_length",
            weight=config.get("bond_length_weight", 1.0),
            enabled=True,
            bond_length_weight=config.get("bond_length_weight", 1.0),
            ideal_bond_lengths={
                "N-CA": 1.45,
                "CA-C": 1.52,
                "C-N": 1.33,
            },
        )
        extensions["bond_length"] = BondLengthExtension(
            bond_length_config,
            rngs=rngs,
        )

    # Add bond angle extension if requested
    if config.get("use_backbone_constraints", True):
        bond_angle_config = ProteinExtensionConfig(
            name="bond_angle",
            weight=config.get("bond_angle_weight", 0.5),
            enabled=True,
            bond_angle_weight=config.get("bond_angle_weight", 0.5),
            ideal_bond_angles={
                "CA-N-C": 2.025,
                "N-CA-C": 2.11,
                "CA-C-N": 1.94,
            },
        )
        extensions["bond_angle"] = BondAngleExtension(
            bond_angle_config,
            rngs=rngs,
        )

    # Add protein mixin extension if requested
    if config.get("use_aa_features", True):
        protein_mixin_config = ProteinMixinConfig(
            name="protein_mixin",
            weight=1.0,
            enabled=True,
            embedding_dim=config.get("model_dim", 64),
            num_aa_types=20,
        )
        extensions["protein_mixin"] = ProteinMixinExtension(
            protein_mixin_config,
            rngs=rngs,
        )

    return extensions


__all__ = [
    "GeometricModelExtension",
    "PointCloudExtension",
    "MeshExtension",
    "VoxelExtension",
    "BondAngleExtension",
    "BondLengthExtension",
    "ProteinMixinExtension",
    "create_protein_extensions",
]
