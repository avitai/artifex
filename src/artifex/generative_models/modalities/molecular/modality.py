"""Molecular modality implementation.

This module provides the molecular modality implementation that adapts
generative models to work with 3D molecular structure data, including
protein-ligand complexes.
"""

from typing import Any, cast

from flax import nnx

from artifex.generative_models.core.configuration import (
    ChemicalConstraintConfig,
    ExtensionConfig,
    ModalityConfig,
)
from artifex.generative_models.extensions.base import ModelExtension
from artifex.generative_models.extensions.chemical.constraints import ChemicalConstraints
from artifex.generative_models.modalities.base import (
    Modality,
    ModelAdapter,
)
from artifex.generative_models.models.base import GenerativeModelProtocol


class PharmacophoreExtension(ModelExtension):
    """Extension for pharmacophore features in molecular models."""

    def __init__(
        self,
        config: ExtensionConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize pharmacophore extension.

        Args:
            config: Extension configuration
            rngs: Random number generator keys
        """
        if not isinstance(config, ExtensionConfig):
            raise TypeError(f"config must be ExtensionConfig, got {type(config).__name__}")
        super().__init__(config, rngs=rngs)
        # Default pharmacophore feature types
        self.feature_types = ("donor", "acceptor", "hydrophobic", "aromatic")

    def apply(self, model_outputs: Any, batch: Any, **kwargs: Any) -> Any:
        """Apply pharmacophore features to model outputs."""
        # Placeholder implementation for pharmacophore features
        return model_outputs


class MolecularModality(Modality):
    """Modality for 3D molecular structures.

    This modality supports protein-ligand complexes, chemical constraints,
    and pharmacophore features for drug discovery applications.
    """

    name = "molecular"

    def __init__(self, *, rngs: nnx.Rngs | None = None) -> None:
        """Initialize the molecular modality.

        Args:
            rngs: Random number generator keys
        """
        super().__init__()
        self.rngs = rngs if rngs is not None else nnx.Rngs(0)

    def get_extensions(
        self, config: ModalityConfig, *, rngs: nnx.Rngs | None = None
    ) -> dict[str, ModelExtension]:
        """Get molecular-specific extensions.

        Args:
            config: Modality configuration
            rngs: Random number generator keys

        Returns:
            dictionary mapping extension names to extension instances
        """
        # Config validation is now handled by dataclass __post_init__
        extensions_dict: dict[str, ModelExtension] = {}
        rngs = rngs if rngs is not None else self.rngs

        # Get extension configs from modality configuration
        # (API changed from extension_configs to extensions)
        extension_configs = config.extensions

        if extension_configs.get("chemical") is not None:
            chemical_config = extension_configs["chemical"]
            if not isinstance(chemical_config, ChemicalConstraintConfig):
                raise TypeError(
                    "chemical extension config must be ChemicalConstraintConfig, "
                    f"got {type(chemical_config).__name__}"
                )
            extensions_dict["chemical"] = ChemicalConstraints(config=chemical_config, rngs=rngs)

        if extension_configs.get("pharmacophore") is not None:
            extensions_dict["pharmacophore"] = PharmacophoreExtension(
                config=extension_configs["pharmacophore"], rngs=rngs
            )

        return cast(dict[str, ModelExtension], nnx.Dict(extensions_dict))

    def get_adapter(
        self, adapter_type: str | type[GenerativeModelProtocol] | None = None
    ) -> ModelAdapter:
        """Get adapter for molecular data.

        Args:
            adapter_type: The adapter type to get, either a string name
                or a model class. If None, returns the default adapter.

        Returns:
            A model adapter for the specified model type.

        Raises:
            ValueError: If the adapter type is not supported.
        """
        # Import adapters locally to avoid circular imports
        from .adapters import (
            MolecularAdapter,
            MolecularDiffusionAdapter,
            MolecularGeometricAdapter,
        )

        if adapter_type is None:
            return MolecularAdapter()

        if isinstance(adapter_type, str):
            if adapter_type == "geometric":
                return MolecularGeometricAdapter()
            if adapter_type == "diffusion":
                return MolecularDiffusionAdapter()
            if adapter_type in {"default", "vae", "gan", "flow", "ebm", "autoregressive"}:
                return MolecularAdapter()
            raise ValueError(f"Unknown adapter type: {adapter_type}")

        # For model classes, use default adapter for now
        # In a real implementation, this would inspect the model type
        return MolecularAdapter()
