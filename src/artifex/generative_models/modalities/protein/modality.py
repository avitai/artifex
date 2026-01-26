"""Protein modality implementation.

This module provides the protein modality implementation that adapts
generative models to work with protein structure data.
"""

from typing import Any, Type

from flax import nnx

from artifex.generative_models.extensions.base import ModelExtension
from artifex.generative_models.extensions.protein import (
    create_protein_extensions,
)
from artifex.generative_models.modalities.base import (
    Modality,
    ModelAdapter,
)
from artifex.generative_models.modalities.protein.adapters import (
    ProteinDiffusionAdapter,
    ProteinGeometricAdapter,
    ProteinModelAdapter,
)
from artifex.generative_models.modalities.protein.utils import (
    get_protein_adapter,
)
from artifex.generative_models.models.base import GenerativeModelProtocol


class ProteinModality(Modality):
    """Protein modality for generative models.

    This modality adapts generative models to work with protein structure
    data.
    """

    name = "protein"

    def __init__(self, **kwargs) -> None:
        """Initialize the protein modality.

        Args:
            **kwargs: Optional arguments (e.g., rngs) that may be passed
                     by the registry but are not used for this modality.
        """
        # Protein modality doesn't require rngs or other args
        # but accepts them for compatibility with the registry
        pass

    def get_extensions(
        self, config: dict[str, Any], *, rngs: nnx.Rngs | None = None
    ) -> dict[str, ModelExtension]:
        """Get protein-specific extensions.

        Args:
            config: Extension configuration.
            rngs: Random number generator keys.

        Returns:
            dictionary mapping extension names to extension instances.
        """
        # For test compatibility, handle extensions key
        extensions_config = config.get("extensions", {})

        # Create default RNGs if not provided
        if rngs is None:
            rngs = nnx.Rngs(0)

        # Create extensions
        return create_protein_extensions(extensions_config, rngs=rngs)

    def get_adapter(
        self, adapter_type: str | Type[GenerativeModelProtocol] | None = None
    ) -> ModelAdapter:
        """Get an adapter for the specified model type.

        Args:
            adapter_type: The adapter type to get, either a string name
                or a model class. If None, returns the default adapter.

        Returns:
            A model adapter for the specified model type.

        Raises:
            ValueError: If the adapter type is not supported.
        """
        if adapter_type is None:
            # Default adapter
            return ProteinModelAdapter()

        if isinstance(adapter_type, str):
            # Adapter selection by name
            if adapter_type == "geometric":
                return ProteinGeometricAdapter()
            elif adapter_type == "diffusion":
                return ProteinDiffusionAdapter()
            elif adapter_type == "vae":
                return ProteinModelAdapter()  # Use the generic model adapter for VAE
            elif adapter_type == "model" or adapter_type == "default":
                return ProteinModelAdapter()
            else:
                raise ValueError(f"Unknown adapter type: {adapter_type}")

        # If it's a model class, use the utility function
        return get_protein_adapter(adapter_type)
