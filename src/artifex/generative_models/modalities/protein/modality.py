"""Protein modality implementation.

This module provides the retained protein modality boundary for typed protein
extension bundles and adapter lookup around the shared model families.
"""

from flax import nnx

from artifex.generative_models.core.configuration import ProteinExtensionsConfig
from artifex.generative_models.extensions.base import ExtensionDict
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

    The shared factory still chooses the base model family from the typed
    config; this modality owns the retained protein adapter and extension
    lookup surface.
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
        self, config: ProteinExtensionsConfig, *, rngs: nnx.Rngs | None = None
    ) -> ExtensionDict:
        """Get protein-specific extensions.

        Args:
            config: Typed protein extension bundle.
            rngs: Random number generator keys.

        Returns:
            dictionary mapping extension names to extension instances.
        """
        if not isinstance(config, ProteinExtensionsConfig):
            raise TypeError(f"config must be ProteinExtensionsConfig, got {type(config).__name__}")

        # Create default RNGs if not provided
        if rngs is None:
            rngs = nnx.Rngs(0)

        return create_protein_extensions(config, rngs=rngs)

    def get_adapter(
        self, adapter_type: str | type[GenerativeModelProtocol] | None = None
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
