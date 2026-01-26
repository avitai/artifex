"""Central registry for modality-specific extensions.

This module provides a unified registry system for managing and discovering
extensions across different modalities in the generative models framework.
"""

import enum
from datetime import datetime
from typing import Any, Type

from flax import nnx

from artifex.generative_models.core.configuration import ExtensionConfig


class ExtensionType(enum.Enum):
    """Enumeration of extension types.

    Each extension type corresponds to a specific role in the framework:
    - MODEL: General model extensions that process outputs
    - CONSTRAINT: Extensions that enforce physical/domain constraints
    - AUGMENTATION: Data augmentation extensions
    - SAMPLING: Extensions that modify generation/sampling
    - LOSS: Modular loss component extensions
    - EVALUATION: Domain-specific evaluation metric extensions
    - CALLBACK: Training lifecycle hook extensions
    - MODALITY: Modality-specific preprocessing extensions
    """

    MODEL = "model"
    CONSTRAINT = "constraint"
    AUGMENTATION = "augmentation"
    SAMPLING = "sampling"
    LOSS = "loss"
    EVALUATION = "evaluation"
    CALLBACK = "callback"
    MODALITY = "modality"


class ExtensionsRegistry:
    """Central registry for modality-specific extensions."""

    def __init__(self):
        """Initialize the extensions registry."""
        self._extensions = {}
        self._register_core_extensions()

    def register_extension(
        self,
        name: str,
        extension_class: Type[nnx.Module],
        modalities: list[str],
        capabilities: list[str],
        extension_type: ExtensionType | None = None,
        description: str = "",
    ):
        """Register an extension for specific modalities.

        Args:
            name: Unique name for the extension
            extension_class: The extension class (must inherit from nnx.Module)
            modalities: List of modalities this extension supports
            capabilities: List of capabilities this extension provides
            extension_type: Type of extension (MODEL, CONSTRAINT, etc.)
            description: Optional description of the extension
        """
        if not issubclass(extension_class, nnx.Module):
            raise ValueError(f"Extension class {extension_class} must inherit from nnx.Module")

        self._extensions[name] = {
            "class": extension_class,
            "modalities": modalities,
            "capabilities": capabilities,
            "extension_type": extension_type,
            "description": description,
            "registered_at": datetime.now(),
        }

    def get_extensions_for_modality(self, modality: str) -> list[str]:
        """Get all extensions available for a modality.

        Args:
            modality: The modality name (e.g., 'image', 'molecular', 'geometric')

        Returns:
            List of extension names that support the specified modality
        """
        return [name for name, info in self._extensions.items() if modality in info["modalities"]]

    def get_extensions_by_capability(self, capability: str) -> list[str]:
        """Get extensions that provide a specific capability.

        Args:
            capability: The capability name (e.g., 'augmentation', 'constraints')

        Returns:
            List of extension names that provide the specified capability
        """
        return [
            name for name, info in self._extensions.items() if capability in info["capabilities"]
        ]

    def get_extensions_by_type(self, extension_type: ExtensionType) -> list[str]:
        """Get extensions of a specific type.

        Args:
            extension_type: The extension type (MODEL, CONSTRAINT, etc.)

        Returns:
            List of extension names of the specified type
        """
        return [
            name
            for name, info in self._extensions.items()
            if info.get("extension_type") == extension_type
        ]

    def create_extension(
        self, name: str, config: ExtensionConfig | None = None, *, rngs: nnx.Rngs
    ) -> nnx.Module:
        """Create extension instance.

        Args:
            name: Name of the extension to create
            config: Extension configuration (must be ExtensionConfig)
            rngs: Random number generator keys

        Returns:
            Instantiated extension module

        Raises:
            ValueError: If extension not registered
            TypeError: If config is not ExtensionConfig
        """
        if name not in self._extensions:
            raise ValueError(f"Extension '{name}' not registered")

        extension_class = self._extensions[name]["class"]

        # Handle configuration
        if config is None:
            # Use default ExtensionConfig with extension name
            config = ExtensionConfig(name=name)
        elif not isinstance(config, ExtensionConfig):
            raise TypeError(f"config must be ExtensionConfig, got {type(config).__name__}")

        # Create extension with typed config
        return extension_class(config, rngs=rngs)

    def list_all_extensions(self) -> dict[str, dict[str, Any]]:
        """List all registered extensions with their metadata.

        Returns:
            Dictionary of extension names to their metadata
        """
        return {
            name: {
                "modalities": info["modalities"],
                "capabilities": info["capabilities"],
                "extension_type": info["extension_type"].value if info["extension_type"] else None,
                "description": info["description"],
                "registered_at": info["registered_at"].isoformat(),
            }
            for name, info in self._extensions.items()
        }

    def get_extension_info(self, name: str) -> dict[str, Any]:
        """Get detailed information about a specific extension.

        Args:
            name: Name of the extension

        Returns:
            Dictionary containing extension metadata
        """
        if name not in self._extensions:
            raise ValueError(f"Extension '{name}' not registered")

        info = self._extensions[name]
        return {
            "class": info["class"].__name__,
            "modalities": info["modalities"],
            "capabilities": info["capabilities"],
            "extension_type": info["extension_type"].value if info["extension_type"] else None,
            "description": info["description"],
            "registered_at": info["registered_at"].isoformat(),
        }

    def _register_core_extensions(self):
        """Register the core extensions provided by the framework."""
        try:
            # Chemical Extensions
            from .chemical.constraints import ChemicalConstraints
            from .chemical.features import MolecularFeatures

            self.register_extension(
                "chemical_constraints",
                ChemicalConstraints,
                modalities=["molecular"],
                capabilities=["validation", "constraints"],
                extension_type=ExtensionType.CONSTRAINT,
                description="Chemical constraint validation for molecular structures",
            )

            self.register_extension(
                "molecular_features",
                MolecularFeatures,
                modalities=["molecular"],
                capabilities=["feature_extraction", "descriptors"],
                extension_type=ExtensionType.MODEL,
                description="Molecular descriptor computation and feature extraction",
            )

        except ImportError:
            # Chemical extensions not available
            pass

        try:
            # Vision Extensions
            from .vision.augmentation import AdvancedImageAugmentation

            self.register_extension(
                "image_augmentation",
                AdvancedImageAugmentation,
                modalities=["image"],
                capabilities=["augmentation", "preprocessing"],
                extension_type=ExtensionType.AUGMENTATION,
                description="Advanced image augmentation for robust training",
            )

        except ImportError:
            # Vision extensions not available
            pass

        try:
            # Audio Processing Extensions
            from .audio_processing.spectral import SpectralAnalysis
            from .audio_processing.temporal import TemporalAnalysis

            self.register_extension(
                "spectral_analysis",
                SpectralAnalysis,
                modalities=["audio"],
                capabilities=["spectral_analysis", "feature_extraction"],
                extension_type=ExtensionType.MODEL,
                description="Spectral analysis and feature extraction for audio",
            )

            self.register_extension(
                "temporal_analysis",
                TemporalAnalysis,
                modalities=["audio"],
                capabilities=["temporal_analysis", "rhythm_detection"],
                extension_type=ExtensionType.MODEL,
                description="Temporal pattern analysis for audio generation",
            )

        except ImportError:
            # Audio extensions not available
            pass

        try:
            # NLP Extensions
            from .nlp.embeddings import TextEmbeddings
            from .nlp.tokenization import AdvancedTokenization

            self.register_extension(
                "advanced_tokenization",
                AdvancedTokenization,
                modalities=["text"],
                capabilities=["tokenization", "preprocessing"],
                extension_type=ExtensionType.MODALITY,
                description="Advanced tokenization for text generation tasks",
            )

            self.register_extension(
                "text_embeddings",
                TextEmbeddings,
                modalities=["text"],
                capabilities=["embeddings", "representation_learning"],
                extension_type=ExtensionType.MODEL,
                description="Text embedding utilities for generation tasks",
            )

        except ImportError:
            # NLP extensions not available
            pass

        # Note: Physics, BlackJAX, and Performance extensions would be registered here

    def validate_extension_compatibility(
        self, extension_names: list[str], modality: str
    ) -> dict[str, bool]:
        """Validate that extensions are compatible with a modality.

        Args:
            extension_names: List of extension names to validate
            modality: Target modality

        Returns:
            Dictionary mapping extension names to compatibility status
        """
        compatibility = {}

        for ext_name in extension_names:
            if ext_name not in self._extensions:
                compatibility[ext_name] = False
            else:
                ext_info = self._extensions[ext_name]
                compatibility[ext_name] = modality in ext_info["modalities"]

        return compatibility

    def create_extension_pipeline(
        self, extension_configs: list[tuple[str, ExtensionConfig | None]], *, rngs: nnx.Rngs
    ) -> list[nnx.Module]:
        """Create a pipeline of extensions from configuration.

        Args:
            extension_configs: List of (name, config) tuples where:
                              - name: Extension name to create
                              - config: ExtensionConfig or None for default
            rngs: Random number generator keys

        Returns:
            List of instantiated extension modules
        """
        pipeline = []

        for ext_name, ext_config in extension_configs:
            extension = self.create_extension(ext_name, ext_config, rngs=rngs)
            pipeline.append(extension)

        return pipeline

    def search_extensions(
        self,
        modality: str | None = None,
        capability: str | None = None,
        extension_type: ExtensionType | None = None,
        description_contains: str | None = None,
    ) -> list[str]:
        """Search for extensions matching criteria.

        Args:
            modality: Filter by modality support
            capability: Filter by capability
            extension_type: Filter by extension type
            description_contains: Filter by description content

        Returns:
            List of extension names matching the criteria
        """
        results = []

        for name, info in self._extensions.items():
            matches = True

            # Filter by modality
            if modality is not None and modality not in info["modalities"]:
                matches = False

            # Filter by capability
            if capability is not None and capability not in info["capabilities"]:
                matches = False

            # Filter by extension type
            if extension_type is not None and info.get("extension_type") != extension_type:
                matches = False

            # Filter by description
            if (
                description_contains is not None
                and description_contains.lower() not in info["description"].lower()
            ):
                matches = False

            if matches:
                results.append(name)

        return results

    def get_available_modalities(self) -> list[str]:
        """Get list of all modalities supported by registered extensions.

        Returns:
            List of unique modality names
        """
        modalities = set()
        for info in self._extensions.values():
            modalities.update(info["modalities"])
        return sorted(list(modalities))

    def get_available_capabilities(self) -> list[str]:
        """Get list of all capabilities provided by registered extensions.

        Returns:
            List of unique capability names
        """
        capabilities = set()
        for info in self._extensions.values():
            capabilities.update(info["capabilities"])
        return sorted(list(capabilities))

    def get_available_extension_types(self) -> list[ExtensionType]:
        """Get list of all extension types present in registered extensions.

        Returns:
            List of unique ExtensionType values
        """
        types: set[ExtensionType] = set()
        for info in self._extensions.values():
            if info.get("extension_type") is not None:
                types.add(info["extension_type"])
        return sorted(list(types), key=lambda t: t.value)


# Global registry instance
_global_registry = None


def get_extensions_registry() -> ExtensionsRegistry:
    """Get the global extensions registry instance.

    Returns:
        Global ExtensionsRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ExtensionsRegistry()
    return _global_registry
