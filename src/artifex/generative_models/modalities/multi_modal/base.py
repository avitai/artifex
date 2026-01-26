"""Base multi-modal modality implementation.

This module provides the core multi-modal modality class that enables
working with multiple data modalities simultaneously.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.protocols.configuration import BaseModalityConfig

# Import the registry which already has all modalities registered
from artifex.generative_models.modalities import registry

# Ensure modalities are imported so they're registered
from artifex.generative_models.modalities.base import BaseModalityImplementation


# Don't copy the registry at module import time, access it dynamically


class MultiModalRepresentation(str, Enum):
    """Multi-modal representation types."""

    CONCATENATED = "concatenated"  # Simple concatenation of modalities
    ALIGNED = "aligned"  # Cross-modal alignment
    FUSED = "fused"  # Deep fusion of modalities
    HIERARCHICAL = "hierarchical"  # Hierarchical fusion


@dataclass
class MultiModalModalityConfig(BaseModalityConfig):
    """Configuration for multi-modal modality."""

    modalities: list[str] = field(default_factory=list)  # List of modality names to combine
    fusion_strategy: str = "concatenate"  # How to combine modalities
    alignment_method: str | None = None  # Cross-modal alignment method
    shared_embedding_dim: int | None = None  # Shared embedding dimension
    modality_weights: dict[str, float] | None = None  # Importance weights
    dropout_rate: float = 0.0  # Modality dropout for robustness

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate modalities
        if len(self.modalities) < 2:
            raise ValueError("Multi-modal requires at least 2 modalities")

        # Check that modalities are valid
        valid_modalities = [
            "image",
            "text",
            "audio",
            "tabular",
            "timeseries",
            "protein",
            "molecular",
        ]
        for modality in self.modalities:
            if modality not in valid_modalities:
                raise ValueError(f"Unknown modality: {modality}")

        # Validate fusion strategy
        valid_strategies = ["concatenate", "attention", "gated", "hierarchical"]
        if self.fusion_strategy not in valid_strategies:
            raise ValueError(
                f"Unknown fusion strategy: {self.fusion_strategy}. "
                f"Valid options: {valid_strategies}"
            )


class MultiModalModality(BaseModalityImplementation):
    """Multi-modal modality for combining multiple data types.

    This class manages multiple modalities and provides functionality for:
    - Cross-modal alignment
    - Modality fusion
    - Joint processing and evaluation
    """

    def __init__(
        self,
        config: MultiModalModalityConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize multi-modal modality.

        Args:
            config: Multi-modal configuration
            rngs: Random number generators
        """
        super().__init__(config=config, rngs=rngs)
        self.config = config
        self.modalities = config.modalities
        self.fusion_strategy = config.fusion_strategy
        self.alignment_method = config.alignment_method
        self.shared_embedding_dim = config.shared_embedding_dim or 256
        self.dropout_rate = config.dropout_rate

        # Initialize modality weights
        if config.modality_weights is None:
            self.modality_weights = {mod: 1.0 / len(self.modalities) for mod in self.modalities}
        else:
            self.modality_weights = config.modality_weights

        # Initialize individual modality instances
        self.modality_instances = {}
        for modality_name in self.modalities:
            # Get modality class from registry
            if modality_name not in registry._MODALITY_REGISTRY:
                raise ValueError(f"Modality '{modality_name}' not found in registry")

            modality_class = registry._MODALITY_REGISTRY[modality_name]

            # Create modality instance with default config
            # In real implementation, we'd use modality-specific configs
            self.modality_instances[modality_name] = modality_class(
                config=self._create_modality_config(modality_name),
                rngs=rngs,
            )

        # Initialize fusion components
        self._setup_fusion_components(rngs)

    def _create_modality_config(self, modality_name: str) -> Any:
        """Create default configuration for a modality.

        Args:
            modality_name: Name of the modality

        Returns:
            Default configuration for the modality
        """
        # Return None to use default config in each modality
        return None

    def _setup_fusion_components(self, rngs: nnx.Rngs):
        """Setup fusion components based on strategy.

        Args:
            rngs: Random number generators
        """
        # This will be implemented with actual fusion layers
        # For now, we store the configuration
        self.fusion_components = {
            "strategy": self.fusion_strategy,
            "alignment": self.alignment_method,
        }

    def process(
        self,
        inputs: dict[str, jax.Array],
        representation: MultiModalRepresentation | None = None,
    ) -> dict[str, jax.Array]:
        """Process multi-modal inputs.

        Args:
            inputs: Dictionary mapping modality names to inputs
            representation: Target representation type

        Returns:
            Processed multi-modal data
        """
        if representation is None:
            representation = MultiModalRepresentation.CONCATENATED

        # Process each modality independently
        processed = {}
        for modality_name, modality_input in inputs.items():
            if modality_name in self.modality_instances:
                modality = self.modality_instances[modality_name]
                processed[modality_name] = modality.process(modality_input)

        # Apply fusion based on representation
        if representation == MultiModalRepresentation.CONCATENATED:
            # Simple concatenation
            features = [processed[mod] for mod in self.modalities if mod in processed]
            fused = jnp.concatenate(features, axis=-1)
        elif representation == MultiModalRepresentation.ALIGNED:
            # Cross-modal alignment (simplified)
            fused = self._align_modalities(processed)
        elif representation == MultiModalRepresentation.FUSED:
            # Deep fusion (simplified)
            fused = self._fuse_modalities(processed)
        else:
            # Hierarchical fusion
            fused = self._hierarchical_fusion(processed)

        return {
            "fused_representation": fused,
            "individual_representations": processed,
        }

    def _align_modalities(self, processed: dict[str, jax.Array]) -> jax.Array:
        """Align modalities in shared space.

        Args:
            processed: Processed modality data

        Returns:
            Aligned representations
        """
        # Simplified alignment - project to shared dimension
        aligned = {}
        for mod, features in processed.items():
            # Project to shared dimension (simplified)
            if features.ndim == 1:
                aligned[mod] = features[: self.shared_embedding_dim]
            else:
                aligned[mod] = features[..., : self.shared_embedding_dim]

        # Stack aligned features
        return jnp.stack([aligned[mod] for mod in self.modalities if mod in aligned])

    def _fuse_modalities(self, processed: dict[str, jax.Array]) -> jax.Array:
        """Deep fusion of modalities.

        Args:
            processed: Processed modality data

        Returns:
            Fused representation
        """
        # Simplified fusion - weighted average
        fused: jax.Array | None = None
        for mod in self.modalities:
            if mod in processed:
                weight = self.modality_weights[mod]
                if fused is None:
                    fused = processed[mod] * weight
                else:
                    fused = fused + processed[mod] * weight

        return fused

    def _hierarchical_fusion(self, processed: dict[str, jax.Array]) -> jax.Array:
        """Hierarchical fusion of modalities.

        Args:
            processed: Processed modality data

        Returns:
            Hierarchically fused representation
        """
        # Simplified hierarchical fusion
        # Group similar modalities first, then fuse groups

        # Example grouping: visual (image) and textual (text) modalities
        visual_mods = ["image", "video"]
        textual_mods = ["text", "audio"]

        groups = []

        # Fuse visual modalities
        visual_features = [
            processed[mod] for mod in visual_mods if mod in processed and mod in self.modalities
        ]
        if visual_features:
            visual_fused = jnp.mean(jnp.stack(visual_features), axis=0)
            groups.append(visual_fused)

        # Fuse textual modalities
        textual_features = [
            processed[mod] for mod in textual_mods if mod in processed and mod in self.modalities
        ]
        if textual_features:
            textual_fused = jnp.mean(jnp.stack(textual_features), axis=0)
            groups.append(textual_fused)

        # Fuse remaining modalities
        remaining_mods = [
            mod for mod in self.modalities if mod not in visual_mods and mod not in textual_mods
        ]
        for mod in remaining_mods:
            if mod in processed:
                groups.append(processed[mod])

        # Final fusion of groups
        if groups:
            return jnp.concatenate(groups, axis=-1)
        else:
            return jnp.zeros((self.shared_embedding_dim,))

    def validate_shapes(self, shapes: dict[str, tuple[int, ...]]) -> bool:
        """Validate input shapes for all modalities.

        Args:
            shapes: Dictionary mapping modality names to shapes

        Returns:
            True if all shapes are valid
        """
        for modality_name, shape in shapes.items():
            if modality_name in self.modality_instances:
                modality = self.modality_instances[modality_name]
                if not modality.validate_shape(shape):
                    return False
        return True

    def get_sample_shape(self) -> dict[str, tuple[int, ...]]:
        """Get sample shapes for all modalities.

        Returns:
            Dictionary mapping modality names to sample shapes
        """
        shapes = {}
        for modality_name, modality in self.modality_instances.items():
            shapes[modality_name] = modality.get_sample_shape()
        return shapes
