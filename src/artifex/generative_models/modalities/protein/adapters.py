"""Adapters for protein models.

This module provides adapters that adapt generative models to work with
protein structure data.
"""

import dataclasses
from typing import Any, Type

import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.modalities.base import ModelAdapter
from artifex.generative_models.models.base import GenerativeModelProtocol


# Type alias for protein data
ProteinData = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class ProteinAdapterConfig:
    """Configuration for protein model adapters.

    Attributes:
        name: Name of the adapter
        model_type: Type of protein model ('graph' or 'point_cloud')
        num_residues: Number of amino acid residues
        num_atoms_per_residue: Number of atoms per residue (default 4 for backbone)
        backbone_indices: Indices of backbone atoms (N, CA, C, O)
        use_constraints: Whether to use protein-specific constraints
        constraint_config: Configuration for constraints
    """

    name: str = "protein_adapter"
    model_type: str = "graph"
    num_residues: int = 10
    num_atoms_per_residue: int = 4
    backbone_indices: tuple[int, ...] = (0, 1, 2, 3)
    use_constraints: bool = True
    constraint_config: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "backbone_weight": 1.0,
            "bond_weight": 1.0,
            "angle_weight": 0.5,
            "dihedral_weight": 0.3,
            "phi_weight": 0.5,
            "psi_weight": 0.5,
        }
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_model_types = {"graph", "point_cloud"}
        if self.model_type not in valid_model_types:
            raise ValueError(
                f"model_type must be one of {valid_model_types}, got {self.model_type}"
            )
        if self.num_residues <= 0:
            raise ValueError(f"num_residues must be positive, got {self.num_residues}")
        if self.num_atoms_per_residue <= 0:
            raise ValueError(
                f"num_atoms_per_residue must be positive, got {self.num_atoms_per_residue}"
            )


class ProteinModelAdapter(ModelAdapter):
    """Base adapter for protein models.

    This adapter provides common functionality for adapting generative models
    to work with protein structure data.
    """

    def __init__(
        self,
        config: ProteinAdapterConfig | None = None,
        model_cls: Type[GenerativeModelProtocol] | None = None,
    ) -> None:
        """Initialize the protein model adapter.

        Args:
            config: Adapter configuration
            model_cls: Optional model class to adapt.
        """
        self.config = config or ProteinAdapterConfig()
        self.name = "protein_model"
        self.modality = "protein"
        self.model_cls = model_cls

    def adapt(self, model: Any, config: Any) -> Any:
        """Adapt a model for protein modality.

        Args:
            model: The model instance to adapt
            config: Model configuration

        Returns:
            The adapted model (currently returns model unchanged)
        """
        # For now, protein models don't need special adaptation
        # as they are created specifically for protein data
        return model

    def adapt_inputs(self, inputs: ProteinData) -> ProteinData:
        """Adapt inputs for the protein model.

        Args:
            inputs: Input protein data.

        Returns:
            Adapted input data.
        """
        # Default implementation just passes through the inputs
        return inputs

    def create(
        self,
        config: Any,
        *,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> GenerativeModelProtocol:
        """Create a protein model.

        Args:
            config: Model configuration (dataclass config)
            rngs: Random number generator keys.
            **kwargs: Additional keyword arguments.

        Returns:
            A generative model for proteins.

        Raises:
            TypeError: If config is not a dataclass
        """
        # Import model classes here to avoid circular imports
        from artifex.generative_models.models.geometric.protein_graph import ProteinGraphModel
        from artifex.generative_models.models.geometric.protein_point_cloud import (
            ProteinPointCloudModel,
        )

        # Determine model type from adapter config or config parameter
        if isinstance(config, ProteinAdapterConfig):
            model_type = config.model_type
        elif dataclasses.is_dataclass(config) and hasattr(config, "model_type"):
            model_type = config.model_type
        else:
            model_type = self.config.model_type

        # Map model types to classes
        model_classes = {
            "graph": ProteinGraphModel,
            "point_cloud": ProteinPointCloudModel,
        }

        # Get the appropriate model class
        if model_type not in model_classes:
            raise ValueError(
                f"Unknown protein model type: {model_type}. "
                f"Available types: {list(model_classes.keys())}"
            )

        model_cls = model_classes[model_type]

        # Create and return model directly
        return model_cls(config, rngs=rngs)


class ProteinGeometricAdapter(ProteinModelAdapter):
    """Adapter for protein geometric models.

    This adapter provides functionality for adapting generative models that
    work with protein structure data in geometric (3D coordinate) format.
    """

    def __init__(
        self,
        config: ProteinAdapterConfig | None = None,
        model_cls: Type[GenerativeModelProtocol] | None = None,
    ) -> None:
        """Initialize the protein geometric adapter.

        Args:
            config: Adapter configuration
            model_cls: Optional model class to adapt.
        """
        super().__init__(config, model_cls)
        self.name = "protein_geometric"

    def adapt_inputs(self, inputs: ProteinData) -> ProteinData:
        """Adapt inputs for the geometric protein model.

        Args:
            inputs: Input protein data.

        Returns:
            Adapted input data with positions field.
        """
        # Create a copy of the inputs to avoid modifying the original
        adapted = dict(inputs)

        # Convert coordinates to positions if necessary
        if "coordinates" in inputs and "positions" not in inputs:
            adapted["positions"] = inputs["coordinates"]

        return adapted

    def create(
        self,
        config: Any,
        *,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> GenerativeModelProtocol:
        """Create a protein geometric model.

        Args:
            config: Model configuration (dataclass config)
            rngs: Random number generator keys.
            **kwargs: Additional keyword arguments.

        Returns:
            A generative model for proteins in geometric format.
        """
        # Import here to avoid circular imports
        from artifex.generative_models.factory import create_model

        # Create and return model with protein modality
        return create_model(config, modality="protein", rngs=rngs)


class ProteinDiffusionAdapter(ProteinModelAdapter):
    """Adapter for protein diffusion models.

    This adapter provides functionality for adapting diffusion-based
    generative models to work with protein structure data.
    """

    def __init__(
        self,
        config: ProteinAdapterConfig | None = None,
        model_cls: Type[GenerativeModelProtocol] | None = None,
    ) -> None:
        """Initialize the protein diffusion adapter.

        Args:
            config: Adapter configuration
            model_cls: Optional model class to adapt.
        """
        super().__init__(config, model_cls)
        self.name = "protein_diffusion"

    def adapt_inputs(self, inputs: ProteinData) -> ProteinData:
        """Adapt inputs for the diffusion protein model.

        Args:
            inputs: Input protein data.

        Returns:
            Adapted input data with noise added.
        """
        # Create a copy of the inputs to avoid modifying the original
        adapted = dict(inputs)

        # Add noise field if not present
        # In a real implementation, this would generate appropriate noise
        if "noise" not in adapted:
            # Create noise with same shape as coordinates if present
            if "coordinates" in inputs:
                shape = inputs["coordinates"].shape
                adapted["noise"] = jnp.zeros(shape)

        return adapted

    def create(
        self,
        config: Any,
        *,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> GenerativeModelProtocol:
        """Create a protein diffusion model.

        Args:
            config: Model configuration (dataclass config)
            rngs: Random number generator keys.
            **kwargs: Additional keyword arguments.

        Returns:
            A diffusion-based generative model for proteins.
        """
        # Import here to avoid circular imports
        from artifex.generative_models.factory import create_model

        # Create diffusion model with protein modality
        return create_model(config, modality="protein", rngs=rngs)
