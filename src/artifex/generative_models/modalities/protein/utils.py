"""Utilities for protein modality.

This module provides utility functions for creating and working with
protein-specific modality components.
"""

from typing import Type

from artifex.generative_models.core.base import GenerativeModel
from artifex.generative_models.modalities.base import ModelAdapter
from artifex.generative_models.modalities.protein.adapters import (
    ProteinDiffusionAdapter,
    ProteinGeometricAdapter,
    ProteinModelAdapter,
)


def get_protein_adapter(
    model_cls: str | Type[GenerativeModel] | None = None,
) -> ModelAdapter:
    """Get the appropriate protein adapter for a model class or name.

    Args:
        model_cls: The model class to adapt or adapter name to get.
            If None, returns the default adapter.

    Returns:
        The appropriate protein adapter.

    Raises:
        ValueError: If the adapter type is not supported.
    """
    # Handle string adapter names
    if isinstance(model_cls, str):
        if model_cls == "geometric":
            return ProteinGeometricAdapter()
        elif model_cls == "diffusion":
            return ProteinDiffusionAdapter()
        elif model_cls == "graph":
            return ProteinGeometricAdapter()  # Graph uses geometric adapter
        elif model_cls == "point_cloud":
            # Point cloud uses geometric adapter
            return ProteinGeometricAdapter()
        elif model_cls in ["model", "default", "generic"]:
            return ProteinModelAdapter()
        else:
            raise ValueError(f"Unknown adapter type: {model_cls}")

    # If None, return default adapter
    if model_cls is None:
        return ProteinModelAdapter()

    # Handle model class types
    model_type = model_cls.__name__

    if "Geometric" in model_type or "Point" in model_type or "Graph" in model_type:
        return ProteinGeometricAdapter(model_cls=model_cls)

    if "Diffusion" in model_type:
        return ProteinDiffusionAdapter(model_cls=model_cls)

    # Default adapter for other model types
    return ProteinModelAdapter(model_cls=model_cls)
