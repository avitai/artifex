"""Modality registry for the generative models framework."""

from .audio.base import AudioModality
from .base import Modality
from .image.base import ImageModality
from .molecular.modality import MolecularModality
from .protein.modality import ProteinModality
from .tabular.base import TabularModality
from .text.base import TextModality
from .timeseries.base import TimeseriesModality


# Registry of available modalities
_MODALITY_REGISTRY: dict[str, type] = {
    "audio": AudioModality,
    "protein": ProteinModality,
    "molecular": MolecularModality,
    "image": ImageModality,
    "text": TextModality,
    "tabular": TabularModality,
    "timeseries": TimeseriesModality,
}

# Alias for backward compatibility
MODALITY_REGISTRY = _MODALITY_REGISTRY


def register_modality(name: str, modality_class: type[Modality]) -> None:
    """Register a new modality.

    Args:
        name: Name of the modality
        modality_class: Modality class to register

    Raises:
        ValueError: If modality is already registered
    """
    if name in _MODALITY_REGISTRY:
        raise ValueError(f"Modality '{name}' is already registered")
    _MODALITY_REGISTRY[name] = modality_class


def get_modality(name: str, **kwargs) -> Modality:
    """Get a modality instance by name.

    Args:
        name: Name of the modality
        **kwargs: Additional arguments to pass to modality constructor (e.g., rngs)

    Returns:
        Modality instance

    Raises:
        ValueError: If modality is not found
    """
    if name not in _MODALITY_REGISTRY:
        available = list(_MODALITY_REGISTRY.keys())
        raise ValueError(f"Unknown modality '{name}'. Available: {available}")

    modality_class = _MODALITY_REGISTRY[name]
    # Return an instance of the modality with provided arguments
    return modality_class(**kwargs)


def list_modalities() -> dict[str, type]:
    """List all available modalities.

    Returns:
        Dictionary mapping modality names to their classes
    """
    return dict(_MODALITY_REGISTRY)


def clear_modalities() -> None:
    """Clear all registered modalities.

    Used primarily for testing.
    """
    _MODALITY_REGISTRY.clear()
