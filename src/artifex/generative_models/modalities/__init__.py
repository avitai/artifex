"""Data modalities for generative models."""

from .audio.base import AudioModality
from .base import (
    BaseDataset,
    BaseEvaluationSuite,
    BaseGenerationProtocol,
    BaseModalityConfig,
    BaseModalityImplementation,
    BaseProcessor,
    create_modality_factory,
    EvaluationMetrics,
    Modality,
    ModalityBatch,
    ModalityConfig,
    ModalityData,
    ModelAdapter,
    validate_modality_interface,
)
from .image.base import ImageModality
from .molecular.modality import MolecularModality
from .protein.modality import ProteinModality
from .registry import get_modality, list_modalities, register_modality
from .tabular.base import TabularModality
from .text.base import TextModality
from .timeseries.base import TimeseriesModality


__all__ = [
    # Core protocols and interfaces
    "Modality",
    "ModelAdapter",
    "BaseGenerationProtocol",
    # Base classes for implementation
    "BaseModalityConfig",
    "BaseModalityImplementation",
    "BaseDataset",
    "BaseEvaluationSuite",
    "BaseProcessor",
    # Type aliases
    "ModalityData",
    "ModalityBatch",
    "ModalityConfig",
    "EvaluationMetrics",
    # Utilities
    "create_modality_factory",
    "validate_modality_interface",
    # Concrete modality implementations
    "AudioModality",
    "ProteinModality",
    "MolecularModality",
    "ImageModality",
    "TabularModality",
    "TextModality",
    "TimeseriesModality",
    # Registry functions
    "get_modality",
    "register_modality",
    "list_modalities",
]
