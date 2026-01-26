"""Multi-modal modality for combining multiple data modalities.

This module provides functionality for working with multiple modalities
simultaneously, including cross-modal alignment, fusion strategies, and
unified evaluation metrics.
"""

from artifex.generative_models.modalities.multi_modal.adapters import (
    create_multi_modal_adapter,
    MultiModalAdapter,
)
from artifex.generative_models.modalities.multi_modal.base import (
    MultiModalModality,
    MultiModalModalityConfig,
    MultiModalRepresentation,
)
from artifex.generative_models.modalities.multi_modal.datasets import (
    create_aligned_dataset,
    create_synthetic_multi_modal_dataset,
    MultiModalDataset,
)
from artifex.generative_models.modalities.multi_modal.evaluation import (
    compute_multi_modal_metrics,
    multi_modal_consistency_loss,
    MultiModalEvaluationSuite,
)
from artifex.generative_models.modalities.multi_modal.representations import (
    CrossModalAttention,
    CrossModalProcessor,
    HierarchicalFusion,
    ModalityDropout,
    ModalityFusionProcessor,
    MultiModalProcessor,
)


# Register multi-modal modality
# from artifex.generative_models.modalities.registry import register_modality
#
# register_modality("multi_modal", MultiModalModality)

__all__ = [
    # Base classes
    "MultiModalModality",
    "MultiModalModalityConfig",
    "MultiModalRepresentation",
    # Datasets
    "MultiModalDataset",
    "create_synthetic_multi_modal_dataset",
    "create_aligned_dataset",
    # Evaluation
    "MultiModalEvaluationSuite",
    "compute_multi_modal_metrics",
    "multi_modal_consistency_loss",
    # Processors
    "MultiModalProcessor",
    "CrossModalProcessor",
    "ModalityFusionProcessor",
    "CrossModalAttention",
    "HierarchicalFusion",
    "ModalityDropout",
    # Adapters
    "MultiModalAdapter",
    "create_multi_modal_adapter",
]
