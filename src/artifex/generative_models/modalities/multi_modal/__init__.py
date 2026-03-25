"""Experimental multi-modal helper package for image, text, and audio.

This helper layer provides aligned dataset generation, fusion helpers, and
evaluation utilities for direct imports from
`artifex.generative_models.modalities.multi_modal`.

It is not registry-backed, is not available through `get_modality(...)`, and
should be treated as an experimental helper surface rather than a shared
factory-ready modality owner.
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
    create_paired_multi_modal_dataset,
    create_synthetic_multi_modal_dataset,
    generate_multi_modal_data,
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


__all__ = [
    # Base classes
    "MultiModalModality",
    "MultiModalModalityConfig",
    "MultiModalRepresentation",
    # Datasets
    "generate_multi_modal_data",
    "create_synthetic_multi_modal_dataset",
    "create_paired_multi_modal_dataset",
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
