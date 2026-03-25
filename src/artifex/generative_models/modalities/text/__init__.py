"""Typed-config text modality helpers for sequence preprocessing.

This package provides tokenization, sequence handling, evaluation helpers, and
processor utilities for text data. `TextModality` supports the default-config
path or an explicit `ModalityConfig`; direct keyword shortcuts like
`vocab_size=` are not part of the supported surface. Standalone text generation
remains model-owned through `TextGenerationProtocol`, public evaluation lives in
`TextEvaluationSuite`, and `TextRepresentation` is an enum rather than an
extractor object.

Example:
    >>> from flax import nnx
    >>> from artifex.generative_models.modalities.text import TextModality
    >>> modality = TextModality(rngs=nnx.Rngs(0))
    >>> tokens = modality.preprocess_text(["hello world"])
"""

from .base import (
    create_text_modality,
    TextGenerationProtocol,
    TextModality,
    TextRepresentation,
    TokenizationStrategy,
)
from .datasets import (
    create_text_dataset,
    generate_synthetic_text_data,
    generate_text_from_strings,
    simple_tokenize,
)
from .evaluation import (
    compute_text_metrics,
    TextEvaluationSuite,
    TextMetrics,
)
from .representations import (
    create_text_processor,
    PositionEncodingProcessor,
    SequenceAugmentationProcessor,
    TextProcessor,
    TokenizationProcessor,
)


__all__ = [
    # Core modality
    "TextGenerationProtocol",
    "TextModality",
    "TextRepresentation",
    "TokenizationStrategy",
    "create_text_modality",
    # Dataset handling
    "generate_synthetic_text_data",
    "generate_text_from_strings",
    "simple_tokenize",
    "create_text_dataset",
    # Evaluation
    "TextEvaluationSuite",
    "TextMetrics",
    "compute_text_metrics",
    # Representation processing
    "PositionEncodingProcessor",
    "SequenceAugmentationProcessor",
    "TextProcessor",
    "TokenizationProcessor",
    "create_text_processor",
]
