"""Text modality for generative models.

This module provides comprehensive text generation capabilities including:
- Multiple tokenization strategies (word-level, subword, character)
- Integration with Transformer, RNN, and Autoregressive models
- Comprehensive text evaluation metrics (BLEU, ROUGE, perplexity)
- Position encoding and attention extensions
- Integration with benchmark framework

Example:
    >>> from artifex.generative_models.modalities.text import TextModality, TextRepresentation
    >>> modality = TextModality(max_length=512, vocab_size=10000)
    >>> text = modality.generate(n_samples=4)
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
    SimpleTextDataset,
    SyntheticTextDataset,
    TextDataset,
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
    "TextDataset",
    "SyntheticTextDataset",
    "SimpleTextDataset",
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
