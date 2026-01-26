"""Core metrics protocol.

This module re-exports the MetricProtocol from artifex.generative_models.core.protocols.metrics
to maintain backward compatibility.
"""

from artifex.generative_models.core.evaluation.metrics.pipeline import (
    EvaluationPipeline,
    MetricComposer,
    ModalityMetrics,
)
from artifex.generative_models.core.protocols.metrics import MetricBase


__all__ = [
    "MetricBase",
    "EvaluationPipeline",
    "MetricComposer",
    "ModalityMetrics",
]
