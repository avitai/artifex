"""Evaluation metrics for artifex.generative_models.core.evaluation."""

from .base import DistributionMetric, FeatureBasedMetric, MetricModule, SequenceMetric
from .general import DensityPrecisionRecall, PrecisionRecall
from .image import FrechetInceptionDistance, InceptionScore
from .pipeline import EvaluationPipeline, MetricComposer, ModalityMetrics
from .registry import MetricsRegistry
from .text import Perplexity


__all__ = [
    # Base classes
    "MetricModule",
    "FeatureBasedMetric",
    "DistributionMetric",
    "SequenceMetric",
    # Pipeline and registry
    "EvaluationPipeline",
    "MetricComposer",
    "ModalityMetrics",
    "MetricsRegistry",
    # Image metrics
    "FrechetInceptionDistance",
    "InceptionScore",
    # Text metrics
    "Perplexity",
    # General metrics
    "PrecisionRecall",
    "DensityPrecisionRecall",
]
