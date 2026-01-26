"""Tabular modality for structured data generation."""

from .base import TabularModality, TabularModalityConfig
from .datasets import (
    create_simple_tabular_dataset,
    create_synthetic_tabular_dataset,
    SyntheticTabularDataset,
)
from .evaluation import compute_tabular_metrics, TabularEvaluationSuite
from .representations import CategoricalEncoder, NumericalProcessor, TabularProcessor


__all__ = [
    "TabularModality",
    "TabularModalityConfig",
    "SyntheticTabularDataset",
    "create_synthetic_tabular_dataset",
    "create_simple_tabular_dataset",
    "TabularEvaluationSuite",
    "compute_tabular_metrics",
    "TabularProcessor",
    "CategoricalEncoder",
    "NumericalProcessor",
]
