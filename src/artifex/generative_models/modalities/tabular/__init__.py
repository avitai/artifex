"""Typed-config tabular modality helpers for structured data generation.

Construct `TabularModality` with `TabularModalityConfig`. The retained public
evaluation surface is lightweight: `compute_tabular_metrics(...)` and
`TabularEvaluationSuite` report numerical KS, correlation, and privacy metrics,
while richer categorical and ordinal helpers stay private implementation
details.
"""

from .base import TabularModality, TabularModalityConfig
from .datasets import (
    compute_feature_statistics,
    create_simple_tabular_dataset,
    create_synthetic_tabular_dataset,
    generate_synthetic_tabular_data,
)
from .evaluation import compute_tabular_metrics, TabularEvaluationSuite
from .representations import CategoricalEncoder, NumericalProcessor, TabularProcessor


__all__ = [
    "TabularModality",
    "TabularModalityConfig",
    "generate_synthetic_tabular_data",
    "compute_feature_statistics",
    "create_synthetic_tabular_dataset",
    "create_simple_tabular_dataset",
    "TabularEvaluationSuite",
    "compute_tabular_metrics",
    "TabularProcessor",
    "CategoricalEncoder",
    "NumericalProcessor",
]
