"""Timeseries modality for temporal sequence generation."""

from .base import (
    DecompositionMethod,
    TimeseriesModality,
    TimeseriesModalityConfig,
    TimeseriesRepresentation,
)
from .datasets import (
    create_simple_timeseries_dataset,
    create_synthetic_timeseries_dataset,
    SyntheticTimeseriesDataset,
)
from .evaluation import compute_timeseries_metrics, TimeseriesEvaluationSuite
from .representations import (
    FourierProcessor,
    MultiScaleProcessor,
    TimeseriesProcessor,
    TrendDecompositionProcessor,
)


__all__ = [
    "TimeseriesModality",
    "TimeseriesModalityConfig",
    "TimeseriesRepresentation",
    "DecompositionMethod",
    "SyntheticTimeseriesDataset",
    "create_synthetic_timeseries_dataset",
    "create_simple_timeseries_dataset",
    "TimeseriesEvaluationSuite",
    "compute_timeseries_metrics",
    "TimeseriesProcessor",
    "FourierProcessor",
    "MultiScaleProcessor",
    "TrendDecompositionProcessor",
]
