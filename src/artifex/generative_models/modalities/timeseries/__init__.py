"""Typed-config timeseries modality helpers for temporal sequence processing.

Construct `TimeseriesModality` with `TimeseriesModalityConfig` plus explicit
`rngs`. The retained public surface uses the `Timeseries*` names only; legacy
`TimeSeries*` aliases are not part of the supported surface. Public evaluation
lives in `TimeseriesEvaluationSuite` and `compute_timeseries_metrics(...)`.
"""

from .base import (
    DecompositionMethod,
    TimeseriesModality,
    TimeseriesModalityConfig,
    TimeseriesRepresentation,
)
from .datasets import (
    create_simple_timeseries_dataset,
    create_synthetic_timeseries_dataset,
    generate_synthetic_timeseries,
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
    "generate_synthetic_timeseries",
    "create_synthetic_timeseries_dataset",
    "create_simple_timeseries_dataset",
    "TimeseriesEvaluationSuite",
    "compute_timeseries_metrics",
    "TimeseriesProcessor",
    "FourierProcessor",
    "MultiScaleProcessor",
    "TrendDecompositionProcessor",
]
