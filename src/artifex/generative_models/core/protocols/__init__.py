"""Protocol definitions for artifex.generative_models.core."""

from artifex.generative_models.core.protocols.evaluation import (
    BatchableDatasetProtocol,
    BenchmarkModelProtocol,
    DatasetProtocol,
)
from artifex.generative_models.core.protocols.metrics import MetricBase
from artifex.generative_models.core.protocols.training import NoiseScheduleProtocol


__all__ = [
    "BatchableDatasetProtocol",
    "BenchmarkModelProtocol",
    "DatasetProtocol",
    "MetricBase",
    "NoiseScheduleProtocol",
]
