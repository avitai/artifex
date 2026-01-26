"""Protocol definitions for artifex.generative_models.core."""

from artifex.generative_models.core.protocols.benchmarks import (
    BenchmarkBase,
    BenchmarkWithValidation,
)
from artifex.generative_models.core.protocols.configuration import (
    BaseConfig,
    ConfigTemplate,
)
from artifex.generative_models.core.protocols.evaluation import (
    BatchableDatasetProtocol,
    DatasetProtocol,
    ModelProtocol,
)
from artifex.generative_models.core.protocols.metrics import MetricBase
from artifex.generative_models.core.protocols.training import NoiseScheduleProtocol


__all__ = [
    "BaseConfig",
    "BatchableDatasetProtocol",
    "BenchmarkBase",
    "BenchmarkWithValidation",
    "ConfigTemplate",
    "DatasetProtocol",
    "MetricBase",
    "ModelProtocol",
    "NoiseScheduleProtocol",
]
