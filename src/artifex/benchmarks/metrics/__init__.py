"""Benchmark metrics: the one metric class layer, computing through calibrax.

Importing this package registers the backbone-based metrics (FID, inception score,
manifold precision and recall, model perplexity) as Tier 1 entries of
``calibrax.metrics.MetricRegistry``.
"""

from artifex.benchmarks.metrics.core import MetricBase
from artifex.benchmarks.metrics.disentanglement import (
    DisentanglementMetric,
    MutualInformationGapMetric,
    SeparationMetric,
)
from artifex.benchmarks.metrics.image import FIDMetric, ISMetric, LPIPSMetric, SSIMMetric
from artifex.benchmarks.metrics.precision_recall import (
    PrecisionRecallBenchmark,
    PrecisionRecallMetric,
)
from artifex.benchmarks.metrics.text import PerplexityMetric


__all__ = [
    "MetricBase",
    "FIDMetric",
    "ISMetric",
    "LPIPSMetric",
    "SSIMMetric",
    "PerplexityMetric",
    "PrecisionRecallBenchmark",
    "PrecisionRecallMetric",
    "MutualInformationGapMetric",
    "SeparationMetric",
    "DisentanglementMetric",
]
