"""Bridge between Artifex benchmark types and CalibraX types.

Converts Artifex benchmark results and configs to and from
`calibrax.core.BenchmarkResult` and `calibrax.core.Metric` for unified
benchmark reporting.
"""

from dataclasses import asdict
from typing import Any

from calibrax.core import BenchmarkResult as CalibraxResult, Metric as CalibraxMetric

from artifex.benchmarks.core.foundation import (
    BenchmarkConfig as ArtifexConfig,
    BenchmarkResult as ArtifexResult,
)


def sanitize_jax_value(value: Any) -> Any:
    """Convert JAX or numpy scalars to Python primitives."""
    if isinstance(value, dict):
        return {k: sanitize_jax_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_jax_value(v) for v in value]
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return value.item()
    return value


def config_to_dict(config: ArtifexConfig) -> dict[str, Any]:
    """Convert an Artifex BenchmarkConfig to a serializable dict."""
    return sanitize_jax_value(asdict(config))


def to_calibrax_result(
    result: ArtifexResult,
    *,
    domain: str = "",
) -> CalibraxResult:
    """Convert an Artifex BenchmarkResult to a CalibraX BenchmarkResult."""
    return CalibraxResult(
        name=result.benchmark_name,
        domain=domain,
        tags={"model_name": result.model_name},
        metrics={name: CalibraxMetric(value=float(val)) for name, val in result.metrics.items()},
        metadata=sanitize_jax_value(result.metadata),
    )


def from_calibrax_result(result: CalibraxResult) -> ArtifexResult:
    """Convert a CalibraX BenchmarkResult to an Artifex BenchmarkResult."""
    return ArtifexResult(
        benchmark_name=result.name,
        model_name=result.tags.get("model_name", ""),
        metrics={name: metric.value for name, metric in result.metrics.items()},
        metadata=dict(result.metadata),
    )
