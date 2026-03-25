"""Core benchmark metric initialization utilities.

The shared ``MetricBase`` stays config-agnostic. Benchmark metrics adapt
``EvaluationConfig`` through the internal helper in this module.
"""

from __future__ import annotations

from typing import Any

import flax.nnx as nnx

from artifex.generative_models.core.configuration import EvaluationConfig
from artifex.generative_models.core.protocols.metrics import MetricBase


def _require_evaluation_config(config: EvaluationConfig) -> EvaluationConfig:
    if not isinstance(config, EvaluationConfig):
        raise TypeError(f"config must be an EvaluationConfig, got {type(config).__name__}")
    return config


def _scoped_metric_params(config: EvaluationConfig, metric_key: str | None) -> dict[str, Any]:
    if metric_key is None:
        return {}
    params = config.metric_params.get(metric_key)
    return dict(params) if isinstance(params, dict) else {}


def _init_metric_from_config(
    metric: MetricBase,
    *,
    config: EvaluationConfig,
    rngs: nnx.Rngs,
    metric_key: str | None,
    modality: str,
    higher_is_better: bool,
    name: str | None = None,
) -> dict[str, Any]:
    config = _require_evaluation_config(config)
    scoped_params = _scoped_metric_params(config, metric_key)
    resolved_modality = scoped_params.get(
        "modality",
        config.metric_params.get("modality", modality),
    )
    resolved_higher_is_better = scoped_params.get(
        "higher_is_better",
        config.metric_params.get("higher_is_better", higher_is_better),
    )

    MetricBase.__init__(
        metric,
        name=config.name if name is None else name,
        batch_size=config.eval_batch_size,
        rngs=rngs,
        modality=resolved_modality,
        higher_is_better=resolved_higher_is_better,
    )
    metric.config = config
    metric.eval_batch_size = config.eval_batch_size
    return scoped_params


__all__ = ["MetricBase"]
