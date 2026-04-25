"""Explicit evaluation pipeline for caller-supplied runtime metrics.

The retained pipeline only supports metric families with caller-supplied
runtime dependencies. Registry ownership lives in ``calibrax.metrics``.
Unsupported metric specs fail fast.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx

from artifex.generative_models.core.configuration import EvaluationConfig
from artifex.generative_models.core.evaluation.metrics.image import (
    FrechetInceptionDistance,
    InceptionScore,
)
from artifex.generative_models.core.evaluation.metrics.text import Perplexity
from artifex.generative_models.core.protocols.metrics import MetricBase


_MetricInvoker = Callable[[MetricBase, dict[str, Any]], dict[str, float]]


def _invoke_fid(metric: MetricBase, payload: dict[str, Any]) -> dict[str, float]:
    return metric.compute(payload["real"], payload["generated"])


def _invoke_inception_score(metric: MetricBase, payload: dict[str, Any]) -> dict[str, float]:
    return metric.compute(payload["generated"])


def _invoke_perplexity(metric: MetricBase, payload: dict[str, Any]) -> dict[str, float]:
    if "inputs" not in payload and "log_probs" not in payload:
        raise ValueError(
            'text evaluation payload for perplexity must include "inputs" or "log_probs".'
        )
    return metric.compute(
        inputs=payload.get("inputs"),
        log_probs=payload.get("log_probs"),
        mask=payload.get("mask"),
    )


@dataclass(frozen=True)
class _MetricSpec:
    spec: str
    modality: str
    metric_name: str
    metric_type: type[MetricBase]
    invoke: _MetricInvoker
    dependency_name: str | None = None
    constructor_defaults: tuple[tuple[str, Any], ...] = ()
    required_payload_keys: tuple[str, ...] = ()


_SUPPORTED_METRICS: dict[str, _MetricSpec] = {
    "image:fid": _MetricSpec(
        spec="image:fid",
        modality="image",
        metric_name="fid",
        metric_type=FrechetInceptionDistance,
        invoke=_invoke_fid,
        dependency_name="feature_extractor",
        required_payload_keys=("real", "generated"),
    ),
    "image:is": _MetricSpec(
        spec="image:is",
        modality="image",
        metric_name="is",
        metric_type=InceptionScore,
        invoke=_invoke_inception_score,
        dependency_name="classifier",
        constructor_defaults=(("splits", 10),),
        required_payload_keys=("generated",),
    ),
    "text:perplexity": _MetricSpec(
        spec="text:perplexity",
        modality="text",
        metric_name="perplexity",
        metric_type=Perplexity,
        invoke=_invoke_perplexity,
        dependency_name="model",
    ),
}


class EvaluationPipeline(nnx.Module):
    """Explicit multi-modality evaluation pipeline."""

    def __init__(self, config: EvaluationConfig, *, rngs: nnx.Rngs):
        """Initialize the evaluation pipeline."""
        super().__init__()
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be EvaluationConfig, got {type(config).__name__}")

        self.config = config
        self.rngs = rngs
        self.metrics = nnx.Dict({})
        self.metric_specs_by_modality: dict[str, list[_MetricSpec]] = {}

        grouped_metrics: dict[str, list[MetricBase]] = {}
        for metric_spec in config.metrics:
            spec = self._resolve_metric_spec(metric_spec)
            grouped_metrics.setdefault(spec.modality, []).append(self._build_metric(spec))
            self.metric_specs_by_modality.setdefault(spec.modality, []).append(spec)

        for modality, metrics in grouped_metrics.items():
            self.metrics[modality] = nnx.List(metrics)

    @staticmethod
    def _resolve_metric_spec(metric_spec: str) -> _MetricSpec:
        if ":" not in metric_spec:
            raise ValueError(
                'EvaluationPipeline metrics must use explicit "modality:metric" specs.'
            )

        spec = _SUPPORTED_METRICS.get(metric_spec)
        if spec is None:
            supported = ", ".join(sorted(_SUPPORTED_METRICS))
            raise ValueError(
                f"Unsupported evaluation metric spec: {metric_spec}. "
                f"Supported specs are {supported}."
            )
        return spec

    def _metric_config(self, spec: _MetricSpec) -> dict[str, Any]:
        return {
            **self.config.metric_params.get(spec.metric_name, {}),
            **self.config.metric_params.get(spec.spec, {}),
        }

    @staticmethod
    def _require_callable_dependency(
        metric_name: str,
        dependency_name: str,
        metric_config: dict[str, Any],
    ) -> Any:
        dependency = metric_config.get(dependency_name)
        if dependency is None:
            raise ValueError(
                f"{metric_name} requires an explicit callable {dependency_name}. "
                "Artifex does not ship a placeholder default."
            )
        if not callable(dependency):
            raise TypeError(
                f"{metric_name} requires {dependency_name} to be callable, "
                f"got {type(dependency).__name__}."
            )
        return dependency

    def _build_metric(self, spec: _MetricSpec) -> MetricBase:
        metric_config = self._metric_config(spec)
        kwargs: dict[str, Any] = {
            "batch_size": metric_config.get("batch_size", 32),
            "name": spec.metric_name,
            "rngs": self.rngs,
        }
        for key, default in spec.constructor_defaults:
            kwargs[key] = metric_config.get(key, default)
        if spec.dependency_name is not None:
            kwargs[spec.dependency_name] = self._require_callable_dependency(
                spec.metric_name,
                spec.dependency_name,
                metric_config,
            )
        return spec.metric_type(**kwargs)

    @staticmethod
    def _require_keys(modality: str, payload: dict[str, Any], keys: tuple[str, ...]) -> None:
        missing = [key for key in keys if key not in payload]
        if missing:
            missing_csv = ", ".join(missing)
            raise ValueError(
                f"{modality} evaluation payload is missing required keys: {missing_csv}."
            )

    def _compute_metric(
        self,
        spec: _MetricSpec,
        metric: MetricBase,
        modality_data: dict[str, Any],
    ) -> dict[str, float]:
        if spec.required_payload_keys:
            self._require_keys(spec.modality, modality_data, spec.required_payload_keys)
        return spec.invoke(metric, modality_data)

    def evaluate(self, data: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
        """Evaluate the configured modalities and metrics."""
        results: dict[str, dict[str, float]] = {}
        for modality, metrics in self.metrics.items():
            if modality not in data:
                raise ValueError(
                    f"Missing evaluation payload for configured modality {modality!r}."
                )

            modality_results: dict[str, float] = {}
            specs = self.metric_specs_by_modality[modality]
            for metric, spec in zip(metrics, specs, strict=False):
                modality_results.update(self._compute_metric(spec, metric, data[modality]))
            results[modality] = modality_results
        return results
