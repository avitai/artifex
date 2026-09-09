"""Precision and recall of a generative model over k-NN manifolds.

The metrics of "Improved Precision and Recall Metric for Assessing Generative
Models" (Kynkäänniemi et al., 2019): precision is the fraction of generated
samples inside the k-nearest-neighbour manifold of the real features, recall the
fraction of real samples inside the generated manifold. The manifold estimates
and their density-weighted variants are calibrax's; this module adds the
feature-extraction step, the config-driven metric class and the benchmark that
samples a model.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, cast

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from calibrax.core.models import MetricDirection
from calibrax.metrics import MetricProperties, MetricSignature, MetricTier, register_metric
from calibrax.metrics.functional.generative import (
    density_weighted_precision,
    density_weighted_recall,
    manifold_precision,
    manifold_recall,
)

from artifex.benchmarks.core import Benchmark, BenchmarkConfig, BenchmarkResult
from artifex.benchmarks.metrics.core import _init_metric_from_config, MetricBase
from artifex.generative_models.core.configuration import EvaluationConfig
from artifex.generative_models.core.protocols.evaluation import (
    BenchmarkModelProtocol,
    DatasetProtocol,
)


FeatureExtractor = Callable[[jax.Array], jax.Array]

_DEFAULT_K = 3
_BACKBONE_PROPERTIES = MetricProperties(is_differentiable=False, is_jit_compatible=False)


def _features(samples: Any, feature_extractor: FeatureExtractor | None) -> jax.Array:
    """Extract features and flatten them to one row per sample."""
    array = jnp.asarray(samples)
    features = array if feature_extractor is None else jnp.asarray(feature_extractor(array))
    return features.reshape(features.shape[0], math.prod(features.shape[1:]))


def _require_neighbours(real: jax.Array, generated: jax.Array, k: int) -> None:
    """Fail when ``k`` leaves no neighbour on either side.

    Args:
        real: Real features or samples, batch first.
        generated: Generated features or samples, batch first.
        k: Neighbour order of the manifold estimate.

    Raises:
        ValueError: If ``k`` is below one or not below both sample counts.
    """
    smallest = min(real.shape[0], generated.shape[0])
    if k < 1 or k >= smallest:
        raise ValueError(
            f"k must be at least 1 and below the smaller sample count ({smallest}), got k={k}"
        )


@register_metric(
    "precision_from_backbone",
    tier=MetricTier.FROZEN_BACKBONE,
    domain="generative",
    direction=MetricDirection.HIGHER,
    description="Manifold precision of generated samples after a caller-supplied feature extractor",
    signature=MetricSignature.CUSTOM,
    properties=_BACKBONE_PROPERTIES,
)
def precision_from_backbone(
    real: Any,
    generated: Any,
    *,
    feature_extractor: FeatureExtractor | None = None,
    k: int = _DEFAULT_K,
    density_weighted: bool = False,
) -> float:
    """Fraction of generated samples inside the real k-NN manifold.

    Args:
        real: Real samples, any shape with the batch first.
        generated: Generated samples with the same trailing shape.
        feature_extractor: Backbone mapping a batch to features; identity when None.
        k: Neighbour order of the manifold estimate.
        density_weighted: Weight by the density of the real manifold.

    Returns:
        Precision in [0, 1].
    """
    real_features = _features(real, feature_extractor)
    generated_features = _features(generated, feature_extractor)
    _require_neighbours(real_features, generated_features, k)
    scorer = density_weighted_precision if density_weighted else manifold_precision
    return float(scorer(real_features, generated_features, k=k))


@register_metric(
    "recall_from_backbone",
    tier=MetricTier.FROZEN_BACKBONE,
    domain="generative",
    direction=MetricDirection.HIGHER,
    description="Manifold recall of real samples after a caller-supplied feature extractor",
    signature=MetricSignature.CUSTOM,
    properties=_BACKBONE_PROPERTIES,
)
def recall_from_backbone(
    real: Any,
    generated: Any,
    *,
    feature_extractor: FeatureExtractor | None = None,
    k: int = _DEFAULT_K,
    density_weighted: bool = False,
) -> float:
    """Fraction of real samples inside the generated k-NN manifold.

    Args:
        real: Real samples, any shape with the batch first.
        generated: Generated samples with the same trailing shape.
        feature_extractor: Backbone mapping a batch to features; identity when None.
        k: Neighbour order of the manifold estimate.
        density_weighted: Weight by the density of the generated manifold.

    Returns:
        Recall in [0, 1].
    """
    real_features = _features(real, feature_extractor)
    generated_features = _features(generated, feature_extractor)
    _require_neighbours(real_features, generated_features, k)
    scorer = density_weighted_recall if density_weighted else manifold_recall
    return float(scorer(real_features, generated_features, k=k))


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; zero when either is zero."""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


class PrecisionRecallMetric(MetricBase):
    """Precision, recall and F1 of generated samples against real ones.

    Config parameters under ``metric_params["precision_recall"]``: ``feature_extractor``
    (a callable, identity when absent), ``k`` (neighbour order, 3) and
    ``density_weighted`` (False).
    """

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig) -> None:
        """Initialise from an evaluation configuration."""
        params = _init_metric_from_config(
            self,
            config=config,
            rngs=rngs,
            metric_key="precision_recall",
            modality="general",
            higher_is_better=True,
        )
        extractor = params.get("feature_extractor")
        if extractor is not None and not callable(extractor):
            raise TypeError("feature_extractor must be callable")
        self.feature_extractor: FeatureExtractor | None = (
            None if extractor is None else cast(FeatureExtractor, extractor)
        )
        self.k = int(params.get("k", _DEFAULT_K))
        self.density_weighted = bool(params.get("density_weighted", False))

    def validate_inputs(self, real_data: Any, generated_data: Any) -> None:
        """Check the two sample sets can be compared.

        Args:
            real_data: Real samples, batch first.
            generated_data: Generated samples, batch first.

        Raises:
            ValueError: If the trailing shapes differ or ``k`` leaves no neighbour.
        """
        real = jnp.asarray(real_data)
        generated = jnp.asarray(generated_data)
        if real.shape[1:] != generated.shape[1:]:
            raise ValueError(
                "real and generated samples must share their trailing shape: "
                f"{real.shape[1:]} != {generated.shape[1:]}"
            )
        _require_neighbours(real, generated, self.k)

    def compute(self, real_data: Any, generated_data: Any, **_kwargs: Any) -> dict[str, float]:
        """Score generated samples against real ones."""
        options = {
            "feature_extractor": self.feature_extractor,
            "k": self.k,
            "density_weighted": self.density_weighted,
        }
        precision = precision_from_backbone(real_data, generated_data, **options)
        recall = recall_from_backbone(real_data, generated_data, **options)
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score(precision, recall),
        }


def create_precision_recall_metric(
    rngs: nnx.Rngs,
    *,
    feature_extractor: FeatureExtractor | None = None,
    k: int = _DEFAULT_K,
    density_weighted: bool = False,
    batch_size: int = 32,
    config_name: str = "precision_recall_metric",
) -> PrecisionRecallMetric:
    """Create a precision-recall metric with a typed configuration."""
    params: dict[str, object] = {"k": k, "density_weighted": density_weighted}
    if feature_extractor is not None:
        params["feature_extractor"] = feature_extractor
    config = EvaluationConfig(
        name=config_name,
        metrics=["precision_recall"],
        metric_params={"precision_recall": params},
        eval_batch_size=batch_size,
    )
    return PrecisionRecallMetric(rngs=rngs, config=config)


class PrecisionRecallBenchmark(Benchmark):
    """Sample a model and score precision, recall and F1 against a dataset."""

    def __init__(
        self,
        k: int = _DEFAULT_K,
        num_samples: int = 1000,
        random_seed: int | None = None,
        *,
        density_weighted: bool = False,
    ) -> None:
        """Configure the sample count, the seed and the manifold estimate."""
        super().__init__(
            config=BenchmarkConfig(
                name="precision_recall",
                description="Manifold precision and recall of a generative model",
                metric_names=["precision", "recall", "f1_score"],
            )
        )
        self.k = k
        self.num_samples = num_samples
        self.random_seed = 42 if random_seed is None else random_seed
        self.density_weighted = density_weighted

    def _real_samples(self, dataset: DatasetProtocol | Sequence[Any] | Any, key: jax.Array) -> Any:
        """Take the dataset as an array, or draw ``num_samples`` items from an indexable one."""
        if hasattr(dataset, "__array__"):
            return dataset
        count = min(len(dataset), self.num_samples)
        indices = jax.random.choice(key, jnp.arange(len(dataset)), shape=(count,), replace=False)
        return jnp.stack([jnp.asarray(dataset[int(index)]) for index in indices])

    def run(
        self,
        model: BenchmarkModelProtocol,
        dataset: DatasetProtocol | None = None,
    ) -> BenchmarkResult:
        """Sample the model and score it against the dataset.

        Args:
            model: Model exposing ``sample(rngs=..., batch_size=...)``.
            dataset: Real samples as an array, or an indexable dataset to subsample.

        Returns:
            Result named ``precision_recall`` with precision, recall and F1.

        Raises:
            ValueError: If no dataset is given.
        """
        if dataset is None:
            raise ValueError("Dataset is required for precision-recall")
        key = jax.random.key(self.random_seed)
        generated = model.sample(rngs=nnx.Rngs(sample=key), batch_size=self.num_samples)
        real = self._real_samples(dataset, key)
        options = {"k": self.k, "density_weighted": self.density_weighted}
        precision = precision_from_backbone(real, generated, **options)
        recall = recall_from_backbone(real, generated, **options)
        return self.result(
            getattr(model, "model_name", "unknown"),
            {"precision": precision, "recall": recall, "f1_score": f1_score(precision, recall)},
        )
