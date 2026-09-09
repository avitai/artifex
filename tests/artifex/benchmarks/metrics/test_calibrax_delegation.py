"""The benchmark metric layer computes through calibrax and registers its backbone metrics."""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest
from calibrax.metrics import MetricRegistry, MetricSignature, MetricTier
from calibrax.metrics.functional.generative import (
    frechet_feature_distance,
    inception_score,
    inception_score_per_split,
)
from calibrax.metrics.functional.text import perplexity

import artifex.benchmarks.metrics as benchmark_metrics
from artifex.benchmarks.metrics.image import (
    create_fid_metric,
    create_is_metric,
    fid_from_backbone,
    inception_score_from_backbone,
)
from artifex.benchmarks.metrics.precision_recall import (
    precision_from_backbone,
    recall_from_backbone,
)
from artifex.benchmarks.metrics.text import create_perplexity_metric, perplexity_from_model
from artifex.generative_models.core.configuration import EvaluationConfig


TIER_ONE_METRICS = {
    "fid_from_backbone": (fid_from_backbone, "image"),
    "inception_score_from_backbone": (inception_score_from_backbone, "image"),
    "precision_from_backbone": (precision_from_backbone, "generative"),
    "recall_from_backbone": (recall_from_backbone, "generative"),
    "perplexity_from_model": (perplexity_from_model, "text"),
}


def _features(images: jax.Array) -> jax.Array:
    means = jnp.mean(images, axis=(1, 2, 3))
    stds = jnp.std(images, axis=(1, 2, 3))
    return jnp.stack([means, stds], axis=1)


def _logits(images: jax.Array) -> jax.Array:
    per_image = jnp.mean(images, axis=(1, 2, 3), keepdims=True)
    return jnp.tile(per_image, (1, 10)) * jnp.linspace(-2.0, 2.0, 10)


def _log_probs(inputs: jax.Array) -> jax.Array:
    return -0.5 * jnp.ones(inputs.shape, dtype=jnp.float32) - 0.1 * inputs


@pytest.fixture
def rngs() -> nnx.Rngs:
    return nnx.Rngs(0)


@pytest.fixture
def images() -> tuple[jax.Array, jax.Array]:
    key = jax.random.key(3)
    real = jax.random.uniform(key, (12, 8, 8, 3))
    generated = jax.random.uniform(jax.random.fold_in(key, 1), (12, 8, 8, 3)) * 0.8
    return real, generated


class TestTierOneRegistration:
    """Importing the benchmark metrics registers the backbone metrics in calibrax's registry."""

    @pytest.mark.parametrize("name", sorted(TIER_ONE_METRICS))
    def test_registered_as_frozen_backbone(self, name: str) -> None:
        assert benchmark_metrics is not None
        entry = MetricRegistry().get(name)
        function, domain = TIER_ONE_METRICS[name]

        assert entry.tier == MetricTier.FROZEN_BACKBONE
        assert entry.domain == domain
        assert entry.signature == MetricSignature.CUSTOM
        assert entry.fn is function
        assert entry.properties.is_jit_compatible is False
        assert entry.description

    def test_tier_one_listing_is_exactly_these(self) -> None:
        names = {entry.name for entry in MetricRegistry().list_by_tier(MetricTier.FROZEN_BACKBONE)}

        assert names == set(TIER_ONE_METRICS)

    def test_no_name_shadows_a_calibrax_function(self) -> None:
        registry = MetricRegistry()
        for name in (
            "frechet_feature_distance",
            "inception_score",
            "manifold_precision",
            "perplexity",
        ):
            assert registry.get(name).tier == MetricTier.PURE_FUNCTION


class TestFrechetInceptionDistance:
    """FID is calibrax's Fréchet feature distance on the extractor's features."""

    def test_function_equals_calibrax(self, images: tuple[jax.Array, jax.Array]) -> None:
        real, generated = images
        expected = float(frechet_feature_distance(_features(real), _features(generated)))

        assert fid_from_backbone(real, generated, feature_extractor=_features) == pytest.approx(
            expected, abs=1e-6
        )

    def test_metric_equals_function(
        self, rngs: nnx.Rngs, images: tuple[jax.Array, jax.Array]
    ) -> None:
        real, generated = images
        metric = create_fid_metric(rngs, feature_extractor=_features)

        result = metric.compute(real, generated)

        assert set(result) == {"fid_score"}
        assert result["fid_score"] == pytest.approx(
            fid_from_backbone(real, generated, feature_extractor=_features), abs=1e-6
        )

    def test_identical_sets_score_zero(self, images: tuple[jax.Array, jax.Array]) -> None:
        real, _ = images

        assert fid_from_backbone(real, real, feature_extractor=_features) == pytest.approx(
            0.0, abs=1e-4
        )


class TestInceptionScore:
    """Inception score is calibrax's score on the softmax of the classifier's logits."""

    def test_function_equals_calibrax(self, images: tuple[jax.Array, jax.Array]) -> None:
        _, generated = images
        probabilities = nnx.softmax(_logits(generated), axis=-1)
        expected = float(inception_score(probabilities, splits=3))

        assert inception_score_from_backbone(
            generated, classifier=_logits, splits=3
        ) == pytest.approx(expected, abs=1e-6)

    def test_metric_reports_mean_and_spread(
        self, rngs: nnx.Rngs, images: tuple[jax.Array, jax.Array]
    ) -> None:
        real, generated = images
        metric = create_is_metric(rngs, classifier=_logits, splits=3)
        probabilities = nnx.softmax(_logits(generated), axis=-1)
        per_split = inception_score_per_split(probabilities, splits=3)

        result = metric.compute(real, generated)

        assert set(result) == {"inception_score", "inception_score_std"}
        assert result["inception_score"] == pytest.approx(float(jnp.mean(per_split)), abs=1e-6)
        assert result["inception_score_std"] == pytest.approx(float(jnp.std(per_split)), abs=1e-6)

    def test_more_splits_than_samples_is_an_error(
        self, rngs: nnx.Rngs, images: tuple[jax.Array, jax.Array]
    ) -> None:
        real, generated = images
        metric = create_is_metric(rngs, classifier=_logits, splits=20)

        with pytest.raises(ValueError, match="splits"):
            metric.compute(real, generated)
        assert metric.splits == 20

    def test_demo_mock_classifier_yields_probabilities(self, rngs: nnx.Rngs) -> None:
        config = EvaluationConfig(
            name="is_demo",
            metrics=["inception_score"],
            metric_params={
                "inception_score": {"mock_inception": True, "demo_mode": True, "splits": 2}
            },
        )
        metric = create_is_metric(rngs, mock_inception=True, splits=2)
        metric_from_config = benchmark_metrics.ISMetric(rngs=rngs, config=config)
        generated = jnp.ones((6, 8, 8, 3)) * 0.4

        for candidate in (metric, metric_from_config):
            result = candidate.compute(generated, generated)
            assert 1.0 <= result["inception_score"] <= 1000.0


class TestPerplexity:
    """Perplexity is calibrax's masked perplexity on the model's log-probabilities."""

    def test_function_equals_calibrax(self) -> None:
        inputs = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        mask = jnp.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
        expected = float(perplexity(_log_probs(inputs), mask=mask))

        assert perplexity_from_model(inputs, model=_log_probs, mask=mask) == pytest.approx(
            expected, abs=1e-5
        )
        assert perplexity_from_model(inputs, model=_log_probs) == pytest.approx(
            float(perplexity(_log_probs(inputs))), abs=1e-5
        )

    def test_metric_scores_through_the_model(self, rngs: nnx.Rngs) -> None:
        inputs = jnp.ones((2, 5), dtype=jnp.int32)
        mask = jnp.array([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
        metric = create_perplexity_metric(rngs=rngs, model=_log_probs)

        result = metric.compute(None, inputs, mask=mask)

        assert set(result) == {"perplexity"}
        assert result["perplexity"] == pytest.approx(
            perplexity_from_model(inputs, model=_log_probs, mask=mask), abs=1e-5
        )
        assert metric.higher_is_better is False

    def test_supported_mode_requires_a_model(self, rngs: nnx.Rngs) -> None:
        with pytest.raises(ValueError, match="model"):
            create_perplexity_metric(rngs=rngs)

    def test_model_must_be_callable(self, rngs: nnx.Rngs) -> None:
        config = EvaluationConfig(
            name="ppl",
            metrics=["perplexity"],
            metric_params={"perplexity": {"model": "gpt2"}},
        )
        with pytest.raises(TypeError, match="callable"):
            benchmark_metrics.PerplexityMetric(rngs=rngs, config=config)

    def test_demo_mock_scores_the_uniform_vocabulary(self, rngs: nnx.Rngs) -> None:
        metric = create_perplexity_metric(rngs=rngs, use_mock=True)

        result = metric.compute(None, ["a short sentence", "another one"])

        assert result["perplexity"] == pytest.approx(10000.0, rel=1e-4)
        assert metric.compute(None, [""])["perplexity"] == jnp.inf
