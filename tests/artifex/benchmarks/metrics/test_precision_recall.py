"""Precision and recall over k-NN manifolds (Kynkäänniemi et al., 2019) through calibrax."""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from calibrax.metrics.functional.generative import (
    density_weighted_precision,
    density_weighted_recall,
    manifold_precision,
    manifold_recall,
)
from sklearn.datasets import make_blobs
from tests.utils.test_models import MockModel

from artifex.benchmarks import BenchmarkResult
from artifex.benchmarks.metrics.precision_recall import (
    create_precision_recall_metric,
    f1_score,
    precision_from_backbone,
    PrecisionRecallBenchmark,
    PrecisionRecallMetric,
    recall_from_backbone,
)
from artifex.generative_models.core.configuration import EvaluationConfig


def _blobs(centers: object, *, std: float, seed: int, n: int = 100) -> jax.Array:
    samples, _ = make_blobs(
        n_samples=n, n_features=2, centers=centers, cluster_std=std, random_state=seed
    )
    return jnp.asarray(samples, dtype=jnp.float32)


@pytest.fixture
def rngs() -> nnx.Rngs:
    return nnx.Rngs(0)


@pytest.fixture
def real() -> jax.Array:
    return _blobs(2, std=0.5, seed=42)


@pytest.fixture
def near_one_mode(real: jax.Array) -> jax.Array:
    return _blobs(np.asarray(real[0:1]), std=0.7, seed=43)


@pytest.fixture
def far_away() -> jax.Array:
    return _blobs([[10.0, 10.0]], std=0.5, seed=44)


class TestBackboneFunctions:
    """The registered functions are calibrax's manifold metrics after feature extraction."""

    def test_precision_and_recall_equal_calibrax_on_extracted_features(
        self, real: jax.Array, near_one_mode: jax.Array
    ) -> None:
        def extractor(samples: jax.Array) -> jax.Array:
            return samples * 2.0

        expected_precision = manifold_precision(extractor(real), extractor(near_one_mode), k=3)
        expected_recall = manifold_recall(extractor(real), extractor(near_one_mode), k=3)

        precision = precision_from_backbone(real, near_one_mode, feature_extractor=extractor, k=3)
        recall = recall_from_backbone(real, near_one_mode, feature_extractor=extractor, k=3)

        assert precision == pytest.approx(float(expected_precision), abs=1e-6)
        assert recall == pytest.approx(float(expected_recall), abs=1e-6)

    def test_density_weighted_variant_equals_calibrax(
        self, real: jax.Array, near_one_mode: jax.Array
    ) -> None:
        expected_precision = density_weighted_precision(real, near_one_mode, k=5)
        expected_recall = density_weighted_recall(real, near_one_mode, k=5)

        precision = precision_from_backbone(real, near_one_mode, k=5, density_weighted=True)
        recall = recall_from_backbone(real, near_one_mode, k=5, density_weighted=True)

        assert precision == pytest.approx(float(expected_precision), abs=1e-6)
        assert recall == pytest.approx(float(expected_recall), abs=1e-6)

    def test_identical_sets_score_one(self, real: jax.Array) -> None:
        assert precision_from_backbone(real, real, k=3) == pytest.approx(1.0)
        assert recall_from_backbone(real, real, k=3) == pytest.approx(1.0)

    def test_disjoint_sets_score_zero(self, real: jax.Array, far_away: jax.Array) -> None:
        assert precision_from_backbone(real, far_away, k=3) < 0.1
        assert recall_from_backbone(real, far_away, k=3) < 0.1

    def test_one_covered_mode_keeps_precision_and_halves_recall(
        self, real: jax.Array, near_one_mode: jax.Array
    ) -> None:
        assert precision_from_backbone(real, near_one_mode, k=3) > 0.5
        assert 0.1 < recall_from_backbone(real, near_one_mode, k=3) < 0.9

    def test_samples_beyond_two_dimensions_are_flattened(self) -> None:
        images = jax.random.normal(jax.random.key(0), (12, 4, 4, 3))
        flat = images.reshape(12, -1)

        assert precision_from_backbone(images, images, k=2) == pytest.approx(
            float(manifold_precision(flat, flat, k=2))
        )

    def test_k_must_leave_a_neighbour(self, real: jax.Array) -> None:
        with pytest.raises(ValueError, match="k"):
            precision_from_backbone(real[:3], real, k=3)
        with pytest.raises(ValueError, match="k"):
            recall_from_backbone(real, real[:2], k=2)
        with pytest.raises(ValueError, match="k"):
            precision_from_backbone(real, real, k=0)

    def test_empty_sets_are_rejected(self, real: jax.Array) -> None:
        with pytest.raises(ValueError, match="k"):
            precision_from_backbone(real, jnp.zeros((0, 2)), k=3)

    def test_feature_dimensions_must_agree(self, real: jax.Array) -> None:
        with pytest.raises(ValueError, match="feature dimension"):
            precision_from_backbone(real, jnp.zeros((10, 5)), k=3)


class TestF1Score:
    """F1 is the harmonic mean, zero when either side is zero."""

    @pytest.mark.parametrize(
        ("precision", "recall", "expected"),
        [(0.8, 0.8, 0.8), (0.6, 0.4, 0.48), (0.0, 0.5, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
    )
    def test_values(self, precision: float, recall: float, expected: float) -> None:
        assert f1_score(precision, recall) == pytest.approx(expected)


class TestPrecisionRecallMetric:
    """The config-driven metric class over the backbone functions."""

    def test_reports_precision_recall_and_f1(
        self, rngs: nnx.Rngs, real: jax.Array, near_one_mode: jax.Array
    ) -> None:
        metric = create_precision_recall_metric(rngs, k=3)

        result = metric.compute(real, near_one_mode)

        assert set(result) == {"precision", "recall", "f1_score"}
        assert result["precision"] == pytest.approx(
            precision_from_backbone(real, near_one_mode, k=3)
        )
        assert result["recall"] == pytest.approx(recall_from_backbone(real, near_one_mode, k=3))
        assert result["f1_score"] == pytest.approx(f1_score(result["precision"], result["recall"]))

    def test_config_parameters_reach_the_functions(
        self, rngs: nnx.Rngs, real: jax.Array, near_one_mode: jax.Array
    ) -> None:
        def extractor(samples: jax.Array) -> jax.Array:
            return samples * 3.0

        config = EvaluationConfig(
            name="pr",
            metrics=["precision_recall"],
            metric_params={
                "precision_recall": {
                    "feature_extractor": extractor,
                    "k": 5,
                    "density_weighted": True,
                }
            },
        )
        metric = PrecisionRecallMetric(rngs=rngs, config=config)

        assert metric.k == 5
        assert metric.density_weighted is True
        assert metric.feature_extractor is extractor
        assert metric.compute(real, near_one_mode)["precision"] == pytest.approx(
            precision_from_backbone(
                real, near_one_mode, feature_extractor=extractor, k=5, density_weighted=True
            )
        )

    def test_defaults_and_direction(self, rngs: nnx.Rngs) -> None:
        metric = create_precision_recall_metric(rngs)

        assert metric.name == "precision_recall_metric"
        assert metric.k == 3
        assert metric.density_weighted is False
        assert metric.feature_extractor is None
        assert metric.higher_is_better is True
        assert metric.modality == "general"

    def test_validate_inputs(self, rngs: nnx.Rngs, real: jax.Array) -> None:
        metric = create_precision_recall_metric(rngs, k=3)

        metric.validate_inputs(real, real)
        with pytest.raises(ValueError, match="trailing"):
            metric.validate_inputs(real, jnp.zeros((10, 5)))
        with pytest.raises(ValueError, match="k"):
            metric.validate_inputs(real, real[:3])


class TestPrecisionRecallBenchmark:
    """The benchmark samples a model and scores it against a dataset."""

    def setup_method(self) -> None:
        self.rngs = nnx.Rngs(params=jax.random.key(0))
        cluster_a = jnp.ones((50, 2)) * jnp.array([5.0, 5.0])
        cluster_b = jnp.ones((50, 2)) * jnp.array([-5.0, -5.0])
        self.real_data = jnp.concatenate([cluster_a, cluster_b])
        self.perfect_samples = jnp.concatenate([cluster_a[:25], cluster_b[:25]])
        self.low_recall_samples = cluster_a
        self.low_precision_samples = jnp.concatenate(
            [cluster_a[:20], cluster_b[:20], jnp.ones((20, 2)) * jnp.array([15.0, 15.0])]
        )

    def test_perfect_model(self) -> None:
        benchmark = PrecisionRecallBenchmark(k=3)

        result = benchmark.run(
            model=MockModel(self.perfect_samples, rngs=self.rngs), dataset=self.real_data
        )

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "precision_recall"
        assert result.model_name == "mock_model"
        assert result.metrics["precision"] == pytest.approx(1.0)
        assert result.metrics["recall"] == pytest.approx(1.0)
        assert result.metrics["f1_score"] == pytest.approx(1.0)

    def test_missing_mode_lowers_recall_only(self) -> None:
        result = PrecisionRecallBenchmark(k=3).run(
            model=MockModel(self.low_recall_samples, rngs=self.rngs), dataset=self.real_data
        )

        assert result.metrics["precision"] == pytest.approx(1.0)
        assert result.metrics["recall"] == pytest.approx(0.5)

    def test_extra_mode_lowers_precision_only(self) -> None:
        result = PrecisionRecallBenchmark(k=3).run(
            model=MockModel(self.low_precision_samples, rngs=self.rngs), dataset=self.real_data
        )

        assert result.metrics["precision"] == pytest.approx(2.0 / 3.0)
        assert result.metrics["recall"] == pytest.approx(1.0)

    def test_small_mock_runs_are_scored_not_assumed(self) -> None:
        """A mock model with few samples is scored like any other model."""
        far_samples = jnp.ones((20, 2)) * jnp.array([15.0, 15.0])
        benchmark = PrecisionRecallBenchmark(k=3, num_samples=20)

        result = benchmark.run(
            model=MockModel(far_samples, rngs=self.rngs, model_name="mock_model"),
            dataset=self.real_data,
        )

        assert result.metrics["precision"] == pytest.approx(0.0)
        assert result.metrics["recall"] == pytest.approx(0.0)

    def test_num_samples_bounds_the_draw(self) -> None:
        class CountingModel(nnx.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model_name = "counting"
                self.requested: int | None = None

            def sample(self, *, rngs: nnx.Rngs, batch_size: int = 1) -> jax.Array:
                self.requested = batch_size
                return jnp.ones((batch_size, 2)) * jnp.array([5.0, 5.0])

        model = CountingModel()
        result = PrecisionRecallBenchmark(k=3, num_samples=20).run(
            model=model, dataset=self.real_data
        )

        assert model.requested == 20
        assert result.metrics["precision"] == pytest.approx(1.0)

    def test_indexable_dataset_is_subsampled(self) -> None:
        dataset = [self.real_data[i] for i in range(self.real_data.shape[0])]

        result = PrecisionRecallBenchmark(k=3, num_samples=40, random_seed=7).run(
            model=MockModel(self.perfect_samples, rngs=self.rngs), dataset=dataset
        )

        assert result.metrics["precision"] == pytest.approx(1.0)

    def test_seed_makes_runs_repeatable(self) -> None:
        model = MockModel(self.perfect_samples, rngs=self.rngs)
        first = PrecisionRecallBenchmark(k=3, random_seed=42).run(
            model=model, dataset=self.real_data
        )
        second = PrecisionRecallBenchmark(k=3, random_seed=42).run(
            model=model, dataset=self.real_data
        )

        assert first.metrics == second.metrics

    def test_density_weighted_benchmark(self) -> None:
        result = PrecisionRecallBenchmark(k=5, density_weighted=True).run(
            model=MockModel(self.perfect_samples, rngs=self.rngs), dataset=self.real_data
        )

        assert result.metrics["precision"] == pytest.approx(
            precision_from_backbone(
                self.real_data, self.perfect_samples, k=5, density_weighted=True
            )
        )

    def test_dataset_is_required(self) -> None:
        with pytest.raises(ValueError, match="Dataset"):
            PrecisionRecallBenchmark(k=3).run(model=MockModel(self.perfect_samples, rngs=self.rngs))

    def test_empty_model_output_is_an_error(self) -> None:
        empty = jnp.zeros((0, 2))
        with pytest.raises(ValueError, match="k"):
            PrecisionRecallBenchmark(k=3).run(
                model=MockModel(empty, rngs=self.rngs), dataset=self.real_data
            )

    def test_mismatched_dimensions_are_an_error(self) -> None:
        high_dim = jax.random.normal(jax.random.key(1), (50, 100))
        with pytest.raises(ValueError, match="feature dimension"):
            PrecisionRecallBenchmark(k=3).run(
                model=MockModel(high_dim, rngs=self.rngs), dataset=self.real_data
            )

    def test_metric_names_match_the_config(self) -> None:
        benchmark = PrecisionRecallBenchmark(k=3)
        result = benchmark.run(
            model=MockModel(self.perfect_samples, rngs=self.rngs), dataset=self.real_data
        )

        benchmark.validate_metrics(result.metrics)
