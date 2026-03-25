"""Tests for evaluation metrics and the narrowed evaluation pipeline."""

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from artifex.benchmarks.metrics.audio import MelCepstralMetric, SpectralMetric
from artifex.benchmarks.metrics.image import FIDMetric, LPIPSMetric
from artifex.benchmarks.metrics.text import BLEUMetric, PerplexityMetric, ROUGEMetric
from artifex.generative_models.core.configuration import EvaluationConfig
from artifex.generative_models.core.evaluation.metrics.image import (
    FrechetInceptionDistance,
    InceptionScore,
)
from artifex.generative_models.core.evaluation.metrics.pipeline import EvaluationPipeline
from artifex.generative_models.core.evaluation.metrics.text import Perplexity
from artifex.generative_models.core.protocols.metrics import MetricBase


def _test_feature_extractor(images):
    means = jnp.mean(images, axis=(1, 2, 3))
    stds = jnp.std(images, axis=(1, 2, 3))
    return jnp.stack([means, stds], axis=1)


def _test_classifier(images):
    means = jnp.mean(images, axis=(1, 2, 3), keepdims=True)
    return jnp.tile(means, (1, 10))


def _test_language_model(inputs):
    return jnp.full(inputs.shape, -0.5)


@pytest.fixture
def rngs():
    """Standard fixture for rngs."""
    return nnx.Rngs(42)


@pytest.fixture
def sample_images():
    """Sample image data for testing."""
    return {"real": jnp.ones((32, 64, 64, 3)), "generated": jnp.ones((32, 64, 64, 3)) * 0.8}


@pytest.fixture
def sample_text():
    """Sample text data for testing."""
    return {
        "reference": ["hello world test", "this is sample text"],
        "generated": ["hello world test", "this sample text is"],
    }


@pytest.fixture
def sample_audio():
    """Sample audio data for testing."""
    return {"real": jnp.ones((32, 16000)), "generated": jnp.ones((32, 16000)) * 0.9}


class TestMetricProtocol:
    """Test the base metric protocol."""

    def test_metric_protocol_creation_without_config(self, rngs):
        """MetricBase should support direct protocol-oriented construction."""

        class TestMetric(MetricBase):
            def __init__(self, *, rngs: nnx.Rngs):
                super().__init__(
                    name="test_metric",
                    batch_size=16,
                    modality="image",
                    higher_is_better=True,
                    rngs=rngs,
                )

            def compute(self, real_data, generated_data, **kwargs):
                return {"score": 0.5}

            def validate_inputs(self, real_data, generated_data):
                pass

        metric = TestMetric(rngs=rngs)
        assert metric.name == "test_metric"
        assert metric.batch_size == 16
        assert metric.modality == "image"
        assert metric.higher_is_better

    def test_metric_protocol_requires_name_without_config(self, rngs):
        """MetricBase should fail fast when direct construction omits a name."""

        class TestMetric(MetricBase):
            def __init__(self, *, rngs: nnx.Rngs):
                super().__init__(rngs=rngs)

            def compute(self, real_data, generated_data, **kwargs):
                return {"score": 0.5}

            def validate_inputs(self, real_data, generated_data):
                pass

        with pytest.raises(TypeError, match="name must be provided"):
            TestMetric(rngs=rngs)

    def test_metric_computation(self, rngs, sample_images):
        """Test metric computation workflow."""

        class TestMetric(MetricBase):
            def __init__(self, *, rngs: nnx.Rngs):
                super().__init__(name="test_metric_eval", modality="image", rngs=rngs)

            def compute(self, real_data, generated_data, **kwargs):
                return {"mse": float(jnp.mean((real_data - generated_data) ** 2))}

            def validate_inputs(self, real_data, generated_data):
                if real_data.shape != generated_data.shape:
                    raise ValueError("Shapes must match")

        metric = TestMetric(rngs=rngs)
        result = metric.compute(sample_images["real"], sample_images["generated"])
        assert "mse" in result
        assert isinstance(result["mse"], float)
        assert result["mse"] > 0

    def test_core_runtime_metrics_inherit_metric_base(self, rngs):
        """The retained core runtime metrics should share the MetricBase protocol."""
        metrics = [
            FrechetInceptionDistance(feature_extractor=_test_feature_extractor, rngs=rngs),
            InceptionScore(classifier=_test_classifier, rngs=rngs),
            Perplexity(model=_test_language_model, rngs=rngs),
        ]

        assert all(isinstance(metric, MetricBase) for metric in metrics)


class TestImageMetrics:
    """Test image-specific metrics."""

    def test_fid_metric(self, rngs, sample_images):
        """Test FID metric computation."""
        config = EvaluationConfig(
            name="fid_test",
            metrics=["fid"],
            metric_params={
                "fid": {
                    "mock_inception": True,
                    "higher_is_better": False,
                }
            },
            eval_batch_size=16,
        )

        fid = FIDMetric(config=config, rngs=rngs)
        result = fid.compute(sample_images["real"], sample_images["generated"])
        assert "fid_score" in result
        assert isinstance(result["fid_score"], float)
        assert result["fid_score"] >= 0

    def test_lpips_metric(self, rngs, sample_images):
        """Test LPIPS perceptual metric."""
        config = EvaluationConfig(
            name="lpips_test",
            metrics=["lpips"],
            metric_params={
                "lpips": {
                    "mock_implementation": True,
                    "higher_is_better": False,
                }
            },
            eval_batch_size=32,
        )

        lpips = LPIPSMetric(config=config, rngs=rngs)
        result = lpips.compute(sample_images["real"], sample_images["generated"])
        assert "lpips_distance" in result
        assert isinstance(result["lpips_distance"], float)


class TestTextMetrics:
    """Test text-specific metrics."""

    def test_bleu_metric(self, rngs, sample_text):
        """Test BLEU score computation."""
        config = EvaluationConfig(
            name="bleu_test",
            metrics=["bleu"],
            metric_params={"bleu": {"max_n": 4, "smoothing": True}},
            eval_batch_size=32,
        )
        bleu = BLEUMetric(config=config, rngs=rngs)
        result = bleu.compute(sample_text["reference"], sample_text["generated"])
        assert "bleu_score" in result
        assert isinstance(result["bleu_score"], float)
        assert 0 <= result["bleu_score"] <= 1

    def test_rouge_metric(self, rngs, sample_text):
        """Test ROUGE score computation."""
        config = EvaluationConfig(
            name="rouge_test",
            metrics=["rouge"],
            metric_params={"rouge": {"rouge_types": ["rouge-1", "rouge-l"], "use_stemmer": True}},
            eval_batch_size=32,
        )
        rouge = ROUGEMetric(config=config, rngs=rngs)
        result = rouge.compute(sample_text["reference"], sample_text["generated"])
        assert "rouge-1" in result
        assert "rouge-l" in result
        assert isinstance(result["rouge-1"], float)

    def test_perplexity_metric_requires_explicit_demo_mock_opt_in(self, rngs, sample_text):
        """Test perplexity computation only through the retained explicit demo path."""
        config = EvaluationConfig(
            name="perplexity_test",
            metrics=["perplexity"],
            metric_params={
                "perplexity": {
                    "model_name": "mock",
                    "max_length": 512,
                    "use_mock": True,
                    "demo_mode": True,
                }
            },
            eval_batch_size=16,
        )
        perplexity = PerplexityMetric(config=config, rngs=rngs)
        result = perplexity.compute(sample_text["reference"], sample_text["generated"])
        assert "perplexity" in result
        assert isinstance(result["perplexity"], float)
        assert result["perplexity"] > 0

    def test_perplexity_metric_rejects_implicit_backend(self, rngs):
        """PerplexityMetric should fail fast without an explicit demo opt-in."""
        config = EvaluationConfig(
            name="perplexity_test",
            metrics=["perplexity"],
            metric_params={"perplexity": {"model_name": "mock", "max_length": 512}},
            eval_batch_size=16,
        )

        with pytest.raises(
            RuntimeError, match="does not ship a benchmark-grade language-model backend"
        ):
            PerplexityMetric(config=config, rngs=rngs)


class TestAudioMetrics:
    """Test audio-specific metrics."""

    def test_spectral_metric(self, rngs, sample_audio):
        """Test spectral convergence metric."""
        config = EvaluationConfig(
            name="spectral_test",
            metrics=["spectral"],
            metric_params={"spectral": {"n_fft": 1024, "hop_length": 256, "sample_rate": 16000}},
            eval_batch_size=32,
        )
        spectral = SpectralMetric(config=config, rngs=rngs)
        result = spectral.compute(sample_audio["real"], sample_audio["generated"])
        assert "spectral_convergence" in result
        assert isinstance(result["spectral_convergence"], float)

    def test_mel_cepstral_metric(self, rngs, sample_audio):
        """Test mel-cepstral distortion."""
        config = EvaluationConfig(
            name="mcd_test",
            metrics=["mcd"],
            metric_params={"mcd": {"n_mels": 80, "sample_rate": 16000, "n_fft": 1024}},
            eval_batch_size=32,
        )
        mcd = MelCepstralMetric(config=config, rngs=rngs)
        result = mcd.compute(sample_audio["real"], sample_audio["generated"])
        assert "mel_cepstral_distortion" in result
        assert isinstance(result["mel_cepstral_distortion"], float)


class TestEvaluationPipeline:
    """Test the narrowed evaluation pipeline."""

    def test_pipeline_creation(self, rngs):
        """Test evaluation pipeline initialization."""
        config = EvaluationConfig(
            name="test_pipeline",
            metrics=["image:fid", "image:is", "text:perplexity"],
            metric_params={
                "fid": {"feature_extractor": _test_feature_extractor},
                "is": {"classifier": _test_classifier},
                "perplexity": {"model": _test_language_model},
            },
            eval_batch_size=32,
        )
        pipeline = EvaluationPipeline(config=config, rngs=rngs)
        assert len(pipeline.metrics) == 2
        assert len(pipeline.metrics["image"]) == 2
        assert len(pipeline.metrics["text"]) == 1

    def test_pipeline_rejects_missing_metric_dependencies(self, rngs):
        """Supported metrics should fail fast when dependencies are missing."""
        with pytest.raises(ValueError, match="feature_extractor"):
            EvaluationPipeline(
                EvaluationConfig(name="missing_fid", metrics=["image:fid"]),
                rngs=rngs,
            )
        with pytest.raises(ValueError, match="classifier"):
            EvaluationPipeline(
                EvaluationConfig(name="missing_is", metrics=["image:is"]),
                rngs=rngs,
            )
        with pytest.raises(ValueError, match="model"):
            EvaluationPipeline(
                EvaluationConfig(name="missing_ppl", metrics=["text:perplexity"]),
                rngs=rngs,
            )

    def test_pipeline_rejects_unprefixed_and_unsupported_metrics(self, rngs):
        """Metrics must be explicit and supported."""
        with pytest.raises(ValueError, match="modality:metric"):
            EvaluationPipeline(
                EvaluationConfig(name="unprefixed", metrics=["fid"]),
                rngs=rngs,
            )
        with pytest.raises(ValueError, match="Unsupported evaluation metric spec"):
            EvaluationPipeline(
                EvaluationConfig(
                    name="unsupported",
                    metrics=["text:bleu"],
                    metric_params={"bleu": {"model": _test_language_model}},
                ),
                rngs=rngs,
            )

    def test_pipeline_evaluation(self, rngs):
        """Test complete pipeline evaluation with explicit dependencies."""
        config = EvaluationConfig(
            name="test_eval",
            metrics=["image:fid", "image:is", "text:perplexity"],
            metric_params={
                "fid": {"feature_extractor": _test_feature_extractor},
                "is": {"classifier": _test_classifier},
                "perplexity": {"model": _test_language_model},
            },
            eval_batch_size=32,
        )
        pipeline = EvaluationPipeline(config=config, rngs=rngs)
        results = pipeline.evaluate(
            {
                "image": {
                    "real": jnp.ones((12, 16, 16, 3)),
                    "generated": jnp.ones((12, 16, 16, 3)) * 0.7,
                },
                "text": {"inputs": jnp.ones((12, 12), dtype=jnp.int32)},
            }
        )
        assert "image" in results
        assert "text" in results
        assert "fid" in results["image"]
        assert "is_mean" in results["image"]
        assert "is_std" in results["image"]
        assert "perplexity" in results["text"]

    def test_pipeline_requires_modality_payloads(self, rngs):
        """Configured modalities should require matching payloads."""
        config = EvaluationConfig(
            name="missing_payload",
            metrics=["image:fid"],
            metric_params={"fid": {"feature_extractor": _test_feature_extractor}},
        )
        pipeline = EvaluationPipeline(config=config, rngs=rngs)
        with pytest.raises(ValueError, match="Missing evaluation payload"):
            pipeline.evaluate({})


class TestInceptionScoreEdgeCases:
    """Test explicit failure modes for Inception Score."""

    def test_inception_score_rejects_nonpositive_splits(self, rngs):
        metric = InceptionScore(classifier=_test_classifier, rngs=rngs)

        with pytest.raises(ValueError, match="positive integer"):
            metric.compute(jnp.ones((4, 8, 8, 3)), splits=0)

    def test_inception_score_rejects_too_many_splits(self, rngs):
        metric = InceptionScore(classifier=_test_classifier, rngs=rngs)

        with pytest.raises(ValueError, match="at least as many samples as splits"):
            metric.compute(jnp.ones((4, 8, 8, 3)), splits=5)


from artifex.generative_models.core.evaluation.metrics.metric_ops import (
    compute_cdf,
    compute_ks_distance,
    nearest_neighbors,
    pairwise_distances,
)


class TestNearestNeighbors:
    """Test JAX nearest neighbors implementation."""

    def test_nearest_neighbors_basic(self):
        query = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        data = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        distances, indices = nearest_neighbors(query, data, k=2)
        assert distances.shape == (2, 2)
        assert indices.shape == (2, 2)
        assert jnp.all(distances >= 0)
        assert indices[0, 0] == 0
        assert distances[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_nearest_neighbors_distance_calculation(self):
        query = jnp.array([[0.0, 0.0]])
        data = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        distances, indices = nearest_neighbors(query, data, k=2)
        expected_distances = jnp.array([1.0, 1.0])
        assert jnp.allclose(distances[0], expected_distances, atol=1e-6)

    def test_nearest_neighbors_k_larger_than_data(self):
        query = jnp.array([[0.0, 0.0]])
        data = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        distances, indices = nearest_neighbors(query, data, k=5)
        assert distances.shape[1] == 2
        assert indices.shape[1] == 2


class TestPairwiseDistances:
    """Test pairwise distance computation."""

    def test_pairwise_distances_basic(self):
        x = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        y = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        distances = pairwise_distances(x, y)
        assert distances.shape == (2, 2)
        assert jnp.all(distances >= 0)
        assert distances[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert distances[0, 1] == pytest.approx(1.0, abs=1e-6)


class TestComputeCDF:
    """Test empirical CDF computation."""

    def test_compute_cdf_basic(self):
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        eval_points = jnp.array([0.0, 2.5, 5.0, 6.0])
        cdf = compute_cdf(data, eval_points)
        assert cdf.shape == eval_points.shape
        assert cdf[0] == 0.0
        assert cdf[1] == 0.4
        assert cdf[2] == 1.0
        assert cdf[3] == 1.0

    def test_compute_cdf_monotonic(self):
        data = jnp.array([1.0, 3.0, 2.0, 5.0, 4.0])
        eval_points = jnp.sort(jnp.array([0.0, 1.5, 2.5, 3.5, 4.5, 6.0]))
        cdf = compute_cdf(data, eval_points)
        for i in range(1, len(cdf)):
            assert cdf[i] >= cdf[i - 1]


class TestComputeKSDistance:
    """Test Kolmogorov-Smirnov distance computation."""

    def test_ks_distance_identical_distributions(self):
        data1 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data2 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ks_dist = compute_ks_distance(data1, data2)
        assert ks_dist == pytest.approx(0.0, abs=1e-6)
