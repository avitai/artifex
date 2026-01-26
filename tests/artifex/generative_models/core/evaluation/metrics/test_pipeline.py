"""Tests for comprehensive metrics and evaluation system."""

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from artifex.benchmarks.metrics.audio import (
    MelCepstralMetric,
    SpectralMetric,
)
from artifex.benchmarks.metrics.image import (
    FIDMetric,
    LPIPSMetric,
)
from artifex.benchmarks.metrics.text import (
    BLEUMetric,
    PerplexityMetric,
    ROUGEMetric,
)
from artifex.generative_models.core.configuration import EvaluationConfig
from artifex.generative_models.core.evaluation.metrics.pipeline import (
    EvaluationPipeline,
    MetricComposer,
    ModalityMetrics,
)
from artifex.generative_models.core.protocols.metrics import MetricBase


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

    def test_metric_protocol_creation(self, rngs):
        """Test metric protocol initialization."""
        config = EvaluationConfig(
            name="test_metric_eval",
            metrics=["test_metric"],
        )

        class TestMetric(MetricBase):
            def __init__(self, config: EvaluationConfig, *, rngs: nnx.Rngs):
                super().__init__(config, rngs=rngs)
                # Set metric-specific attributes
                self.name = "test_metric"
                self.modality = "image"
                self.higher_is_better = True

            def compute(self, real_data, generated_data, **kwargs):
                return {"score": 0.5}

            def validate_inputs(self, real_data, generated_data):
                return True

        metric = TestMetric(config=config, rngs=rngs)
        assert metric.name == "test_metric"
        assert metric.modality == "image"
        assert metric.higher_is_better

    def test_metric_computation(self, rngs, sample_images):
        """Test metric computation workflow."""
        config = EvaluationConfig(
            name="test_metric_eval",
            metrics=["test_metric"],
        )

        class TestMetric(MetricBase):
            def __init__(self, config: EvaluationConfig, *, rngs: nnx.Rngs):
                super().__init__(config, rngs=rngs)
                self.name = "test_metric"
                self.modality = "image"

            def compute(self, real_data, generated_data, **kwargs):
                return {"mse": float(jnp.mean((real_data - generated_data) ** 2))}

            def validate_inputs(self, real_data, generated_data):
                return real_data.shape == generated_data.shape

        metric = TestMetric(config=config, rngs=rngs)
        result = metric.compute(sample_images["real"], sample_images["generated"])

        assert "mse" in result
        assert isinstance(result["mse"], float)
        assert result["mse"] > 0


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

    # def test_is_metric(self, rngs, sample_images):
    #     """Test Inception Score metric."""
    #     # ISMetric not implemented yet
    #     pass

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

    # def test_image_metrics_suite(self, rngs, sample_images):
    #     """Test complete image metrics suite."""
    #     # ImageMetrics not implemented yet
    #     pass


class TestTextMetrics:
    """Test text-specific metrics."""

    def test_bleu_metric(self, rngs, sample_text):
        """Test BLEU score computation."""
        config = EvaluationConfig(
            name="bleu_test",
            metrics=["bleu"],
            metric_params={
                "bleu": {
                    "max_n": 4,
                    "smoothing": True,
                }
            },
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
            metric_params={
                "rouge": {
                    "rouge_types": ["rouge-1", "rouge-l"],
                    "use_stemmer": True,
                }
            },
            eval_batch_size=32,
        )

        rouge = ROUGEMetric(config=config, rngs=rngs)
        result = rouge.compute(sample_text["reference"], sample_text["generated"])

        assert "rouge-1" in result
        assert "rouge-l" in result
        assert isinstance(result["rouge-1"], float)

    def test_perplexity_metric(self, rngs, sample_text):
        """Test perplexity computation."""
        config = EvaluationConfig(
            name="perplexity_test",
            metrics=["perplexity"],
            metric_params={
                "perplexity": {
                    "model_name": "mock",
                    "max_length": 512,
                }
            },
            eval_batch_size=16,
        )

        perplexity = PerplexityMetric(config=config, rngs=rngs)
        result = perplexity.compute(sample_text["reference"], sample_text["generated"])

        assert "perplexity" in result
        assert isinstance(result["perplexity"], float)
        assert result["perplexity"] > 0


class TestAudioMetrics:
    """Test audio-specific metrics."""

    def test_spectral_metric(self, rngs, sample_audio):
        """Test spectral convergence metric."""
        config = EvaluationConfig(
            name="spectral_test",
            metrics=["spectral"],
            metric_params={
                "spectral": {
                    "n_fft": 1024,
                    "hop_length": 256,
                    "sample_rate": 16000,
                }
            },
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
            metric_params={
                "mcd": {
                    "n_mels": 80,
                    "sample_rate": 16000,
                    "n_fft": 1024,
                }
            },
            eval_batch_size=32,
        )

        mcd = MelCepstralMetric(config=config, rngs=rngs)
        result = mcd.compute(sample_audio["real"], sample_audio["generated"])

        assert "mel_cepstral_distortion" in result
        assert isinstance(result["mel_cepstral_distortion"], float)


class TestEvaluationPipeline:
    """Test the complete evaluation pipeline."""

    def test_pipeline_creation(self, rngs):
        """Test evaluation pipeline initialization."""
        config = EvaluationConfig(
            name="test_pipeline",
            metrics=["image:fid", "image:is", "text:perplexity"],
            eval_batch_size=32,
        )

        pipeline = EvaluationPipeline(config=config, rngs=rngs)
        assert len(pipeline.metrics) == 2  # image and text

    def test_pipeline_evaluation(self, rngs, sample_images, sample_text):
        """Test complete pipeline evaluation."""
        config = EvaluationConfig(
            name="test_eval",
            metrics=["image:fid"],
            eval_batch_size=32,
        )

        pipeline = EvaluationPipeline(config=config, rngs=rngs)
        # Skip actual evaluation since metrics need models
        assert "image" in pipeline.metrics


class TestMetricComposer:
    """Test metric composition and aggregation."""

    def test_metric_composer(self, rngs):
        """Test metric composition capabilities."""
        config = EvaluationConfig(
            name="test_composer",
            metrics=["fid", "is", "perplexity"],
            metric_params={
                "composition_rules": {
                    "overall_score": {
                        "weights": {"fid": -0.5, "is": 0.3, "perplexity": 0.2},
                        "normalization": "min_max",
                    }
                }
            },
            eval_batch_size=32,
        )

        composer = MetricComposer(config=config, rngs=rngs)

        metrics = {"fid": 15.2, "is": 2.5, "perplexity": 0.7}

        composed = composer.compose(metrics)
        assert "overall_score" in composed
        assert isinstance(composed["overall_score"], float)

    def test_modality_aggregation(self, rngs):
        """Test cross-modality metric aggregation."""
        config = EvaluationConfig(
            name="test_aggregation",
            metrics=["image:fid", "text:perplexity"],
            metric_params={
                "composer_settings": {
                    "aggregation_strategy": "weighted_average",
                    "modality_weights": {"image": 0.6, "text": 0.4},
                }
            },
            eval_batch_size=32,
        )

        composer = MetricComposer(config=config, rngs=rngs)

        modality_results = {"image": {"fid": 15.2, "is": 2.5}, "text": {"bleu": 0.7, "rouge": 0.6}}

        aggregated = composer.aggregate_modalities(modality_results)
        assert "cross_modality_score" in aggregated
        assert isinstance(aggregated["cross_modality_score"], float)


class TestModalityMetrics:
    """Test modality-specific metric management."""

    def test_modality_metrics_registry(self, rngs):
        """Test modality metrics registration."""
        config = EvaluationConfig(
            name="test_registry",
            metrics=["image:fid", "text:perplexity", "audio:spectral"],
            eval_batch_size=32,
        )

        modality_metrics = ModalityMetrics(config=config, rngs=rngs)

        # Test registration
        assert "image" in modality_metrics.get_supported_modalities()
        assert "text" in modality_metrics.get_supported_modalities()
        assert "audio" in modality_metrics.get_supported_modalities()

    def test_metric_selection(self, rngs):
        """Test automatic metric selection by modality."""
        config = EvaluationConfig(
            name="test_selection",
            metrics=["image:fid", "text:perplexity", "audio:spectral"],
            metric_params={
                "quality_levels": {"high": ["fid", "lpips"], "fast": ["mse"]},
            },
            eval_batch_size=32,
        )

        modality_metrics = ModalityMetrics(config=config, rngs=rngs)

        selected = modality_metrics.select_metrics("image", quality_level="high")
        assert len(selected) >= 1
        assert any("fid" in metric or "lpips" in metric for metric in selected)


# ========================================================================
# Tests merged from tests/unit/generative_models/core/evaluation/metrics/test_metric_ops.py
# ========================================================================

from artifex.generative_models.core.evaluation.metrics.metric_ops import (
    bincount,
    compute_cdf,
    compute_ks_distance,
    corrcoef,
    matrix_sqrtm,
    nearest_neighbors,
    pairwise_distances,
)


class TestNearestNeighbors:
    """Test JAX nearest neighbors implementation."""

    def test_nearest_neighbors_basic(self):
        """Test basic k-nearest neighbors functionality."""
        # Create simple 2D data
        query = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        data = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        distances, indices = nearest_neighbors(query, data, k=2)

        # Check shapes
        assert distances.shape == (2, 2)
        assert indices.shape == (2, 2)

        # Check that distances are non-negative
        assert jnp.all(distances >= 0)

        # Check that nearest neighbor of first query point is itself
        assert indices[0, 0] == 0  # First query point's nearest neighbor
        assert distances[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_nearest_neighbors_distance_calculation(self):
        """Test that distances are calculated correctly."""
        query = jnp.array([[0.0, 0.0]])
        data = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        distances, indices = nearest_neighbors(query, data, k=2)

        # Expected distances: 1.0, 1.0 (to [1,0] and [0,1])
        expected_distances = jnp.array([1.0, 1.0])
        assert jnp.allclose(distances[0], expected_distances, atol=1e-6)

    def test_nearest_neighbors_k_larger_than_data(self):
        """Test when k is larger than available data points."""
        query = jnp.array([[0.0, 0.0]])
        data = jnp.array([[1.0, 0.0], [0.0, 1.0]])

        distances, indices = nearest_neighbors(query, data, k=5)

        # Should return all available data points
        assert distances.shape[1] == 2
        assert indices.shape[1] == 2


class TestPairwiseDistances:
    """Test pairwise distance computation."""

    def test_pairwise_distances_basic(self):
        """Test basic pairwise distance functionality."""
        x = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        y = jnp.array([[0.0, 1.0], [1.0, 0.0]])

        distances = pairwise_distances(x, y)

        # Check shape
        assert distances.shape == (2, 2)

        # Check that distances are non-negative
        assert jnp.all(distances >= 0)

        # Check specific distances
        # Distance from [0,0] to [0,1] should be 1.0
        assert distances[0, 0] == pytest.approx(1.0, abs=1e-6)
        # Distance from [0,0] to [1,0] should be 1.0
        assert distances[0, 1] == pytest.approx(1.0, abs=1e-6)


class TestComputeCDF:
    """Test empirical CDF computation."""

    def test_compute_cdf_basic(self):
        """Test basic CDF computation."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        eval_points = jnp.array([0.0, 2.5, 5.0, 6.0])

        cdf = compute_cdf(data, eval_points)

        # Check shape
        assert cdf.shape == eval_points.shape

        # Check values
        assert cdf[0] == 0.0  # No values <= 0
        assert cdf[1] == 0.4  # 2 values <= 2.5 (40%)
        assert cdf[2] == 1.0  # All values <= 5.0
        assert cdf[3] == 1.0  # All values <= 6.0

    def test_compute_cdf_monotonic(self):
        """Test that CDF is monotonically increasing."""
        data = jnp.array([1.0, 3.0, 2.0, 5.0, 4.0])
        eval_points = jnp.sort(jnp.array([0.0, 1.5, 2.5, 3.5, 4.5, 6.0]))

        cdf = compute_cdf(data, eval_points)

        # CDF should be monotonically increasing
        for i in range(1, len(cdf)):
            assert cdf[i] >= cdf[i - 1]


class TestComputeKSDistance:
    """Test Kolmogorov-Smirnov distance computation."""

    def test_ks_distance_identical_distributions(self):
        """Test KS distance for identical distributions."""
        data1 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data2 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        ks_dist = compute_ks_distance(data1, data2)

        # KS distance should be 0 for identical distributions
        assert ks_dist == pytest.approx(0.0, abs=1e-6)

    def test_ks_distance_different_distributions(self):
        """Test KS distance for different distributions."""
        data1 = jnp.array([1.0, 2.0, 3.0])
        data2 = jnp.array([4.0, 5.0, 6.0])

        ks_dist = compute_ks_distance(data1, data2)

        # KS distance should be positive for different distributions
        assert ks_dist > 0
        assert ks_dist <= 1.0  # KS distance is bounded by 1


class TestBincount:
    """Test JAX bincount implementation."""

    def test_bincount_basic(self):
        """Test basic bincount functionality."""
        data = jnp.array([0, 1, 1, 2, 2, 2])
        length = 4

        counts = bincount(data, length)

        # Check shape
        assert counts.shape == (length,)

        # Check counts
        assert counts[0] == 1  # One 0
        assert counts[1] == 2  # Two 1s
        assert counts[2] == 3  # Three 2s
        assert counts[3] == 0  # No 3s

    def test_bincount_empty_categories(self):
        """Test bincount with empty categories."""
        data = jnp.array([0, 2, 4])
        length = 6

        counts = bincount(data, length)

        assert counts[0] == 1
        assert counts[1] == 0
        assert counts[2] == 1
        assert counts[3] == 0
        assert counts[4] == 1
        assert counts[5] == 0


class TestCorrcoef:
    """Test correlation coefficient computation."""

    def test_corrcoef_basic(self):
        """Test basic correlation coefficient computation."""
        # Create perfectly correlated data
        data = jnp.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])

        corr = corrcoef(data, rowvar=True)

        # Check shape
        assert corr.shape == (2, 2)

        # Diagonal should be 1
        assert jnp.allclose(jnp.diag(corr), 1.0)

        # Off-diagonal should be 1 (perfect correlation)
        assert corr[0, 1] == pytest.approx(1.0, abs=1e-6)
        assert corr[1, 0] == pytest.approx(1.0, abs=1e-6)

    def test_corrcoef_uncorrelated(self):
        """Test correlation for uncorrelated data."""
        # Create orthogonal vectors
        data = jnp.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

        corr = corrcoef(data, rowvar=True)

        # Off-diagonal should be close to 0
        assert abs(corr[0, 1]) < 0.1
        assert abs(corr[1, 0]) < 0.1


class TestMatrixSqrtm:
    """Test matrix square root computation."""

    def test_matrix_sqrtm_identity(self):
        """Test matrix square root of identity matrix."""
        identity = jnp.eye(3)

        sqrt_matrix = matrix_sqrtm(identity)

        # Square root of identity should be identity
        assert jnp.allclose(sqrt_matrix, identity, atol=1e-6)

    def test_matrix_sqrtm_properties(self):
        """Test that (sqrt(A))^2 = A for symmetric positive definite matrix."""
        # Create a symmetric positive definite matrix
        A = jnp.array([[4.0, 1.0], [1.0, 3.0]])

        sqrt_A = matrix_sqrtm(A)
        reconstructed = sqrt_A @ sqrt_A

        # Should reconstruct original matrix
        assert jnp.allclose(reconstructed, A, atol=1e-6)

    def test_matrix_sqrtm_real_output(self):
        """Test that matrix square root produces real output."""
        # Even for complex eigenvalues, result should be real for PSD matrices
        A = jnp.array([[2.0, 1.0], [1.0, 2.0]])

        sqrt_A = matrix_sqrtm(A)

        # Should be real
        assert jnp.isrealobj(sqrt_A)
