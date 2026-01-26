"""Tests for timeseries modality implementation."""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.modalities.timeseries import (
    compute_timeseries_metrics,
    create_simple_timeseries_dataset,
    create_synthetic_timeseries_dataset,
    DecompositionMethod,
    FourierProcessor,
    MultiScaleProcessor,
    SyntheticTimeseriesDataset,
    TimeseriesEvaluationSuite,
    TimeseriesModality,
    TimeseriesModalityConfig,
    TimeseriesProcessor,
    TimeseriesRepresentation,
    TrendDecompositionProcessor,
)


@pytest.fixture
def rngs():
    """Standard RNG fixture."""
    return nnx.Rngs(42)


class TestTimeseriesModalityConfig:
    """Test cases for TimeseriesModalityConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TimeseriesModalityConfig()

        assert config.sequence_length == 100
        assert config.num_features == 1
        assert config.sampling_rate == 1.0
        assert config.representation == TimeseriesRepresentation.RAW
        assert not config.use_fourier_features
        assert config.num_frequencies == 64
        assert config.max_frequency == 100.0
        assert not config.use_trend_decomposition
        assert config.decomposition_method == DecompositionMethod.SEASONAL
        assert config.decomposition_period == 24
        assert config.multi_scale_factors == [1, 2, 4]
        assert config.univariate is True
        assert config.stationary is False
        assert config.seasonal_period is None

    def test_custom_config_validation(self):
        """Test custom configuration with validation."""
        config = TimeseriesModalityConfig(
            sequence_length=200,
            num_features=3,
            univariate=False,
            use_fourier_features=True,
            num_frequencies=32,
            seasonal_period=12,
        )

        assert config.sequence_length == 200
        assert config.num_features == 3
        assert not config.univariate
        assert config.use_fourier_features
        assert config.num_frequencies == 32
        assert config.seasonal_period == 12

    def test_feature_names_auto_generation(self):
        """Test automatic generation of feature names."""
        # Univariate case
        config = TimeseriesModalityConfig(num_features=1, univariate=True)
        assert config.feature_names == ["value"]

        # Multivariate case
        config = TimeseriesModalityConfig(num_features=3, univariate=False)
        assert config.feature_names == ["feature_0", "feature_1", "feature_2"]

    def test_config_validation_errors(self):
        """Test configuration validation errors."""
        # Invalid sequence length
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            TimeseriesModalityConfig(sequence_length=0)

        # Invalid num_features
        with pytest.raises(ValueError, match="num_features must be positive"):
            TimeseriesModalityConfig(num_features=0)

        # Invalid sampling_rate
        with pytest.raises(ValueError, match="sampling_rate must be positive"):
            TimeseriesModalityConfig(sampling_rate=0.0)

        # Invalid Fourier features
        with pytest.raises(ValueError, match="num_frequencies must be positive"):
            TimeseriesModalityConfig(use_fourier_features=True, num_frequencies=0)

        # Invalid univariate setting
        with pytest.raises(ValueError, match="num_features must be 1 for univariate"):
            TimeseriesModalityConfig(univariate=True, num_features=3)

    def test_expected_shape_property(self):
        """Test expected_shape property."""
        config = TimeseriesModalityConfig(sequence_length=50, num_features=2, univariate=False)
        assert config.expected_shape == (50, 2)


class TestTimeseriesModality:
    """Test cases for TimeseriesModality."""

    def test_initialization(self, rngs):
        """Test modality initialization."""
        config = TimeseriesModalityConfig(sequence_length=100, num_features=2, univariate=False)
        modality = TimeseriesModality(config, rngs=rngs)

        assert modality.name == "timeseries"
        assert modality.sequence_length == 100
        assert modality.num_features == 2
        assert modality.sampling_rate == 1.0

    def test_preprocess_data(self, rngs):
        """Test data preprocessing."""
        config = TimeseriesModalityConfig(sequence_length=50, num_features=1)
        modality = TimeseriesModality(config, rngs=rngs)

        # Create test data
        data = jnp.ones((4, 50, 1)) * 2.0  # Batch size 4

        processed = modality.preprocess(data)

        # Check output shape
        assert processed.shape == (4, 50, 1)

        # Check normalization was applied (should be zero mean)
        assert jnp.abs(jnp.mean(processed)) < 1e-6

    def test_preprocess_validation(self, rngs):
        """Test preprocessing validation."""
        config = TimeseriesModalityConfig(sequence_length=50, num_features=1)
        modality = TimeseriesModality(config, rngs=rngs)

        # Wrong dimensions
        with pytest.raises(ValueError, match="Expected 3D input"):
            modality.preprocess(jnp.ones((50, 1)))  # Missing batch dimension

        # Wrong sequence length
        with pytest.raises(ValueError, match="Expected sequence_length"):
            modality.preprocess(jnp.ones((4, 40, 1)))  # Wrong sequence length

        # Wrong number of features
        with pytest.raises(ValueError, match="Expected 1 features"):
            modality.preprocess(jnp.ones((4, 50, 2)))  # Wrong features

    def test_postprocess_data(self, rngs):
        """Test data postprocessing."""
        config = TimeseriesModalityConfig(sequence_length=50, num_features=1, stationary=True)
        modality = TimeseriesModality(config, rngs=rngs)

        # Create data with extreme values and NaNs
        data = jnp.array([[[10.0]], [[jnp.inf]], [[-jnp.inf]], [[jnp.nan]]])
        data = jnp.tile(data, (1, 50, 1))

        processed = modality.postprocess(data)

        # Check that all values are finite and clipped
        assert jnp.all(jnp.isfinite(processed))
        assert jnp.all(jnp.abs(processed) <= 5.0)

    def test_validate_data(self, rngs):
        """Test data validation."""
        config = TimeseriesModalityConfig(sequence_length=50, num_features=1)
        modality = TimeseriesModality(config, rngs=rngs)

        # Valid data
        valid_data = jnp.ones((4, 50, 1))
        assert modality.validate_data(valid_data) is True

        # Invalid shape
        assert modality.validate_data(jnp.ones((50, 1))) is False

        # Wrong sequence length
        assert modality.validate_data(jnp.ones((4, 40, 1))) is False

        # Non-finite values
        invalid_data = jnp.array([[[jnp.nan]]])
        invalid_data = jnp.tile(invalid_data, (4, 50, 1))
        assert modality.validate_data(invalid_data) is False

    def test_get_feature_info(self, rngs):
        """Test feature information retrieval."""
        config = TimeseriesModalityConfig(
            sequence_length=100,
            num_features=2,
            sampling_rate=0.5,
            seasonal_period=24,
            univariate=False,
            stationary=True,
        )
        modality = TimeseriesModality(config, rngs=rngs)

        info = modality.get_feature_info()

        assert info["feature_names"] == ["feature_0", "feature_1"]
        assert info["num_features"] == 2
        assert info["sequence_length"] == 100
        assert info["sampling_rate"] == 0.5
        assert info["is_univariate"] is False
        assert info["is_stationary"] is True
        assert info["seasonal_period"] == 24
        assert info["representation"] == "raw"


class TestTimeseriesProcessor:
    """Test cases for TimeseriesProcessor."""

    def test_initialization(self, rngs):
        """Test processor initialization."""
        processor = TimeseriesProcessor(
            config=TimeseriesModalityConfig(
                sequence_length=100,
                num_features=2,
                univariate=False,
            ),
            sequence_length=100,
            num_features=2,
            normalize=True,
            rngs=rngs,
        )

        assert processor.sequence_length == 100
        assert processor.num_features == 2
        assert processor.normalize is True

    def test_processing(self, rngs):
        """Test data processing."""
        processor = TimeseriesProcessor(
            config=TimeseriesModalityConfig(
                sequence_length=50,
                num_features=1,
                univariate=True,
            ),
            sequence_length=50,
            num_features=1,
            normalize=True,
            rngs=rngs,
        )

        # Create test data with non-zero mean
        data = jnp.ones((4, 50, 1)) * 5.0

        processed = processor(data)

        # Check normalization was applied
        batch_means = jnp.mean(processed, axis=1)
        assert jnp.allclose(batch_means, 0.0, atol=1e-6)

    def test_reverse(self, rngs):
        """Test reverse processing."""
        processor = TimeseriesProcessor(
            config=TimeseriesModalityConfig(
                sequence_length=50,
                num_features=1,
                univariate=True,
            ),
            sequence_length=50,
            num_features=1,
            rngs=rngs,
        )

        data = jnp.ones((4, 50, 1))
        reversed_data = processor.reverse(data)

        # Base implementation should return input unchanged
        assert jnp.array_equal(data, reversed_data)


class TestFourierProcessor:
    """Test cases for FourierProcessor."""

    def test_initialization(self, rngs):
        """Test Fourier processor initialization."""
        processor = FourierProcessor(
            num_frequencies=32,
            max_frequency=50.0,
            include_original=True,
            rngs=rngs,
        )

        assert processor.num_frequencies == 32
        assert processor.max_frequency == 50.0
        assert processor.include_original is True

    def test_fourier_features(self, rngs):
        """Test Fourier feature computation."""
        processor = FourierProcessor(
            num_frequencies=16,
            max_frequency=10.0,
            include_original=True,
            rngs=rngs,
        )

        # Create simple test data
        data = jnp.ones((2, 32, 1))

        result = processor(data)

        # Should include original data plus Fourier features
        # 16 frequencies * 2 (sin/cos) * 1 feature = 32 Fourier features
        # Plus 1 original feature = 33 total features
        assert result.shape == (2, 32, 33)

    def test_fourier_features_without_original(self, rngs):
        """Test Fourier features without original data."""
        processor = FourierProcessor(
            num_frequencies=8,
            max_frequency=5.0,
            include_original=False,
            rngs=rngs,
        )

        data = jnp.ones((2, 16, 2))

        result = processor(data)

        # 8 frequencies * 2 (sin/cos) * 2 features = 32 Fourier features
        assert result.shape == (2, 16, 32)


class TestMultiScaleProcessor:
    """Test cases for MultiScaleProcessor."""

    def test_initialization(self, rngs):
        """Test multi-scale processor initialization."""
        processor = MultiScaleProcessor(
            scale_factors=[1, 2, 4],
            aggregation_method="mean",
            rngs=rngs,
        )

        assert processor.scale_factors == [1, 2, 4]
        assert processor.aggregation_method == "mean"

    def test_invalid_aggregation_method(self, rngs):
        """Test invalid aggregation method."""
        with pytest.raises(ValueError, match="Invalid aggregation method"):
            MultiScaleProcessor(
                scale_factors=[1, 2],
                aggregation_method="invalid",
                rngs=rngs,
            )

    def test_multi_scale_representations(self, rngs):
        """Test multi-scale representation creation."""
        processor = MultiScaleProcessor(
            scale_factors=[1, 2, 4],
            aggregation_method="mean",
            rngs=rngs,
        )

        # Create test data divisible by all scale factors
        data = jnp.ones((2, 16, 1))

        representations = processor(data)

        assert "scale_1" in representations
        assert "scale_2" in representations
        assert "scale_4" in representations

        # Check shapes
        assert representations["scale_1"].shape == (2, 16, 1)  # Original
        assert representations["scale_2"].shape == (2, 8, 1)  # Downsampled by 2
        assert representations["scale_4"].shape == (2, 4, 1)  # Downsampled by 4

    def test_reconstruction(self, rngs):
        """Test reconstruction from multi-scale representations."""
        processor = MultiScaleProcessor(
            scale_factors=[1, 2],
            rngs=rngs,
        )

        data = jnp.ones((2, 8, 1))
        representations = processor(data)

        # Reconstruct from representations
        reconstructed = processor.reconstruct(representations)

        # Should return the original scale (scale_1)
        assert jnp.array_equal(reconstructed, data)


class TestTrendDecompositionProcessor:
    """Test cases for TrendDecompositionProcessor."""

    def test_initialization(self, rngs):
        """Test trend decomposition processor initialization."""
        processor = TrendDecompositionProcessor(
            period=12,
            method="seasonal",
            rngs=rngs,
        )

        assert processor.period == 12
        assert processor.method == "seasonal"

    def test_invalid_decomposition_method(self, rngs):
        """Test invalid decomposition method."""
        with pytest.raises(ValueError, match="Invalid decomposition method"):
            TrendDecompositionProcessor(
                period=12,
                method="invalid",
                rngs=rngs,
            )

    def test_decomposition(self, rngs):
        """Test trend and seasonal decomposition."""
        processor = TrendDecompositionProcessor(
            period=4,
            method="seasonal",
            rngs=rngs,
        )

        # Create test data with trend and seasonality
        t = jnp.arange(16)
        trend = 0.1 * t
        seasonal = jnp.sin(2 * jnp.pi * t / 4)
        data = (trend + seasonal)[None, :, None]  # Add batch and feature dims
        data = jnp.tile(data, (2, 1, 1))  # Batch size 2

        components = processor(data)

        assert "trend" in components
        assert "seasonal" in components
        assert "residual" in components
        assert "original" in components

        # Check shapes
        assert components["trend"].shape == (2, 16, 1)
        assert components["seasonal"].shape == (2, 16, 1)
        assert components["residual"].shape == (2, 16, 1)

    def test_reconstruction_from_components(self, rngs):
        """Test reconstruction from decomposed components."""
        processor = TrendDecompositionProcessor(
            period=4,
            method="seasonal",
            rngs=rngs,
        )

        data = jnp.ones((2, 16, 1))
        components = processor(data)

        # Test reconstruction
        reconstructed = processor.reconstruct(components)

        # Should return the original data
        assert jnp.array_equal(reconstructed, data)


class TestSyntheticTimeseriesDataset:
    """Test cases for SyntheticTimeseriesDataset."""

    def test_initialization(self, rngs):
        """Test dataset initialization."""
        dataset = SyntheticTimeseriesDataset(
            config=TimeseriesModalityConfig(
                sequence_length=50,
                num_features=2,
                univariate=False,
            ),
            sequence_length=50,
            num_features=2,
            num_samples=100,
            pattern_type="sinusoidal",
            noise_level=0.1,
            rngs=rngs,
        )

        assert dataset.sequence_length == 50
        assert dataset.num_features == 2
        assert dataset.num_samples == 100
        assert dataset.pattern_type == "sinusoidal"
        assert dataset.noise_level == 0.1

    def test_dataset_length(self, rngs):
        """Test dataset length."""
        dataset = SyntheticTimeseriesDataset(
            config=TimeseriesModalityConfig(
                sequence_length=20,
                num_features=1,
                univariate=True,
            ),
            sequence_length=20,
            num_samples=50,
            rngs=rngs,
        )

        assert len(dataset) == 50

    def test_dataset_indexing(self, rngs):
        """Test dataset indexing."""
        dataset = SyntheticTimeseriesDataset(
            config=TimeseriesModalityConfig(
                sequence_length=20,
                num_features=1,
                univariate=True,
            ),
            sequence_length=20,
            num_features=1,
            num_samples=10,
            rngs=rngs,
        )

        # Test valid indexing
        sample = dataset[0]
        assert sample.shape == (20, 1)

        # Test invalid indexing
        with pytest.raises(IndexError):
            dataset[20]

    def test_different_patterns(self, rngs):
        """Test different pattern types."""
        patterns = ["sinusoidal", "random_walk", "ar", "seasonal", "mixed"]

        for pattern in patterns:
            dataset = SyntheticTimeseriesDataset(
                config=TimeseriesModalityConfig(
                    sequence_length=50,
                    num_features=1,
                    univariate=True,
                ),
                sequence_length=50,
                num_features=1,
                num_samples=10,
                pattern_type=pattern,
                rngs=rngs,
            )

            assert len(dataset) == 10
            sample = dataset[0]
            assert sample.shape == (50, 1)
            assert jnp.all(jnp.isfinite(sample))

    def test_invalid_pattern_type(self, rngs):
        """Test invalid pattern type."""
        with pytest.raises(ValueError, match="Unknown pattern type"):
            SyntheticTimeseriesDataset(
                config=TimeseriesModalityConfig(
                    sequence_length=20,
                    num_features=1,
                    univariate=True,
                ),
                sequence_length=20,
                num_samples=10,
                pattern_type="invalid_pattern",
                rngs=rngs,
            )

    def test_batch_iterator(self, rngs):
        """Test batch iterator."""
        dataset = SyntheticTimeseriesDataset(
            config=TimeseriesModalityConfig(
                sequence_length=20,
                num_features=1,
                univariate=True,
            ),
            sequence_length=20,
            num_features=1,
            num_samples=25,
            rngs=rngs,
        )

        batches = list(dataset.batch_iterator(batch_size=10))

        assert len(batches) == 3  # 25 / 10 = 2 full batches + 1 partial
        assert batches[0].shape == (10, 20, 1)
        assert batches[1].shape == (10, 20, 1)
        assert batches[2].shape == (5, 20, 1)  # Last partial batch

    def test_statistics(self, rngs):
        """Test dataset statistics."""
        dataset = SyntheticTimeseriesDataset(
            config=TimeseriesModalityConfig(
                sequence_length=50,
                num_features=2,
                univariate=False,
            ),
            sequence_length=50,
            num_features=2,
            num_samples=100,
            rngs=rngs,
        )

        stats = dataset.get_statistics()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["sequence_length"] == 50
        assert stats["num_features"] == 2
        assert stats["num_samples"] == 100

        # Check shapes of statistical measures
        assert stats["mean"].shape == (2,)
        assert stats["std"].shape == (2,)


class TestTimeseriesEvaluationSuite:
    """Test cases for TimeseriesEvaluationSuite."""

    def test_initialization(self, rngs):
        """Test evaluation suite initialization."""
        evaluator = TimeseriesEvaluationSuite(
            config=TimeseriesModalityConfig(
                sequence_length=100,
                num_features=2,
                univariate=False,
            ),
            sequence_length=100,
            num_features=2,
            max_lag=20,
            rngs=rngs,
        )

        assert evaluator.sequence_length == 100
        assert evaluator.num_features == 2
        assert evaluator.max_lag == 20

    def test_compute_metrics(self, rngs):
        """Test comprehensive metrics computation."""
        evaluator = TimeseriesEvaluationSuite(
            config=TimeseriesModalityConfig(
                sequence_length=50,
                num_features=1,
                univariate=True,
            ),
            sequence_length=50,
            num_features=1,
            rngs=rngs,
        )

        # Create test data
        real_data = jnp.ones((10, 50, 1))
        generated_data = jnp.ones((10, 50, 1)) * 1.1  # Slightly different

        metrics = evaluator.compute_metrics(real_data, generated_data)

        # Check that all expected metrics are present
        expected_metrics = [
            "mse",
            "mae",
            "rmse",  # Basic metrics
            "dtw_distance",
            "trend_correlation",
            "temporal_consistency",  # Temporal
            "spectral_distance",
            "frequency_correlation",  # Spectral
            "mean_error",
            "variance_error",
            "skewness_error",  # Statistical
            "autocorr_distance",
            "autocorr_correlation",  # Autocorrelation
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)

    def test_shape_mismatch_error(self, rngs):
        """Test error on shape mismatch."""
        evaluator = TimeseriesEvaluationSuite(
            config=TimeseriesModalityConfig(
                sequence_length=50,
                num_features=1,
                univariate=True,
            ),
            sequence_length=50,
            num_features=1,
            rngs=rngs,
        )

        real_data = jnp.ones((10, 50, 1))
        generated_data = jnp.ones((5, 50, 1))  # Different batch size

        with pytest.raises(ValueError, match="Shape mismatch"):
            evaluator.compute_metrics(real_data, generated_data)

    def test_perfect_match(self, rngs):
        """Test metrics for perfectly matching data."""
        evaluator = TimeseriesEvaluationSuite(
            config=TimeseriesModalityConfig(
                sequence_length=50,
                num_features=1,
                univariate=True,
            ),
            sequence_length=50,
            num_features=1,
            rngs=rngs,
        )

        data = jnp.ones((5, 50, 1))
        metrics = evaluator.compute_metrics(data, data)

        # Basic metrics should be zero for identical data
        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def test_create_synthetic_timeseries_dataset(self):
        """Test synthetic dataset factory function."""
        dataset = create_synthetic_timeseries_dataset(
            config=TimeseriesModalityConfig(
                sequence_length=30,
                num_features=2,
                univariate=False,
            ),
            sequence_length=30,
            num_features=2,
            num_samples=50,
            pattern_type="sinusoidal",
            noise_level=0.05,
        )

        assert isinstance(dataset, SyntheticTimeseriesDataset)
        assert dataset.sequence_length == 30
        assert dataset.num_features == 2
        assert dataset.num_samples == 50
        assert dataset.pattern_type == "sinusoidal"
        assert dataset.noise_level == 0.05

    def test_create_simple_timeseries_dataset(self):
        """Test simple dataset factory function."""
        dataset = create_simple_timeseries_dataset(
            config=TimeseriesModalityConfig(
                sequence_length=25,
                num_features=1,
                univariate=True,
            ),
            sequence_length=25,
            num_samples=20,
        )

        assert isinstance(dataset, SyntheticTimeseriesDataset)
        assert dataset.sequence_length == 25
        assert dataset.num_features == 1
        assert dataset.num_samples == 20
        assert dataset.pattern_type == "sinusoidal"
        assert dataset.noise_level == 0.05

    def test_compute_timeseries_metrics(self):
        """Test metrics computation factory function."""
        real_data = jnp.ones((5, 20, 1))
        generated_data = jnp.ones((5, 20, 1)) * 1.2

        metrics = compute_timeseries_metrics(real_data, generated_data)

        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert "dtw_distance" in metrics
        assert metrics["mse"] > 0  # Should have non-zero error


class TestTimeseriesIntegration:
    """Integration tests for timeseries modality."""

    def test_end_to_end_workflow(self, rngs):
        """Test complete workflow from data creation to evaluation."""
        # Create configuration
        config = TimeseriesModalityConfig(
            sequence_length=50,
            num_features=1,
            use_fourier_features=True,
            num_frequencies=16,
        )

        # Create modality
        modality = TimeseriesModality(config, rngs=rngs)

        # Create dataset
        dataset = create_synthetic_timeseries_dataset(
            config=TimeseriesModalityConfig(
                sequence_length=50,
                num_features=1,
                univariate=True,
            ),
            sequence_length=50,
            num_features=1,
            num_samples=20,
            pattern_type="sinusoidal",
        )

        # Get some data
        batch = next(dataset.batch_iterator(batch_size=10))

        # Preprocess data
        processed = modality.preprocess(batch)

        # Validate data
        assert modality.validate_data(processed)

        # Create processor
        processor = FourierProcessor(
            num_frequencies=16,
            include_original=True,
            rngs=rngs,
        )

        # Apply processing
        features = processor(processed)

        # Check shape transformation
        # Original: (10, 50, 1), Fourier: 16*2*1 = 32, Total: 1+32 = 33
        assert features.shape == (10, 50, 33)

        # Evaluate using a slightly modified version as "generated"
        generated = processed + 0.1

        metrics = compute_timeseries_metrics(processed, generated)
        assert metrics["mse"] > 0

    def test_multi_scale_workflow(self, rngs):
        """Test multi-scale processing workflow."""
        config = TimeseriesModalityConfig(
            sequence_length=32,
            num_features=2,
            univariate=False,
            use_trend_decomposition=True,
            decomposition_period=8,
        )

        TimeseriesModality(config, rngs=rngs)

        # Create test data
        data = jnp.ones((5, 32, 2))

        # Multi-scale processing
        multi_scale_processor = MultiScaleProcessor(
            scale_factors=[1, 2, 4, 8],
            aggregation_method="mean",
            rngs=rngs,
        )

        representations = multi_scale_processor(data)

        # Check all scales are present
        for scale in [1, 2, 4, 8]:
            assert f"scale_{scale}" in representations

        # Reconstruction
        reconstructed = multi_scale_processor.reconstruct(representations)
        assert jnp.array_equal(reconstructed, data)

        # Trend decomposition
        trend_processor = TrendDecompositionProcessor(
            period=8,
            method="seasonal",
            rngs=rngs,
        )

        components = trend_processor(data)
        trend_reconstructed = trend_processor.reconstruct(components)
        assert jnp.array_equal(trend_reconstructed, data)
