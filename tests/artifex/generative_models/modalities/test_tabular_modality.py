"""Tests for the tabular modality implementation."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.evaluation.metrics.statistical import (
    compute_chi2_statistic,
    compute_ks_distance,
)
from artifex.generative_models.modalities.tabular import (
    CategoricalEncoder,
    compute_tabular_metrics,
    create_simple_tabular_dataset,
    create_synthetic_tabular_dataset,
    NumericalProcessor,
    SyntheticTabularDataset,
    TabularEvaluationSuite,
    TabularModality,
    TabularModalityConfig,
    TabularProcessor,
)
from artifex.generative_models.modalities.tabular.base import ColumnType


@pytest.fixture
def rngs():
    """Standard fixture for rngs."""
    return nnx.Rngs(42)


@pytest.fixture
def simple_config():
    """Create a simple tabular configuration for testing."""
    return TabularModalityConfig(
        num_features=5,
        numerical_features=["age", "income"],
        categorical_features=["category"],
        ordinal_features=["education"],
        binary_features=["is_member"],
        categorical_vocab_sizes={"category": 4},
        ordinal_orders={"education": ["high_school", "bachelor", "master", "phd"]},
        normalization_type="standard",
        handle_missing="impute",
        max_categorical_cardinality=10,
    )


@pytest.fixture
def sample_tabular_data():
    """Create sample tabular data for testing."""
    return {
        "age": jnp.array([25.0, 30.0, 35.0, 40.0, 45.0]),
        "income": jnp.array([50000.0, 60000.0, 70000.0, 80000.0, 90000.0]),
        "category": jnp.array([0, 1, 2, 3, 0]),
        "education": jnp.array([0, 1, 2, 3, 2]),
        "is_member": jnp.array([0, 1, 1, 0, 1]),
    }


class TestTabularModalityConfig:
    """Test TabularModalityConfig."""

    def test_config_initialization(self, simple_config):
        """Test basic configuration initialization."""
        assert simple_config.num_features == 5
        assert len(simple_config.numerical_features) == 2
        assert len(simple_config.categorical_features) == 1
        assert len(simple_config.ordinal_features) == 1
        assert len(simple_config.binary_features) == 1

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration should pass
        config = TabularModalityConfig(
            num_features=3,
            numerical_features=["num1"],
            categorical_features=["cat1"],
            binary_features=["bin1"],
            categorical_vocab_sizes={"cat1": 5},
        )
        validated = config.validate_feature_consistency()
        assert validated.num_features == 3

    def test_config_validation_duplicate_features(self):
        """Test validation fails with duplicate feature names."""
        with pytest.raises(ValueError, match="Feature names must be unique"):
            config = TabularModalityConfig(
                num_features=2,
                numerical_features=["feature1"],
                categorical_features=["feature1"],  # Duplicate name
                categorical_vocab_sizes={"feature1": 5},
            )
            config.validate_feature_consistency()

    def test_config_validation_feature_count_mismatch(self):
        """Test validation fails when feature count doesn't match."""
        with pytest.raises(ValueError, match="doesn't match num_features"):
            config = TabularModalityConfig(
                num_features=5,  # Claims 5 features
                numerical_features=["num1"],
                categorical_features=["cat1"],  # Only 2 features total
                categorical_vocab_sizes={"cat1": 5},
            )
            config.validate_feature_consistency()

    def test_config_validation_missing_vocab_size(self):
        """Test validation fails when categorical vocab size is missing."""
        with pytest.raises(ValueError, match="Missing vocab size"):
            config = TabularModalityConfig(
                num_features=1,
                categorical_features=["cat1"],
                categorical_vocab_sizes={},  # Missing vocab size
            )
            config.validate_feature_consistency()

    def test_config_validation_missing_ordinal_order(self):
        """Test validation fails when ordinal order is missing."""
        with pytest.raises(ValueError, match="Missing order information"):
            config = TabularModalityConfig(
                num_features=1,
                ordinal_features=["ord1"],
                ordinal_orders={},  # Missing order
            )
            config.validate_feature_consistency()


class TestTabularModality:
    """Test TabularModality."""

    def test_modality_initialization(self, simple_config, rngs):
        """Test tabular modality initialization."""
        modality = TabularModality(config=simple_config, rngs=rngs)
        assert modality.config.num_features == 5
        assert isinstance(modality.rngs, nnx.Rngs)

    def test_get_feature_info(self, simple_config, rngs):
        """Test getting feature information."""
        modality = TabularModality(config=simple_config, rngs=rngs)
        feature_info = modality.get_feature_info()

        # Check numerical features
        assert feature_info["age"]["type"] == ColumnType.NUMERICAL
        assert feature_info["age"]["encoding_dim"] == 1

        # Check categorical features
        assert feature_info["category"]["type"] == ColumnType.CATEGORICAL
        assert feature_info["category"]["encoding_dim"] == 4  # vocab_size

        # Check ordinal features
        assert feature_info["education"]["type"] == ColumnType.ORDINAL
        assert feature_info["education"]["encoding_dim"] == 1

        # Check binary features
        assert feature_info["is_member"]["type"] == ColumnType.BINARY
        assert feature_info["is_member"]["encoding_dim"] == 1

    def test_get_total_encoding_dim(self, simple_config, rngs):
        """Test getting total encoding dimension."""
        modality = TabularModality(config=simple_config, rngs=rngs)
        total_dim = modality.get_total_encoding_dim()
        # 2 numerical + 4 categorical + 1 ordinal + 1 binary = 8
        assert total_dim == 8

    def test_validate_input_valid(self, simple_config, sample_tabular_data, rngs):
        """Test input validation with valid data."""
        modality = TabularModality(config=simple_config, rngs=rngs)
        # Should not raise an exception
        modality.validate_input(sample_tabular_data)

    def test_validate_input_missing_features(self, simple_config, rngs):
        """Test input validation with missing features."""
        modality = TabularModality(config=simple_config, rngs=rngs)
        incomplete_data = {
            "age": jnp.array([25.0, 30.0]),
            # Missing other features
        }
        with pytest.raises(ValueError, match="Feature mismatch"):
            modality.validate_input(incomplete_data)

    def test_validate_input_categorical_out_of_range(self, simple_config, rngs):
        """Test input validation with categorical values out of range."""
        modality = TabularModality(config=simple_config, rngs=rngs)
        invalid_data = {
            "age": jnp.array([25.0, 30.0]),
            "income": jnp.array([50000.0, 60000.0]),
            "category": jnp.array([5, 1]),  # 5 is out of range (vocab_size=4)
            "education": jnp.array([0, 1]),
            "is_member": jnp.array([0, 1]),
        }
        with pytest.raises(ValueError, match="values must be in"):
            modality.validate_input(invalid_data)

    def test_validate_input_binary_invalid_values(self, simple_config, rngs):
        """Test input validation with invalid binary values."""
        modality = TabularModality(config=simple_config, rngs=rngs)
        invalid_data = {
            "age": jnp.array([25.0, 30.0]),
            "income": jnp.array([50000.0, 60000.0]),
            "category": jnp.array([0, 1]),
            "education": jnp.array([0, 1]),
            "is_member": jnp.array([2, 1]),  # 2 is invalid for binary
        }
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            modality.validate_input(invalid_data)


class TestNumericalProcessor:
    """Test NumericalProcessor."""

    def test_processor_initialization(self, rngs):
        """Test numerical processor initialization."""
        processor = NumericalProcessor(
            features=["feature1", "feature2"],
            normalization_type="standard",
            rngs=rngs,
        )
        assert processor.features == ["feature1", "feature2"]
        assert processor.normalization_type == "standard"
        assert not processor._fitted

    def test_fit_and_transform_standard(self, rngs):
        """Test fit and transform with standard normalization."""
        processor = NumericalProcessor(
            features=["age", "income"],
            normalization_type="standard",
            rngs=rngs,
        )

        data = {
            "age": jnp.array([20.0, 30.0, 40.0, 50.0]),
            "income": jnp.array([40000.0, 50000.0, 60000.0, 70000.0]),
        }

        processor.fit(data)
        assert processor._fitted
        assert processor.mean is not None
        assert processor.std is not None

        transformed = processor.transform(data)
        assert transformed.shape == (4, 2)

        # Check that transformed data has approximately zero mean and unit std
        assert jnp.allclose(jnp.mean(transformed, axis=0), 0.0, atol=1e-6)
        assert jnp.allclose(jnp.std(transformed, axis=0), 1.0, atol=1e-6)

    def test_fit_and_transform_minmax(self, rngs):
        """Test fit and transform with minmax normalization."""
        processor = NumericalProcessor(
            features=["age"],
            normalization_type="minmax",
            rngs=rngs,
        )

        data = {"age": jnp.array([20.0, 30.0, 40.0, 50.0])}

        processor.fit(data)
        transformed = processor.transform(data)

        # Check that transformed data is in [0, 1] range
        assert jnp.min(transformed) >= 0.0
        assert jnp.max(transformed) <= 1.0

    def test_inverse_transform(self, rngs):
        """Test inverse transformation."""
        processor = NumericalProcessor(
            features=["age"],
            normalization_type="standard",
            rngs=rngs,
        )

        original_data = {"age": jnp.array([20.0, 30.0, 40.0, 50.0])}

        processor.fit(original_data)
        transformed = processor.transform(original_data)
        reconstructed = processor.inverse_transform(transformed)

        assert jnp.allclose(original_data["age"], reconstructed.squeeze(), atol=1e-6)

    def test_empty_features(self, rngs):
        """Test processor with no features."""
        processor = NumericalProcessor(features=[], rngs=rngs)
        data = {"other_feature": jnp.array([1.0, 2.0])}

        processor.fit(data)
        transformed = processor.transform(data)

        assert transformed.shape == (2, 0)  # Empty features


class TestCategoricalEncoder:
    """Test CategoricalEncoder."""

    def test_encoder_initialization(self, rngs):
        """Test categorical encoder initialization."""
        encoder = CategoricalEncoder(
            features=["cat1", "cat2"],
            vocab_sizes={"cat1": 3, "cat2": 4},
            rngs=rngs,
        )
        assert encoder.features == ["cat1", "cat2"]
        assert encoder.total_dim == 7  # 3 + 4

    def test_encode_and_decode(self, rngs):
        """Test categorical encoding and decoding."""
        encoder = CategoricalEncoder(
            features=["category"],
            vocab_sizes={"category": 4},
            rngs=rngs,
        )

        data = {"category": jnp.array([0, 1, 2, 3, 0])}

        encoded = encoder.encode(data)
        assert encoded.shape == (5, 4)  # 5 samples, 4 vocab size

        # Check one-hot encoding
        assert jnp.sum(encoded[0]) == 1  # Each row should sum to 1
        assert jnp.argmax(encoded[0]) == 0  # First sample should be category 0

        decoded = encoder.decode(encoded)
        assert jnp.array_equal(decoded["category"], data["category"])

    def test_empty_features(self, rngs):
        """Test encoder with no features."""
        encoder = CategoricalEncoder(features=[], vocab_sizes={}, rngs=rngs)
        data = {"other_feature": jnp.array([1, 2])}

        encoded = encoder.encode(data)
        assert encoded.shape == (2, 0)  # Empty encoding

        decoded = encoder.decode(encoded)
        assert len(decoded) == 0


class TestTabularProcessor:
    """Test TabularProcessor."""

    def test_processor_initialization(self, simple_config, rngs):
        """Test tabular processor initialization."""
        processor = TabularProcessor(config=simple_config, rngs=rngs)
        assert processor.config == simple_config
        assert isinstance(processor.numerical_processor, NumericalProcessor)
        assert isinstance(processor.categorical_encoder, CategoricalEncoder)

    def test_fit_encode_decode_cycle(self, simple_config, sample_tabular_data, rngs):
        """Test complete fit, encode, decode cycle."""
        processor = TabularProcessor(config=simple_config, rngs=rngs)

        # Fit processor
        processor.fit(sample_tabular_data)

        # Encode data
        encoded = processor.encode(sample_tabular_data)
        expected_dim = 2 + 4 + 1 + 1  # numerical + categorical + ordinal + binary
        assert encoded.shape == (5, expected_dim)

        # Decode data
        decoded = processor.decode(encoded)

        # Check that decoded data has the right structure
        assert set(decoded.keys()) == set(sample_tabular_data.keys())

        # Check numerical features (should be approximately equal due to normalization)
        for feature in simple_config.numerical_features:
            assert decoded[feature].shape == sample_tabular_data[feature].shape

        # Check categorical features (should be exactly equal)
        for feature in simple_config.categorical_features:
            assert jnp.array_equal(decoded[feature], sample_tabular_data[feature])

    def test_get_encoding_dimensions(self, simple_config, rngs):
        """Test getting encoding dimensions."""
        processor = TabularProcessor(config=simple_config, rngs=rngs)
        dimensions = processor.get_encoding_dimensions()

        assert "numerical" in dimensions
        assert "categorical" in dimensions
        assert "ordinal" in dimensions
        assert "binary" in dimensions

        # Check dimension ranges
        num_start, num_end = dimensions["numerical"]
        assert num_end - num_start == 2  # 2 numerical features

        cat_start, cat_end = dimensions["categorical"]
        assert cat_end - cat_start == 4  # 1 categorical feature with vocab_size=4


class TestSyntheticTabularDataset:
    """Test SyntheticTabularDataset."""

    def test_dataset_initialization(self, simple_config, rngs):
        """Test synthetic dataset initialization."""
        dataset = SyntheticTabularDataset(
            config=simple_config,
            num_samples=100,
            rngs=rngs,
        )
        assert len(dataset) == 100
        assert set(dataset.data.keys()) == set(
            ["age", "income", "category", "education", "is_member"]
        )

    def test_dataset_get_item(self, simple_config, rngs):
        """Test getting individual samples."""
        dataset = SyntheticTabularDataset(
            config=simple_config,
            num_samples=10,
            rngs=rngs,
        )

        sample = dataset[0]
        assert set(sample.keys()) == set(dataset.data.keys())
        for feature_name in sample.keys():
            assert sample[feature_name].shape == ()  # Scalar

    def test_dataset_get_batch(self, simple_config, rngs):
        """Test getting batches."""
        dataset = SyntheticTabularDataset(
            config=simple_config,
            num_samples=100,
            rngs=rngs,
        )

        # Get full batch
        full_batch = dataset.get_batch()
        assert len(full_batch) == 5  # 5 features

        # Get partial batch
        batch = dataset.get_batch(batch_size=20)
        for feature_data in batch.values():
            assert feature_data.shape[0] == 20

    def test_dataset_feature_statistics(self, simple_config, rngs):
        """Test feature statistics computation."""
        dataset = SyntheticTabularDataset(
            config=simple_config,
            num_samples=1000,
            rngs=rngs,
        )

        stats = dataset.get_feature_statistics()

        # Check numerical feature stats
        assert "age" in stats
        assert stats["age"]["type"] == "numerical"
        assert "mean" in stats["age"]
        assert "std" in stats["age"]

        # Check categorical feature stats
        assert "category" in stats
        assert stats["category"]["type"] == "categorical"
        assert stats["category"]["vocab_size"] == 4
        assert len(stats["category"]["counts"]) == 4

        # Check binary feature stats
        assert "is_member" in stats
        assert stats["is_member"]["type"] == "binary"
        assert "positive_rate" in stats["is_member"]


class TestTabularDatasetFactories:
    """Test tabular dataset factory functions."""

    def test_create_synthetic_tabular_dataset(self, rngs):
        """Test synthetic dataset creation."""
        dataset, config = create_synthetic_tabular_dataset(
            num_features=10,
            num_samples=100,
            rngs=rngs,
        )

        assert isinstance(dataset, SyntheticTabularDataset)
        assert isinstance(config, TabularModalityConfig)
        assert len(dataset) == 100
        assert config.num_features == 10

    def test_create_simple_tabular_dataset(self, rngs):
        """Test simple dataset creation."""
        dataset, config = create_simple_tabular_dataset(num_samples=50, rngs=rngs)

        assert isinstance(dataset, SyntheticTabularDataset)
        assert len(dataset) == 50
        assert config.num_features == 5

    def test_invalid_feature_ratios(self, rngs):
        """Test that invalid feature ratios raise an error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            create_synthetic_tabular_dataset(
                numerical_ratio=0.5,
                categorical_ratio=0.3,
                ordinal_ratio=0.1,
                binary_ratio=0.05,  # Sum = 0.95, not 1.0
                rngs=rngs,
            )


class TestTabularEvaluationSuite:
    """Test TabularEvaluationSuite."""

    def test_evaluation_suite_initialization(self, simple_config):
        """Test evaluation suite initialization."""
        suite = TabularEvaluationSuite(config=simple_config)
        assert suite.config == simple_config

    def test_evaluate_batch(self, simple_config, sample_tabular_data, rngs):
        """Test batch evaluation."""
        suite = TabularEvaluationSuite(config=simple_config)

        # Create slightly different generated data
        generated_data = {
            "age": sample_tabular_data["age"] + jax.random.normal(rngs.evaluation(), (5,)) * 0.1,
            "income": sample_tabular_data["income"]
            + jax.random.normal(rngs.evaluation(), (5,)) * 1000,
            "category": sample_tabular_data["category"],  # Keep categorical same
            "education": sample_tabular_data["education"],  # Keep ordinal same
            "is_member": sample_tabular_data["is_member"],  # Keep binary same
        }

        metrics = suite.evaluate_batch(sample_tabular_data, generated_data)

        # Check that metrics are computed
        assert "overall_quality" in metrics
        assert isinstance(metrics["overall_quality"], float)

        # Check distribution metrics for numerical features
        assert "ks_distance_age" in metrics
        assert "ks_distance_income" in metrics

        # Check feature preservation metrics
        assert "correlation_preservation" in metrics

        # Check privacy metrics
        assert "dcr_score" in metrics
        assert "memorization_score" in metrics

    def test_ks_distance_computation(self, simple_config):
        """Test Kolmogorov-Smirnov distance computation."""
        TabularEvaluationSuite(config=simple_config)

        # Identical distributions should have KS distance 0
        data1 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data2 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ks_dist = compute_ks_distance(data1, data2)
        assert jnp.allclose(ks_dist, 0.0, atol=1e-6)

        # Different distributions should have KS distance > 0
        data3 = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        ks_dist = compute_ks_distance(data1, data3)
        assert ks_dist > 0.0

    def test_chi2_statistic_computation(self, simple_config):
        """Test chi-square statistic computation."""
        TabularEvaluationSuite(config=simple_config)

        # Identical categorical distributions should have low chi2
        data1 = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])
        data2 = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])
        chi2 = compute_chi2_statistic(data1, data2, vocab_size=4)
        assert chi2 < 1.0  # Should be very low for identical distributions


class TestTabularMetricsFactory:
    """Test tabular metrics factory function."""

    def test_compute_tabular_metrics(self, simple_config, sample_tabular_data, rngs):
        """Test tabular metrics computation."""
        # Create slightly perturbed generated data
        generated_data = {}
        for feature, data in sample_tabular_data.items():
            if feature in simple_config.numerical_features:
                # Add small noise to numerical features
                noise = jax.random.normal(rngs.metrics(), data.shape) * 0.01
                generated_data[feature] = data + noise
            else:
                # Keep categorical/ordinal/binary features the same
                generated_data[feature] = data

        metrics = compute_tabular_metrics(
            real_data=sample_tabular_data,
            generated_data=generated_data,
            config=simple_config,
        )

        assert isinstance(metrics, dict)
        assert "overall_quality" in metrics
        assert metrics["overall_quality"] >= 0.0
        assert metrics["overall_quality"] <= 1.0


class TestTabularModalityIntegration:
    """Test end-to-end tabular modality integration."""

    def test_end_to_end_workflow(self, rngs):
        """Test complete end-to-end workflow."""
        # Create dataset and config
        dataset, config = create_simple_tabular_dataset(num_samples=100, rngs=rngs)

        # Create modality
        modality = TabularModality(config=config, rngs=rngs)

        # Get data batch
        data_batch = dataset.get_batch(batch_size=20)

        # Validate input
        modality.validate_input(data_batch)

        # Create and fit processor
        processor = TabularProcessor(config=config, rngs=rngs)
        processor.fit(data_batch)

        # Encode and decode
        encoded = processor.encode(data_batch)
        decoded = processor.decode(encoded)

        # Evaluate quality (comparing original with decoded)
        metrics = compute_tabular_metrics(
            real_data=data_batch,
            generated_data=decoded,
            config=config,
        )

        # Check that the workflow produces reasonable results
        assert metrics["overall_quality"] > 0.3  # Should be reasonable quality for reconstruction
