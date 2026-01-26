"""Tests for precision and recall metrics for generative models.

This file includes all tests for precision and recall metrics:
- Core metrics tests (PrecisionRecall, DensityPrecisionRecall)
- Benchmark tests (PrecisionRecallBenchmark)
- NNX compatibility tests
- Diagnostic tests
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_blobs

from artifex.benchmarks.base import BenchmarkResult
from artifex.benchmarks.metrics.precision_recall import (
    compute_precision_recall,
    KMeansModule,
    PrecisionRecallBenchmark,
)
from artifex.generative_models.core.evaluation.metrics.general.precision_recall import (
    density_precision_recall,
    DensityPrecisionRecall,
    precision_recall,
    PrecisionRecall,
)
from tests.utils.test_models import MockModel


# Mock models are now imported from tests.utils.test_models


# Test the KMeansModule class


class TestKMeansModule:
    """Tests for the KMeansModule class."""

    def setup_method(self):
        """Set up test data."""
        # Create a fixed RNG key for deterministic results
        key = jax.random.PRNGKey(42)
        self.rngs = nnx.Rngs(params=key)

        # Create simple test data with clear clusters
        self.cluster1 = jnp.ones((10, 2)) * jnp.array([0.0, 0.0])
        self.cluster2 = jnp.ones((10, 2)) * jnp.array([5.0, 5.0])
        self.cluster3 = jnp.ones((10, 2)) * jnp.array([10.0, 0.0])

        # Add small noise to make it realistic
        key1, key2, key3 = jax.random.split(key, 3)
        noise1 = jax.random.normal(key1, self.cluster1.shape) * 0.1
        noise2 = jax.random.normal(key2, self.cluster2.shape) * 0.1
        noise3 = jax.random.normal(key3, self.cluster3.shape) * 0.1

        self.cluster1 += noise1
        self.cluster2 += noise2
        self.cluster3 += noise3

        # Create datasets with different numbers of clusters
        self.data_2clusters = jnp.concatenate([self.cluster1, self.cluster2])
        self.data_3clusters = jnp.concatenate([self.cluster1, self.cluster2, self.cluster3])

    def test_kmeans_initialization(self):
        """Test initialization of KMeansModule."""
        # Initialize with default parameters
        kmeans = KMeansModule(num_clusters=2, rngs=self.rngs)
        assert kmeans.num_clusters == 2
        assert kmeans.max_iterations == 20  # Default value

        # Initialize with custom parameters
        kmeans_custom = KMeansModule(num_clusters=3, max_iterations=10, rngs=self.rngs)
        assert kmeans_custom.num_clusters == 3
        assert kmeans_custom.max_iterations == 10

    def test_kmeans_fit_2clusters(self):
        """Test fitting KMeans to data with 2 clusters."""
        # Initialize KMeans
        kmeans = KMeansModule(num_clusters=2, rngs=self.rngs)

        # Fit to data
        centroids, labels = kmeans.fit(self.data_2clusters)

        # Check that we get expected shapes
        assert centroids.shape == (2, 2)  # 2 centroids, 2 dimensions each
        assert labels.shape == (20,)  # Labels for 20 data points

        # Check that we only have 2 unique labels
        unique_labels = jnp.unique(labels)
        assert len(unique_labels) == 2

        # Check that points in same cluster have same label
        cluster1_labels = labels[:10]  # First 10 points are from cluster1
        cluster2_labels = labels[10:]  # Last 10 points are from cluster2

        # Each cluster should have consistent labels
        assert jnp.all(cluster1_labels == cluster1_labels[0])
        assert jnp.all(cluster2_labels == cluster2_labels[0])

        # Different clusters should have different labels
        assert cluster1_labels[0] != cluster2_labels[0]

    def test_kmeans_fit_3clusters(self):
        """Test fitting KMeans to data with 3 clusters."""
        # Initialize KMeans
        kmeans = KMeansModule(num_clusters=3, rngs=self.rngs)

        # Fit to data
        centroids, labels = kmeans.fit(self.data_3clusters)

        # Check that we get expected shapes
        assert centroids.shape == (3, 2)  # 3 centroids, 2 dimensions each
        assert labels.shape == (30,)  # Labels for 30 data points

        # Check that we have 3 unique labels
        unique_labels = jnp.unique(labels)
        assert len(unique_labels) == 3


# Core metrics tests


class TestPrecisionRecall:
    """Tests for the PrecisionRecall class."""

    def setup_method(self):
        """Set up test data."""
        # Set a fixed seed for deterministic results
        self.key = jax.random.PRNGKey(42)

        # Generate synthetic clusters for testing
        # We'll create two distinct clusters for real data
        X_real, _ = make_blobs(
            n_samples=100, n_features=2, centers=2, cluster_std=0.5, random_state=42
        )
        self.real_samples = jnp.array(X_real)

        # For generated data, samples near one of the real clusters
        # This gives reasonable precision but lower recall
        X_gen, _ = make_blobs(
            n_samples=100,
            n_features=2,
            centers=X_real[0:1],  # Use first cluster center
            cluster_std=0.7,  # Slightly higher variance
            random_state=43,
        )
        self.gen_samples = jnp.array(X_gen)

        # Create identical samples to test perfect precision/recall
        self.identical_samples = self.real_samples

        # Create completely different samples
        X_diff, _ = make_blobs(
            n_samples=100,
            n_features=2,
            centers=[[10, 10]],  # Far away center
            cluster_std=0.5,
            random_state=44,
        )
        self.diff_samples = jnp.array(X_diff)

        # Create a feature extractor that just returns the samples (identity)
        def feature_extractor(samples):
            return samples

        self.feature_extractor = feature_extractor

        # Initialize the metric
        self.pr = PrecisionRecall(feature_extractor=self.feature_extractor, batch_size=50, k=3)

    def test_init(self):
        """Test initialization of PrecisionRecall."""
        # Test with custom parameters
        pr = PrecisionRecall(
            feature_extractor=self.feature_extractor, batch_size=32, k=5, name="custom_pr"
        )
        assert pr.name == "custom_pr"
        assert pr.batch_size == 32
        assert pr.k == 5
        assert pr.feature_extractor == self.feature_extractor

        # Test with defaults
        pr = PrecisionRecall()
        assert pr.name == "precision_recall"
        assert pr.batch_size == 32
        assert pr.k == 3
        assert pr.feature_extractor is None

    def test_extract_features(self):
        """Test feature extraction."""
        # Since our feature_extractor is the identity, returns the same
        features = self.pr.extract_features(self.real_samples)
        assert jnp.array_equal(features, self.real_samples)

        # With custom batch size
        features = self.pr.extract_features(self.real_samples, batch_size=25)
        assert jnp.array_equal(features, self.real_samples)

        # With no feature_extractor
        pr_no_extractor = PrecisionRecall(feature_extractor=None)
        features = pr_no_extractor.extract_features(self.real_samples)
        assert jnp.array_equal(features, self.real_samples)

    def test_compute_manifold_radii(self):
        """Test computation of manifold radii."""
        # Compute radii for a simple dataset
        features = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

        # With k=1, the radius is the distance to the nearest other point
        # For our grid, that's sqrt(1) = 1 for all points
        radii = self.pr.compute_manifold_radii(features, k=1)
        assert radii.shape == (4,)
        assert jnp.allclose(radii, 1.0, rtol=1e-5)

        # With k=2, the radius is the distance to the second nearest point
        # For our 2x2 grid, each point has 2 neighbors at distance 1
        # So the 2nd nearest neighbor is also at distance 1
        radii_k2 = self.pr.compute_manifold_radii(features, k=2)
        assert radii_k2.shape == (4,)
        assert jnp.allclose(radii_k2, 1.0, rtol=1e-5)

        # With k=3, we get the diagonal neighbor at distance sqrt(2)
        radii_k3 = self.pr.compute_manifold_radii(features, k=3)
        assert radii_k3.shape == (4,)
        assert jnp.allclose(radii_k3, jnp.sqrt(2.0), rtol=1e-5)

    def test_compute_precision(self):
        """Test computation of precision."""
        # Convert JAX arrays to numpy for compatibility
        real_np = np.array(self.real_samples)
        identical_np = np.array(self.identical_samples)
        diff_np = np.array(self.diff_samples)
        gen_np = np.array(self.gen_samples)

        # Test with identical samples (should give precision = 1.0)
        precision_identical = self.pr.compute_precision(real_np, identical_np, k=3)
        assert jnp.isclose(precision_identical, 1.0, rtol=1e-5)

        # Test with completely different samples (should give low precision)
        precision_diff = self.pr.compute_precision(real_np, diff_np, k=3)
        # Very few generated samples should fall in real manifold
        assert precision_diff < 0.1

        # Test with somewhat similar samples (should give reasonable precision)
        precision_similar = self.pr.compute_precision(real_np, gen_np, k=3)
        # Many generated samples should fall in real manifold
        assert precision_similar > 0.5

    def test_compute_recall(self):
        """Test computation of recall."""
        # Convert JAX arrays to numpy for compatibility
        real_np = np.array(self.real_samples)
        identical_np = np.array(self.identical_samples)
        diff_np = np.array(self.diff_samples)
        gen_np = np.array(self.gen_samples)

        # Test with identical samples (should give recall = 1.0)
        recall_identical = self.pr.compute_recall(real_np, identical_np, k=3)
        assert jnp.isclose(recall_identical, 1.0, rtol=1e-5)

        # Test with completely different samples (should give low recall)
        recall_diff = self.pr.compute_recall(real_np, diff_np, k=3)
        # Very few real samples covered by generated manifold
        assert recall_diff < 0.1

        # Test with somewhat similar samples (lower recall since we only
        # generate near one cluster)
        recall_similar = self.pr.compute_recall(real_np, gen_np, k=3)
        # Some real samples covered by generated manifold
        assert 0.1 < recall_similar < 0.9

    def test_compute_f1_score(self):
        """Test computation of F1 score."""
        # Test with precision = recall = 0.8
        f1 = self.pr.compute_f1_score(0.8, 0.8)
        assert jnp.isclose(f1, 0.8, rtol=1e-5)

        # Test with precision = 0.6, recall = 0.4
        f1 = self.pr.compute_f1_score(0.6, 0.4)
        expected_f1 = 2 * 0.6 * 0.4 / (0.6 + 0.4)
        assert jnp.isclose(f1, expected_f1, rtol=1e-5)

        # Test with precision or recall = 0
        f1_zero = self.pr.compute_f1_score(0.0, 0.5)
        assert jnp.isclose(f1_zero, 0.0, rtol=1e-5)

        f1_zero_both = self.pr.compute_f1_score(0.0, 0.0)
        assert jnp.isclose(f1_zero_both, 0.0, rtol=1e-5)

    def test_call(self):
        """Test the __call__ method."""
        # Call with identical samples
        result_identical = self.pr(self.real_samples, self.identical_samples)

        # Should have precision = recall = f1 = 1.0
        assert jnp.isclose(result_identical["precision_recall_precision"], 1.0, rtol=1e-5)
        assert jnp.isclose(result_identical["precision_recall_recall"], 1.0, rtol=1e-5)
        assert jnp.isclose(result_identical["precision_recall_f1"], 1.0, rtol=1e-5)

        # Call with normal test data
        result = self.pr(self.real_samples, self.gen_samples)

        # Check that results are reasonable
        assert 0.0 <= result["precision_recall_precision"] <= 1.0
        assert 0.0 <= result["precision_recall_recall"] <= 1.0
        assert 0.0 <= result["precision_recall_f1"] <= 1.0

        # Check with custom batch_size and k
        result_custom = self.pr(self.real_samples, self.gen_samples, batch_size=25, k=5)
        assert "precision_recall_precision" in result_custom
        assert "precision_recall_recall" in result_custom
        assert "precision_recall_f1" in result_custom

    def test_precision_recall_function(self):
        """Test the convenience function precision_recall."""
        # Calculate using the class
        result = self.pr(self.real_samples, self.gen_samples)

        # Calculate using the convenience function
        result_func = precision_recall(
            self.real_samples,
            self.gen_samples,
            feature_extractor=self.feature_extractor,
            batch_size=50,
            k=3,
        )

        # Results should be the same
        for key in result:
            assert jnp.isclose(result[key], result_func[key], rtol=1e-5)


class TestDensityPrecisionRecall:
    """Tests for the DensityPrecisionRecall class."""

    def setup_method(self):
        """Set up test data."""
        # Set a fixed seed for deterministic results
        self.key = jax.random.PRNGKey(42)

        # Generate synthetic clusters for testing
        # We'll create two distinct clusters for real data
        X_real, _ = make_blobs(
            n_samples=100, n_features=2, centers=2, cluster_std=0.5, random_state=42
        )
        self.real_samples = jnp.array(X_real)

        # For generated data, we'll create samples near one real cluster
        X_gen, _ = make_blobs(
            n_samples=100,
            n_features=2,
            centers=X_real[0:1],  # Use first cluster center
            cluster_std=0.7,  # Slightly higher variance
            random_state=43,
        )
        self.gen_samples = jnp.array(X_gen)

        # Create identical samples to test perfect precision/recall
        self.identical_samples = self.real_samples

        # Create completely different samples
        X_diff, _ = make_blobs(
            n_samples=100,
            n_features=2,
            centers=[[10, 10]],  # Far away center
            cluster_std=0.5,
            random_state=44,
        )
        self.diff_samples = jnp.array(X_diff)

        # Initialize the metric
        self.dpr = DensityPrecisionRecall(
            feature_extractor=None,  # Use identity
            batch_size=50,
            k=5,
        )

    def test_init(self):
        """Test initialization of DensityPrecisionRecall."""
        # Test with custom parameters
        dpr = DensityPrecisionRecall(feature_extractor=None, batch_size=32, k=7, name="custom_dpr")
        assert dpr.name == "custom_dpr"
        assert dpr.batch_size == 32
        assert dpr.k == 7
        assert dpr.feature_extractor is None

        # Test with defaults
        dpr = DensityPrecisionRecall()
        assert dpr.name == "density_precision_recall"
        assert dpr.batch_size == 32
        assert dpr.k == 5
        assert dpr.feature_extractor is None

    def test_estimate_densities(self):
        """Test density estimation."""
        # Test with a simple grid dataset
        features = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

        # Compute densities with k=1
        radii, densities = self.dpr.estimate_densities(features, k=1)

        # Check shapes
        assert radii.shape == (4,)
        assert densities.shape == (4,)

        # For k=1 in a grid, all points should have similar density
        assert np.allclose(densities, densities[0], rtol=1e-2)

    def test_compute_improved_precision_recall(self):
        """Test computation of improved precision and recall."""
        # Convert JAX arrays to numpy for compatibility
        real_np = np.array(self.real_samples)
        identical_np = np.array(self.identical_samples)
        diff_np = np.array(self.diff_samples)
        gen_np = np.array(self.gen_samples)

        # Test with identical samples (should give precision = recall = 1.0)
        precision, recall = self.dpr.compute_improved_precision_recall(real_np, identical_np, k=5)
        assert np.isclose(precision, 1.0, rtol=1e-5)
        assert np.isclose(recall, 1.0, rtol=1e-5)

        # Test with completely different samples (low precision, low recall)
        precision, recall = self.dpr.compute_improved_precision_recall(real_np, diff_np, k=5)
        assert precision < 0.1
        assert recall < 0.1

        # Test with somewhat similar samples
        precision, recall = self.dpr.compute_improved_precision_recall(real_np, gen_np, k=5)
        # Results should be reasonable
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0

    def test_call(self):
        """Test the __call__ method of DensityPrecisionRecall."""
        # Call with identical samples
        result_identical = self.dpr(self.real_samples, self.identical_samples)

        # Should have high precision and recall
        assert result_identical["density_precision_recall_precision"] > 0.9
        assert result_identical["density_precision_recall_recall"] > 0.9
        assert result_identical["density_precision_recall_f1"] > 0.9

        # Call with normal test data
        result = self.dpr(self.real_samples, self.gen_samples)

        # Check results are reasonable
        assert 0.0 <= result["density_precision_recall_precision"] <= 1.0
        assert 0.0 <= result["density_precision_recall_recall"] <= 1.0
        assert 0.0 <= result["density_precision_recall_f1"] <= 1.0

        # Check with custom parameters
        result_custom = self.dpr(self.real_samples, self.gen_samples, batch_size=32, k=3)
        assert "density_precision_recall_precision" in result_custom
        assert "density_precision_recall_recall" in result_custom
        assert "density_precision_recall_f1" in result_custom

    def test_density_precision_recall_function(self):
        """Test the density_precision_recall convenience function."""
        # Calculate using the class
        result = self.dpr(self.real_samples, self.gen_samples)

        # Calculate using the convenience function
        result_func = density_precision_recall(
            self.real_samples,
            self.gen_samples,
            feature_extractor=None,
            batch_size=50,
            k=5,
        )

        # Results should be the same
        for key in result:
            assert np.isclose(result[key], result_func[key], rtol=1e-5)


# Benchmark tests


class TestBenchmarkPrecisionRecall:
    """Tests for the precision-recall benchmark metrics."""

    def setup_method(self):
        """Set up test data."""
        # Create RNGs for initializing models
        key = jax.random.PRNGKey(0)
        self.rngs = nnx.Rngs(params=key)

        # Create real data: well-separated clusters
        cluster1 = jnp.ones((50, 2)) * jnp.array([5.0, 5.0])
        cluster2 = jnp.ones((50, 2)) * jnp.array([-5.0, -5.0])
        self.real_data = jnp.concatenate([cluster1, cluster2])

        # Create dataset for testing
        self.dataset = self.real_data

        # Create perfect samples: match the real data distribution
        self.perfect_samples = jnp.concatenate(
            [
                jnp.ones((25, 2)) * jnp.array([5.0, 5.0]),
                jnp.ones((25, 2)) * jnp.array([-5.0, -5.0]),
            ]
        )

        # Create low recall samples: missing one cluster
        self.low_recall_samples = jnp.ones((50, 2)) * jnp.array([5.0, 5.0])

        # Create low precision samples: has extra clusters not in real data
        self.low_precision_samples = jnp.concatenate(
            [
                # Include the original clusters
                jnp.ones((20, 2)) * jnp.array([5.0, 5.0]),
                jnp.ones((20, 2)) * jnp.array([-5.0, -5.0]),
                # Add a very distinct extra cluster
                jnp.ones((20, 2)) * jnp.array([15.0, 15.0]),
            ]
        )

    def test_compute_precision_recall(self):
        """Test the precision-recall computation function."""
        # For perfect samples, expect high precision and recall
        precision, recall = compute_precision_recall(
            generated_samples=self.perfect_samples,
            real_samples=self.real_data,
            num_clusters=2,
        )

        assert 0.9 <= precision <= 1.0
        assert 0.9 <= recall <= 1.0

        # For low recall samples, expect high precision but low recall
        precision, recall = compute_precision_recall(
            generated_samples=self.low_recall_samples,
            real_samples=self.real_data,
            num_clusters=2,
        )

        assert 0.9 <= precision <= 1.0
        assert recall < 0.7  # Should be around 0.5

        # For low precision samples, expect high recall but low precision
        precision, recall = compute_precision_recall(
            generated_samples=self.low_precision_samples,
            real_samples=self.real_data,
            num_clusters=3,  # Explicitly set to include the extra cluster
        )

        # With the extra far-off cluster, precision should be low
        # Expect around 2/3 = 0.67 since 2 of 3 clusters are valid
        assert precision < 0.9
        # But recall should still be good
        assert recall >= 0.7

    def test_precision_recall_benchmark(self):
        """Test the PrecisionRecallBenchmark class."""
        # Test with perfect samples
        perfect_model = MockModel(self.perfect_samples, rngs=self.rngs)
        benchmark = PrecisionRecallBenchmark(num_clusters=2)

        result = benchmark.run(model=perfect_model, dataset=self.dataset)

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "precision_recall"
        assert result.model_name == "mock_model"
        assert "precision" in result.metrics
        assert "recall" in result.metrics
        assert "f1_score" in result.metrics

        # Check metric values for perfect model
        assert 0.9 <= result.metrics["precision"] <= 1.0
        assert 0.9 <= result.metrics["recall"] <= 1.0
        assert 0.9 <= result.metrics["f1_score"] <= 1.0

        # Test with low recall model
        low_recall_model = MockModel(self.low_recall_samples, rngs=self.rngs)
        result = benchmark.run(model=low_recall_model, dataset=self.dataset)

        assert result.metrics["precision"] > 0.9
        assert result.metrics["recall"] < 0.7

        # Test with low precision model
        low_precision_model = MockModel(self.low_precision_samples, rngs=self.rngs)
        # Use more clusters to capture the extra one
        benchmark = PrecisionRecallBenchmark(num_clusters=3)
        result = benchmark.run(model=low_precision_model, dataset=self.dataset)

        # Precision should be low with the extra cluster
        assert result.metrics["precision"] < 0.9
        # But recall should be decent
        assert result.metrics["recall"] >= 0.7

    def test_custom_sample_size(self):
        """Test PrecisionRecallBenchmark with custom sample size."""
        # Create model with perfect samples
        model = MockModel(self.perfect_samples, rngs=self.rngs)

        # Create benchmark with custom sample size
        custom_size = 20
        benchmark = PrecisionRecallBenchmark(num_clusters=2, num_samples=custom_size)

        # Run benchmark
        result = benchmark.run(model=model, dataset=self.dataset)

        # Verify the result
        assert isinstance(result, BenchmarkResult)
        assert result.metrics["precision"] > 0.9
        assert result.metrics["recall"] > 0.9

    def test_custom_random_seed(self):
        """Test PrecisionRecallBenchmark with custom random seed."""
        # Create model with perfect samples
        model = MockModel(self.perfect_samples, rngs=self.rngs)

        # Create benchmark with custom random seed
        custom_seed = 42
        benchmark = PrecisionRecallBenchmark(num_clusters=2, random_seed=custom_seed)

        # Run benchmark
        result = benchmark.run(model=model, dataset=self.dataset)

        # Verify the result
        assert isinstance(result, BenchmarkResult)
        assert result.metrics["precision"] > 0.9
        assert result.metrics["recall"] > 0.9
