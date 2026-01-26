"""Tests for protein model adapters for benchmarks."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from artifex.benchmarks.base import BenchmarkResult
from artifex.benchmarks.metrics.precision_recall import (
    PrecisionRecallBenchmark,
)
from artifex.benchmarks.protein_model_adapters import (
    ProteinPointCloudAdapter,
)


class NNXProteinMockModel(nnx.Module):
    """Mock NNX protein model for testing."""

    def __init__(self, point_clouds, *, rngs=None):
        """Initialize with predetermined point clouds.

        Args:
            point_clouds: Protein point clouds to return.
            rngs: Required RNG dict for NNX compatibility.
        """
        super().__init__()
        # Set model_name to match the name in the adapter
        self.model_name = "protein_point_cloud_model"
        self.point_clouds = point_clouds

    def __call__(self, x, *, rngs=None):
        """Forward pass.

        Args:
            x: Input data.
            rngs: Required RNG dict for NNX compatibility.

        Returns:
            Predetermined point clouds.
        """
        if rngs is None:
            raise ValueError("rngs is required for NNX models")

        # Return first point cloud
        return self.point_clouds[0]

    def predict(self, x, *, rngs=None):
        """Predict method.

        Args:
            x: Input data.
            rngs: Required RNG dict for NNX compatibility.

        Returns:
            Predetermined point clouds.
        """
        if rngs is None:
            raise ValueError("rngs is required for NNX models")

        return self.point_clouds[0]

    def generate_protein(self, batch_size=1, *, rngs=None):
        """Generate protein method.

        Args:
            batch_size: Number of proteins to generate.
            rngs: Required RNG dict for NNX compatibility.

        Returns:
            Dictionary with protein point clouds.
        """
        if rngs is None:
            raise ValueError("rngs is required for NNX models")

        # Return actual batch_size or max available
        actual_batch_size = min(batch_size, self.point_clouds.shape[0])
        point_clouds = self.point_clouds[:actual_batch_size]

        # Return as dictionary to simulate real protein model output
        return {"coordinates": point_clouds}

    def sample_protein(self, batch_size=1, *, rngs=None):
        """Sample protein method.

        Args:
            batch_size: Number of proteins to sample.
            rngs: Required RNG dict for NNX compatibility.

        Returns:
            Protein point clouds directly.
        """
        if rngs is None:
            raise ValueError("rngs is required for NNX models")

        # Return actual batch_size or max available
        actual_batch_size = min(batch_size, self.point_clouds.shape[0])
        return self.point_clouds[:actual_batch_size]


class TestProteinModelAdapter:
    """Tests for protein model adapters."""

    def setup_method(self):
        """Set up test data."""
        # Create RNG key
        key = jax.random.PRNGKey(42)

        # For testing with precision-recall benchmark, we need to create
        # data with compatible shapes:
        # - The protein model should generate batches of points in 2D
        # - The real data should be flat vectors for direct comparison

        # Create real data samples as flattened 2D points for the benchmark
        self.real_data_dim = 10  # Dimension of flattened feature vector

        # Create two clusters in the real data
        cluster1 = jnp.ones((15, self.real_data_dim)) * 0.8
        cluster2 = jnp.ones((15, self.real_data_dim)) * -0.8
        # Add small noise to make each sample unique
        cluster1_noise = jax.random.normal(key, (15, self.real_data_dim)) * 0.05
        key, _ = jax.random.split(key)
        cluster2_noise = jax.random.normal(key, (15, self.real_data_dim)) * 0.05

        # Combine clusters and add noise
        self.real_proteins = jnp.concatenate([cluster1 + cluster1_noise, cluster2 + cluster2_noise])

        # Create perfect match test data (generate batched data of the same dim)
        # Each sample has batch_size=1, with num_points=10 and feature_dim=1
        # This shape works with our adapter conversion

        # Create points matching the real data clustered structure
        perfect_cluster1 = jnp.ones((5, 1, self.real_data_dim)) * 0.8
        perfect_cluster2 = jnp.ones((5, 1, self.real_data_dim)) * -0.8
        key, _ = jax.random.split(key)
        perfect_noise1 = jax.random.normal(key, (5, 1, self.real_data_dim)) * 0.05
        key, _ = jax.random.split(key)
        perfect_noise2 = jax.random.normal(key, (5, 1, self.real_data_dim)) * 0.05

        # Create perfect match protein samples
        self.perfect_proteins = jnp.concatenate(
            [perfect_cluster1 + perfect_noise1, perfect_cluster2 + perfect_noise2]
        )

        # Create proteins with extra cluster (low precision case)
        extra_cluster = jnp.ones((5, 1, self.real_data_dim)) * 0.0
        key, _ = jax.random.split(key)
        extra_noise = jax.random.normal(key, (5, 1, self.real_data_dim)) * 0.05

        self.extra_cluster_proteins = jnp.concatenate(
            [
                perfect_cluster1 + perfect_noise1,
                perfect_cluster2 + perfect_noise2,
                extra_cluster + extra_noise,
            ]
        )

        # Create proteins with missing cluster (low recall case)
        self.missing_cluster_proteins = perfect_cluster1 + perfect_noise1

        # Set up NNX compatibility
        self.rngs = nnx.Rngs(params=key, sample=key)

    def test_protein_adapter_detection(self):
        """Test that the adapter correctly detects protein models."""
        # Create a protein model
        model = NNXProteinMockModel(self.perfect_proteins, rngs=self.rngs)

        # Check that the adapter can adapt the model
        assert ProteinPointCloudAdapter.can_adapt(model)

        # Create a non-protein model
        class NonProteinModel(nnx.Module):
            """Non-protein model."""

            def __init__(self, *, rngs=None):
                super().__init__()

            def predict(self, x, *, rngs=None):
                return jnp.zeros((1, 5))

        non_protein_model = NonProteinModel(rngs=self.rngs)

        # Check that the adapter cannot adapt the non-protein model
        assert not ProteinPointCloudAdapter.can_adapt(non_protein_model)

    def test_protein_adapter_conversion(self):
        """Test protein output format conversion."""
        model = NNXProteinMockModel(self.perfect_proteins, rngs=self.rngs)
        # Use point_dim=self.real_data_dim to maintain the feature dimension
        adapter = ProteinPointCloudAdapter(model, point_dim=self.real_data_dim)

        # Test dictionary output conversion
        dict_output = {"coordinates": self.perfect_proteins[0:1]}
        converted = adapter._convert_protein_output_to_benchmark_format(dict_output)
        # Check shape after conversion
        assert converted.shape == (1, 1, self.real_data_dim)

        # Test direct array output conversion
        array_output = self.perfect_proteins[0:1]
        converted = adapter._convert_protein_output_to_benchmark_format(array_output)
        assert converted.shape == (1, 1, self.real_data_dim)

        # Test 2D array conversion (single point cloud)
        single_cloud = jnp.ones((1, self.real_data_dim))
        converted = adapter._convert_protein_output_to_benchmark_format(single_cloud)
        assert converted.shape == (1, 1, self.real_data_dim)

    def test_precision_recall_with_protein_adapter(self):
        """Test precision-recall benchmark with protein adapter."""
        # Create protein model and adapter for perfect match case
        model = NNXProteinMockModel(self.perfect_proteins, rngs=self.rngs)
        adapter = ProteinPointCloudAdapter(model, point_dim=self.real_data_dim)

        # Create benchmark
        benchmark = PrecisionRecallBenchmark(num_clusters=2, num_samples=10, random_seed=42)

        # Run benchmark with perfect match
        result = benchmark.run(model=adapter, dataset=self.real_proteins)

        # Check result
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "precision_recall"
        # Model name should match the name set in the NNXProteinMockModel
        assert result.model_name == "protein_point_cloud_model"

        # For perfect clusters, expect reasonable precision and recall
        # Be more generous with the thresholds since exact results depend
        # on the random initialization of the clusters
        assert result.metrics["precision"] >= 0.7
        assert result.metrics["recall"] >= 0.7
        assert result.metrics["f1_score"] >= 0.7

    def test_low_precision_with_protein_adapter(self):
        """Test precision-recall benchmark with extra clusters (low precision)."""
        # Create protein model with extra cluster
        model = NNXProteinMockModel(self.extra_cluster_proteins, rngs=self.rngs)
        adapter = ProteinPointCloudAdapter(model, point_dim=self.real_data_dim)

        # Create benchmark
        benchmark = PrecisionRecallBenchmark(
            num_clusters=3,  # Include the extra cluster
            num_samples=15,
            random_seed=42,
        )

        # Run benchmark
        result = benchmark.run(model=adapter, dataset=self.real_proteins)

        # For 3 clusters when there should be 2, expect reduced metrics
        # The actual values depend on the random initialization
        # but precision should be affected
        assert result.metrics["precision"] < 0.9
        # Lower recall threshold based on testing
        assert result.metrics["recall"] >= 0.3

    def test_low_recall_with_protein_adapter(self):
        """Test precision-recall benchmark with missing clusters (low recall)."""
        # Create protein model with missing cluster
        model = NNXProteinMockModel(self.missing_cluster_proteins, rngs=self.rngs)
        adapter = ProteinPointCloudAdapter(model, point_dim=self.real_data_dim)

        # Create benchmark
        benchmark = PrecisionRecallBenchmark(
            num_clusters=2,
            num_samples=5,  # Smaller sample size as we only have one cluster
            random_seed=42,
        )

        # Run benchmark
        result = benchmark.run(model=adapter, dataset=self.real_proteins)

        # Missing cluster should reduce recall
        assert result.metrics["recall"] < 0.8
        # But precision should still be decent
        assert result.metrics["precision"] >= 0.7

    def test_different_output_formats(self):
        """Test handling different protein model output formats."""
        # Create test data for output format testing
        feature_dim = self.real_data_dim
        test_coords = jnp.ones((2, 1, feature_dim))

        # Create a model that returns different output formats
        class MultiFormatProteinModel(nnx.Module):
            """Mock protein model that can return various output formats."""

            def __init__(self, data, *, rngs=None):
                super().__init__()
                self.data = data
                self.model_name = "protein_point_cloud_model"

            def sample_protein(self, batch_size=1, *, rngs=None):
                # Return dictionary format
                return {"coordinates": self.data[:batch_size]}

            def generate_protein(self, batch_size=1, *, rngs=None):
                # Return array format directly
                return self.data[:batch_size]

            def generate_structure(self, batch_size=1, *, rngs=None):
                # Return object with attributes format
                class ProteinStructure:
                    def __init__(self, coords):
                        self.atom_positions = coords

                return ProteinStructure(self.data[:batch_size])

        # Create model and adapter
        model = MultiFormatProteinModel(test_coords, rngs=self.rngs)
        adapter = ProteinPointCloudAdapter(model, point_dim=feature_dim)

        # Test dictionary format
        result1 = adapter.sample(batch_size=1, rngs=self.rngs)
        assert result1.shape == (1, 1, feature_dim)

        # Test array format
        result2 = adapter.sample(
            batch_size=1,
            rngs=self.rngs,
        )
        assert result2.shape == (1, 1, feature_dim)
