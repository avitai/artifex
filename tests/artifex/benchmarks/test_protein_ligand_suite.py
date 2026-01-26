"""Tests for protein-ligand co-design benchmark suite."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.benchmarks.datasets.crossdocked import CrossDockedDataset
from artifex.benchmarks.metrics.protein_ligand import (
    BindingAffinityMetric,
    DrugLikenessMetric,
    MolecularValidityMetric,
)
from artifex.benchmarks.suites.protein_ligand_suite import (
    ProteinLigandBenchmarkSuite,
    ProteinLigandCoDesignBenchmark,
)
from artifex.generative_models.core.configuration import DataConfig


@pytest.fixture
def rngs():
    """Fixture providing RNGs for tests."""
    return nnx.Rngs(42)


@pytest.fixture
def small_dataset(rngs):
    """Fixture providing a small dataset for testing."""
    config = DataConfig(
        name="crossdocked_small",
        dataset_name="CrossDocked2020",
        split="train",
        metadata={
            "num_samples": 10,
            "max_protein_atoms": 50,
            "max_ligand_atoms": 20,
            "pocket_radius": 10.0,
            "batch_size": 3,
        },
    )
    return CrossDockedDataset(
        data_path="test_results/test_data/mock_crossdocked", config=config, rngs=rngs
    )


class MockProteinLigandModel:
    """Mock model for testing protein-ligand benchmarks."""

    def __init__(self, rngs):
        self.rngs = rngs

    def predict_binding_affinity(
        self, protein_coords, protein_types, ligand_coords, ligand_types, **kwargs
    ):
        """Mock binding affinity prediction."""
        batch_size = protein_coords.shape[0]
        key = jax.random.key(123)
        # Generate mock predictions in reasonable range
        return jax.random.uniform(key, (batch_size,), minval=-10.0, maxval=-3.0)

    def generate_ligand(self, protein_coords, protein_types, **kwargs):
        """Mock ligand generation."""
        batch_size = protein_coords.shape[0]
        key = jax.random.key(456)
        keys = jax.random.split(key, 2)

        # Generate mock ligand coordinates near protein center
        protein_centers = jnp.mean(protein_coords, axis=1)  # (batch_size, 3)

        # Mock ligand with 15 atoms per molecule
        n_atoms = 15
        ligand_coords = (
            jax.random.normal(keys[0], (batch_size, n_atoms, 3)) * 1.5 + protein_centers[:, None, :]
        )

        # Mock atom types (C, N, O, etc.)
        ligand_types = jax.random.randint(keys[1], (batch_size, n_atoms), minval=1, maxval=8)

        # Create mask (all True since we're generating fixed number of atoms)
        ligand_mask = jnp.ones((batch_size, n_atoms), dtype=jnp.bool_)

        return {
            "coordinates": ligand_coords,
            "atom_types": ligand_types,
            "mask": ligand_mask,
        }


class TestCrossDockedDataset:
    """Test CrossDocked dataset implementation."""

    def test_dataset_initialization(self, rngs):
        """Test dataset can be initialized."""
        config = DataConfig(
            name="crossdocked_test",
            dataset_name="CrossDocked2020",
            split="train",
            metadata={
                "num_samples": 5,
                "max_protein_atoms": 100,
                "max_ligand_atoms": 30,
                "pocket_radius": 10.0,
                "batch_size": 32,
            },
        )
        dataset = CrossDockedDataset(
            data_path="test_results/test_data/mock_crossdocked", config=config, rngs=rngs
        )
        assert len(dataset) == 5
        assert dataset.max_protein_atoms == 100
        assert dataset.max_ligand_atoms == 30

    def test_dataset_getitem(self, small_dataset):
        """Test dataset indexing works correctly."""
        sample = small_dataset[0]

        # Check required fields
        required_fields = [
            "protein_coords",
            "protein_types",
            "ligand_coords",
            "ligand_types",
            "binding_affinity",
            "complex_id",
        ]
        for field in required_fields:
            assert field in sample

        # Check shapes
        assert sample["protein_coords"].shape[1] == 3  # 3D coordinates
        assert sample["ligand_coords"].shape[1] == 3
        assert len(sample["protein_coords"]) == len(sample["protein_types"])
        assert len(sample["ligand_coords"]) == len(sample["ligand_types"])

        # Check affinity is in reasonable range
        assert -15.0 <= sample["binding_affinity"] <= 0.0

    def test_dataset_batch_generation(self, small_dataset):
        """Test batch generation with proper padding."""
        batch = small_dataset.get_batch(batch_size=3)

        # Check batch fields
        required_fields = [
            "protein_coords",
            "protein_types",
            "protein_masks",
            "ligand_coords",
            "ligand_types",
            "ligand_masks",
            "binding_affinities",
            "complex_ids",
        ]
        for field in required_fields:
            assert field in batch

        # Check batch dimensions
        assert batch["protein_coords"].shape[0] == 3  # batch size
        assert batch["ligand_coords"].shape[0] == 3
        assert batch["binding_affinities"].shape == (3,)
        assert len(batch["complex_ids"]) == 3

        # Check masks are boolean
        assert batch["protein_masks"].dtype == jnp.bool_
        assert batch["ligand_masks"].dtype == jnp.bool_

    def test_pocket_extraction(self, small_dataset):
        """Test pocket extraction functionality."""
        sample = small_dataset[0]
        protein_coords = sample["protein_coords"]
        ligand_coords = sample["ligand_coords"]

        pocket_coords, pocket_indices = small_dataset.extract_pocket(
            protein_coords, ligand_coords, radius=5.0
        )

        # Check that pocket coordinates are subset of protein coordinates
        assert len(pocket_coords) <= len(protein_coords)
        assert len(pocket_indices) == len(pocket_coords)

        # Check that pocket atoms are within radius
        ligand_center = jnp.mean(ligand_coords, axis=0)
        distances = jnp.linalg.norm(pocket_coords - ligand_center, axis=1)
        assert jnp.all(distances <= 5.0)


class TestProteinLigandMetrics:
    """Test protein-ligand specific metrics."""

    def test_binding_affinity_metric(self, rngs):
        """Test binding affinity metric computation."""
        metric = BindingAffinityMetric(rngs=rngs)

        # Mock predictions and targets
        predictions = jnp.array([-8.5, -6.2, -9.1, -7.3])
        targets = jnp.array([-8.0, -6.5, -9.0, -7.0])

        results = metric.compute(predictions, targets)

        # Check metric fields
        expected_fields = ["rmse", "mae", "r2", "pearson_r"]
        for field in expected_fields:
            assert field in results
            assert isinstance(results[field], float)

        # Check RMSE is reasonable
        assert 0.0 <= results["rmse"] <= 1.0  # Should be small for close predictions
        assert 0.0 <= results["mae"] <= 1.0
        assert -1.0 <= results["pearson_r"] <= 1.0

    def test_molecular_validity_metric(self, rngs):
        """Test molecular validity metric computation."""
        metric = MolecularValidityMetric(rngs=rngs)

        # Create mock molecular data
        batch_size = 3
        num_atoms = 10

        # Generate reasonable molecular coordinates
        coordinates = jax.random.normal(rngs.default(), (batch_size, num_atoms, 3)) * 2.0
        atom_types = jax.random.randint(rngs.default(), (batch_size, num_atoms), 1, 8)
        masks = jnp.ones((batch_size, num_atoms), dtype=jnp.bool_)

        results = metric.compute(coordinates, atom_types, masks)

        # Check metric fields
        expected_fields = ["validity_rate", "bond_validity", "angle_validity", "clash_free"]
        for field in expected_fields:
            assert field in results
            assert isinstance(results[field], float)
            assert 0.0 <= results[field] <= 1.0  # All should be rates/fractions

    def test_drug_likeness_metric(self, rngs):
        """Test drug-likeness metric computation."""
        metric = DrugLikenessMetric(rngs=rngs)

        # Create mock ligand data
        batch_size = 2
        num_atoms = 25  # Drug-like size

        coordinates = jax.random.normal(rngs.default(), (batch_size, num_atoms, 3)) * 3.0
        atom_types = jax.random.randint(rngs.default(), (batch_size, num_atoms), 1, 8)
        masks = jnp.ones((batch_size, num_atoms), dtype=jnp.bool_)

        results = metric.compute(coordinates, atom_types, masks)

        # Check metric fields
        expected_fields = [
            "qed_score",
            "lipinski_compliance",
            "molecular_weight",
            "num_rotatable_bonds",
        ]
        for field in expected_fields:
            assert field in results
            assert isinstance(results[field], float)

        # Check ranges
        assert 0.0 <= results["qed_score"] <= 1.0
        assert 0.0 <= results["lipinski_compliance"] <= 1.0
        assert results["molecular_weight"] > 0.0


class TestProteinLigandBenchmark:
    """Test the main protein-ligand co-design benchmark."""

    def test_benchmark_initialization(self, small_dataset, rngs):
        """Test benchmark can be initialized."""
        benchmark = ProteinLigandCoDesignBenchmark(
            dataset=small_dataset, num_samples=5, batch_size=2, rngs=rngs
        )

        assert benchmark.config.name == "protein_ligand_codesign"
        assert benchmark.num_samples == 5
        assert benchmark.batch_size == 2
        assert "binding_affinity_rmse" in benchmark.config.metric_names
        assert "molecular_validity_rate" in benchmark.config.metric_names

        # Check target metrics are set
        target_metrics = benchmark.config.metadata.get("target_metrics", {})
        assert target_metrics["binding_affinity_rmse"] == 1.0
        assert target_metrics["molecular_validity_rate"] == 0.95
        assert target_metrics["qed_score"] == 0.7

    def test_benchmark_run_with_mock_model(self, small_dataset, rngs):
        """Test benchmark execution with mock model."""
        benchmark = ProteinLigandCoDesignBenchmark(
            dataset=small_dataset, num_samples=4, batch_size=2, rngs=rngs
        )

        model = MockProteinLigandModel(rngs)
        result = benchmark.run(model)

        # Check result structure
        assert result.benchmark_name == "protein_ligand_codesign"
        assert isinstance(result.metrics, dict)
        assert isinstance(result.metadata, dict)

        # Check key metrics are present
        assert "binding_affinity_rmse" in result.metrics
        assert "molecular_validity_rate" in result.metrics
        assert "qed_score" in result.metrics

        # Check metadata
        assert result.metadata["num_samples"] == 4
        assert result.metadata["batch_size"] == 2

    def test_mock_prediction_generation(self, small_dataset, rngs):
        """Test mock prediction generation for fallback."""
        benchmark = ProteinLigandCoDesignBenchmark(
            dataset=small_dataset, num_samples=3, batch_size=3, rngs=rngs
        )

        true_affinities = jnp.array([-8.0, -6.5, -9.2])
        mock_predictions = benchmark._generate_mock_predictions(true_affinities)

        assert mock_predictions.shape == true_affinities.shape
        # Predictions should be close but not identical
        assert not jnp.allclose(mock_predictions, true_affinities)


class TestProteinLigandBenchmarkSuite:
    """Test the complete protein-ligand benchmark suite."""

    def test_suite_initialization(self, rngs):
        """Test suite can be initialized with default config."""
        suite = ProteinLigandBenchmarkSuite(
            dataset_config={"num_samples": 20},
            benchmark_config={"num_samples": 5, "batch_size": 2},
            rngs=rngs,
        )

        assert suite.name == "protein_ligand_codesign_suite"
        assert len(suite.benchmarks) == 1  # Should have one main benchmark
        assert len(suite.dataset) == 20

    def test_suite_run_all(self, rngs):
        """Test running all benchmarks in the suite."""
        suite = ProteinLigandBenchmarkSuite(
            dataset_config={"num_samples": 8},
            benchmark_config={"num_samples": 4, "batch_size": 2},
            rngs=rngs,
        )

        model = MockProteinLigandModel(rngs)
        results = suite.run_all(model)

        # Check results structure
        assert isinstance(results, dict)
        assert len(results) == len(suite.benchmarks)

        # Check each result
        for benchmark_name, result in results.items():
            assert isinstance(result.metrics, dict)
            assert "binding_affinity_rmse" in result.metrics
            assert "molecular_validity_rate" in result.metrics
            assert "qed_score" in result.metrics


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_pipeline(self, rngs):
        """Test complete end-to-end pipeline with small data."""
        # Initialize suite with minimal configuration
        suite = ProteinLigandBenchmarkSuite(
            dataset_config={
                "num_samples": 6,
                "max_protein_atoms": 100,  # Use higher value to accommodate mock data
                "max_ligand_atoms": 30,  # Use higher value to accommodate mock data
            },
            benchmark_config={
                "num_samples": 4,
                "batch_size": 2,
            },
            rngs=rngs,
        )

        # Create mock model
        model = MockProteinLigandModel(rngs)

        # Run complete evaluation
        results = suite.run_all(model)

        # Verify comprehensive results
        assert len(results) > 0

        for benchmark_name, result in results.items():
            metrics = result.metrics

            # Check Week 5-8 target metrics are evaluated
            assert "binding_affinity_rmse" in metrics
            assert "molecular_validity_rate" in metrics
            assert "qed_score" in metrics

            # Check values are reasonable
            assert metrics["binding_affinity_rmse"] >= 0.0
            assert 0.0 <= metrics["molecular_validity_rate"] <= 1.0
            assert 0.0 <= metrics["qed_score"] <= 1.0

            # Check additional metrics
            assert "binding_affinity_mae" in metrics
            assert "bond_validity" in metrics
            assert "lipinski_compliance" in metrics

    def test_performance_targets_validation(self, rngs):
        """Test that the benchmark properly validates against performance targets."""
        suite = ProteinLigandBenchmarkSuite(
            dataset_config={"num_samples": 4, "max_protein_atoms": 100, "max_ligand_atoms": 30},
            benchmark_config={"num_samples": 2, "batch_size": 1},
            rngs=rngs,
        )

        model = MockProteinLigandModel(rngs)
        results = suite.run_all(model)

        # Check target metrics are defined and reasonable
        for result in results.values():
            target_metrics = result.metadata.get("target_metrics", {})

            # Week 5-8 targets should be present
            assert target_metrics.get("binding_affinity_rmse") == 1.0
            assert target_metrics.get("molecular_validity_rate") == 0.95
            assert target_metrics.get("qed_score") == 0.7

            # Actual metrics should be evaluated
            actual_metrics = result.metrics
            assert "binding_affinity_rmse" in actual_metrics
            assert "molecular_validity_rate" in actual_metrics
            assert "qed_score" in actual_metrics
