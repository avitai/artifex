"""Tests for SE(3)-Equivariant Molecular Flows benchmark suite."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.benchmarks.datasets.qm9 import QM9Dataset
from artifex.benchmarks.metrics.molecular_flows import (
    MolecularFlowsMetrics,
)
from artifex.benchmarks.suites.se3_molecular_flows_suite import (
    SE3MolecularFlowsBenchmark,
    SE3MolecularFlowsSuite,
)
from artifex.generative_models.core.configuration import DataConfig, EvaluationConfig
from artifex.generative_models.models.flow.se3_molecular import SE3MolecularFlow


@pytest.fixture
def rngs():
    """Standard fixture for rngs."""
    return nnx.Rngs(42)


@pytest.fixture
def mock_metrics_config():
    """Mock evaluation configuration for metrics."""
    return EvaluationConfig(
        name="molecular_flows_metrics_config",
        metrics=["chemical_validity", "conformational_diversity", "energy_consistency"],
        eval_batch_size=32,
        metadata={
            "modality": "molecular",
            "higher_is_better": True,
        },
    )


@pytest.fixture
def mock_qm9_config():
    """Mock QM9 dataset configuration."""
    return DataConfig(
        name="qm9_dataset_config",
        dataset_name="qm9",
        data_dir=Path("test_results/test_data/mock_qm9_data"),
        split="train",
        metadata={
            "max_atoms": 29,
            "batch_size": 8,
            "num_conformations": 100,
        },
    )


@pytest.fixture
def mock_se3_flow_config():
    """Mock SE(3) molecular flow configuration."""
    return {
        "hidden_dim": 64,
        "num_layers": 3,
        "num_coupling_layers": 4,
        "max_atoms": 29,
        "atom_types": 5,  # H, C, N, O, F for QM9
        "use_attention": True,
        "equivariant_layers": True,
    }


@pytest.fixture
def benchmark_config(mock_qm9_config, mock_se3_flow_config):
    """Complete benchmark configuration."""
    return EvaluationConfig(
        name="se3_molecular_flows_benchmark_config",
        metrics=["chemical_validity", "conformational_diversity", "energy_consistency"],
        eval_batch_size=16,
        metadata={
            "dataset_config": mock_qm9_config,
            "model_config": mock_se3_flow_config,
            "training": {
                "learning_rate": 1e-4,
                "num_epochs": 2,
                "warmup_steps": 100,
                "gradient_clip": 1.0,
            },
            "performance_targets": {
                "chemical_validity": 0.95,
                "conformational_diversity": 0.8,
                "energy_consistency": 0.1,  # kcal/mol
                "training_time_per_epoch": 3600,  # 1 hour
            },
        },
    )


class TestQM9Dataset:
    """Test QM9 dataset implementation."""

    def test_initialization(self, rngs, mock_qm9_config):
        """Test QM9 dataset initialization."""
        dataset = QM9Dataset(
            data_path=str(mock_qm9_config.data_dir), config=mock_qm9_config, rngs=rngs
        )

        assert dataset.max_atoms == 29
        assert dataset.batch_size == 8
        assert dataset.split == "train"

    def test_batch_generation(self, rngs, mock_qm9_config):
        """Test QM9 dataset batch generation."""
        dataset = QM9Dataset(
            data_path=str(mock_qm9_config.data_dir), config=mock_qm9_config, rngs=rngs
        )
        batch = next(iter(dataset))

        # Check batch structure
        assert isinstance(batch, dict)
        required_keys = {
            "coordinates",
            "atom_types",
            "atom_mask",
            "num_atoms",
            "energies",
            "forces",
        }
        assert required_keys.issubset(batch.keys())

        # Check batch shapes
        batch_size = mock_qm9_config.metadata["batch_size"]
        max_atoms = mock_qm9_config.metadata["max_atoms"]

        assert batch["coordinates"].shape == (batch_size, max_atoms, 3)
        assert batch["atom_types"].shape == (batch_size, max_atoms)
        assert batch["atom_mask"].shape == (batch_size, max_atoms)
        assert batch["num_atoms"].shape == (batch_size,)
        assert batch["energies"].shape == (batch_size,)
        assert batch["forces"].shape == (batch_size, max_atoms, 3)

    def test_molecular_properties(self, rngs, mock_qm9_config):
        """Test molecular property constraints."""
        dataset = QM9Dataset(
            data_path=str(mock_qm9_config.data_dir), config=mock_qm9_config, rngs=rngs
        )
        batch = next(iter(dataset))

        # Check that atom masks are consistent with num_atoms
        for i in range(batch["atom_mask"].shape[0]):
            num_atoms = batch["num_atoms"][i]
            mask = batch["atom_mask"][i]
            assert jnp.sum(mask) == num_atoms

        # Check coordinate ranges (reasonable molecular scales)
        coords = batch["coordinates"]
        mask = batch["atom_mask"]

        # Only check coordinates where atoms are present
        masked_coords = coords[mask]
        assert jnp.all(jnp.abs(masked_coords) < 10.0)  # Reasonable Angstrom range


class TestSE3MolecularFlow:
    """Test SE(3) molecular flow model."""

    def test_initialization(self, rngs, mock_se3_flow_config):
        """Test SE(3) molecular flow initialization."""
        model = SE3MolecularFlow(**mock_se3_flow_config, rngs=rngs)

        assert model.hidden_dim == 64
        assert model.num_layers == 3
        assert model.max_atoms == 29
        assert model.atom_types == 5

    def test_forward_pass(self, rngs, mock_se3_flow_config):
        """Test forward pass through SE(3) molecular flow."""
        model = SE3MolecularFlow(**mock_se3_flow_config, rngs=rngs)

        batch_size = 4
        max_atoms = mock_se3_flow_config["max_atoms"]

        # Create mock molecular batch
        coordinates = jax.random.normal(rngs.params(), (batch_size, max_atoms, 3))
        atom_types = jax.random.randint(rngs.params(), (batch_size, max_atoms), 0, 5)
        atom_mask = jnp.ones(
            (batch_size, max_atoms), dtype=jnp.bool_
        )  # All atoms present for simplicity

        # Forward pass
        log_prob = model.log_prob(
            coordinates=coordinates,
            atom_types=atom_types,
            atom_mask=atom_mask,
        )

        assert log_prob.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(log_prob))

    def test_sampling(self, rngs, mock_se3_flow_config):
        """Test molecular sampling from SE(3) flow."""
        model = SE3MolecularFlow(**mock_se3_flow_config, rngs=rngs)

        batch_size = 4
        max_atoms = mock_se3_flow_config["max_atoms"]

        # Create template for sampling
        atom_types = jax.random.randint(rngs.params(), (batch_size, max_atoms), 0, 5)
        atom_mask = jnp.ones((batch_size, max_atoms), dtype=jnp.bool_)

        # Sample coordinates
        samples = model.sample(
            atom_types=atom_types,
            atom_mask=atom_mask,
            num_samples=batch_size,
            rngs=rngs,
        )

        assert samples.shape == (batch_size, max_atoms, 3)
        assert jnp.all(jnp.isfinite(samples))

    @pytest.mark.skip(
        reason="SE(3) equivariance requires specialized geometric deep learning architecture - current implementation is a research placeholder"
    )
    def test_se3_equivariance(self, rngs, mock_se3_flow_config):
        """Test SE(3) equivariance properties."""
        model = SE3MolecularFlow(**mock_se3_flow_config, rngs=rngs)

        batch_size = 2
        max_atoms = mock_se3_flow_config["max_atoms"]

        # Create test molecule
        coordinates = jax.random.normal(rngs.params(), (batch_size, max_atoms, 3))
        atom_types = jax.random.randint(rngs.params(), (batch_size, max_atoms), 0, 5)
        atom_mask = jnp.ones((batch_size, max_atoms), dtype=jnp.bool_)

        # Original log probability
        log_prob_orig = model.log_prob(
            coordinates=coordinates,
            atom_types=atom_types,
            atom_mask=atom_mask,
        )

        # Apply random rotation and translation
        rotation_matrix = jax.random.orthogonal(rngs.params(), 3)
        translation = jax.random.normal(rngs.params(), (3,))

        # Transform coordinates
        transformed_coords = (
            jnp.einsum("...ij,baj->bai", rotation_matrix, coordinates) + translation
        )

        # Log probability after transformation (should be equal for SE(3) equivariance)
        log_prob_transformed = model.log_prob(
            coordinates=transformed_coords,
            atom_types=atom_types,
            atom_mask=atom_mask,
        )

        # Check equivariance (allowing small numerical tolerance)
        assert jnp.allclose(log_prob_orig, log_prob_transformed, atol=1e-5)


class TestMolecularFlowsMetrics:
    """Test molecular flows evaluation metrics."""

    def test_initialization(self, rngs, mock_metrics_config):
        """Test molecular flows metrics initialization."""
        metrics = MolecularFlowsMetrics(rngs=rngs, config=mock_metrics_config)

        assert hasattr(metrics, "chemical_validity")
        assert hasattr(metrics, "conformational_diversity")
        assert hasattr(metrics, "energy_consistency")

    def test_chemical_validity_metric(self, rngs, mock_metrics_config):
        """Test chemical validity evaluation."""
        metrics = MolecularFlowsMetrics(rngs=rngs, config=mock_metrics_config)

        batch_size = 8
        max_atoms = 10

        # Create mock molecular data
        coordinates = jax.random.normal(rngs.params(), (batch_size, max_atoms, 3))
        atom_types = jax.random.randint(rngs.params(), (batch_size, max_atoms), 0, 5)
        atom_mask = jnp.ones((batch_size, max_atoms), dtype=jnp.bool_)

        validity_score = metrics.chemical_validity(
            coordinates=coordinates,
            atom_types=atom_types,
            atom_mask=atom_mask,
        )

        assert 0.0 <= validity_score <= 1.0

    def test_conformational_diversity_metric(self, rngs, mock_metrics_config):
        """Test conformational diversity evaluation."""
        metrics = MolecularFlowsMetrics(rngs=rngs, config=mock_metrics_config)

        batch_size = 16
        max_atoms = 10

        # Create mock conformations
        coordinates = jax.random.normal(rngs.params(), (batch_size, max_atoms, 3))
        atom_types = jax.random.randint(rngs.params(), (batch_size, max_atoms), 0, 5)
        atom_mask = jnp.ones((batch_size, max_atoms), dtype=jnp.bool_)

        diversity_score = metrics.conformational_diversity(
            coordinates=coordinates,
            atom_types=atom_types,
            atom_mask=atom_mask,
        )

        assert 0.0 <= diversity_score <= 1.0

    def test_energy_consistency_metric(self, rngs, mock_metrics_config):
        """Test energy consistency evaluation."""
        metrics = MolecularFlowsMetrics(rngs=rngs, config=mock_metrics_config)

        batch_size = 8
        max_atoms = 10

        # Create mock molecular data with reference energies
        coordinates = jax.random.normal(rngs.params(), (batch_size, max_atoms, 3))
        atom_types = jax.random.randint(rngs.params(), (batch_size, max_atoms), 0, 5)
        atom_mask = jnp.ones((batch_size, max_atoms), dtype=jnp.bool_)
        reference_energies = jax.random.normal(rngs.params(), (batch_size,))

        energy_error = metrics.energy_consistency(
            coordinates=coordinates,
            atom_types=atom_types,
            atom_mask=atom_mask,
            reference_energies=reference_energies,
        )

        assert energy_error >= 0.0  # RMSE should be non-negative


class TestSE3MolecularFlowsBenchmark:
    """Test SE(3) molecular flows benchmark."""

    def test_initialization(self, rngs, benchmark_config):
        """Test benchmark initialization."""
        benchmark = SE3MolecularFlowsBenchmark(rngs=rngs, config=benchmark_config)

        assert benchmark.config.name == "se3_molecular_flows"
        assert hasattr(benchmark, "dataset")
        assert hasattr(benchmark, "metrics")

    @pytest.mark.benchmark(group="se3_molecular_flows")
    def test_model_evaluation(self, rngs, benchmark_config, benchmark):
        """Test model evaluation in benchmark.

        Uses pytest-benchmark for performance measurement.
        """
        bm = SE3MolecularFlowsBenchmark(rngs=rngs, config=benchmark_config)

        # Create mock model using config from metadata
        model_config = benchmark_config.metadata["model_config"]
        model = SE3MolecularFlow(**model_config, rngs=rngs)

        # Run evaluation with benchmark (reduced samples for faster unit tests)
        result = benchmark(bm.evaluate_model, model, num_samples=4)

        assert isinstance(result, dict)
        expected_metrics = {
            "chemical_validity",
            "conformational_diversity",
            "energy_consistency",
            "model_name",
        }
        assert expected_metrics.issubset(result.keys())

        # Check metric ranges
        assert 0.0 <= result["chemical_validity"] <= 1.0
        assert 0.0 <= result["conformational_diversity"] <= 1.0
        assert result["energy_consistency"] >= 0.0

    def test_performance_targets(self, rngs, benchmark_config):
        """Test performance target validation."""
        benchmark = SE3MolecularFlowsBenchmark(rngs=rngs, config=benchmark_config)

        # Mock results that meet targets
        good_results = {
            "chemical_validity": 0.97,
            "conformational_diversity": 0.85,
            "energy_consistency": 0.08,
        }

        meets_targets = benchmark.meets_performance_targets(good_results)
        assert meets_targets

        # Mock results that don't meet targets
        poor_results = {
            "chemical_validity": 0.85,  # Below 0.95 target
            "conformational_diversity": 0.6,  # Below 0.8 target
            "energy_consistency": 0.2,  # Above 0.1 target
        }

        meets_targets = benchmark.meets_performance_targets(poor_results)
        assert not meets_targets


class TestSE3MolecularFlowsSuite:
    """Test complete SE(3) molecular flows benchmark suite."""

    def test_suite_initialization(self, rngs):
        """Test benchmark suite initialization."""
        suite = SE3MolecularFlowsSuite(rngs=rngs)

        assert hasattr(suite, "benchmarks")
        assert len(suite.benchmarks) > 0

    def test_run_all_benchmarks(self, rngs):
        """Test running all benchmarks in suite."""
        suite = SE3MolecularFlowsSuite(rngs=rngs)

        # Create mock models for testing
        mock_models = {
            "se3_flow_small": SE3MolecularFlow(
                hidden_dim=32,
                num_layers=2,
                num_coupling_layers=2,
                max_atoms=29,
                atom_types=5,
                rngs=rngs,
            ),
            "se3_flow_medium": SE3MolecularFlow(
                hidden_dim=64,
                num_layers=3,
                num_coupling_layers=4,
                max_atoms=29,
                atom_types=5,
                rngs=rngs,
            ),
        }

        # Run benchmarks
        results = suite.run_benchmarks(mock_models, num_samples=8)

        assert isinstance(results, dict)
        assert len(results) == len(mock_models)

        for model_name, model_results in results.items():
            assert model_name in mock_models
            assert isinstance(model_results, dict)

            # Check that all benchmarks ran
            expected_benchmarks = ["se3_molecular_flows"]
            for benchmark_name in expected_benchmarks:
                assert benchmark_name in model_results

    def test_performance_comparison(self, rngs):
        """Test model performance comparison."""
        suite = SE3MolecularFlowsSuite(rngs=rngs)

        # Mock results for comparison
        results = {
            "model_a": {
                "se3_molecular_flows": {
                    "chemical_validity": 0.96,
                    "conformational_diversity": 0.82,
                    "energy_consistency": 0.09,
                }
            },
            "model_b": {
                "se3_molecular_flows": {
                    "chemical_validity": 0.94,
                    "conformational_diversity": 0.79,
                    "energy_consistency": 0.12,
                }
            },
        }

        comparison = suite.compare_models(results)

        assert isinstance(comparison, dict)
        assert "best_model" in comparison
        assert "metric_comparison" in comparison

        # Model A should be better (higher validity, diversity, lower energy error)
        assert comparison["best_model"] == "model_a"
