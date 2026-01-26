"""Tests for geometric benchmark suite."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from artifex.benchmarks.datasets.geometric import (
    GeometricDatasetRegistry,
    ShapeNetDataset,
)
from artifex.benchmarks.metrics.geometric import PointCloudMetrics
from artifex.benchmarks.suites.geometric_suite import (
    GeometricBenchmarkSuite,
    PointCloudGenerationBenchmark,
)
from artifex.generative_models.core.configuration import (
    DataConfig,
    EvaluationConfig,
    ModelConfig,
)


@pytest.fixture
def rngs():
    """Standard RNG fixture."""
    return nnx.Rngs(42)


@pytest.fixture
def sample_config():
    """Sample configuration for benchmarks."""
    dataset_config = DataConfig(
        name="test_shapenet",
        dataset_name="shapenet",
        metadata={
            "num_points": 512,
            "mock_samples": 100,
            "normalize": True,
        },
    )

    model_config = ModelConfig(
        name="test_pointcloud_model",
        model_class="artifex.generative_models.models.geometric.PointCloudModel",
        input_dim=(512, 3),
        hidden_dims=[128, 256, 128],
        output_dim=(512, 3),
        activation="relu",
        metadata={
            "geometric_params": {
                "embed_dim": 128,
                "num_points": 512,
                "num_layers": 3,
                "num_heads": 4,
                "dropout": 0.1,
            }
        },
    )

    eval_config = EvaluationConfig(
        name="test_geometric_eval",
        metrics=["1nn_accuracy", "coverage", "chamfer_distance"],
        metric_params={
            "1nn_accuracy": {"k": 1},
            "coverage": {"threshold": 0.01},
            "chamfer_distance": {"reduction": "mean"},
        },
        eval_batch_size=8,
    )

    return {
        "dataset_path": "data/test_shapenet",
        "dataset_config": dataset_config,
        "model_config": model_config,
        "eval_config": eval_config,
        "training_config": {
            "num_epochs": 2,
            "batch_size": 8,
            "learning_rate": 1e-4,
        },
        "performance_targets": {
            "1nn_accuracy": 0.95,
            "coverage": 0.8,
            "training_time_per_epoch": 4.0,
        },
    }


@pytest.fixture
def sample_point_clouds(rngs):
    """Generate sample point clouds for testing."""
    key = rngs.params() if hasattr(rngs, "params") else jax.random.key(42)

    # Generate simple sphere and cube point clouds
    batch_size = 4
    num_points = 256

    # Sphere points
    sphere_key, cube_key = jax.random.split(key)

    sphere_points = jax.random.normal(sphere_key, (batch_size // 2, num_points, 3))
    sphere_norms = jnp.linalg.norm(sphere_points, axis=2, keepdims=True)
    sphere_points = sphere_points / (sphere_norms + 1e-8)

    # Cube points
    cube_points = jax.random.uniform(
        cube_key, (batch_size // 2, num_points, 3), minval=-1, maxval=1
    )

    return jnp.concatenate([sphere_points, cube_points], axis=0)


class TestShapeNetDataset:
    """Tests for ShapeNet dataset implementation."""

    def test_dataset_initialization(self, sample_config, rngs):
        """Test dataset initialization with mock data."""
        dataset = ShapeNetDataset(
            data_path=sample_config["dataset_path"],
            config=sample_config["dataset_config"],
            rngs=rngs,
        )

        assert dataset.num_points == 512
        assert dataset.normalize
        assert hasattr(dataset, "data")
        assert "train" in dataset.data
        assert "val" in dataset.data
        assert "test" in dataset.data

    def test_dataset_batch_generation(self, sample_config, rngs):
        """Test batch generation from dataset."""
        dataset = ShapeNetDataset(
            data_path=sample_config["dataset_path"],
            config=sample_config["dataset_config"],
            rngs=rngs,
        )

        batch_size = 8
        batch = dataset.get_batch(batch_size, split="train")

        assert "point_clouds" in batch
        assert "labels" in batch
        assert batch["point_clouds"].shape == (batch_size, 512, 3)
        assert batch["labels"].shape == (batch_size,)

    def test_dataset_info(self, sample_config, rngs):
        """Test dataset information retrieval."""
        dataset = ShapeNetDataset(
            data_path=sample_config["dataset_path"],
            config=sample_config["dataset_config"],
            rngs=rngs,
        )

        info = dataset.get_dataset_info()

        assert info["name"] == "ShapeNet"
        assert info["num_points"] == 512
        assert "train_size" in info
        assert "val_size" in info
        assert "test_size" in info

    def test_mock_data_generation(self, sample_config, rngs):
        """Test mock data generation with different shapes."""
        dataset = ShapeNetDataset(
            data_path=sample_config["dataset_path"],
            config=sample_config["dataset_config"],
            rngs=rngs,
        )

        # Check that different shape types are generated
        train_data = dataset.data["train"]
        labels = train_data["labels"]

        # Should have 3 different shape types (0, 1, 2)
        unique_labels = jnp.unique(labels)
        assert len(unique_labels) == 3
        assert set(unique_labels.tolist()) == {0, 1, 2}


class TestPointCloudMetrics:
    """Tests for point cloud metrics implementation."""

    def test_metrics_initialization(self, rngs):
        """Test metrics initialization."""
        # Create evaluation configuration
        eval_config = EvaluationConfig(
            name="test_point_cloud_metrics",
            metrics=["1nn_accuracy", "coverage", "chamfer_distance"],
            metric_params={
                "point_cloud": {
                    "higher_is_better": True,
                    "coverage_threshold": 0.1,
                    "metric_weights": {
                        "1nn_accuracy": 0.4,
                        "coverage": 0.3,
                        "geometric_fidelity": 0.2,
                        "chamfer_distance": 0.1,
                    },
                }
            },
            eval_batch_size=32,
        )

        metrics = PointCloudMetrics(rngs=rngs, config=eval_config)

        assert metrics.metric_name == "point_cloud_metrics"
        assert hasattr(metrics, "rngs")

    def test_1nn_accuracy_computation(self, rngs, sample_point_clouds):
        """Test 1-NN accuracy computation."""
        eval_config = EvaluationConfig(
            name="test_1nn_accuracy",
            metrics=["1nn_accuracy"],
            metric_params={"point_cloud": {"higher_is_better": True}},
            eval_batch_size=32,
        )
        metrics = PointCloudMetrics(rngs=rngs, config=eval_config)

        # Use same data for both generated and real (should give high accuracy)
        generated = sample_point_clouds
        real = sample_point_clouds

        accuracy = metrics._compute_1nn_accuracy(generated, real)

        # Should be high accuracy since we're using identical data
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_chamfer_distance_computation(self, rngs, sample_point_clouds):
        """Test Chamfer distance computation."""
        eval_config = EvaluationConfig(
            name="test_chamfer_distance",
            metrics=["chamfer_distance"],
            metric_params={"point_cloud": {"higher_is_better": False}},
            eval_batch_size=32,
        )
        metrics = PointCloudMetrics(rngs=rngs, config=eval_config)

        generated = sample_point_clouds
        real = sample_point_clouds

        chamfer_dist = metrics._compute_chamfer_distance(generated, real)

        # Should be very small for identical data
        assert isinstance(chamfer_dist, float)
        assert chamfer_dist >= 0.0
        assert chamfer_dist < 0.1  # Should be small for identical data

    def test_coverage_computation(self, rngs, sample_point_clouds):
        """Test coverage metric computation."""
        eval_config = EvaluationConfig(
            name="test_coverage",
            metrics=["coverage"],
            metric_params={"point_cloud": {"higher_is_better": True, "coverage_threshold": 0.5}},
            eval_batch_size=32,
        )
        metrics = PointCloudMetrics(rngs=rngs, config=eval_config)

        generated = sample_point_clouds
        real = sample_point_clouds

        coverage = metrics._compute_coverage(generated, real, coverage_threshold=0.5)

        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0

    def test_geometric_fidelity_computation(self, rngs, sample_point_clouds):
        """Test geometric fidelity computation."""
        eval_config = EvaluationConfig(
            name="test_geometric_fidelity",
            metrics=["geometric_fidelity"],
            metric_params={"point_cloud": {"higher_is_better": True}},
            eval_batch_size=32,
        )
        metrics = PointCloudMetrics(rngs=rngs, config=eval_config)

        generated = sample_point_clouds
        real = sample_point_clouds

        fidelity = metrics._compute_geometric_fidelity(generated, real)

        assert isinstance(fidelity, float)
        assert 0.0 <= fidelity <= 1.0

    def test_complete_metrics_computation(self, rngs, sample_point_clouds):
        """Test complete metrics computation."""
        eval_config = EvaluationConfig(
            name="test_complete_metrics",
            metrics=[
                "1nn_accuracy",
                "coverage",
                "chamfer_distance",
                "geometric_fidelity",
                "earth_movers_distance",
            ],
            metric_params={"point_cloud": {"higher_is_better": True}},
            eval_batch_size=32,
        )
        metrics = PointCloudMetrics(rngs=rngs, config=eval_config)

        generated = sample_point_clouds
        real = sample_point_clouds

        results = metrics.compute_metrics(generated, real)

        expected_metrics = [
            "1nn_accuracy",
            "coverage",
            "chamfer_distance",
            "geometric_fidelity",
            "earth_movers_distance",
        ]

        for metric_name in expected_metrics:
            assert metric_name in results
            assert isinstance(results[metric_name], float)

    def test_combined_metric_computation(self, rngs, sample_point_clouds):
        """Test combined metric score computation."""
        eval_config = EvaluationConfig(
            name="test_combined_metric",
            metrics=["1nn_accuracy", "coverage", "chamfer_distance"],
            metric_params={"point_cloud": {"higher_is_better": True}},
            eval_batch_size=32,
        )
        metrics = PointCloudMetrics(rngs=rngs, config=eval_config)

        generated = sample_point_clouds
        real = sample_point_clouds

        combined_score = metrics.compute_metric(generated, real)

        assert isinstance(combined_score, float)
        assert 0.0 <= combined_score <= 1.0


class TestPointCloudGenerationBenchmark:
    """Tests for point cloud generation benchmark."""

    def test_benchmark_initialization(self, sample_config, rngs):
        """Test benchmark initialization."""
        benchmark = PointCloudGenerationBenchmark(
            config=sample_config,
            rngs=rngs,
        )

        assert hasattr(benchmark, "dataset")
        assert hasattr(benchmark, "model")
        assert hasattr(benchmark, "metrics")
        assert hasattr(benchmark, "performance_targets")

    def test_benchmark_info(self, sample_config, rngs):
        """Test benchmark information retrieval."""
        benchmark = PointCloudGenerationBenchmark(
            config=sample_config,
            rngs=rngs,
        )

        info = benchmark.get_benchmark_info()

        assert info["name"] == "Point Cloud Generation"
        assert info["dataset"] == "ShapeNet"
        assert info["model"] == "PointCloudModel"
        assert "targets" in info
        assert "config" in info

    def test_training_execution(self, sample_config, rngs):
        """Test training phase execution."""
        # Use smaller config for faster testing
        small_config = sample_config.copy()
        small_config["training_config"]["num_epochs"] = 1
        small_config["training_config"]["batch_size"] = 4

        benchmark = PointCloudGenerationBenchmark(
            config=small_config,
            rngs=rngs,
        )

        training_results = benchmark.run_training()

        assert "final_loss" in training_results
        assert "avg_epoch_time_hours" in training_results
        assert "total_training_time_hours" in training_results
        assert isinstance(training_results["final_loss"], float)

    def test_evaluation_execution(self, sample_config, rngs):
        """Test evaluation phase execution."""
        small_config = sample_config.copy()
        small_config["eval_batch_size"] = 4

        benchmark = PointCloudGenerationBenchmark(
            config=small_config,
            rngs=rngs,
        )

        evaluation_results = benchmark.run_evaluation()

        # Should contain all the metrics computed
        expected_metrics = [
            "1nn_accuracy",
            "coverage",
            "chamfer_distance",
            "geometric_fidelity",
            "earth_movers_distance",
        ]

        for metric_name in expected_metrics:
            assert metric_name in evaluation_results

    def test_performance_validation(self, sample_config, rngs):
        """Test performance validation against targets."""
        benchmark = PointCloudGenerationBenchmark(
            config=sample_config,
            rngs=rngs,
        )

        # Mock results
        mock_results = {
            "1nn_accuracy": 0.96,  # Above target (0.95)
            "coverage": 0.85,  # Above target (0.8)
            "avg_epoch_time_hours": 3.5,  # Below target (4.0)
        }

        validation = benchmark.validate_performance(mock_results)

        assert "1nn_accuracy_target" in validation
        assert "coverage_target" in validation
        assert "training_time_target" in validation
        assert validation["1nn_accuracy_target"]
        assert validation["coverage_target"]
        assert validation["training_time_target"]


class TestGeometricBenchmarkSuite:
    """Tests for geometric benchmark suite."""

    def test_suite_initialization(self, rngs):
        """Test suite initialization with properly typed configurations."""
        from artifex.generative_models.core.configuration import (
            DataConfig,
            EvaluationConfig,
            ModelConfig,
            OptimizerConfig,
            TrainingConfig,
        )

        # Create proper typed configurations
        dataset_config = DataConfig(
            name="test_shapenet",
            dataset_name="shapenet",
            metadata={"num_points": 256, "mock_samples": 50},
        )

        model_config = ModelConfig(
            name="test_pointcloud_model",
            model_class="artifex.generative_models.models.geometric.PointCloudModel",
            input_dim=(256, 3),
            hidden_dims=[64],
            output_dim=(256, 3),
            activation="relu",
            metadata={"geometric_params": {"embed_dim": 64, "num_layers": 2}},
        )

        eval_config = EvaluationConfig(
            name="test_geometric_eval",
            metrics=["1nn_accuracy", "coverage", "chamfer_distance"],
            metric_params={
                "1nn_accuracy": {"k": 1},
                "coverage": {"threshold": 0.01},
                "chamfer_distance": {"reduction": "mean"},
            },
            eval_batch_size=4,
        )

        # Create optimizer configuration
        optimizer_config = OptimizerConfig(
            name="test_optimizer", optimizer_type="adam", learning_rate=1e-4
        )

        # Create training configuration
        training_config = TrainingConfig(
            name="test_training", batch_size=4, num_epochs=1, optimizer=optimizer_config
        )

        config = {
            "point_cloud_generation": {
                "dataset_path": "data/test_shapenet",
                "dataset_config": dataset_config,
                "model_config": model_config,
                "eval_config": eval_config,
                "training_config": training_config,
            }
        }

        suite = GeometricBenchmarkSuite(config=config, rngs=rngs)

        assert hasattr(suite, "benchmarks")
        assert "point_cloud_generation" in suite.benchmarks

    def test_suite_benchmark_execution(self, rngs):
        """Test running all benchmarks in suite with properly typed configurations."""
        from artifex.generative_models.core.configuration import (
            DataConfig,
            EvaluationConfig,
            ModelConfig,
            OptimizerConfig,
            TrainingConfig,
        )

        # Create proper typed configurations
        dataset_config = DataConfig(
            name="test_shapenet",
            dataset_name="shapenet",
            metadata={"num_points": 128, "mock_samples": 20},
        )

        model_config = ModelConfig(
            name="test_pointcloud_model",
            model_class="artifex.generative_models.models.geometric.PointCloudModel",
            input_dim=(128, 3),
            hidden_dims=[32],
            output_dim=(128, 3),
            activation="relu",
            metadata={"geometric_params": {"embed_dim": 32, "num_layers": 1}},
        )

        eval_config = EvaluationConfig(
            name="test_geometric_eval",
            metrics=["1nn_accuracy", "coverage", "chamfer_distance"],
            metric_params={
                "1nn_accuracy": {"k": 1},
                "coverage": {"threshold": 0.01},
                "chamfer_distance": {"reduction": "mean"},
            },
            eval_batch_size=2,
        )

        # Create optimizer configuration
        optimizer_config = OptimizerConfig(
            name="test_optimizer", optimizer_type="adam", learning_rate=1e-4
        )

        # Create training configuration
        training_config = TrainingConfig(
            name="test_training", batch_size=2, num_epochs=1, optimizer=optimizer_config
        )

        config = {
            "point_cloud_generation": {
                "dataset_path": "data/test_shapenet",
                "dataset_config": dataset_config,
                "model_config": model_config,
                "eval_config": eval_config,
                "training_config": training_config,
            }
        }

        suite = GeometricBenchmarkSuite(config=config, rngs=rngs)

        results = suite.run_all_benchmarks()

        assert "point_cloud_generation" in results
        benchmark_result = results["point_cloud_generation"]

        assert "training" in benchmark_result
        assert "evaluation" in benchmark_result
        assert "validation" in benchmark_result
        assert "info" in benchmark_result

    def test_suite_summary_generation(self, rngs):
        """Test suite summary generation."""
        # Mock results for testing
        mock_results = {
            "point_cloud_generation": {
                "training": {"final_loss": 0.1},
                "evaluation": {"1nn_accuracy": 0.9, "coverage": 0.8},
                "validation": {
                    "1nn_accuracy_target": True,
                    "coverage_target": True,
                    "training_time_target": False,
                },
                "info": {"name": "Point Cloud Generation"},
            }
        }

        from artifex.generative_models.core.configuration import EvaluationConfig

        # Create proper evaluation configuration
        eval_config = EvaluationConfig(
            name="test_point_cloud_metrics",
            metrics=["1nn_accuracy", "coverage", "chamfer_distance"],
            metric_params={
                "1nn_accuracy": {"k": 1},
                "coverage": {"threshold": 0.01},
                "chamfer_distance": {"reduction": "mean"},
            },
            eval_batch_size=32,
        )

        config = {"point_cloud_generation": {"eval_config": eval_config}}
        suite = GeometricBenchmarkSuite(config=config, rngs=rngs)

        summary = suite.get_suite_summary(mock_results)

        assert "total_benchmarks" in summary
        assert "completed_benchmarks" in summary
        assert "benchmark_results" in summary
        assert "overall_success_rate" in summary

        # Check individual benchmark summary
        pc_summary = summary["benchmark_results"]["point_cloud_generation"]
        assert "targets_met" in pc_summary
        assert "total_targets" in pc_summary
        assert "success_rate" in pc_summary


class TestGeometricDatasetRegistry:
    """Tests for geometric dataset registry."""

    def test_registry_listing(self):
        """Test listing available datasets."""
        datasets = GeometricDatasetRegistry.list_datasets()

        assert "shapenet" in datasets
        assert isinstance(datasets, list)

    def test_dataset_retrieval(self, rngs):
        """Test retrieving dataset by name."""
        from artifex.generative_models.core.configuration import DataConfig

        config = DataConfig(
            name="test_shapenet",
            dataset_name="shapenet",
            metadata={"num_points": 256, "mock_samples": 10},
        )

        dataset = GeometricDatasetRegistry.get_dataset(
            name="shapenet",
            data_path="data/test",
            config=config,
            rngs=rngs,
        )

        assert isinstance(dataset, ShapeNetDataset)

    def test_invalid_dataset_name(self, rngs):
        """Test error handling for invalid dataset name."""
        from artifex.generative_models.core.configuration import DataConfig

        config = DataConfig(name="test_invalid", dataset_name="invalid", metadata={})

        with pytest.raises(ValueError, match="Dataset 'invalid' not registered"):
            GeometricDatasetRegistry.get_dataset(
                name="invalid",
                data_path="data/test",
                config=config,
                rngs=rngs,
            )

    def test_registry_rejects_dict_config(self, rngs):
        """Test that GeometricDatasetRegistry rejects dict configs."""
        dict_config = {"num_points": 256, "mock_samples": 10}

        with pytest.raises(TypeError, match="config must be DataConfig"):
            GeometricDatasetRegistry.get_dataset(
                name="shapenet",
                data_path="data/test",
                config=dict_config,
                rngs=rngs,
            )

    def test_dataset_registration(self, rngs):
        """Test registering new dataset class."""

        class MockDataset:
            def __init__(self, data_path, config, *, rngs):
                pass

        GeometricDatasetRegistry.register_dataset("mock", MockDataset)

        datasets = GeometricDatasetRegistry.list_datasets()
        assert "mock" in datasets
