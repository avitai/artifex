"""Geometric benchmarks suite for point cloud and 3D data generation."""

from typing import Any

import jax.numpy as jnp
from flax import nnx

from artifex.benchmarks.base import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
)
from artifex.benchmarks.datasets.geometric import ShapeNetDataset
from artifex.benchmarks.metrics.geometric import (
    PointCloudMetrics,
)
from artifex.generative_models.core.configuration import EvaluationConfig
from artifex.generative_models.core.protocols.evaluation import (
    BatchableDatasetProtocol,
    DatasetProtocol,
    ModelProtocol,
)
from artifex.generative_models.models.geometric.point_cloud import PointCloudModel


class PointCloudGenerationBenchmark(Benchmark):
    """Benchmark for point cloud generation using ShapeNet dataset.

    This benchmark evaluates point cloud generation models using metrics like
    1-NN accuracy, coverage, and geometric fidelity.

    Target Performance:
    - 1-NN accuracy >95%
    - Training time <4h/epoch on A100 GPU
    - Coverage >0.8
    - Geometric consistency validated
    """

    def __init__(self, *, rngs: nnx.Rngs, config):
        """Initialize point cloud generation benchmark.

        Args:
            config: Benchmark configuration (EvaluationConfig or dict containing eval_config)
            rngs: NNX Rngs for stochastic operations

        Raises:
            TypeError: If config is not EvaluationConfig or dict with eval_config
        """
        # Handle both EvaluationConfig and dict config patterns
        if isinstance(config, EvaluationConfig):
            self.eval_config = config
            self.original_config = None
        elif isinstance(config, dict):
            # Extract EvaluationConfig from dict config
            if "eval_config" in config:
                self.eval_config = config["eval_config"]
                if not isinstance(self.eval_config, EvaluationConfig):
                    raise TypeError(
                        f"eval_config must be EvaluationConfig, "
                        f"got {type(self.eval_config).__name__}"
                    )
                self.original_config = config
            else:
                raise TypeError(
                    "Dict config must contain 'eval_config' field with EvaluationConfig"
                )
        else:
            raise TypeError(f"config must be EvaluationConfig or dict, got {type(config).__name__}")

        # Create BenchmarkConfig for parent class
        benchmark_config = BenchmarkConfig(
            name="point_cloud_generation",
            description="Point cloud generation benchmark using ShapeNet dataset",
            metric_names=["1nn_accuracy", "coverage", "chamfer_distance", "geometric_fidelity"],
            metadata={"evaluation_config": self.eval_config},
        )
        super().__init__(config=benchmark_config)
        self.rngs = rngs
        self.performance_targets: dict[str, float] = {}
        self._setup_benchmark_components()

    def run(
        self,
        model: ModelProtocol,
        dataset: DatasetProtocol | BatchableDatasetProtocol | None = None,
    ) -> BenchmarkResult:
        """Run the point cloud generation benchmark.

        Args:
            model: Model to benchmark
            dataset: Optional dataset to use (will use default if None)

        Returns:
            BenchmarkResult with computed metrics
        """
        # Use provided dataset or default ShapeNet dataset
        if dataset is None:
            dataset = self.dataset

        # Run evaluation
        evaluation_results = self.run_evaluation(model, dataset)

        # Create and return benchmark result
        return BenchmarkResult(
            model_name=getattr(model, "name", "unknown_model"),
            benchmark_name=self.config.name,
            metrics=evaluation_results,
            metadata={
                "performance_targets": self.performance_targets,
                "model_config": getattr(model, "config", {}),
            },
        )

    def _setup_benchmark_components(self) -> None:
        """Setup benchmark-specific components."""
        from pathlib import Path

        from artifex.generative_models.core.configuration import DataConfig

        # Get dataset configuration - must be DataConfig
        dataset_config = getattr(self.eval_config, "dataset_config", None)
        if dataset_config is None:
            # Create default DataConfig
            dataset_config = DataConfig(
                name="shapenet_dataset",
                dataset_name="shapenet",
                data_dir=Path("./data"),
                split="train",
                num_workers=4,
                metadata={
                    "dataset_type": "geometric",
                    "batch_size": 32,
                    "shuffle": True,
                },
            )
        elif not isinstance(dataset_config, DataConfig):
            raise TypeError(
                f"dataset_config must be DataConfig, got {type(dataset_config).__name__}"
            )

        dataset_path = getattr(self.eval_config, "dataset_path", "data/shapenet")

        self.dataset = ShapeNetDataset(
            data_path=dataset_path,
            config=dataset_config,
            rngs=self.rngs,
        )

        # Initialize model
        if self.original_config:
            model_config = self.original_config.get(
                "model_config",
                {
                    "embed_dim": 256,
                    "num_points": 1024,
                    "num_layers": 6,
                    "num_heads": 8,
                    "dropout": 0.1,
                },
            )
        else:
            model_config = getattr(
                self.eval_config,
                "model_config",
                {
                    "embed_dim": 256,
                    "num_points": 1024,
                    "num_layers": 6,
                    "num_heads": 8,
                    "dropout": 0.1,
                },
            )

        # Create PointCloudConfig using dataclass defaults
        # The dataclass has sensible defaults, we only override num_points from dataset
        from artifex.generative_models.core.configuration import (
            PointCloudConfig,
            PointCloudNetworkConfig,
        )

        # Get num_points from dataset if available
        num_points = getattr(self.dataset, "num_points", PointCloudConfig.num_points)

        # Check if model_config is already a PointCloudConfig
        if isinstance(model_config, PointCloudConfig):
            # Use the provided config directly
            point_cloud_config = model_config
        else:
            # Create network config with dataclass defaults
            network_config = PointCloudNetworkConfig(
                name="point_cloud_benchmark_network",
                hidden_dims=(256, 256),  # Required field from BaseNetworkConfig
                activation="relu",  # Required field from BaseNetworkConfig
            )

            # Create PointCloudConfig with dataset's num_points
            point_cloud_config = PointCloudConfig(
                name="point_cloud_benchmark_model",
                network=network_config,
                num_points=num_points,
            )

        self.model = PointCloudModel(
            config=point_cloud_config,
            rngs=self.rngs,
        )

        # Initialize metrics with the evaluation configuration
        self.metrics = PointCloudMetrics(rngs=self.rngs, config=self.eval_config)

        # Performance targets
        if self.original_config:
            self.performance_targets = self.original_config.get(
                "performance_targets",
                {
                    "1nn_accuracy": 0.95,
                    "coverage": 0.8,
                    "training_time_per_epoch": 4.0,  # hours
                },
            )
        else:
            self.performance_targets = getattr(
                self.eval_config,
                "performance_targets",
                {
                    "1nn_accuracy": 0.95,
                    "coverage": 0.8,
                    "training_time_per_epoch": 4.0,  # hours
                },
            )

    def run_training(self) -> dict[str, float | int]:
        """Execute the training phase of the benchmark.

        Returns:
            Dictionary containing training metrics and performance measures
        """
        import time

        if self.original_config:
            training_config = self.original_config.get("training_config")
            # Handle both TrainingConfig objects and dicts for now
            if training_config is None:
                # Default values if no config provided
                num_epochs = 10
                batch_size = 32
            elif hasattr(training_config, "num_epochs"):
                # TrainingConfig object
                num_epochs = training_config.num_epochs
                batch_size = training_config.batch_size
            else:
                # Legacy dict format (should be eliminated eventually)
                num_epochs = training_config.get("num_epochs", 10)
                batch_size = training_config.get("batch_size", 32)
        else:
            # Default values
            num_epochs = 10
            batch_size = 32

        # Training loop (simplified for benchmark)
        training_metrics = []
        epoch_times = []

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Get training batch
            batch = self.dataset.get_batch(batch_size, split="train")

            # Simplified training step
            # In real implementation, this would include proper optimization
            model_output = self.model(batch["point_clouds"], deterministic=False)

            # Calculate basic training metrics
            loss = jnp.mean((model_output["positions"] - batch["point_clouds"]) ** 2)

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            training_metrics.append(
                {
                    "epoch": epoch,
                    "loss": float(loss),
                    "epoch_time_hours": epoch_time / 3600.0,
                }
            )

        # Calculate average training time
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
        avg_epoch_time_hours = avg_epoch_time / 3600.0

        return {
            "final_loss": float(training_metrics[-1]["loss"]) if training_metrics else 0.0,
            "avg_epoch_time_hours": avg_epoch_time_hours,
            "total_training_time_hours": sum(epoch_times) / 3600.0,
        }

    def run_evaluation(
        self,
        model: ModelProtocol | None = None,
        dataset: DatasetProtocol | BatchableDatasetProtocol | None = None,
    ) -> dict[str, float | int]:
        """Execute the evaluation phase of the benchmark.

        Args:
            model: Optional model to evaluate (uses internal model if None)
            dataset: Optional dataset to use (uses internal dataset if None)

        Returns:
            Dictionary containing evaluation metrics and performance measures
        """
        # Use provided model/dataset or default to internal instances
        if model is None:
            model = self.model
        if dataset is None:
            dataset = self.dataset

        # Get evaluation data
        eval_batch = dataset.get_batch(batch_size=self.eval_config.eval_batch_size, split="test")

        # Generate samples (using the existing point clouds as input for demo)
        generated_samples = model(eval_batch["point_clouds"], deterministic=True)

        # Calculate evaluation metrics
        metrics_results = self.metrics.compute_metrics(
            generated=generated_samples["positions"], real=eval_batch["point_clouds"]
        )

        return metrics_results

    def validate_performance(self, results: dict[str, float | int]) -> dict[str, bool]:
        """Validate performance against targets.

        Args:
            results: Combined training and evaluation results

        Returns:
            Dictionary indicating which targets were met
        """
        validation: dict[str, bool] = {}

        # Check 1-NN accuracy target
        if "1nn_accuracy" in results:
            validation["1nn_accuracy_target"] = (
                float(results["1nn_accuracy"]) >= self.performance_targets["1nn_accuracy"]
            )

        # Check coverage target
        if "coverage" in results:
            validation["coverage_target"] = (
                float(results["coverage"]) >= self.performance_targets["coverage"]
            )

        # Check training time target
        if "avg_epoch_time_hours" in results:
            validation["training_time_target"] = (
                float(results["avg_epoch_time_hours"])
                <= self.performance_targets["training_time_per_epoch"]
            )

        return validation

    def get_performance_targets(self) -> dict[str, float]:
        """Return performance targets for this benchmark.

        Returns:
            Dictionary mapping metric names to target values
        """
        return self.performance_targets.copy()

    def get_benchmark_info(self) -> dict[str, Any]:
        """Get comprehensive benchmark information.

        Returns:
            Dictionary with benchmark metadata and configuration
        """
        return {
            "name": "Point Cloud Generation",
            "dataset": "ShapeNet",
            "model": "PointCloudModel",
            "targets": self.performance_targets,
            "config": self.config,
            "description": "Evaluates point cloud generation quality and efficiency",
        }


class GeometricBenchmarkSuite:
    """Complete suite of geometric benchmarks.

    This suite includes all geometric generation benchmarks with proper
    orchestration and result aggregation.
    """

    def __init__(self, *, rngs: nnx.Rngs, config):
        """Initialize geometric benchmark suite.

        Args:
            config: Suite configuration as EvaluationConfig or dict.
                If dict is provided, it will be adapted to EvaluationConfig.
            rngs: NNX Rngs for all benchmarks
        """
        self.config = config
        self.rngs = rngs
        self.benchmarks = self._initialize_benchmarks()

    def _initialize_benchmarks(self) -> dict[str, PointCloudGenerationBenchmark]:
        """Initialize all benchmarks in the suite.

        NOTE: This method only accepts properly typed configurations.
        Dict configs are rejected in accordance with the unified configuration system.
        """
        benchmarks: dict[str, PointCloudGenerationBenchmark] = {}

        # Point Cloud Generation Benchmark
        if "point_cloud_generation" in self.config:
            pc_config = self.config["point_cloud_generation"]

            # Enforce unified configuration system - reject dict configs
            if isinstance(pc_config, dict) and "eval_config" not in pc_config:
                raise TypeError(
                    "GeometricBenchmarkSuite no longer accepts dict configurations. "
                    "Please use properly typed configurations (DataConfig, "
                    "ModelConfiguration, EvaluationConfig) instead. "
                    "Dict configs have been eliminated from the unified configuration system."
                )

            benchmarks["point_cloud_generation"] = PointCloudGenerationBenchmark(
                rngs=self.rngs,
                config=pc_config,
            )

        return benchmarks

    def run_all_benchmarks(self) -> dict[str, dict[str, Any]]:
        """Run all benchmarks in the suite.

        Returns:
            Dictionary with results from all benchmarks
        """
        results: dict[str, dict[str, Any]] = {}

        for benchmark_name, benchmark in self.benchmarks.items():
            print(f"Running {benchmark_name} benchmark...")

            # Run training
            training_results = benchmark.run_training()

            # Run evaluation
            evaluation_results = benchmark.run_evaluation()

            # Combine results
            combined_results = {**training_results, **evaluation_results}

            # Validate performance (all our benchmarks have this method)
            validation_results = benchmark.validate_performance(combined_results)

            # Get benchmark info
            benchmark_info = benchmark.get_benchmark_info()

            # Store complete results
            results[benchmark_name] = {
                "training": training_results,
                "evaluation": evaluation_results,
                "validation": validation_results,
                "info": benchmark_info,
            }

            print(f"Completed {benchmark_name} benchmark")

        return results

    def get_suite_summary(self, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Generate summary of suite performance.

        Args:
            results: Results from run_all_benchmarks()

        Returns:
            Dictionary with suite-level summary
        """
        summary: dict[str, Any] = {
            "total_benchmarks": len(self.benchmarks),
            "completed_benchmarks": len(results),
            "benchmark_results": {},
        }

        # Process each benchmark's results
        for benchmark_name, benchmark_results in results.items():
            validation = benchmark_results.get("validation", {})
            if isinstance(validation, dict):
                targets_met = sum(1 for v in validation.values() if v)
                total_targets = len(validation)
            else:
                targets_met = 0
                total_targets = 0

            evaluation_results = benchmark_results.get("evaluation", {})
            key_metrics = {}
            if isinstance(evaluation_results, dict):
                key_metrics = {
                    k: v
                    for k, v in evaluation_results.items()
                    if k in ["1nn_accuracy", "coverage", "geometric_fidelity"]
                }

            summary["benchmark_results"][benchmark_name] = {
                "targets_met": targets_met,
                "total_targets": total_targets,
                "success_rate": targets_met / total_targets if total_targets > 0 else 0.0,
                "key_metrics": key_metrics,
            }

        # Calculate overall success rate
        benchmark_results = summary["benchmark_results"]
        if isinstance(benchmark_results, dict):
            total_targets_met = sum(br.get("targets_met", 0) for br in benchmark_results.values())
            total_targets_all = sum(br.get("total_targets", 0) for br in benchmark_results.values())
        else:
            total_targets_met = 0
            total_targets_all = 0

        summary["overall_success_rate"] = (
            total_targets_met / total_targets_all if total_targets_all > 0 else 0.0
        )

        return summary
