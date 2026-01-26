"""Optimization benchmark for generative models.

This module provides benchmarks for evaluating the training performance and
optimization strategies for generative models, measuring convergence rates,
loss curves, and training efficiency.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from artifex.benchmarks.base import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
    DatasetProtocol,
)


@dataclass
class OptimizationMetricsConfig:
    """Configuration for optimization metrics.

    Attributes:
        metric_names: Names of the metrics to compute.
        loss_name: Name of the loss function to track.
        target_loss: Target loss value for convergence measurement.
        max_iterations: Maximum number of iterations to run.
        eval_frequency: Frequency of evaluation during training.
        early_stopping_patience: Number of evaluations without improvement.
        early_stopping_min_delta: Minimum change to qualify as improvement.
    """

    metric_names: list[str] = field(default_factory=lambda: ["loss"])
    loss_name: str = "loss"
    target_loss: float | None = None
    max_iterations: int = 1000
    eval_frequency: int = 10
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4


@runtime_checkable
class TrainerProtocol(Protocol):
    """Protocol defining the interface for trainers that can be benchmarked.

    The trainer must implement three methods:
    - init: Initialize the model and optimizer
    - train_step: Perform a single training step
    - evaluate: Evaluate the model on a dataset

    All methods must handle the NNX RNG objects correctly.
    """

    def init(self, rng_key: jax.Array, *, rngs: nnx.Rngs | None = None) -> Any:
        """Initialize the model and optimizer.

        Args:
            rng_key: JAX random key.
            rngs: NNX Rngs object.

        Returns:
            The initialized training state.

        Raises:
            TypeError: If RNG objects are not handled correctly.
            ValueError: If initialization fails due to invalid parameters.
        """
        ...

    def train_step(
        self, state: Any, batch: Any, *, rngs: nnx.Rngs | None = None
    ) -> tuple[Any, dict[str, float]]:
        """Perform a single training step.

        Args:
            state: Current training state.
            batch: Batch of training data.
            rngs: NNX Rngs object.

        Returns:
            tuple of (new state, metrics dictionary).

        Raises:
            TypeError: If RNG objects are not handled correctly.
            ValueError: If the training step fails due to invalid parameters.
        """
        ...

    def evaluate(
        self, state: Any, dataset: DatasetProtocol, *, rngs: nnx.Rngs | None = None
    ) -> dict[str, float]:
        """Evaluate the model on a dataset.

        Args:
            state: Current training state.
            dataset: Dataset for evaluation.
            rngs: NNX Rngs object.

        Returns:
            dictionary of evaluation metrics.

        Raises:
            TypeError: If RNG objects are not handled correctly.
            ValueError: If evaluation fails due to invalid parameters or dataset issues.
        """
        ...


@dataclass
class TrainingCurvePoint:
    """Point on a training curve.

    Attributes:
        iteration: Training iteration.
        timestamp: Time at which the metrics were recorded.
        metrics: dictionary of metric values.
    """

    iteration: int
    timestamp: float
    metrics: dict[str, float]


class OptimizationBenchmark(Benchmark):
    """Benchmark for evaluating training optimization strategies.

    This benchmark measures how quickly models converge during training, tracking loss
    curves, optimization efficiency, and computational performance.
    """

    def __init__(
        self,
        metrics_config: OptimizationMetricsConfig | None = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        random_seed: int | None = None,
    ) -> None:
        """Initialize the optimization benchmark.

        Args:
            metrics_config: Configuration for optimization metrics.
            batch_size: Batch size for training.
            num_epochs: Number of epochs to train for.
            random_seed: Random seed for training.

        Raises:
            ValueError: If batch_size or num_epochs is non-positive.
        """
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        if num_epochs <= 0:
            raise ValueError(f"Number of epochs must be positive, got {num_epochs}")

        if metrics_config is None:
            metrics_config = OptimizationMetricsConfig()

        config = BenchmarkConfig(
            name="optimization",
            description=("Training optimization performance for generative models"),
            metric_names=[
                "iterations_to_convergence",
                "time_to_convergence",
                "final_loss",
                "training_throughput",
                *metrics_config.metric_names,
            ],
        )
        super().__init__(config=config)

        self.metrics_config = metrics_config
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.random_seed = random_seed

    def run(self, model: Any, dataset: DatasetProtocol | None = None) -> BenchmarkResult:
        """Run the optimization benchmark.

        Args:
            model: Model trainer to benchmark.
            dataset: Dataset for training.

        Returns:
            Benchmark result with optimization metrics.

        Raises:
            TypeError: If model does not implement TrainerProtocol or if dataset
                is not a DatasetProtocol.
            ValueError: If dataset is None, empty, or if there are issues with
                the model initialization, training, or evaluation steps.
            RuntimeError: If the benchmark fails due to unexpected issues during execution.
        """
        # Check that model implements TrainerProtocol
        if not isinstance(model, TrainerProtocol):
            raise TypeError(
                f"Model must implement TrainerProtocol for optimization benchmarks, "
                f"got {type(model).__name__}. Ensure your model implements all "
                f"required methods with correct signatures."
            )

        # Validate dataset
        if dataset is None:
            raise ValueError(
                "Dataset cannot be None for optimization benchmarks. "
                "Please provide a valid dataset."
            )

        try:
            dataset_len = len(dataset)
        except (TypeError, AttributeError) as e:
            raise TypeError(
                f"Dataset must have a __len__ method, got {type(dataset).__name__}"
            ) from e

        if dataset_len == 0:
            raise ValueError("Dataset cannot be empty for optimization benchmarks")

        try:
            # Check if dataset has __getitem__ method
            _ = dataset[0]
        except (TypeError, AttributeError) as e:
            raise TypeError(
                f"Dataset must have a __getitem__ method, got {type(dataset).__name__}"
            ) from e
        except IndexError:
            # This is fine, it means the dataset is valid but empty (which we checked above)
            pass

        # Dataset is already validated above

        # Create RNG key
        if self.random_seed is not None:
            key = jax.random.PRNGKey(self.random_seed)
        else:
            key = jax.random.PRNGKey(0)

        # Create rngs object for NNX models
        try:
            # Create NNX Rngs object directly
            rngs = nnx.Rngs(params=key)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Error creating NNX RNG object: {e}") from e

        # Initialize the model and optimizer
        try:
            state = model.init(key, rngs=rngs)
        except Exception as e:
            raise ValueError(
                f"Error initializing trainer: {e}. Ensure the trainer properly "
                "handles NNX RNG objects."
            ) from e

        # Create data loader for training
        # Note: In a real implementation, we would use a proper data loader
        # Here we're just using a simple implementation for the benchmark
        data_size = len(dataset)
        num_batches = data_size // self.batch_size

        if num_batches == 0:
            raise ValueError(
                f"Batch size ({self.batch_size}) is larger than dataset size "
                f"({data_size}). Please use a smaller batch size."
            )

        # Initialize training curve tracking
        training_curve: list[TrainingCurvePoint] = []
        start_time = time.time()
        best_loss = float("inf")
        patience_counter = 0

        # Track convergence metrics
        iterations_to_convergence = None
        time_to_convergence = None

        # Training loop
        iteration = 0
        for epoch in range(self.num_epochs):
            # Shuffle data indices
            epoch_key = jax.random.fold_in(key, epoch + (self.random_seed or 0))
            indices = jax.random.permutation(epoch_key, data_size)

            # Mini-batch training
            for batch_idx in range(num_batches):
                # Create batch
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, data_size)
                batch_indices = indices[start_idx:end_idx]

                try:
                    batch = jnp.stack([dataset[int(i)] for i in batch_indices])
                except Exception as e:
                    raise ValueError(
                        f"Error creating batch from dataset: {e}. Ensure the dataset "
                        "returns valid tensors that can be stacked."
                    ) from e

                # Training step
                step_key = jax.random.fold_in(key, iteration)
                step_rngs = nnx.Rngs(dropout=step_key)
                try:
                    state, metrics = model.train_step(state, batch, rngs=step_rngs)
                except Exception as e:
                    raise ValueError(
                        f"Error during training step: {e}. Ensure the trainer "
                        "correctly handles batches and RNG objects."
                    ) from e

                # Record metrics periodically
                if iteration % self.metrics_config.eval_frequency == 0:
                    # Evaluate on the full dataset
                    eval_key = jax.random.fold_in(key, 10000 + iteration)
                    eval_rngs = nnx.Rngs(dropout=eval_key)
                    try:
                        eval_metrics = model.evaluate(state, dataset, rngs=eval_rngs)
                    except Exception as e:
                        raise ValueError(
                            f"Error during evaluation: {e}. Ensure the evaluation "
                            "method correctly processes the dataset."
                        ) from e

                    # Record training curve point
                    timestamp = time.time() - start_time
                    point = TrainingCurvePoint(
                        iteration=iteration,
                        timestamp=timestamp,
                        metrics=eval_metrics.copy(),
                    )
                    training_curve.append(point)

                    # Verify that loss exists in the metrics
                    if self.metrics_config.loss_name not in eval_metrics:
                        available_metrics = ", ".join(sorted(eval_metrics.keys()))
                        raise ValueError(
                            f"Loss metric '{self.metrics_config.loss_name}' "
                            f"not found in evaluation metrics. Available metrics: "
                            f"{available_metrics}"
                        )

                    # Check for convergence to target loss
                    current_loss = eval_metrics[self.metrics_config.loss_name]
                    if self.metrics_config.target_loss is not None:
                        if current_loss <= self.metrics_config.target_loss:
                            if iterations_to_convergence is None:
                                iterations_to_convergence = iteration
                                time_to_convergence = timestamp
                                break

                    # Early stopping check
                    delta = self.metrics_config.early_stopping_min_delta
                    if current_loss < best_loss - delta:
                        best_loss = current_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= self.metrics_config.early_stopping_patience:
                        break

                iteration += 1
                if iteration >= self.metrics_config.max_iterations:
                    break

            # Break if we've reached the target or max iterations
            target_reached = iterations_to_convergence is not None
            max_iter_reached = iteration >= self.metrics_config.max_iterations
            if target_reached or max_iter_reached:
                break

        # Calculate final metrics
        end_time = time.time()
        total_time = end_time - start_time

        # Get final evaluation
        final_eval_key = jax.random.fold_in(key, 20000)
        final_eval_rngs = nnx.Rngs(dropout=final_eval_key)
        try:
            final_metrics = model.evaluate(state, dataset, rngs=final_eval_rngs)
        except Exception as e:
            raise ValueError(
                f"Error during final evaluation: {e}. Unable to compute final metrics."
            ) from e

        # Verify that loss exists in the final metrics
        if self.metrics_config.loss_name not in final_metrics:
            available_metrics = ", ".join(sorted(final_metrics.keys()))
            raise ValueError(
                f"Loss metric '{self.metrics_config.loss_name}' not found in final "
                f"evaluation metrics. Available metrics: {available_metrics}"
            )

        final_loss = final_metrics[self.metrics_config.loss_name]

        # If we didn't reach the target, record iterations as max
        if iterations_to_convergence is None:
            iterations_to_convergence = iteration

        # If we didn't reach the target, record time as total time
        if time_to_convergence is None:
            time_to_convergence = total_time

        # Calculate training throughput (examples/second)
        examples_processed = iteration * self.batch_size
        training_throughput = examples_processed / total_time if total_time > 0 else 0

        # Create result metrics
        metrics = {
            "iterations_to_convergence": float(iterations_to_convergence),
            "time_to_convergence": time_to_convergence,
            "final_loss": final_loss,
            "training_throughput": training_throughput,
        }

        # Add any additional metrics from final evaluation
        for name in self.metrics_config.metric_names:
            if name in final_metrics:
                metrics[name] = final_metrics[name]

        # Add all metrics from final evaluation
        for name, value in final_metrics.items():
            if name not in metrics:
                metrics[name] = value

        # Create metadata with training curve
        metadata = {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "training_curve": [
                {
                    "iteration": point.iteration,
                    "timestamp": point.timestamp,
                    "metrics": point.metrics,
                }
                for point in training_curve
            ],
            "total_iterations": iteration,
            "total_time": total_time,
            "examples_processed": examples_processed,
            "total_epochs": epoch + 1,
        }

        # Create result
        result = BenchmarkResult(
            benchmark_name=self.config.name,
            model_name=getattr(model, "model_name", "unknown"),
            metrics=metrics,
            metadata=metadata,
        )

        return result


class TrainingConvergenceBenchmark(OptimizationBenchmark):
    """Benchmark focusing on training convergence speed.

    This benchmark measures how quickly models converge to a target loss value,
    which is useful for comparing optimization algorithms and learning rate schedules.
    """

    def __init__(
        self,
        target_loss: float,
        max_iterations: int = 10000,
        metrics_config: OptimizationMetricsConfig | None = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        random_seed: int | None = None,
    ) -> None:
        """Initialize the training convergence benchmark.

        Args:
            target_loss: Target loss value to measure convergence to.
            max_iterations: Maximum number of iterations to run.
            metrics_config: Configuration for optimization metrics.
            batch_size: Batch size for training.
            num_epochs: Number of epochs to train for.
            random_seed: Random seed for training.

        Raises:
            ValueError: If target_loss is not a valid float or if max_iterations is non-positive.
        """
        if not isinstance(target_loss, (int, float)):
            raise ValueError(f"Target loss must be a number, got {type(target_loss).__name__}")

        if max_iterations <= 0:
            raise ValueError(f"Maximum iterations must be positive, got {max_iterations}")

        if metrics_config is None:
            metrics_config = OptimizationMetricsConfig(
                target_loss=target_loss,
                max_iterations=max_iterations,
            )
        else:
            metrics_config.target_loss = target_loss
            metrics_config.max_iterations = max_iterations

        super().__init__(
            metrics_config=metrics_config,
            batch_size=batch_size,
            num_epochs=num_epochs,
            random_seed=random_seed,
        )

        # Override the benchmark name
        self.config.name = "training_convergence"
        self.config.description = "Training convergence speed for generative models"


class OptimizerComparisonBenchmark(Benchmark):
    """Benchmark for comparing multiple optimizers on the same model and dataset.

    This benchmark runs multiple optimization benchmarks with different optimizer
    configurations and compares their performance.
    """

    def __init__(
        self,
        optimizer_configs: list[dict[str, Any]],
        trainer_factory: Callable[[dict[str, Any]], TrainerProtocol],
        metrics_config: OptimizationMetricsConfig | None = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        random_seed: int | None = None,
    ) -> None:
        """Initialize the optimizer comparison benchmark.

        Args:
            optimizer_configs: List of optimizer configurations to compare.
            trainer_factory: Function that creates a trainer from a config.
            metrics_config: Configuration for optimization metrics.
            batch_size: Batch size for training.
            num_epochs: Number of epochs to train for.
            random_seed: Random seed for training.

        Raises:
            ValueError: If optimizer_configs is empty or if batch_size or num_epochs is
                non-positive.
        """
        if not optimizer_configs:
            raise ValueError("optimizer_configs cannot be empty")

        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        if num_epochs <= 0:
            raise ValueError(f"Number of epochs must be positive, got {num_epochs}")

        config = BenchmarkConfig(
            name="optimizer_comparison",
            description=("Comparison of optimizer performance for generative models"),
            metric_names=[
                "best_optimizer",
                "iterations_to_convergence",
                "time_to_convergence",
                "final_loss",
                "training_throughput",
            ],
        )
        super().__init__(config=config)

        self.optimizer_configs = optimizer_configs
        self.trainer_factory = trainer_factory
        self.metrics_config = metrics_config or OptimizationMetricsConfig()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.random_seed = random_seed

    def run(self, model: Any, dataset: DatasetProtocol | None = None) -> BenchmarkResult:
        """Run the optimizer comparison benchmark.

        Args:
            model: Model to benchmark (unused - trainers are created from factory).
            dataset: Dataset for training.

        Returns:
            Benchmark result with comparison of optimizers.

        Raises:
            ValueError: If dataset is None or if there are issues creating or running trainers.
            TypeError: If any trainer created does not implement TrainerProtocol.
        """
        if dataset is None:
            raise ValueError("Dataset is required for optimizer comparison benchmark")

        # Verify dataset implements the DatasetProtocol interface
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise TypeError("Dataset must implement __len__ and __getitem__ methods")

        # Verify dataset has at least one item
        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty")

        # Run optimization benchmark for each optimizer
        results = []
        for i, optimizer_config in enumerate(self.optimizer_configs):
            # Create trainer for this optimizer
            try:
                trainer = self.trainer_factory(optimizer_config)
            except Exception as e:
                raise ValueError(f"Error creating trainer for optimizer config {i}: {e}") from e

            # Verify that the trainer implements TrainerProtocol
            if not isinstance(trainer, TrainerProtocol):
                raise TypeError(
                    f"Trainer created for optimizer config {i} does not implement "
                    f"TrainerProtocol, got {type(trainer).__name__}"
                )

            # Create a separate random seed for each optimizer
            seed = None
            if self.random_seed is not None:
                seed = self.random_seed + i * 1000

            # Create and run optimization benchmark
            benchmark = OptimizationBenchmark(
                metrics_config=self.metrics_config,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                random_seed=seed,
            )

            # Run benchmark for this optimizer
            try:
                result = benchmark.run(trainer, dataset)
            except Exception as e:
                raise ValueError(f"Error running benchmark for optimizer config {i}: {e}") from e

            # Add optimizer configuration to metadata
            result.metadata["optimizer_config"] = optimizer_config.copy()
            optimizer_name = optimizer_config.get("name", f"optimizer_{i}")
            result.metadata["optimizer_name"] = optimizer_name

            results.append(result)

        if not results:
            raise ValueError("No valid results obtained from any optimizer")

        # Find the best optimizer based on final loss
        try:
            best_result = min(results, key=lambda r: r.metrics["final_loss"])
            best_optimizer_idx = results.index(best_result)
            best_optimizer_config = self.optimizer_configs[best_optimizer_idx]
        except (KeyError, ValueError) as e:
            raise ValueError(
                "Error determining best optimizer. Ensure all results contain "
                f"'final_loss' metric: {e}"
            ) from e

        # Aggregate metrics
        metrics = {
            "best_optimizer": best_optimizer_idx,
            "iterations_to_convergence": best_result.metrics["iterations_to_convergence"],
            "time_to_convergence": best_result.metrics["time_to_convergence"],
            "final_loss": best_result.metrics["final_loss"],
            "training_throughput": best_result.metrics["training_throughput"],
        }

        # Create aggregated metadata
        metadata = {
            "optimizer_configs": self.optimizer_configs,
            "individual_results": [r.metrics for r in results],
            "best_optimizer_config": best_optimizer_config,
            "num_optimizers": len(self.optimizer_configs),
        }

        # Create result
        result = BenchmarkResult(
            benchmark_name=self.config.name,
            model_name=getattr(model, "model_name", "unknown"),
            metrics=metrics,
            metadata=metadata,
        )

        return result
