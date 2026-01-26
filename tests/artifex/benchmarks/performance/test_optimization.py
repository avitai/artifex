"""Tests for optimization benchmarks."""

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from artifex.benchmarks.base import DatasetProtocol
from artifex.benchmarks.performance.optimization import (
    OptimizationBenchmark,
    OptimizationMetricsConfig,
    OptimizerComparisonBenchmark,
    TrainerProtocol,
    TrainingConvergenceBenchmark,
)


class SimpleTrainerForTest:
    """Simple trainer for testing optimization benchmarks."""

    def __init__(
        self,
        optimizer_config: dict[str, Any] | None = None,
        model_name: str = "test_model",
        target_loss: float = 0.01,
        loss_decay_rate: float = 0.95,
    ):
        """Initialize the trainer.

        Args:
            optimizer_config: Configuration for the optimizer.
            model_name: Name of the model.
            target_loss: Target loss to converge to.
            loss_decay_rate: Rate at which loss decreases (smaller = faster).
        """
        self.optimizer_config = optimizer_config or {"learning_rate": 0.01}
        self.model_name = model_name
        self.target_loss = target_loss
        self.loss_decay_rate = loss_decay_rate
        self.eval_loss = 1.0

    def init(self, rng_key: jax.Array, *, rngs: nnx.Rngs | None = None) -> dict[str, Any]:
        """Initialize the model and optimizer.

        Args:
            rng_key: JAX random key.
            rngs: NNX Rngs object.

        Returns:
            The initialized training state.
        """
        del rng_key, rngs  # Unused
        return {
            "step": 0,
            "loss": 1.0,
            "learning_rate": self.optimizer_config.get("learning_rate", 0.01),
        }

    def train_step(
        self, state: dict[str, Any], batch: jax.Array, *, rngs: nnx.Rngs | None = None
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Perform a single training step.

        Args:
            state: Current training state.
            batch: Batch of training data.
            rngs: NNX Rngs object.

        Returns:
            Tuple of (new state, metrics dictionary).
        """
        del batch, rngs  # Unused

        # Simulate training with exponential decay towards target loss
        current_loss = state["loss"]
        decay_factor = self.loss_decay_rate

        # Apply learning rate to decay speed
        lr_factor = state["learning_rate"] * 10.0  # Scale for test purposes

        # Calculate new loss (decreasing exponentially towards target)
        new_loss = max(self.target_loss, current_loss * decay_factor**lr_factor)

        new_state = {
            "step": state["step"] + 1,
            "loss": new_loss,
            "learning_rate": state["learning_rate"],
        }

        metrics = {"loss": new_loss}
        return new_state, metrics

    def evaluate(
        self, state: dict[str, Any], dataset: DatasetProtocol, *, rngs: nnx.Rngs | None = None
    ) -> dict[str, float]:
        """Evaluate the model on a dataset.

        Args:
            state: Current training state.
            dataset: Dataset for evaluation.
            rngs: NNX Rngs object.

        Returns:
            Dictionary of evaluation metrics.
        """
        del dataset, rngs  # Unused

        # For testing purposes, we'll make eval loss slightly better than training
        # loss
        eval_loss = state["loss"] * 0.95
        self.eval_loss = eval_loss

        return {
            "loss": eval_loss,
            "accuracy": 1.0 - eval_loss,  # Simulated accuracy
        }


class SimpleDatasetForTest:
    """Simple dataset for testing optimization benchmarks."""

    def __init__(self, size: int = 100, dim: int = 10):
        """Initialize the dataset.

        Args:
            size: Number of examples in the dataset.
            dim: Dimension of each example.
        """
        # Create synthetic data for testing
        rng = np.random.RandomState(42)
        self.data = rng.randn(size, dim).astype(np.float32)

    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> jax.Array:
        """Get an example from the dataset."""
        return jnp.array(self.data[idx])


# Dataset without __getitem__ implementation
class InvalidDataset:
    def __init__(self, size: int = 10):
        self.size = size

    def __len__(self) -> int:
        return self.size


# Empty dataset
class EmptyDataset:
    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> jax.Array:
        raise IndexError("Dataset is empty")


# Dataset that returns non-tensor items
class NonTensorDataset:
    def __init__(self, size: int = 20):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> str:
        return f"item_{idx}"


@pytest.fixture
def optimizer_config():
    """Create a test optimizer configuration."""
    return {"learning_rate": 0.1, "name": "test_optimizer"}


@pytest.fixture
def trainer():
    """Create a test trainer."""
    return SimpleTrainerForTest(target_loss=0.01, loss_decay_rate=0.5)


@pytest.fixture
def dataset():
    """Create a test dataset."""
    return SimpleDatasetForTest(size=100, dim=10)


def test_optimization_benchmark_initialization():
    """Test initialization of the optimization benchmark."""
    benchmark = OptimizationBenchmark(
        batch_size=16,
        num_epochs=5,
        random_seed=42,
    )

    assert benchmark.config.name == "optimization"
    assert benchmark.batch_size == 16
    assert benchmark.num_epochs == 5
    assert benchmark.random_seed == 42


def test_optimization_benchmark_initialization_invalid_params():
    """Test that initialization with invalid parameters raises errors."""
    # Test negative batch size
    with pytest.raises(ValueError, match="Batch size must be positive"):
        OptimizationBenchmark(batch_size=-5)

    # Test zero batch size
    with pytest.raises(ValueError, match="Batch size must be positive"):
        OptimizationBenchmark(batch_size=0)

    # Test negative num_epochs
    with pytest.raises(ValueError, match="Number of epochs must be positive"):
        OptimizationBenchmark(num_epochs=-1)

    # Test zero num_epochs
    with pytest.raises(ValueError, match="Number of epochs must be positive"):
        OptimizationBenchmark(num_epochs=0)


def test_optimization_benchmark_run(trainer, dataset):
    """Test running the optimization benchmark."""
    benchmark = OptimizationBenchmark(
        batch_size=16,
        num_epochs=2,
        random_seed=42,
    )

    result = benchmark.run(trainer, dataset)

    assert result.benchmark_name == "optimization"
    assert result.model_name == "test_model"
    assert "iterations_to_convergence" in result.metrics
    assert "time_to_convergence" in result.metrics
    assert "final_loss" in result.metrics
    assert "training_throughput" in result.metrics
    assert "loss" in result.metrics
    assert "accuracy" in result.metrics

    # Verify training curve in metadata
    assert "training_curve" in result.metadata
    assert len(result.metadata["training_curve"]) > 0
    assert "iteration" in result.metadata["training_curve"][0]
    assert "metrics" in result.metadata["training_curve"][0]


def test_optimization_benchmark_invalid_dataset(trainer):
    """Test that running with invalid datasets raises errors."""
    benchmark = OptimizationBenchmark(
        batch_size=16,
        num_epochs=2,
        random_seed=42,
    )

    # Test with no dataset
    with pytest.raises(ValueError, match="Dataset cannot be None"):
        benchmark.run(trainer, None)

    # Test with dataset that doesn't implement the protocol
    with pytest.raises(TypeError, match="Dataset must have a __getitem__ method"):
        benchmark.run(trainer, InvalidDataset())

    # Test with empty dataset
    with pytest.raises(ValueError, match="Dataset cannot be empty"):
        benchmark.run(trainer, EmptyDataset())

    # Test with batch size larger than dataset
    small_dataset = SimpleDatasetForTest(size=5)
    with pytest.raises(ValueError, match="Batch size .* is larger than dataset size"):
        benchmark.run(trainer, small_dataset)

    # Test with dataset that returns non-tensor items
    with pytest.raises(ValueError, match="Error creating batch.*"):
        benchmark.run(trainer, NonTensorDataset())


def test_optimization_benchmark_invalid_model(dataset):
    """Test that running with invalid model raises TypeErrors."""
    benchmark = OptimizationBenchmark(
        batch_size=16,
        num_epochs=2,
        random_seed=42,
    )

    # Test with non-protocol implementing model
    class InvalidModel:
        pass

    with pytest.raises(TypeError, match="must implement TrainerProtocol"):
        benchmark.run(InvalidModel(), dataset)


def test_training_convergence_benchmark(dataset):
    """Test the training convergence benchmark."""
    # Create a trainer with very aggressive loss decay specifically for this test
    fast_trainer = SimpleTrainerForTest(target_loss=0.01, loss_decay_rate=0.1)

    benchmark = TrainingConvergenceBenchmark(
        target_loss=0.1,
        max_iterations=100,
        batch_size=16,
        num_epochs=2,
        random_seed=42,
    )

    result = benchmark.run(fast_trainer, dataset)

    assert result.benchmark_name == "training_convergence"
    assert "iterations_to_convergence" in result.metrics
    assert result.metrics["iterations_to_convergence"] > 0
    assert result.metrics["final_loss"] <= 0.1  # Should reach target loss


def test_training_convergence_invalid_params():
    """Test that initialization with invalid parameters raises errors."""
    # Test non-numeric target loss
    with pytest.raises(ValueError, match="Target loss must be a number"):
        TrainingConvergenceBenchmark(target_loss="not a number")

    # Test negative max_iterations
    with pytest.raises(ValueError, match="Maximum iterations must be positive"):
        TrainingConvergenceBenchmark(target_loss=0.1, max_iterations=-100)

    # Test zero max_iterations
    with pytest.raises(ValueError, match="Maximum iterations must be positive"):
        TrainingConvergenceBenchmark(target_loss=0.1, max_iterations=0)

    # Test negative batch size
    with pytest.raises(ValueError, match="Batch size must be positive"):
        TrainingConvergenceBenchmark(target_loss=0.1, batch_size=-5)

    # Test negative num_epochs
    with pytest.raises(ValueError, match="Number of epochs must be positive"):
        TrainingConvergenceBenchmark(target_loss=0.1, num_epochs=-1)


def test_optimizer_comparison_benchmark(dataset, optimizer_config):
    """Test the optimizer comparison benchmark."""

    # Create a function to build trainers with different optimizers
    def trainer_factory(config):
        """Create a trainer with the given optimizer config."""
        model_name = f"model_with_{config.get('name', 'unknown')}"
        return SimpleTrainerForTest(optimizer_config=config, model_name=model_name)

    # Create multiple optimizer configs to compare
    optimizer_configs = [
        {"learning_rate": 0.01, "name": "slow"},
        {"learning_rate": 0.1, "name": "medium"},
        {"learning_rate": 0.5, "name": "fast"},
    ]

    benchmark = OptimizerComparisonBenchmark(
        optimizer_configs=optimizer_configs,
        trainer_factory=trainer_factory,
        batch_size=16,
        num_epochs=2,
        random_seed=42,
    )

    result = benchmark.run(None, dataset)

    assert result.benchmark_name == "optimizer_comparison"
    assert "best_optimizer" in result.metrics
    # Should be the fastest optimizer (index 2)
    assert result.metrics["best_optimizer"] == 2

    # Check that the metadata contains information about all optimizers
    assert "optimizer_configs" in result.metadata
    assert len(result.metadata["optimizer_configs"]) == 3
    assert "individual_results" in result.metadata
    assert len(result.metadata["individual_results"]) == 3


def test_optimizer_comparison_invalid_params():
    """Test that initialization with invalid parameters raises errors."""

    def trainer_factory(config):
        return SimpleTrainerForTest(optimizer_config=config)

    # Test empty optimizer_configs
    with pytest.raises(ValueError, match="optimizer_configs cannot be empty"):
        OptimizerComparisonBenchmark(optimizer_configs=[], trainer_factory=trainer_factory)

    # Test negative batch size
    with pytest.raises(ValueError, match="Batch size must be positive"):
        OptimizerComparisonBenchmark(
            optimizer_configs=[{"lr": 0.1}], trainer_factory=trainer_factory, batch_size=-5
        )

    # Test zero num_epochs
    with pytest.raises(ValueError, match="Number of epochs must be positive"):
        OptimizerComparisonBenchmark(
            optimizer_configs=[{"lr": 0.1}], trainer_factory=trainer_factory, num_epochs=0
        )


def test_optimizer_comparison_invalid_factory(dataset):
    """Test that an invalid trainer factory raises errors."""

    # Create a factory that returns invalid trainers
    def invalid_trainer_factory(config):
        return object()  # Not a TrainerProtocol

    benchmark = OptimizerComparisonBenchmark(
        optimizer_configs=[{"lr": 0.1}],
        trainer_factory=invalid_trainer_factory,
        batch_size=16,
        num_epochs=2,
    )

    with pytest.raises(TypeError, match="does not implement TrainerProtocol"):
        benchmark.run(None, dataset)

    # Test factory that raises exceptions
    def failing_factory(config):
        raise ValueError("Factory error")

    benchmark = OptimizerComparisonBenchmark(
        optimizer_configs=[{"lr": 0.1}],
        trainer_factory=failing_factory,
        batch_size=16,
        num_epochs=2,
    )

    with pytest.raises(ValueError, match="Error creating trainer for optimizer config"):
        benchmark.run(None, dataset)


def test_early_stopping(trainer, dataset):
    """Test that early stopping works correctly."""
    # Set up a benchmark with aggressive early stopping
    metrics_config = OptimizationMetricsConfig(
        early_stopping_patience=1,
        early_stopping_min_delta=0.001,
    )

    benchmark = OptimizationBenchmark(
        metrics_config=metrics_config,
        batch_size=16,
        num_epochs=10,  # More epochs than should be needed
        random_seed=42,
    )

    result = benchmark.run(trainer, dataset)

    # Verify that early stopping occurred
    max_iter = benchmark.metrics_config.max_iterations
    assert result.metadata["total_iterations"] < max_iter
    # Early stopping might finish all epochs but fewer iterations
    assert result.metadata["total_iterations"] < (
        benchmark.num_epochs * len(dataset) // benchmark.batch_size
    )


def test_error_handling(dataset):
    """Test that error handling works correctly."""

    # Create a trainer that raises an error during initialization
    class ErrorInInitTrainer:
        def init(self, rng_key, *, rngs=None):
            raise ValueError("Test error in init")

        def train_step(self, state, batch, *, rngs=None):
            return state, {}

        def evaluate(self, state, dataset, *, rngs=None):
            return {"loss": 0.0}

    # Create a trainer that raises an error during training
    class ErrorInTrainStepTrainer:
        def __init__(self):
            self.model_name = "error_trainer"

        def init(self, rng_key, *, rngs=None):
            return {}

        def train_step(self, state, batch, *, rngs=None):
            raise ValueError("Test error in train_step")

        def evaluate(self, state, dataset, *, rngs=None):
            return {"loss": 0.0}

    # Create a trainer that raises an error during evaluation
    class ErrorInEvaluateTrainer:
        def __init__(self):
            self.model_name = "error_trainer"

        def init(self, rng_key, *, rngs=None):
            return {}

        def train_step(self, state, batch, *, rngs=None):
            return state, {"loss": 0.1}

        def evaluate(self, state, dataset, *, rngs=None):
            raise ValueError("Test error in evaluate")

    # Create a trainer that returns metrics without the loss key
    class MissingLossTrainer:
        def __init__(self):
            self.model_name = "missing_loss_trainer"

        def init(self, rng_key, *, rngs=None):
            return {}

        def train_step(self, state, batch, *, rngs=None):
            return state, {"accuracy": 0.9}

        def evaluate(self, state, dataset, *, rngs=None):
            return {"accuracy": 0.9}  # No loss metric

    # Test initialization error
    benchmark = OptimizationBenchmark(random_seed=42)
    with pytest.raises(ValueError, match="Error initializing trainer"):
        benchmark.run(ErrorInInitTrainer(), dataset)

    # Test train_step error
    benchmark = OptimizationBenchmark(random_seed=42)
    with pytest.raises(ValueError, match="Error during training step"):
        benchmark.run(ErrorInTrainStepTrainer(), dataset)

    # Test evaluate error
    benchmark = OptimizationBenchmark(random_seed=42)
    with pytest.raises(ValueError, match="Error during evaluation"):
        benchmark.run(ErrorInEvaluateTrainer(), dataset)

    # Test missing loss metric
    benchmark = OptimizationBenchmark(random_seed=42)
    error_match = "Loss metric .* not found in evaluation metrics"
    with pytest.raises(ValueError, match=error_match):
        benchmark.run(MissingLossTrainer(), dataset)


def test_rng_handling(dataset):
    """Test that RNG handling is correctly implemented with NNX Rngs."""

    # Create a trainer that verifies the RNG types
    class RngValidatingTrainer:
        def __init__(self):
            self.model_name = "rng_validator"
            self.init_called = False
            self.train_step_called = False
            self.evaluate_called = False

        def init(self, rng_key, *, rngs):
            self.init_called = True
            # Verify rng_key is a JAX PRNGKey
            assert isinstance(rng_key, jax.Array), "rng_key should be a JAX array"
            assert rng_key.shape == (2,), "rng_key should be a JAX PRNGKey with shape (2,)"

            # Verify rngs is an NNX Rngs object
            assert isinstance(rngs, nnx.Rngs), "rngs should be an nnx.Rngs object"
            assert "params" in rngs, "rngs should contain 'params' key"

            return {"state": "initialized"}

        def train_step(self, state, batch, *, rngs):
            self.train_step_called = True
            # Verify state is passed correctly
            assert state == {"state": "initialized"}, "State not passed correctly"

            # Verify rngs is an NNX Rngs object
            assert isinstance(rngs, nnx.Rngs), "rngs should be an nnx.Rngs object"
            assert "dropout" in rngs, "rngs should contain 'dropout' key"

            return state, {"loss": 0.5}

        def evaluate(self, state, dataset, *, rngs):
            self.evaluate_called = True
            # Verify state is passed correctly
            assert state == {"state": "initialized"}, "State not passed correctly"

            # Verify rngs is an NNX Rngs object
            assert isinstance(rngs, nnx.Rngs), "rngs should be an nnx.Rngs object"
            assert "dropout" in rngs, "rngs should contain 'dropout' key"

            return {"loss": 0.5, "accuracy": 0.9}

    # Run benchmark with the validating trainer
    trainer = RngValidatingTrainer()
    benchmark = OptimizationBenchmark(
        batch_size=4,
        num_epochs=1,
        random_seed=42,
    )

    # Should run without assertion errors if RNG handling is correct
    result = benchmark.run(trainer, dataset)

    # Verify that all methods were called
    assert trainer.init_called, "init method was not called"
    assert trainer.train_step_called, "train_step method was not called"
    assert trainer.evaluate_called, "evaluate method was not called"

    # Basic verification of benchmark results
    assert "iterations_to_convergence" in result.metrics
    assert "final_loss" in result.metrics
    assert result.metrics["final_loss"] == 0.5


def test_none_rngs_handling(dataset):
    """Test that None rngs parameter is handled correctly."""

    # Create a trainer that handles None rngs
    class NoneRngsTrainer:
        def __init__(self):
            self.model_name = "none_rngs_trainer"

        def init(self, rng_key, *, rngs=None):
            # Should handle None rngs gracefully
            return {"state": "initialized"}

        def train_step(self, state, batch, *, rngs=None):
            # Should handle None rngs gracefully
            return state, {"loss": 0.5}

        def evaluate(self, state, dataset, *, rngs=None):
            # Should handle None rngs gracefully
            return {"loss": 0.5}

    # Explicitly set the trainer's type to be recognized as a TrainerProtocol
    trainer = NoneRngsTrainer()

    # This class doesn't work with runtime_checkable due to signature mismatch
    # So we'll test only with the class that follows the protocol exactly
    assert isinstance(trainer, TrainerProtocol)

    benchmark = OptimizationBenchmark(
        batch_size=4,
        num_epochs=1,
        random_seed=42,
    )

    # This would raise a TypeError if protocol implementation was incorrect
    result = benchmark.run(trainer, dataset)

    # Basic verification of benchmark results
    assert "final_loss" in result.metrics
    assert result.metrics["final_loss"] == 0.5
