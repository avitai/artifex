"""Benchmark protocol definitions for artifex.generative_models.core."""

from abc import ABC, abstractmethod
from typing import Any, Protocol

import flax.nnx as nnx

from artifex.generative_models.core.configuration import EvaluationConfig


class BenchmarkBase(nnx.Module, ABC):
    """Abstract base class for all benchmarks following NNX patterns.

    This class defines the standard interface for benchmarks in the
    comprehensive generative models benchmark system. All benchmarks
    must exclusively support NNX models and follow proper RNG handling.

    Attributes:
        config: Benchmark configuration dictionary
        rngs: NNX Rngs for stochastic operations
    """

    _registry: dict[str, type["BenchmarkBase"]] = {}

    def __new__(cls, name: str | None = None, **kwargs: Any) -> "BenchmarkBase":
        """Create a new benchmark protocol instance.

        Args:
            name: Name of the protocol to create
            **kwargs: Additional arguments to pass to the protocol constructor

        Returns:
            An instance of the requested benchmark protocol
        """
        if name is not None and name in cls._registry:
            return cls._registry[name](**kwargs)
        return super().__new__(cls)

    @classmethod
    def register(cls, name: str, protocol_cls: type["BenchmarkBase"]) -> None:
        """Register a benchmark implementation.

        Args:
            name: Name to register the protocol under
            protocol_cls: Protocol class to register
        """
        cls._registry[name] = protocol_cls

    def __init__(self, config: EvaluationConfig, *, rngs: nnx.Rngs):
        """Initialize benchmark with configuration and RNG state.

        Args:
            config: Evaluation configuration including metrics targets
            rngs: NNX Rngs for all stochastic operations

        Raises:
            TypeError: If config is not EvaluationConfig
        """
        super().__init__()
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be EvaluationConfig, got {type(config).__name__}")
        self.config = config
        self.rngs = rngs
        self._setup_benchmark_components()

    @abstractmethod
    def _setup_benchmark_components(self) -> None:
        """Setup benchmark-specific components.

        Subclasses must implement this to initialize:
        - Model instances
        - Performance trackers
        - Dataset loaders
        - Any other benchmark-specific components
        """
        pass

    @abstractmethod
    def run_training(self) -> dict[str, float]:
        """Execute the training phase of the benchmark.

        Returns:
            dictionary containing training metrics like:
            - training_loss: Final training loss
            - training_time: Training duration in seconds
            - epochs: Number of training epochs
            - Any other training-specific metrics
        """
        pass

    @abstractmethod
    def run_evaluation(self) -> dict[str, float]:
        """Execute the evaluation phase of the benchmark.

        Returns:
            dictionary containing evaluation metrics like:
            - accuracy: Model accuracy
            - latency_ms: Inference latency in milliseconds
            - throughput: Samples per second
            - Any other evaluation-specific metrics
        """
        pass

    @abstractmethod
    def get_performance_targets(self) -> dict[str, float]:
        """Return performance targets for this benchmark.

        Returns:
            dictionary mapping metric names to target values
        """
        pass

    def validate_targets_achieved(self, metrics: dict[str, float]) -> bool:
        """Check if metrics meet performance targets.

        Args:
            metrics: Current metrics to validate

        Returns:
            True if all targets are met, False otherwise
        """
        targets = self.get_performance_targets()

        for metric_name, target_value in targets.items():
            if metric_name not in metrics:
                return False

            current_value = metrics[metric_name]

            # For latency metrics, lower is better
            if "latency" in metric_name.lower() or "time" in metric_name.lower():
                if current_value > target_value:
                    return False
            else:
                # For most metrics (accuracy, etc.), higher is better
                if current_value < target_value:
                    return False

        return True


class BenchmarkWithValidation(Protocol):
    """Protocol for benchmarks that support performance validation.

    This protocol defines the interface for benchmarks that can validate
    their performance against predefined targets and provide comprehensive
    benchmark information.
    """

    def validate_performance(self, results: dict[str, float | int]) -> dict[str, bool]:
        """Validate performance against targets.

        Args:
            results: Combined training and evaluation results

        Returns:
            dictionary indicating which targets were met
        """
        ...

    def get_benchmark_info(self) -> dict[str, Any]:
        """Get comprehensive benchmark information.

        Returns:
            dictionary with benchmark metadata and configuration
        """
        ...
