"""Metric protocol definitions for artifex.generative_models.core."""

from abc import ABC, abstractmethod

import flax.nnx as nnx

from artifex.generative_models.core.configuration import EvaluationConfig


class MetricBase(nnx.Module, ABC):
    """Abstract base class for all metrics following NNX patterns.

    This class defines the standard interface for metrics in the
    comprehensive generative models benchmark system. All metrics
    must follow proper RNG handling and JAX-compatible operations.

    Attributes:
        name: Metric name identifier
        modality: Target modality (image, text, audio, etc.)
        higher_is_better: Whether higher values indicate better performance
        config: Metric-specific configuration
        rngs: NNX Rngs for stochastic operations
    """

    def __init__(self, config: EvaluationConfig, *, rngs: nnx.Rngs):
        """Initialize metric protocol.

        Args:
            config: Evaluation configuration
            rngs: NNX Rngs for stochastic operations
        """
        if not isinstance(config, EvaluationConfig):
            # Also check for class name to handle dynamic module loading
            if type(config).__name__ != "EvaluationConfig":
                raise TypeError(f"config must be EvaluationConfig, got {type(config).__name__}")

        # Extract metric information from EvaluationConfig
        self.name = config.name
        self.config = config
        self.rngs = rngs

        # Extract metric-specific parameters from config.metric_params
        metric_params = config.metric_params or {}
        self.modality = metric_params.get("modality", "unknown")
        self.higher_is_better = metric_params.get("higher_is_better", True)

    @abstractmethod
    def compute(self, real_data, generated_data, **kwargs) -> dict[str, float]:
        """Compute metric values.

        Args:
            real_data: Ground truth data
            generated_data: Generated/predicted data
            **kwargs: Additional metric-specific parameters

        Returns:
            Dictionary of computed metric values
        """
        pass

    @abstractmethod
    def validate_inputs(self, real_data, generated_data) -> bool:
        """Validate input data compatibility.

        Args:
            real_data: Ground truth data
            generated_data: Generated/predicted data

        Returns:
            True if inputs are valid for this metric
        """
        pass
