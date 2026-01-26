"""Base protocols and abstract classes for modality components.

This module defines the base protocols, abstract classes, and common patterns
for modality components that adapt generic model architectures to work with
specific types of data.
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel
from artifex.generative_models.core.protocols.configuration import BaseModalityConfig
from artifex.generative_models.extensions.base import ModelExtension


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol defining interface for model adapters.

    Model adapters adapt generic model classes to work with specific
    data modalities by adding extensions and modifying behavior.
    """

    def create(self, config: Any, *, rngs: nnx.Rngs, **kwargs: Any) -> GenerativeModel:
        """Create a model with modality-specific adaptations.

        Args:
            config: Model configuration (must be ModelConfiguration).
            rngs: Random number generator keys.
            **kwargs: Additional keyword arguments for model creation.

        Returns:
            An initialized model instance.
        """
        ...


@runtime_checkable
class Modality(Protocol):
    """Protocol defining interface for data modalities.

    Modalities represent domain-specific data types and contain the
    logic needed to adapt generic models to work with that data type.
    """

    name: str

    def get_extensions(self, config: Any, *, rngs: nnx.Rngs) -> dict[str, ModelExtension]:
        """Get modality-specific extensions.

        Args:
            config: Extension configuration (must be a typed configuration).
            rngs: Random number generator keys.

        Returns:
            dictionary mapping extension names to extension instances.
        """
        ...

    def get_adapter(self, model_cls: type[GenerativeModel]) -> ModelAdapter:
        """Get an adapter for the specified model class.

        Args:
            model_cls: The model class to adapt.

        Returns:
            A model adapter for the specified model class.
        """
        ...


class BaseGenerationProtocol(Protocol):
    """Base protocol for generation models across all modalities."""

    def generate(
        self,
        n_samples: int = 1,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> jax.Array:
        """Generate samples from the model.

        Args:
            n_samples: Number of samples to generate
            rngs: Random number generators
            **kwargs: Modality-specific generation parameters

        Returns:
            Generated samples
        """
        ...

    def compute_likelihood(self, data: jax.Array) -> jax.Array:
        """Compute likelihood of data samples.

        Args:
            data: Data to evaluate

        Returns:
            Log-likelihood values
        """
        ...


class BaseDataset(ABC):
    """Abstract base class for modality datasets.

    Note: This is a data container class and does NOT inherit from nnx.Module.
    Datasets store JAX arrays in list/dict attributes which would cause
    "unexpected Arrays in static attribute" errors if they inherited from nnx.Module.
    """

    def __init__(
        self,
        config: BaseModalityConfig,
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize dataset.

        Args:
            config: Modality configuration
            split: Dataset split ('train', 'val', 'test')
            rngs: Random number generators
        """
        self.config = config
        self.split = split
        self.rngs = rngs

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over dataset samples."""
        ...

    @abstractmethod
    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Get a batch of samples.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Batch dictionary with modality-specific data
        """
        ...

    def get_sample(self, index: int) -> dict[str, jax.Array]:
        """Get a single sample by index.

        Args:
            index: Sample index

        Returns:
            Sample data
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        # Default implementation - subclasses can override for efficiency
        for i, sample in enumerate(self):
            if i == index:
                return sample
        raise IndexError(f"Sample {index} not found")

    def get_data_statistics(self) -> dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        return {
            "size": len(self),
            "split": self.split,
            "config": self.config,
        }


class BaseEvaluationSuite(nnx.Module, ABC):
    """Abstract base class for modality evaluation suites."""

    def __init__(
        self,
        config: BaseModalityConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize evaluation suite.

        Args:
            config: Modality configuration
            rngs: Random number generators
        """
        super().__init__()
        self.config = config
        self.rngs = rngs

    @abstractmethod
    def evaluate_batch(
        self,
        generated_data: jax.Array,
        reference_data: jax.Array | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate a batch of generated data.

        Args:
            generated_data: Generated data to evaluate
            reference_data: Reference data for comparison (optional)
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary of evaluation metrics
        """
        ...

    def compute_quality_metrics(self, data: jax.Array) -> dict[str, float]:
        """Compute intrinsic quality metrics.

        Args:
            data: Data to evaluate

        Returns:
            Quality metrics
        """
        # Default implementation - subclasses should override
        return {
            "data_shape": str(data.shape),
            "data_mean": float(jnp.mean(data)),
            "data_std": float(jnp.std(data)),
        }

    def compute_diversity_metrics(self, data_batch: jax.Array) -> dict[str, float]:
        """Compute diversity metrics for a batch.

        Args:
            data_batch: Batch of data samples

        Returns:
            Diversity metrics
        """
        # Default implementation - subclasses should override
        batch_size = data_batch.shape[0]
        return {
            "batch_size": float(batch_size),
            "unique_samples": float(batch_size),  # Placeholder
        }


class BaseProcessor(nnx.Module, ABC):
    """Abstract base class for modality processors."""

    def __init__(
        self,
        config: BaseModalityConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize processor.

        Args:
            config: Modality configuration
            rngs: Random number generators
        """
        super().__init__()
        self.config = config
        self.rngs = rngs

    @abstractmethod
    def process(
        self,
        data: jax.Array,
        **kwargs,
    ) -> jax.Array:
        """Process input data.

        Args:
            data: Input data to process
            **kwargs: Processing parameters

        Returns:
            Processed data
        """
        ...

    def preprocess(self, data: jax.Array) -> jax.Array:
        """Preprocess data for model input.

        Args:
            data: Raw data

        Returns:
            Preprocessed data
        """
        # Default implementation - subclasses can override
        return self.process(data)

    def postprocess(self, data: jax.Array) -> jax.Array:
        """Postprocess model output.

        Args:
            data: Model output

        Returns:
            Postprocessed data
        """
        # Default implementation - subclasses can override
        return data


class BaseModalityImplementation:
    """Base implementation class for modalities.

    This provides common functionality that most modalities need,
    reducing code duplication across different modality implementations.

    Note: This is a container/utility class and does NOT inherit from nnx.Module.
    Modality classes may store JAX arrays in dict/list attributes which would
    cause "unexpected Arrays in static attribute" errors if they inherited from nnx.Module.
    """

    def __init__(
        self,
        config: BaseModalityConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize modality.

        Args:
            config: Modality configuration (must be BaseModalityConfig)
            rngs: Random number generators

        Raises:
            TypeError: If config is not a BaseModalityConfig
        """
        if not isinstance(config, BaseModalityConfig):
            raise TypeError(f"config must be a BaseModalityConfig, got {type(config).__name__}")
        self.config = config
        self.rngs = rngs
        self.name = config.name

    def get_extensions(self, config: Any) -> dict[str, Any]:
        """Get modality-specific extensions.

        Default implementation returns empty dict.
        Subclasses should override to provide actual extensions.

        Args:
            config: Extension configuration (must be a typed configuration)

        Returns:
            Dictionary of extension configurations

        Raises:
            TypeError: If config is not a BaseConfiguration subclass
        """
        # Config validation is now handled by dataclass __post_init__
        return {}

    def validate_data_shape(self, data: jax.Array, expected_shape: tuple) -> bool:
        """Validate data has expected shape.

        Args:
            data: Data to validate
            expected_shape: Expected shape (can include None for variable dims)

        Returns:
            True if shape is valid
        """
        if len(data.shape) != len(expected_shape):
            return False

        for actual, expected in zip(data.shape, expected_shape):
            if expected is not None and actual != expected:
                return False

        return True

    def create_batch_from_samples(self, samples: list[jax.Array]) -> jax.Array:
        """Create batch from list of samples.

        Args:
            samples: List of individual samples

        Returns:
            Batched data
        """
        if not samples:
            raise ValueError("Cannot create batch from empty samples list")

        # Check if all samples have the same shape
        first_shape = samples[0].shape
        for i, sample in enumerate(samples[1:], 1):
            if sample.shape != first_shape:
                raise ValueError(
                    f"Sample {i} shape {sample.shape} doesn't match "
                    f"first sample shape {first_shape}"
                )

        return jnp.stack(samples)

    def split_batch_to_samples(self, batch: jax.Array) -> list[jax.Array]:
        """Split batch into individual samples.

        Args:
            batch: Batched data

        Returns:
            List of individual samples
        """
        return [batch[i] for i in range(batch.shape[0])]


# Type aliases for common patterns
ModalityData = jax.Array
ModalityBatch = dict[str, jax.Array]
ModalityConfig = BaseModalityConfig
EvaluationMetrics = dict[str, float]


# Factory function type
def create_modality_factory(
    modality_class: type,
    default_config: BaseModalityConfig,
):
    """Create a factory function for a modality.

    Args:
        modality_class: The modality class to instantiate
        default_config: Default configuration

    Returns:
        Factory function
    """

    def factory(
        config: BaseModalityConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        """Factory function for modality creation.

        Args:
            config: Modality configuration (uses default if None, must be BaseModalityConfig)
            rngs: Random number generators
            **kwargs: Additional arguments

        Returns:
            Modality instance

        Raises:
            TypeError: If config is not None and not a BaseModalityConfig
        """
        if config is not None and not isinstance(config, BaseModalityConfig):
            raise TypeError(f"config must be a BaseModalityConfig, got {type(config).__name__}")
        final_config = config or default_config
        return modality_class(config=final_config, rngs=rngs, **kwargs)

    return factory


def validate_modality_interface(modality_instance: Any) -> bool:
    """Validate that an instance implements the Modality protocol.

    Args:
        modality_instance: Instance to validate

    Returns:
        True if instance implements Modality protocol
    """
    required_attrs = ["name", "get_extensions", "get_adapter"]

    for attr in required_attrs:
        if not hasattr(modality_instance, attr):
            return False

        if attr != "name" and not callable(getattr(modality_instance, attr)):
            return False

    return True


# Export commonly used base classes and protocols
__all__ = [
    # Protocols
    "ModelAdapter",
    "Modality",
    "BaseGenerationProtocol",
    # Abstract base classes
    "BaseModalityConfig",
    "BaseDataset",
    "BaseEvaluationSuite",
    "BaseProcessor",
    "BaseModalityImplementation",
    # Type aliases
    "ModalityData",
    "ModalityBatch",
    "ModalityConfig",
    "EvaluationMetrics",
    # Utilities
    "create_modality_factory",
    "validate_modality_interface",
]
