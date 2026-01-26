"""Base dataset management protocols and infrastructure."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from artifex.generative_models.core.configuration import DataConfig
from artifex.utils.file_utils import get_valid_output_dir


class DatasetProtocol(ABC):
    """Abstract base class for all benchmark datasets.

    This class defines the standard interface for datasets in the
    benchmark system. All datasets must follow proper RNG handling
    and JAX-compatible operations.

    Note: This class does NOT inherit from nnx.Module because datasets
    are data containers, not neural network modules. JAX arrays should
    be stored as regular attributes since datasets don't need JIT
    compilation or gradient computation.

    Attributes:
        data_path: Path to dataset files
        config: Dataset configuration
        rngs: NNX Rngs for stochastic operations
    """

    def __init__(self, data_path: str, config: DataConfig, *, rngs: nnx.Rngs):
        """Initialize dataset with path, configuration and RNG state.

        Args:
            data_path: Path to dataset directory
            config: DataConfig instance with dataset settings
            rngs: NNX Rngs for all stochastic operations
        """
        if not isinstance(config, DataConfig):
            raise TypeError(f"config must be DataConfig, got {type(config).__name__}")

        # Check if the path exists before trying to create it
        path_obj = Path(data_path)
        if not path_obj.exists() and path_obj.is_absolute() and str(path_obj).startswith("/"):
            # If it's an absolute path that doesn't exist, check if we can access the parent
            try:
                # Try to check if parent exists
                if not path_obj.parent.exists():
                    raise FileNotFoundError(f"Dataset path does not exist: {data_path}")
            except (PermissionError, OSError):
                raise FileNotFoundError(f"Dataset path does not exist: {data_path}") from None

        # Ensure data path is in the proper directory (test_results or benchmark_results)
        validated_path = get_valid_output_dir(
            data_path,
            base_dir="test_results" if "test" in data_path else "benchmark_results",
            create_dir=True,
        )
        self.data_path = Path(validated_path)
        self.config = config
        self.rngs = rngs
        self._load_dataset()

    @abstractmethod
    def _load_dataset(self):
        """Load the dataset into memory or prepare lazy loading.

        Subclasses must implement this to:
        - Load data according to configuration
        - Set up data structures
        - Prepare indices for batching
        """
        pass

    @abstractmethod
    def get_batch(self, batch_size: int | None = None) -> dict[str, jax.Array]:
        """Get a batch of data samples.

        Args:
            batch_size: Size of batch to return, uses config default if None

        Returns:
            Dictionary containing batch data (e.g., "points", "labels")
        """
        pass

    @abstractmethod
    def get_dataset_info(self) -> dict[str, Any]:
        """Get comprehensive dataset information.

        Returns:
            Dictionary containing:
            - name: Dataset name
            - modality: Data modality (geometric, image, etc.)
            - n_samples: Number of samples
            - data_shape: Shape information for each data field
            - Any other relevant metadata
        """
        pass

    def get_sample_count(self) -> int:
        """Get total number of samples in dataset.

        Returns:
            Number of samples
        """
        info = self.get_dataset_info()
        return info.get("n_samples", 0)

    def validate_batch(self, batch: dict[str, jax.Array]) -> bool:
        """Validate a batch has expected structure and properties.

        Args:
            batch: Batch dictionary to validate

        Returns:
            True if batch is valid, False otherwise
        """
        # Check for required fields if specified in config metadata
        validation_config = self.config.metadata.get("validation", {})
        required_fields = validation_config.get("required_fields", [])
        for field in required_fields:
            if field not in batch:
                return False

        # Check that all arrays are finite
        for key, array in batch.items():
            if isinstance(array, jax.Array):
                if not jnp.isfinite(array).all():
                    return False

        return True


class DatasetRegistry:
    """Singleton registry for managing dataset instances.

    This registry provides centralized management of datasets,
    allowing registration, retrieval, and metadata access.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Create a new dataset registry instance.

        Returns:
            An instance of the DatasetRegistry
        """
        if cls._instance is None:
            cls._instance = super(DatasetRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the dataset registry.

        Sets up an empty dictionary to store registered datasets.
        """
        if not self._initialized:
            self.datasets: dict[str, DatasetProtocol] = {}
            self._initialized = True

    def register_dataset(self, name: str, dataset: DatasetProtocol):
        """Register a dataset instance.

        Args:
            name: Unique name for the dataset
            dataset: Dataset instance to register
        """
        self.datasets[name] = dataset

    def get_dataset(self, name: str) -> DatasetProtocol:
        """Retrieve a registered dataset.

        Args:
            name: Name of dataset to retrieve

        Returns:
            Dataset instance

        Raises:
            KeyError: If dataset is not registered
        """
        if name not in self.datasets:
            raise KeyError(f"Dataset '{name}' not registered")
        return self.datasets[name]

    def list_datasets(self) -> list[str]:
        """Get list of registered dataset names.

        Returns:
            List of dataset names
        """
        return list(self.datasets.keys())

    def has_dataset(self, name: str) -> bool:
        """Check if dataset is registered.

        Args:
            name: Dataset name to check

        Returns:
            True if dataset is registered, False otherwise
        """
        return name in self.datasets

    def get_dataset_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a registered dataset.

        Args:
            name: Name of dataset

        Returns:
            Dataset metadata dictionary

        Raises:
            KeyError: If dataset is not registered
        """
        dataset = self.get_dataset(name)
        return dataset.get_dataset_info()

    def clear_registry(self):
        """Clear all registered datasets."""
        self.datasets.clear()


class DatasetValidator:
    """Validates dataset structure and data quality.

    This class provides comprehensive validation for datasets,
    including structure checks, data quality validation,
    and compliance with benchmark requirements.
    """

    def __init__(self):
        """Initialize validator with standard validation rules."""
        self.validation_rules = {
            "check_finite_values": True,
            "check_shape_consistency": True,
            "check_required_fields": True,
            "check_data_ranges": True,
        }

    def validate_structure(self, dataset: DatasetProtocol) -> tuple[bool, list[str]]:
        """Validate dataset structure and configuration.

        Args:
            dataset: Dataset to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check if dataset has required methods
        required_methods = [
            "get_batch",
            "get_dataset_info",
            "_validate_dataset_path",
            "_load_dataset",
        ]
        for method in required_methods:
            if not hasattr(dataset, method):
                errors.append(f"Dataset missing required method: {method}")

        # Check if dataset has required attributes
        required_attrs = ["data_path", "config", "rngs"]
        for attr in required_attrs:
            if not hasattr(dataset, attr):
                errors.append(f"Dataset missing required attribute: {attr}")

        # Check if dataset path exists
        if hasattr(dataset, "data_path") and not dataset.data_path.exists():
            errors.append(f"Dataset path does not exist: {dataset.data_path}")

        # Check if dataset has loaded data
        if hasattr(dataset, "data") and not dataset.data:
            errors.append("Dataset data not loaded")

        return len(errors) == 0, errors

    def validate_data_quality(self, batch: dict[str, jax.Array]) -> tuple[bool, list[str]]:
        """Validate data quality of a batch.

        Args:
            batch: Batch dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if not batch:
            issues.append("Batch is empty")
            return False, issues

        # Check for finite values
        if self.validation_rules["check_finite_values"]:
            for key, array in batch.items():
                if isinstance(array, jax.Array):
                    if not jnp.isfinite(array).all():
                        issues.append(f"Non-finite values found in {key}")

        # Check shape consistency within batch
        if self.validation_rules["check_shape_consistency"]:
            batch_sizes = []
            for key, array in batch.items():
                if isinstance(array, jax.Array) and array.ndim > 0:
                    batch_sizes.append(array.shape[0])

            if batch_sizes and not all(bs == batch_sizes[0] for bs in batch_sizes):
                issues.append("Inconsistent batch sizes across arrays")

        return len(issues) == 0, issues

    def check_minimum_samples(self, dataset: DatasetProtocol, min_samples: int) -> bool:
        """Check if dataset has minimum required samples.

        Args:
            dataset: Dataset to check
            min_samples: Minimum required samples

        Returns:
            True if dataset has enough samples, False otherwise
        """
        sample_count = dataset.get_sample_count()
        return sample_count >= min_samples

    def validate_config(self, config: DataConfig) -> tuple[bool, list[str]]:
        """Validate dataset configuration.

        Args:
            config: DataConfig instance to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if not isinstance(config, DataConfig):
            errors.append(f"config must be DataConfig, got {type(config).__name__}")
            return False, errors

        # Check required fields
        if not config.dataset_name:
            errors.append("dataset_name is required")

        # Validate batch size if in metadata
        if "batch_size" in config.metadata:
            batch_size = config.metadata["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                errors.append("batch_size must be a positive integer")

        # Validate num_workers
        if config.num_workers < 0:
            errors.append("num_workers must be non-negative")

        # Validate prefetch_factor
        if config.prefetch_factor < 1:
            errors.append("prefetch_factor must be at least 1")

        return len(errors) == 0, errors

    def set_validation_rule(self, rule_name: str, enabled: bool):
        """Enable or disable a validation rule.

        Args:
            rule_name: Name of validation rule
            enabled: Whether to enable the rule
        """
        if rule_name in self.validation_rules:
            self.validation_rules[rule_name] = enabled


class DatasetLoader:
    """Factory for loading datasets from configurations.

    This class provides functionality to create dataset instances
    from configuration dictionaries, supporting multiple dataset types
    and formats.
    """

    def __init__(self):
        """Initialize loader with supported formats and types."""
        self.supported_formats = ["mock", "shapenet", "crossdocked", "celeba"]
        self.dataset_types: dict[str, type[DatasetProtocol]] = {}

    def register_dataset_type(self, type_name: str, dataset_class: Type[DatasetProtocol]):
        """Register a new dataset type.

        Args:
            type_name: Name for the dataset type
            dataset_class: Dataset class to register
        """
        self.dataset_types[type_name] = dataset_class

    def load_from_config(self, config: DataConfig, *, rngs: nnx.Rngs) -> DatasetProtocol:
        """Load dataset from configuration.

        Args:
            config: DataConfig instance with dataset settings
            rngs: NNX Rngs for dataset initialization

        Returns:
            Initialized dataset instance

        Raises:
            TypeError: If config is not DataConfig
            ValueError: If dataset type is not registered
            KeyError: If required config fields are missing
        """
        if not isinstance(config, DataConfig):
            raise TypeError(f"config must be DataConfig, got {type(config).__name__}")

        # Get dataset type from metadata
        dataset_type = config.metadata.get("type")
        if dataset_type is None:
            raise KeyError("DataConfig metadata must specify 'type'")

        if dataset_type not in self.dataset_types:
            raise ValueError(f"Dataset type '{dataset_type}' not registered")

        # Use data_dir from DataConfig
        data_path = str(config.data_dir)

        dataset_class = self.dataset_types[dataset_type]

        # Create dataset instance with DataConfig
        dataset = dataset_class(data_path=data_path, config=config, rngs=rngs)

        return dataset

    def list_supported_types(self) -> list[str]:
        """Get list of supported dataset types.

        Returns:
            List of registered dataset type names
        """
        return list(self.dataset_types.keys())

    def has_dataset_type(self, type_name: str) -> bool:
        """Check if dataset type is supported.

        Args:
            type_name: Dataset type name to check

        Returns:
            True if type is registered, False otherwise
        """
        return type_name in self.dataset_types

    def get_dataset_requirements(self, type_name: str) -> dict[str, Any]:
        """Get requirements for a specific dataset type.

        Args:
            type_name: Dataset type name

        Returns:
            Dictionary of requirements (config fields, file structure, etc.)

        Raises:
            ValueError: If dataset type is not registered
        """
        if type_name not in self.dataset_types:
            raise ValueError(f"Dataset type '{type_name}' not registered")

        # Basic requirements - can be extended per dataset type
        return {
            "required_config_fields": ["name", "data_path", "batch_size"],
            "optional_config_fields": ["split", "preprocessing", "validation"],
            "supported_formats": self.supported_formats,
        }
