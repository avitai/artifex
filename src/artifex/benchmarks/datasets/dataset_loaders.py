"""Dataset loaders for benchmark datasets.

This module implements the base dataset class for benchmarks, along with
utility functions for loading and registering datasets.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

from artifex.benchmarks.base import DatasetProtocol


@dataclass
class BenchmarkDatasetConfig:
    """Configuration for a benchmark dataset.

    Attributes:
        name: Name of the dataset.
        description: Description of the dataset.
        data_type: Type of data in the dataset (e.g., 'continuous', 'discrete').
        dimensions: Dimensions of each data point.
        metadata: Additional metadata for the dataset.
    """

    name: str
    description: str
    data_type: str
    dimensions: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkDataset(DatasetProtocol):
    """Base class for benchmark datasets.

    This class provides a standardized interface for datasets used in benchmarks.
    It implements the DatasetProtocol interface.
    """

    def __init__(
        self,
        config: BenchmarkDatasetConfig,
        data: np.ndarray | jnp.ndarray,
    ) -> None:
        """Initialize a benchmark dataset.

        Args:
            config: Configuration for the dataset.
            data: Data array for the dataset.
        """
        self.config = config
        # Convert to jax array if numpy array is provided
        if isinstance(data, np.ndarray):
            self.data = jnp.array(data)
        else:
            self.data = data

    def __len__(self) -> int:
        """Get the number of examples in the dataset.

        Returns:
            Number of examples.
        """
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Any:
        """Get an example from the dataset.

        Args:
            idx: Index of the example.

        Returns:
            The example.
        """
        return self.data[idx]

    def save(self, path: str) -> None:
        """Save the dataset to a file.

        Args:
            path: Path to save the dataset to.
        """
        # Convert jax array to numpy for saving
        data_np = np.array(self.data)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Save data and config separately
        np.savez(
            path,
            data=data_np,
            name=self.config.name,
            description=self.config.description,
            data_type=self.config.data_type,
            dimensions=np.array(self.config.dimensions),
            metadata=json.dumps(self.config.metadata),
        )

    @classmethod
    def load(cls, path: str) -> "BenchmarkDataset":
        """Load a dataset from a file.

        Args:
            path: Path to load the dataset from.

        Returns:
            The loaded dataset.
        """
        # Load data and config
        with np.load(path, allow_pickle=True) as data:
            # Extract config parameters
            config = BenchmarkDatasetConfig(
                name=str(data["name"]),
                description=str(data["description"]),
                data_type=str(data["data_type"]),
                dimensions=data["dimensions"].tolist(),
                metadata=json.loads(str(data["metadata"])),
            )

            # Create dataset
            return cls(config=config, data=jnp.array(data["data"]))


# Registry to store datasets
_dataset_registry: dict[str, BenchmarkDataset] = {}


def register_dataset(name: str, dataset: BenchmarkDataset) -> None:
    """Register a dataset with the given name.

    Args:
        name: Name to register the dataset under.
        dataset: Dataset to register.
    """
    _dataset_registry[name] = dataset


def get_dataset(name: str) -> BenchmarkDataset:
    """Get a dataset by name.

    Args:
        name: Name of the dataset to get.

    Returns:
        The dataset.

    Raises:
        KeyError: If the dataset isn't registered.
    """
    if name not in _dataset_registry:
        raise KeyError(f"Dataset '{name}' not found in registry")
    return _dataset_registry[name]


def list_datasets() -> list[str]:
    """List all registered datasets.

    Returns:
        List of dataset names.
    """
    return list(_dataset_registry.keys())


def load_dataset(dataset_path_or_name: str) -> BenchmarkDataset:
    """Load a dataset from a file or get it from the registry.

    Args:
        dataset_path_or_name: Path to a dataset file or name of a registered
            dataset.

    Returns:
        The dataset.

    Raises:
        ValueError: If the dataset can't be found.
    """
    # Check if it's a registered dataset
    if dataset_path_or_name in _dataset_registry:
        return _dataset_registry[dataset_path_or_name]

    # Check if it's a file path
    if os.path.exists(dataset_path_or_name):
        return BenchmarkDataset.load(dataset_path_or_name)

    # Not found
    raise ValueError(f"Dataset '{dataset_path_or_name}' not found in registry or as file")
