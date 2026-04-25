"""Dataset loaders for benchmark datasets.

Implements utility functions for loading and registering datasets.
Structurally conforms to calibrax DatasetProtocol.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from artifex.benchmarks.datasets.base import DatasetRegistry


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkDatasetConfig:
    """Configuration for a benchmark dataset."""

    name: str
    description: str
    data_type: str
    dimensions: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkDataset:
    """Benchmark dataset with save/load support.

    Structurally conforms to calibrax's DatasetProtocol
    (implements ``__len__`` and ``__getitem__``).
    """

    def __init__(
        self,
        config: BenchmarkDatasetConfig,
        data: np.ndarray | jnp.ndarray,
    ) -> None:
        """Initialize the in-memory benchmark dataset."""
        self.config = config
        self.data = jnp.array(data) if isinstance(data, np.ndarray) else data

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Any:
        """Return one dataset sample."""
        return self.data[idx]

    def save(self, path: str | Path) -> None:
        """Save the dataset to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data_np = np.array(self.data)
        np.savez(
            save_path,
            data=data_np,
            name=self.config.name,
            description=self.config.description,
            data_type=self.config.data_type,
            dimensions=np.array(self.config.dimensions),
            metadata=json.dumps(self.config.metadata),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkDataset":
        """Load a dataset from a file.

        Note: Uses np.load with allow_pickle for backward compatibility
        with existing saved datasets. Only load files from trusted sources.
        """
        with np.load(str(path), allow_pickle=True) as data:  # noqa: S301
            config = BenchmarkDatasetConfig(
                name=str(data["name"]),
                description=str(data["description"]),
                data_type=str(data["data_type"]),
                dimensions=data["dimensions"].tolist(),
                metadata=json.loads(str(data["metadata"])),
            )
            return cls(config=config, data=jnp.array(data["data"]))


_dataset_registry = DatasetRegistry()


def register_dataset(name: str, dataset: BenchmarkDataset) -> None:
    """Register a dataset with the given name."""
    _dataset_registry.register_dataset(name, dataset)


def get_dataset(name: str) -> BenchmarkDataset:
    """Get a dataset by name."""
    return _dataset_registry.get_dataset(name)


def list_datasets() -> list[str]:
    """List all registered datasets."""
    return _dataset_registry.list_datasets()


def load_benchmark_dataset(dataset_path_or_name: str) -> BenchmarkDataset:
    """Load a dataset from a file or get it from the registry."""
    if _dataset_registry.has_dataset(dataset_path_or_name):
        return _dataset_registry.get_dataset(dataset_path_or_name)

    file_path = Path(dataset_path_or_name)
    if file_path.exists():
        return BenchmarkDataset.load(file_path)

    raise ValueError(f"Dataset '{dataset_path_or_name}' not found in registry or as file")
