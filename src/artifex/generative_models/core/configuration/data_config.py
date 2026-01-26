"""Data configuration using frozen dataclasses.

This module provides a frozen dataclass-based configuration for data loading
and preprocessing, replacing the Pydantic-based DataConfig.
"""

import dataclasses
from pathlib import Path
from typing import Any

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.validation import (
    validate_non_negative_int,
    validate_positive_int,
    validate_range,
)


@dataclasses.dataclass(frozen=True)
class DataConfig(BaseConfig):
    """Configuration for data loading and preprocessing.

    This dataclass provides a type-safe, immutable configuration for data
    handling in generative models. It replaces the Pydantic DataConfig.

    Attributes:
        dataset_name: Name of the dataset (required, validated in __post_init__)
        data_dir: Directory containing the data
        split: Data split to use (train/val/test)
        num_workers: Number of data loading workers
        prefetch_factor: Number of batches to prefetch per worker
        pin_memory: Whether to pin memory for faster GPU transfer
        augmentation: Whether to use data augmentation
        augmentation_params: Parameters for data augmentation
        validation_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
    """

    # Required field with dummy default (validated in __post_init__)
    # Python dataclass limitation: can't have required fields after optional parent fields
    dataset_name: str = ""

    # Dataset paths and splits
    data_dir: Path = Path("./data")
    split: str = "train"

    # Data loading settings
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True

    # Augmentation
    augmentation: bool = False
    augmentation_params: dict[str, Any] = dataclasses.field(default_factory=dict)

    # Split ratios
    validation_split: float = 0.1
    test_split: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
        """
        super().__post_init__()

        # Validate dataset_name (required field with dummy default)
        if not self.dataset_name or not self.dataset_name.strip():
            raise ValueError("dataset_name cannot be empty")

        # Validate num_workers
        validate_non_negative_int(self.num_workers, "num_workers")

        # Validate prefetch_factor
        validate_positive_int(self.prefetch_factor, "prefetch_factor")

        # Validate validation_split
        validate_range(self.validation_split, "validation_split", 0.0, 1.0)

        # Validate test_split
        validate_range(self.test_split, "test_split", 0.0, 1.0)

        # Validate that splits don't exceed 1
        if self.validation_split + self.test_split > 1.0:
            raise ValueError(
                f"validation_split + test_split must not exceed 1, "
                f"got {self.validation_split} + {self.test_split} = "
                f"{self.validation_split + self.test_split}"
            )

        # Convert data_dir to Path if it's a string (for direct instantiation)
        if isinstance(self.data_dir, str):
            object.__setattr__(self, "data_dir", Path(self.data_dir))
