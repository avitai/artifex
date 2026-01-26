from pydantic import Field, field_validator

from artifex.configs.schema.base import BaseConfig


class DatasetConfig(BaseConfig):
    """Base configuration for datasets."""

    dataset_type: str = Field(..., description="Type of dataset")
    data_path: str | None = Field(None, description="Path to dataset")
    cache_dir: str | None = Field(None, description="Directory for caching data")

    @field_validator("data_path")
    @classmethod
    def validate_data_path(cls, v: str | None) -> str | None:
        """Validate data path."""
        if v is not None and not v:
            raise ValueError("Data path cannot be empty")
        return v


class ProteinDatasetConfig(DatasetConfig):
    """Configuration for protein datasets."""

    dataset_type: str = Field("protein", description="Dataset type (always 'protein')")
    max_seq_length: int = Field(128, description="Maximum sequence length")
    atom_types: list[str] = Field([], description="Atom types to include (empty for all)")
    backbone_atom_indices: list[int] = Field(
        [0, 1, 2, 4], description="Indices of backbone atoms to use"
    )
    residue_types: list[str] | None = Field(
        None, description="Residue types to include (None for all)"
    )
    splits_file: str | None = Field(None, description="Path to file defining train/val/test splits")
    precompute_statistics: bool = Field(
        True, description="Whether to precompute dataset statistics"
    )

    @field_validator("max_seq_length")
    @classmethod
    def validate_max_seq_length(cls, v: int) -> int:
        """Validate max sequence length."""
        if v <= 0:
            raise ValueError("Max sequence length must be positive")
        return v


class DataConfig(BaseConfig):
    """Configuration for data loading and processing."""

    # Dataset configuration
    validation_dataset: ProteinDatasetConfig | None = Field(
        None, description="Validation dataset configuration"
    )
    test_dataset: ProteinDatasetConfig | None = Field(
        None, description="Test dataset configuration"
    )

    # Data processing
    normalize_coordinates: bool = Field(True, description="Whether to normalize coordinates")
    center_coordinates: bool = Field(
        True, description="Whether to center coordinates on center of mass"
    )

    # Data augmentation
    augmentation_enabled: bool = Field(False, description="Whether to use data augmentation")
    random_rotation: bool = Field(
        False, description="Apply random 3D rotations to protein structures"
    )

    # Multiprocessing
    num_workers: int = Field(4, description="Number of workers for data loading")
    prefetch_factor: int = Field(2, description="Prefetch factor for data loading")
    persistent_workers: bool = Field(
        False,
        description="Whether to maintain worker processes between batches",
    )

    @field_validator("num_workers", "prefetch_factor")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate that value is a positive integer."""
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v
