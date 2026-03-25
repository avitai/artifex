"""Benchmark dataset helpers backed by CalibraX primitives."""

from typing import Any

import jax
import jax.numpy as jnp
from calibrax.core import SingletonRegistry


class DatasetRegistry(SingletonRegistry[Any]):
    """CalibraX-backed singleton registry for benchmark datasets.

    The canonical registry semantics come from `calibrax.core.SingletonRegistry`.
    Artifex keeps only narrow dataset-specific convenience helpers here.
    """

    def register_dataset(self, name: str, dataset: Any) -> None:
        """Register a dataset instance."""
        self.register(name, dataset)

    def get_dataset(self, name: str) -> Any:
        """Retrieve a registered dataset."""
        return self.get(name)

    def list_datasets(self) -> list[str]:
        """Return the registered dataset names."""
        return self.list_names()

    def has_dataset(self, name: str) -> bool:
        """Check whether a dataset is registered."""
        return self.has(name)

    def get_dataset_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a registered dataset."""
        dataset = self.get(name)
        return dataset.get_dataset_info()

    def clear_registry(self) -> None:
        """Remove all registered datasets."""
        self.clear()


class DatasetValidator:
    """Validates dataset structure and data quality.

    Provides structure checks (required methods/attributes) and
    data quality validation (finite values, shape consistency).
    """

    def __init__(self) -> None:
        """Initialize validator with default validation rules."""
        self.validation_rules: dict[str, bool] = {
            "check_finite_values": True,
            "check_shape_consistency": True,
            "check_required_fields": True,
            "check_data_ranges": True,
        }

    def validate_structure(self, dataset: Any) -> tuple[bool, list[str]]:
        """Validate that a dataset has the required interface."""
        errors: list[str] = []

        required_methods = ["get_batch", "get_dataset_info", "_load_dataset"]
        for method in required_methods:
            if not hasattr(dataset, method):
                errors.append(f"Dataset missing required method: {method}")

        required_attrs = ["config"]
        for attr in required_attrs:
            if not hasattr(dataset, attr):
                errors.append(f"Dataset missing required attribute: {attr}")

        return len(errors) == 0, errors

    def validate_data_quality(self, batch: dict[str, jax.Array]) -> tuple[bool, list[str]]:
        """Validate data quality of a batch."""
        issues: list[str] = []

        if not batch:
            issues.append("Batch is empty")
            return False, issues

        if self.validation_rules["check_finite_values"]:
            for key, array in batch.items():
                if isinstance(array, jax.Array) and not jnp.isfinite(array).all():
                    issues.append(f"Non-finite values found in {key}")

        if self.validation_rules["check_shape_consistency"]:
            batch_sizes: list[int] = []
            for array in batch.values():
                if isinstance(array, jax.Array) and array.ndim > 0:
                    batch_sizes.append(array.shape[0])
            if batch_sizes and not all(bs == batch_sizes[0] for bs in batch_sizes):
                issues.append("Inconsistent batch sizes across arrays")

        return len(issues) == 0, issues

    def check_minimum_samples(self, dataset: Any, min_samples: int) -> bool:
        """Check whether a dataset meets a minimum sample count."""
        return dataset.get_sample_count() >= min_samples

    def set_validation_rule(self, rule_name: str, *, enabled: bool) -> None:
        """Enable or disable a validation rule."""
        if rule_name in self.validation_rules:
            self.validation_rules[rule_name] = enabled
