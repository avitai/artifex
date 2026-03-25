"""Base tabular modality implementation.

The retained public constructor is the typed `TabularModalityConfig` path; the
removed quick-start keyword constructor is not part of the supported surface.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import BaseModalityConfig

from ..base import BaseModalityImplementation


class ColumnType(Enum):
    """Types of columns in tabular data."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BINARY = "binary"


@dataclass(frozen=True, slots=True, kw_only=True)
class TabularModalityConfig(BaseModalityConfig):
    """Configuration for tabular modality.

    Attributes:
        num_features: Total number of features
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        ordinal_features: List of ordinal feature names
        binary_features: List of binary feature names
        categorical_vocab_sizes: Vocabulary sizes for categorical features
        ordinal_orders: Ordering information for ordinal features
        normalization_type: Type of normalization ('standard', 'minmax', 'robust')
        handle_missing: How to handle missing values ('drop', 'impute', 'mask')
        max_categorical_cardinality: Maximum cardinality for categorical features
    """

    num_features: int = 0
    numerical_features: tuple[str, ...] = field(default_factory=tuple)
    categorical_features: tuple[str, ...] = field(default_factory=tuple)
    ordinal_features: tuple[str, ...] = field(default_factory=tuple)
    binary_features: tuple[str, ...] = field(default_factory=tuple)
    categorical_vocab_sizes: dict[str, int] = field(default_factory=dict)
    ordinal_orders: dict[str, tuple[str, ...]] = field(default_factory=dict)
    normalization_type: str = "standard"
    handle_missing: str = "impute"
    max_categorical_cardinality: int = 100

    def __post_init__(self) -> None:
        """Initialize and validate the configuration."""
        BaseModalityConfig.__post_init__(self)
        object.__setattr__(self, "numerical_features", tuple(self.numerical_features))
        object.__setattr__(self, "categorical_features", tuple(self.categorical_features))
        object.__setattr__(self, "ordinal_features", tuple(self.ordinal_features))
        object.__setattr__(self, "binary_features", tuple(self.binary_features))
        object.__setattr__(
            self,
            "ordinal_orders",
            {feature: tuple(order) for feature, order in self.ordinal_orders.items()},
        )
        if self.num_features <= 0:
            raise ValueError("num_features must be positive")

    def validate_feature_consistency(self) -> "TabularModalityConfig":
        """Validate that feature lists are consistent."""
        all_features = (
            self.numerical_features
            + self.categorical_features
            + self.ordinal_features
            + self.binary_features
        )

        if len(set(all_features)) != len(all_features):
            raise ValueError("Feature names must be unique across all feature types")

        if len(all_features) != self.num_features:
            raise ValueError(
                f"Total features ({len(all_features)}) doesn't match "
                f"num_features ({self.num_features})"
            )

        # Validate categorical vocab sizes
        for feature in self.categorical_features:
            if feature not in self.categorical_vocab_sizes:
                raise ValueError(f"Missing vocab size for categorical feature: {feature}")
            if self.categorical_vocab_sizes[feature] > self.max_categorical_cardinality:
                raise ValueError(
                    f"Categorical feature {feature} has cardinality "
                    f"{self.categorical_vocab_sizes[feature]} > max allowed "
                    f"{self.max_categorical_cardinality}"
                )

        # Validate ordinal orders
        for feature in self.ordinal_features:
            if feature not in self.ordinal_orders:
                raise ValueError(f"Missing order information for ordinal feature: {feature}")

        return self


class TabularModality(BaseModalityImplementation):
    """Tabular modality for structured data generation and processing.

    This modality handles typed-config validation and encoding for mixed-type
    tabular data. Public evaluation remains narrower than the internal helper
    set and currently stays on the numerical/correlation/privacy surface.
    """

    def __init__(
        self,
        config: TabularModalityConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize tabular modality.

        Args:
            config: Tabular modality configuration
            rngs: Random number generators
        """
        super().__init__(config=config, rngs=rngs)
        self.config = config.validate_feature_consistency()
        self.rngs = rngs

        # Initialize feature processors
        self._init_processors()

    def _init_processors(self) -> None:
        """Initialize feature-specific processors."""
        # Will be implemented in representations.py
        pass

    def get_feature_info(self) -> dict[str, dict[str, Any]]:
        """Get information about features.

        Returns:
            Dictionary mapping feature names to their metadata
        """
        feature_info = {}

        for feature in self.config.numerical_features:
            feature_info[feature] = {
                "type": ColumnType.NUMERICAL,
                "encoding_dim": 1,
                "preprocessing": self.config.normalization_type,
            }

        for feature in self.config.categorical_features:
            vocab_size = self.config.categorical_vocab_sizes[feature]
            feature_info[feature] = {
                "type": ColumnType.CATEGORICAL,
                "vocab_size": vocab_size,
                "encoding_dim": vocab_size,  # One-hot encoding
            }

        for feature in self.config.ordinal_features:
            order_size = len(self.config.ordinal_orders[feature])
            feature_info[feature] = {
                "type": ColumnType.ORDINAL,
                "order_size": order_size,
                "encoding_dim": 1,  # Ordinal encoding to single value
                "order": self.config.ordinal_orders[feature],
            }

        for feature in self.config.binary_features:
            feature_info[feature] = {
                "type": ColumnType.BINARY,
                "encoding_dim": 1,
            }

        return feature_info

    def get_total_encoding_dim(self) -> int:
        """Get total dimensionality after encoding all features.

        Returns:
            Total encoding dimensionality
        """
        feature_info = self.get_feature_info()
        return sum(info["encoding_dim"] for info in feature_info.values())

    def validate_input(self, data: dict[str, jnp.ndarray]) -> None:
        """Validate input tabular data.

        Args:
            data: Dictionary mapping feature names to arrays

        Raises:
            ValueError: If data doesn't match configuration
        """
        expected_features = set(
            self.config.numerical_features
            + self.config.categorical_features
            + self.config.ordinal_features
            + self.config.binary_features
        )

        if set(data.keys()) != expected_features:
            missing = expected_features - set(data.keys())
            extra = set(data.keys()) - expected_features
            raise ValueError(f"Feature mismatch. Missing: {missing}, Extra: {extra}")

        # Validate shapes are consistent
        batch_sizes = [arr.shape[0] for arr in data.values()]
        if len(set(batch_sizes)) > 1:
            raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")

        # Validate categorical feature ranges
        for feature in self.config.categorical_features:
            arr = data[feature]
            vocab_size = self.config.categorical_vocab_sizes[feature]
            if jnp.any(arr < 0) or jnp.any(arr >= vocab_size):
                raise ValueError(
                    f"Categorical feature {feature} values must be in [0, {vocab_size})"
                )

        # Validate binary features
        for feature in self.config.binary_features:
            arr = data[feature]
            if not jnp.all(jnp.isin(arr, jnp.array([0, 1]))):
                raise ValueError(f"Binary feature {feature} must contain only 0 and 1")
