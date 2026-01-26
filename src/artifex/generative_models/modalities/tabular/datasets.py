"""Tabular datasets for synthetic data generation."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from ..base import BaseDataset
from .base import TabularModalityConfig


class SyntheticTabularDataset(BaseDataset):
    """Dataset for generating synthetic tabular data."""

    def __init__(
        self,
        config: TabularModalityConfig,
        num_samples: int = 1000,
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize synthetic tabular dataset.

        Args:
            config: Tabular modality configuration
            num_samples: Number of samples to generate
            split: Dataset split
            rngs: Random number generators
        """
        super().__init__(config=config, split=split, rngs=rngs)
        self.config = config.validate_feature_consistency()
        self.num_samples = num_samples

        # Generate synthetic data and wrap in nnx.Dict for NNX compatibility
        self.data = nnx.Dict(self._generate_data())

    def _generate_data(self) -> dict[str, jnp.ndarray]:
        """Generate synthetic tabular data.

        Returns:
            Dictionary mapping feature names to data arrays
        """
        data = {}

        # Generate keys for different feature types
        if "tabular" in self.rngs:
            key = self.rngs.tabular()
        else:
            key = jax.random.key(0)

        key, *subkeys = jax.random.split(key, 5)

        # Generate numerical features
        if self.config.numerical_features:
            numerical_key = subkeys[0]
            for i, feature in enumerate(self.config.numerical_features):
                feature_key = jax.random.fold_in(numerical_key, i)

                # Generate data with different distributions
                if i % 3 == 0:
                    # Normal distribution
                    feature_data = jax.random.normal(feature_key, (self.num_samples,))
                elif i % 3 == 1:
                    # Uniform distribution
                    feature_data = jax.random.uniform(
                        feature_key, (self.num_samples,), minval=-2.0, maxval=2.0
                    )
                else:
                    # Exponential-like distribution
                    feature_data = jax.random.exponential(feature_key, (self.num_samples,))

                data[feature] = feature_data

        # Generate categorical features
        if self.config.categorical_features:
            categorical_key = subkeys[1]
            for i, feature in enumerate(self.config.categorical_features):
                feature_key = jax.random.fold_in(categorical_key, i)
                vocab_size = self.config.categorical_vocab_sizes[feature]

                # Generate categorical data with uniform distribution
                feature_data = jax.random.randint(feature_key, (self.num_samples,), 0, vocab_size)
                data[feature] = feature_data

        # Generate ordinal features
        if self.config.ordinal_features:
            ordinal_key = subkeys[2]
            for i, feature in enumerate(self.config.ordinal_features):
                feature_key = jax.random.fold_in(ordinal_key, i)
                order_size = len(self.config.ordinal_orders[feature])

                # Generate ordinal data with slight bias toward middle values
                # Use beta distribution to create realistic ordinal patterns
                alpha, beta = 2.0, 2.0
                uniform_samples = jax.random.beta(feature_key, alpha, beta, (self.num_samples,))
                feature_data = jnp.floor(uniform_samples * order_size).astype(jnp.int32)
                # Ensure values are in valid range
                feature_data = jnp.clip(feature_data, 0, order_size - 1)
                data[feature] = feature_data

        # Generate binary features
        if self.config.binary_features:
            binary_key = subkeys[3]
            for i, feature in enumerate(self.config.binary_features):
                feature_key = jax.random.fold_in(binary_key, i)

                # Generate binary data with slightly unbalanced distribution
                prob = 0.3 + 0.4 * (i % 2)  # Alternate between 0.3 and 0.7
                feature_data = jax.random.bernoulli(feature_key, prob, (self.num_samples,))
                data[feature] = feature_data.astype(jnp.int32)

        return data

    def __len__(self) -> int:
        """Return dataset size."""
        return self.num_samples

    def __iter__(self):
        """Iterate over dataset samples."""
        for i in range(self.num_samples):
            yield self.__getitem__(i)

    def __getitem__(self, idx: int) -> dict[str, jnp.ndarray]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary mapping feature names to single sample values
        """
        sample = {}
        for feature_name, feature_data in self.data.items():
            sample[feature_name] = feature_data[idx]
        return sample

    def get_batch(self, batch_size: int | None = None) -> dict[str, jnp.ndarray]:
        """Get a batch of samples.

        Args:
            batch_size: Size of batch (if None, returns full dataset)

        Returns:
            Dictionary mapping feature names to batch data
        """
        if batch_size is None:
            return self.data

        # Generate random indices
        key = self.rngs.sample()

        indices = jax.random.choice(key, self.num_samples, (batch_size,), replace=True)

        batch = {}
        for feature_name, feature_data in self.data.items():
            batch[feature_name] = feature_data[indices]
        return batch

    def get_feature_statistics(self) -> dict[str, dict[str, Any]]:
        """Get statistics about features.

        Returns:
            Dictionary mapping feature names to their statistics
        """
        stats = {}

        for feature in self.config.numerical_features:
            feature_data = self.data[feature]
            stats[feature] = {
                "type": "numerical",
                "mean": float(jnp.mean(feature_data)),
                "std": float(jnp.std(feature_data)),
                "min": float(jnp.min(feature_data)),
                "max": float(jnp.max(feature_data)),
            }

        for feature in self.config.categorical_features:
            feature_data = self.data[feature]
            vocab_size = self.config.categorical_vocab_sizes[feature]
            # Calculate frequency of each category
            counts = jnp.bincount(feature_data, length=vocab_size)
            stats[feature] = {
                "type": "categorical",
                "vocab_size": vocab_size,
                "counts": counts.tolist(),
                "frequencies": (counts / self.num_samples).tolist(),
            }

        for feature in self.config.ordinal_features:
            feature_data = self.data[feature]
            order_size = len(self.config.ordinal_orders[feature])
            counts = jnp.bincount(feature_data, length=order_size)
            stats[feature] = {
                "type": "ordinal",
                "order_size": order_size,
                "order": self.config.ordinal_orders[feature],
                "counts": counts.tolist(),
                "frequencies": (counts / self.num_samples).tolist(),
            }

        for feature in self.config.binary_features:
            feature_data = self.data[feature]
            positive_rate = float(jnp.mean(feature_data))
            stats[feature] = {
                "type": "binary",
                "positive_rate": positive_rate,
                "negative_rate": 1.0 - positive_rate,
            }

        return stats


def create_synthetic_tabular_dataset(
    num_features: int = 10,
    num_samples: int = 1000,
    numerical_ratio: float = 0.4,
    categorical_ratio: float = 0.3,
    ordinal_ratio: float = 0.2,
    binary_ratio: float = 0.1,
    max_categorical_cardinality: int = 10,
    *,
    rngs: nnx.Rngs,
) -> tuple[SyntheticTabularDataset, TabularModalityConfig]:
    """Create a synthetic tabular dataset with mixed feature types.

    Args:
        num_features: Total number of features
        num_samples: Number of samples to generate
        numerical_ratio: Proportion of numerical features
        categorical_ratio: Proportion of categorical features
        ordinal_ratio: Proportion of ordinal features
        binary_ratio: Proportion of binary features
        max_categorical_cardinality: Maximum vocabulary size for categorical features
        rngs: Random number generators

    Returns:
        Tuple of (dataset, config)
    """
    # Validate ratios
    total_ratio = numerical_ratio + categorical_ratio + ordinal_ratio + binary_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Feature ratios must sum to 1.0, got {total_ratio}")

    # Calculate number of features for each type
    num_numerical = int(num_features * numerical_ratio)
    num_categorical = int(num_features * categorical_ratio)
    num_ordinal = int(num_features * ordinal_ratio)
    num_binary = num_features - num_numerical - num_categorical - num_ordinal

    # Generate feature names
    numerical_features = [f"num_{i}" for i in range(num_numerical)]
    categorical_features = [f"cat_{i}" for i in range(num_categorical)]
    ordinal_features = [f"ord_{i}" for i in range(num_ordinal)]
    binary_features = [f"bin_{i}" for i in range(num_binary)]

    # Generate categorical vocab sizes
    categorical_vocab_sizes = {}
    if "config" in rngs:
        key = rngs.config()
    else:
        key = jax.random.key(123)

    for i, feature in enumerate(categorical_features):
        feature_key = jax.random.fold_in(key, i)
        vocab_size = jax.random.randint(feature_key, (), 2, max_categorical_cardinality + 1)
        categorical_vocab_sizes[feature] = int(vocab_size)

    # Generate ordinal orders
    ordinal_orders = {}
    for i, feature in enumerate(ordinal_features):
        # Create realistic ordinal categories
        if i % 3 == 0:
            # Rating scale
            ordinal_orders[feature] = ["poor", "fair", "good", "excellent"]
        elif i % 3 == 1:
            # Size scale
            ordinal_orders[feature] = ["small", "medium", "large"]
        else:
            # Education level
            ordinal_orders[feature] = ["high_school", "bachelor", "master", "phd"]

    # Create configuration
    config = TabularModalityConfig(
        num_features=num_features,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        ordinal_features=ordinal_features,
        binary_features=binary_features,
        categorical_vocab_sizes=categorical_vocab_sizes,
        ordinal_orders=ordinal_orders,
        normalization_type="standard",
        handle_missing="impute",
        max_categorical_cardinality=max_categorical_cardinality,
    )

    # Create dataset
    dataset = SyntheticTabularDataset(
        config=config,
        num_samples=num_samples,
        rngs=rngs,
    )

    return dataset, config


def create_simple_tabular_dataset(
    num_samples: int = 500,
    split: str = "train",
    *,
    rngs: nnx.Rngs,
) -> tuple[SyntheticTabularDataset, TabularModalityConfig]:
    """Create a simple tabular dataset for testing.

    Args:
        num_samples: Number of samples to generate
        split: Dataset split
        rngs: Random number generators

    Returns:
        Tuple of (dataset, config)
    """
    config = TabularModalityConfig(
        num_features=5,
        numerical_features=["age", "income"],
        categorical_features=["category"],
        ordinal_features=["education"],
        binary_features=["is_member"],
        categorical_vocab_sizes={"category": 4},
        ordinal_orders={"education": ["high_school", "bachelor", "master", "phd"]},
        normalization_type="standard",
        handle_missing="impute",
        max_categorical_cardinality=10,
    )

    dataset = SyntheticTabularDataset(
        config=config,
        num_samples=num_samples,
        split=split,
        rngs=rngs,
    )

    return dataset, config
