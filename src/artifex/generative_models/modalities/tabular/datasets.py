"""Tabular datasets backed by datarax MemorySource.

Provides pure data generation functions and factory functions that wrap
generated data in datarax MemorySource for pipeline integration.
"""

from typing import Any

import jax
import jax.numpy as jnp
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx

from .base import TabularModalityConfig


# ---------------------------------------------------------------------------
# Data generation (pure functions)
# ---------------------------------------------------------------------------


def generate_synthetic_tabular_data(
    modality_config: TabularModalityConfig,
    num_samples: int,
    *,
    key: jax.Array | None = None,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic tabular data with mixed feature types.

    Args:
        modality_config: Tabular modality configuration with feature definitions.
        num_samples: Number of samples to generate.
        key: Optional RNG key. If None, uses jax.random.key(0).

    Returns:
        Dictionary mapping feature names to data arrays.
    """
    mc = modality_config.validate_feature_consistency()
    data: dict[str, jnp.ndarray] = {}

    if key is None:
        key = jax.random.key(0)

    key, *subkeys = jax.random.split(key, 5)

    if mc.numerical_features:
        numerical_key = subkeys[0]
        for i, feature in enumerate(mc.numerical_features):
            feature_key = jax.random.fold_in(numerical_key, i)
            if i % 3 == 0:
                feature_data = jax.random.normal(feature_key, (num_samples,))
            elif i % 3 == 1:
                feature_data = jax.random.uniform(
                    feature_key,
                    (num_samples,),
                    minval=-2.0,
                    maxval=2.0,
                )
            else:
                feature_data = jax.random.exponential(feature_key, (num_samples,))
            data[feature] = feature_data

    if mc.categorical_features:
        categorical_key = subkeys[1]
        for i, feature in enumerate(mc.categorical_features):
            feature_key = jax.random.fold_in(categorical_key, i)
            vocab_size = mc.categorical_vocab_sizes[feature]
            data[feature] = jax.random.randint(feature_key, (num_samples,), 0, vocab_size)

    if mc.ordinal_features:
        ordinal_key = subkeys[2]
        for i, feature in enumerate(mc.ordinal_features):
            feature_key = jax.random.fold_in(ordinal_key, i)
            order_size = len(mc.ordinal_orders[feature])
            alpha, beta = 2.0, 2.0
            uniform_samples = jax.random.beta(feature_key, alpha, beta, (num_samples,))
            feature_data = jnp.floor(uniform_samples * order_size).astype(jnp.int32)
            data[feature] = jnp.clip(feature_data, 0, order_size - 1)

    if mc.binary_features:
        binary_key = subkeys[3]
        for i, feature in enumerate(mc.binary_features):
            feature_key = jax.random.fold_in(binary_key, i)
            prob = 0.3 + 0.4 * (i % 2)
            feature_data = jax.random.bernoulli(feature_key, prob, (num_samples,))
            data[feature] = feature_data.astype(jnp.int32)

    return data


def compute_feature_statistics(
    data: dict[str, jnp.ndarray],
    modality_config: TabularModalityConfig,
    num_samples: int,
) -> dict[str, dict[str, Any]]:
    """Compute statistics about tabular features.

    Args:
        data: Dictionary mapping feature names to data arrays.
        modality_config: Tabular modality configuration.
        num_samples: Number of samples in the dataset.

    Returns:
        Dictionary mapping feature names to their statistics.
    """
    stats: dict[str, dict[str, Any]] = {}
    mc = modality_config

    for feature in mc.numerical_features:
        feature_data = data[feature]
        stats[feature] = {
            "type": "numerical",
            "mean": float(jnp.mean(feature_data)),
            "std": float(jnp.std(feature_data)),
            "min": float(jnp.min(feature_data)),
            "max": float(jnp.max(feature_data)),
        }

    for feature in mc.categorical_features:
        feature_data = data[feature]
        vocab_size = mc.categorical_vocab_sizes[feature]
        counts = jnp.bincount(feature_data, length=vocab_size)
        stats[feature] = {
            "type": "categorical",
            "vocab_size": vocab_size,
            "counts": counts.tolist(),
            "frequencies": (counts / num_samples).tolist(),
        }

    for feature in mc.ordinal_features:
        feature_data = data[feature]
        order_size = len(mc.ordinal_orders[feature])
        counts = jnp.bincount(feature_data, length=order_size)
        stats[feature] = {
            "type": "ordinal",
            "order_size": order_size,
            "order": mc.ordinal_orders[feature],
            "counts": counts.tolist(),
            "frequencies": (counts / num_samples).tolist(),
        }

    for feature in mc.binary_features:
        feature_data = data[feature]
        positive_rate = float(jnp.mean(feature_data))
        stats[feature] = {
            "type": "binary",
            "positive_rate": positive_rate,
            "negative_rate": 1.0 - positive_rate,
        }

    return stats


# ---------------------------------------------------------------------------
# Factory functions — return MemorySource instances
# ---------------------------------------------------------------------------


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
    shuffle: bool = False,
) -> tuple[MemorySource, TabularModalityConfig]:
    """Create a synthetic tabular dataset with mixed feature types.

    Args:
        num_features: Total number of features.
        num_samples: Number of samples to generate.
        numerical_ratio: Proportion of numerical features.
        categorical_ratio: Proportion of categorical features.
        ordinal_ratio: Proportion of ordinal features.
        binary_ratio: Proportion of binary features.
        max_categorical_cardinality: Maximum vocabulary size for categorical.
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.

    Returns:
        Tuple of (MemorySource, TabularModalityConfig).

    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
    total_ratio = numerical_ratio + categorical_ratio + ordinal_ratio + binary_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Feature ratios must sum to 1.0, got {total_ratio}")

    num_numerical = int(num_features * numerical_ratio)
    num_categorical = int(num_features * categorical_ratio)
    num_ordinal = int(num_features * ordinal_ratio)
    num_binary = num_features - num_numerical - num_categorical - num_ordinal

    numerical_features = [f"num_{i}" for i in range(num_numerical)]
    categorical_features = [f"cat_{i}" for i in range(num_categorical)]
    ordinal_features = [f"ord_{i}" for i in range(num_ordinal)]
    binary_features = [f"bin_{i}" for i in range(num_binary)]

    categorical_vocab_sizes: dict[str, int] = {}
    if "config" in rngs:
        key = rngs.config()
    else:
        key = jax.random.key(123)

    for i, feature in enumerate(categorical_features):
        feature_key = jax.random.fold_in(key, i)
        vocab_size = jax.random.randint(feature_key, (), 2, max_categorical_cardinality + 1)
        categorical_vocab_sizes[feature] = int(vocab_size)

    ordinal_orders: dict[str, list[str]] = {}
    for i, feature in enumerate(ordinal_features):
        if i % 3 == 0:
            ordinal_orders[feature] = ["poor", "fair", "good", "excellent"]
        elif i % 3 == 1:
            ordinal_orders[feature] = ["small", "medium", "large"]
        else:
            ordinal_orders[feature] = [
                "high_school",
                "bachelor",
                "master",
                "phd",
            ]

    modality_config = TabularModalityConfig(
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

    if "tabular" in rngs:
        gen_key = rngs.tabular()
    else:
        gen_key = jax.random.key(0)

    data = generate_synthetic_tabular_data(modality_config, num_samples, key=gen_key)

    source_config = MemorySourceConfig(shuffle=shuffle)
    source = MemorySource(source_config, data, rngs=rngs)

    return source, modality_config


def create_simple_tabular_dataset(
    num_samples: int = 500,
    split: str = "train",
    *,
    rngs: nnx.Rngs,
    shuffle: bool = False,
) -> tuple[MemorySource, TabularModalityConfig]:
    """Create a simple tabular dataset for testing.

    Args:
        num_samples: Number of samples to generate.
        split: Dataset split (unused, kept for API compatibility).
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.

    Returns:
        Tuple of (MemorySource, TabularModalityConfig).
    """
    modality_config = TabularModalityConfig(
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

    if "tabular" in rngs:
        gen_key = rngs.tabular()
    else:
        gen_key = jax.random.key(0)

    data = generate_synthetic_tabular_data(modality_config, num_samples, key=gen_key)

    source_config = MemorySourceConfig(shuffle=shuffle)
    source = MemorySource(source_config, data, rngs=rngs)

    return source, modality_config
