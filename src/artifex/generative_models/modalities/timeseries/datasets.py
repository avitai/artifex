"""Timeseries datasets backed by datarax MemorySource.

Provides pure data generation functions and factory functions that wrap
generated data in datarax MemorySource for pipeline integration.
"""

from typing import Any

import jax
import jax.numpy as jnp
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx


# ---------------------------------------------------------------------------
# Data generation (pure functions)
# ---------------------------------------------------------------------------


def generate_synthetic_timeseries(
    num_samples: int,
    *,
    sequence_length: int = 100,
    num_features: int = 1,
    pattern_type: str = "sinusoidal",
    noise_level: float = 0.1,
    trend_strength: float = 0.0,
    seasonal_period: int | None = None,
    key: jax.Array | None = None,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic timeseries data.

    Args:
        num_samples: Number of time series to generate.
        sequence_length: Length of each time series.
        num_features: Number of features per timestep.
        pattern_type: Type of pattern ('sinusoidal', 'random_walk',
            'ar', 'seasonal', 'mixed').
        noise_level: Standard deviation of noise to add.
        trend_strength: Strength of linear trend component.
        seasonal_period: Period for seasonal patterns.
        key: Optional RNG key. If None, uses jax.random.key(0).

    Returns:
        Dictionary with 'timeseries' array of shape
        (num_samples, sequence_length, num_features).

    Raises:
        ValueError: If sequence_length, num_features, or num_samples is non-positive.
        ValueError: If noise_level is negative.
        ValueError: If pattern_type is unknown.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    if key is None:
        key = jax.random.key(0)

    keys = jax.random.split(key, num_samples)

    def generate_single(series_key: jax.Array) -> jnp.ndarray:
        return _generate_single_timeseries(
            series_key,
            sequence_length=sequence_length,
            num_features=num_features,
            pattern_type=pattern_type,
            noise_level=noise_level,
            trend_strength=trend_strength,
            seasonal_period=seasonal_period,
        )

    data = jax.vmap(generate_single)(keys)
    return {"timeseries": data}


def _generate_single_timeseries(
    key: jax.Array,
    *,
    sequence_length: int,
    num_features: int,
    pattern_type: str,
    noise_level: float,
    trend_strength: float,
    seasonal_period: int | None,
) -> jnp.ndarray:
    """Generate a single timeseries.

    Args:
        key: Random key for this series.
        sequence_length: Length of the series.
        num_features: Number of features.
        pattern_type: Type of pattern.
        noise_level: Noise level.
        trend_strength: Trend strength.
        seasonal_period: Seasonal period.

    Returns:
        Single timeseries of shape (sequence_length, num_features).
    """
    keys = jax.random.split(key, 4)
    t = jnp.arange(sequence_length, dtype=jnp.float32)

    if pattern_type == "sinusoidal":
        data = _generate_sinusoidal(t, keys[0], num_features)
    elif pattern_type == "random_walk":
        data = _generate_random_walk(keys[0], sequence_length, num_features)
    elif pattern_type == "ar":
        data = _generate_ar_process(keys[0], sequence_length, num_features)
    elif pattern_type == "seasonal":
        data = _generate_seasonal(t, keys[0], num_features, sequence_length, seasonal_period)
    elif pattern_type == "mixed":
        data = _generate_mixed_pattern(t, keys[0], num_features, sequence_length, seasonal_period)
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    if trend_strength > 0:
        trend = trend_strength * t[:, None] / sequence_length
        data = data + trend

    if noise_level > 0:
        noise = jax.random.normal(keys[1], (sequence_length, num_features)) * noise_level
        data = data + noise

    return data


def _generate_sinusoidal(t: jnp.ndarray, key: jax.Array, num_features: int) -> jnp.ndarray:
    """Generate sinusoidal patterns."""
    keys = jax.random.split(key, num_features + 2)
    frequencies = jax.random.uniform(keys[0], (num_features,), minval=0.1, maxval=2.0)
    phases = jax.random.uniform(keys[1], (num_features,), minval=0, maxval=2 * jnp.pi)
    t_expanded = t[:, None]
    return jnp.sin(2 * jnp.pi * frequencies[None, :] * t_expanded + phases[None, :])


def _generate_random_walk(key: jax.Array, sequence_length: int, num_features: int) -> jnp.ndarray:
    """Generate random walk patterns."""
    steps = jax.random.normal(key, (sequence_length, num_features))
    return jnp.cumsum(steps, axis=0)


def _generate_ar_process(key: jax.Array, sequence_length: int, num_features: int) -> jnp.ndarray:
    """Generate AR(1) process."""
    keys = jax.random.split(key, 3)
    ar_coefs = jax.random.uniform(keys[0], (num_features,), minval=-0.9, maxval=0.9)
    data = jnp.zeros((sequence_length, num_features))
    initial_value = jax.random.normal(keys[1], (num_features,))
    data = data.at[0].set(initial_value)
    noise = jax.random.normal(keys[2], (sequence_length - 1, num_features))
    for i in range(1, sequence_length):
        data = data.at[i].set(ar_coefs * data[i - 1] + noise[i - 1])
    return data


def _generate_seasonal(
    t: jnp.ndarray,
    key: jax.Array,
    num_features: int,
    sequence_length: int,
    seasonal_period: int | None,
) -> jnp.ndarray:
    """Generate seasonal patterns."""
    if seasonal_period is None:
        period = sequence_length // 4
    else:
        period = seasonal_period

    keys = jax.random.split(key, num_features + 1)
    amplitudes = jax.random.uniform(keys[0], (num_features,), minval=0.5, maxval=2.0)
    t_expanded = t[:, None]
    amplitudes_expanded = amplitudes[None, :]
    seasonal = amplitudes_expanded * jnp.sin(2 * jnp.pi * t_expanded / period)
    harmonics = amplitudes_expanded * 0.3 * jnp.sin(4 * jnp.pi * t_expanded / period + jnp.pi / 4)
    return seasonal + harmonics


def _generate_mixed_pattern(
    t: jnp.ndarray,
    key: jax.Array,
    num_features: int,
    sequence_length: int,
    seasonal_period: int | None,
) -> jnp.ndarray:
    """Generate mixed patterns combining multiple components."""
    keys = jax.random.split(key, 4)
    sinusoidal = _generate_sinusoidal(t, keys[0], num_features) * 0.4
    seasonal = _generate_seasonal(t, keys[1], num_features, sequence_length, seasonal_period) * 0.3
    rw_steps = jax.random.normal(keys[2], (sequence_length, num_features)) * 0.1
    random_walk = jnp.cumsum(rw_steps, axis=0) * 0.3
    return sinusoidal + seasonal + random_walk


# ---------------------------------------------------------------------------
# Factory functions — return MemorySource instances
# ---------------------------------------------------------------------------


def create_synthetic_timeseries_dataset(
    sequence_length: int = 100,
    num_features: int = 1,
    num_samples: int = 1000,
    pattern_type: str = "sinusoidal",
    noise_level: float = 0.1,
    *,
    rngs: nnx.Rngs | None = None,
    shuffle: bool = False,
    **kwargs: Any,
) -> Any:
    """Create a synthetic timeseries dataset as a MemorySource.

    Args:
        sequence_length: Length of each time series.
        num_features: Number of features per timestep.
        num_samples: Number of time series to generate.
        pattern_type: Type of pattern to generate.
        noise_level: Level of noise to add.
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.
        **kwargs: Additional parameters (trend_strength, seasonal_period).

    Returns:
        MemorySource backed by generated timeseries data.
    """
    if rngs is None:
        rngs = nnx.Rngs(42)

    key = rngs.sample() if rngs is not None else jax.random.key(0)
    data = generate_synthetic_timeseries(
        num_samples,
        sequence_length=sequence_length,
        num_features=num_features,
        pattern_type=pattern_type,
        noise_level=noise_level,
        key=key,
        **kwargs,
    )

    source_config = MemorySourceConfig(shuffle=shuffle)
    return MemorySource(source_config, data, rngs=rngs)


def create_simple_timeseries_dataset(
    sequence_length: int = 50,
    num_samples: int = 100,
    *,
    rngs: nnx.Rngs | None = None,
    shuffle: bool = False,
    **kwargs: Any,
) -> Any:
    """Create a simple timeseries dataset for testing.

    Args:
        sequence_length: Length of each time series.
        num_samples: Number of time series to generate.
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.
        **kwargs: Additional parameters.

    Returns:
        MemorySource backed by generated timeseries data.
    """
    return create_synthetic_timeseries_dataset(
        sequence_length=sequence_length,
        num_features=1,
        num_samples=num_samples,
        pattern_type="sinusoidal",
        noise_level=0.05,
        rngs=rngs,
        shuffle=shuffle,
        **kwargs,
    )
