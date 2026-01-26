"""Synthetic datasets for timeseries modality."""

from typing import Iterator

import jax
import jax.numpy as jnp
from flax import nnx

from ..base import BaseDataset


class SyntheticTimeseriesDataset(BaseDataset):
    """Synthetic timeseries dataset for testing and development.

    Generates time series with various patterns including:
    - Sinusoidal patterns with noise
    - Trend components
    - Seasonal patterns
    - Random walks
    - AR/MA processes
    """

    def __init__(
        self,
        config: dict,
        split: str = "train",
        sequence_length: int = 100,
        num_features: int = 1,
        num_samples: int = 1000,
        pattern_type: str = "sinusoidal",
        noise_level: float = 0.1,
        sampling_rate: float = 1.0,
        seasonal_period: int | None = None,
        trend_strength: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the synthetic timeseries dataset.

        Args:
            config: Dataset configuration
            split: Dataset split ('train', 'val', 'test')
            sequence_length: Length of each time series
            num_features: Number of features per timestep
            num_samples: Number of time series to generate
            pattern_type: Type of pattern ('sinusoidal', 'random_walk', 'ar', 'seasonal')
            noise_level: Standard deviation of noise to add
            sampling_rate: Sampling rate of the time series
            seasonal_period: Period for seasonal patterns
            trend_strength: Strength of linear trend component
            rngs: Random number generator keys
        """
        super().__init__(config, split=split, rngs=rngs)
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_samples = num_samples
        self.pattern_type = pattern_type
        self.noise_level = noise_level
        self.sampling_rate = sampling_rate
        self.seasonal_period = seasonal_period
        self.trend_strength = trend_strength

        # Validate parameters
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if noise_level < 0:
            raise ValueError("noise_level must be non-negative")

        # Generate the dataset
        key = rngs.sample()

        self._data = self._generate_data(key)

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Get a batch of samples.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Batch dictionary with timeseries data
        """
        # Generate random indices for the batch
        key = jax.random.key(42)  # TODO: Use proper RNG from self.rngs
        indices = jax.random.choice(key, self.num_samples, shape=(batch_size,), replace=True)

        batch_data = self._data[indices]

        return {
            "timeseries": batch_data,
            "sequence_length": jnp.array(self.sequence_length),
            "num_features": jnp.array(self.num_features),
        }

    def get_sample(self, index: int) -> dict[str, jax.Array]:
        """Get a single sample by index.

        Args:
            index: Sample index

        Returns:
            Sample data
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        return {
            "timeseries": self._data[index],
            "sequence_length": jnp.array(self.sequence_length),
            "num_features": jnp.array(self.num_features),
        }

    def _generate_data(self, key: jax.Array) -> jnp.ndarray:
        """Generate synthetic timeseries data.

        Args:
            key: Random key for generation

        Returns:
            Generated timeseries data of shape (num_samples, sequence_length, num_features)
        """
        keys = jax.random.split(key, self.num_samples)

        def generate_single_series(series_key):
            return self._generate_single_timeseries(series_key)

        # Generate all series
        data = jax.vmap(generate_single_series)(keys)

        return data

    def _generate_single_timeseries(self, key: jax.Array) -> jnp.ndarray:
        """Generate a single timeseries.

        Args:
            key: Random key for this series

        Returns:
            Single timeseries of shape (sequence_length, num_features)
        """
        keys = jax.random.split(key, 4)

        # Time indices
        t = jnp.arange(self.sequence_length, dtype=jnp.float32)

        if self.pattern_type == "sinusoidal":
            data = self._generate_sinusoidal(t, keys[0])
        elif self.pattern_type == "random_walk":
            data = self._generate_random_walk(keys[0])
        elif self.pattern_type == "ar":
            data = self._generate_ar_process(keys[0])
        elif self.pattern_type == "seasonal":
            data = self._generate_seasonal(t, keys[0])
        elif self.pattern_type == "mixed":
            data = self._generate_mixed_pattern(t, keys[0])
        else:
            raise ValueError(f"Unknown pattern type: {self.pattern_type}")

        # Add trend component
        if self.trend_strength > 0:
            trend = self.trend_strength * t[:, None] / self.sequence_length
            data = data + trend

        # Add noise
        if self.noise_level > 0:
            noise = (
                jax.random.normal(keys[1], (self.sequence_length, self.num_features))
                * self.noise_level
            )
            data = data + noise

        return data

    def _generate_sinusoidal(self, t: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        """Generate sinusoidal patterns.

        Args:
            t: Time indices
            key: Random key

        Returns:
            Sinusoidal timeseries
        """
        keys = jax.random.split(key, self.num_features + 2)

        # Random frequencies and phases for each feature
        frequencies = jax.random.uniform(keys[0], (self.num_features,), minval=0.1, maxval=2.0)
        phases = jax.random.uniform(keys[1], (self.num_features,), minval=0, maxval=2 * jnp.pi)

        # Generate sinusoidal patterns
        t_expanded = t[:, None]  # Shape: (sequence_length, 1)
        frequencies_expanded = frequencies[None, :]  # Shape: (1, num_features)
        phases_expanded = phases[None, :]  # Shape: (1, num_features)

        data = jnp.sin(2 * jnp.pi * frequencies_expanded * t_expanded + phases_expanded)

        return data

    def _generate_random_walk(self, key: jax.Array) -> jnp.ndarray:
        """Generate random walk patterns.

        Args:
            key: Random key

        Returns:
            Random walk timeseries
        """
        # Generate random steps
        steps = jax.random.normal(key, (self.sequence_length, self.num_features))

        # Compute cumulative sum for random walk
        data = jnp.cumsum(steps, axis=0)

        return data

    def _generate_ar_process(self, key: jax.Array) -> jnp.ndarray:
        """Generate AR(1) process.

        Args:
            key: Random key

        Returns:
            AR process timeseries
        """
        keys = jax.random.split(key, 3)

        # AR coefficients (between -0.9 and 0.9 for stability)
        ar_coefs = jax.random.uniform(keys[0], (self.num_features,), minval=-0.9, maxval=0.9)

        # Initialize the series
        data = jnp.zeros((self.sequence_length, self.num_features))

        # Initial value
        initial_value = jax.random.normal(keys[1], (self.num_features,))
        data = data.at[0].set(initial_value)

        # Generate noise for the entire series
        noise = jax.random.normal(keys[2], (self.sequence_length - 1, self.num_features))

        # Generate AR process iteratively
        for i in range(1, self.sequence_length):
            data = data.at[i].set(ar_coefs * data[i - 1] + noise[i - 1])

        return data

    def _generate_seasonal(self, t: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        """Generate seasonal patterns.

        Args:
            t: Time indices
            key: Random key

        Returns:
            Seasonal timeseries
        """
        if self.seasonal_period is None:
            period = self.sequence_length // 4  # Default to 1/4 of sequence length
        else:
            period = self.seasonal_period

        keys = jax.random.split(key, self.num_features + 1)

        # Random amplitudes for each feature
        amplitudes = jax.random.uniform(keys[0], (self.num_features,), minval=0.5, maxval=2.0)

        # Generate seasonal pattern
        t_expanded = t[:, None]
        amplitudes_expanded = amplitudes[None, :]

        seasonal_component = amplitudes_expanded * jnp.sin(2 * jnp.pi * t_expanded / period)

        # Add some harmonics for more complex seasonal patterns
        harmonics = (
            amplitudes_expanded * 0.3 * jnp.sin(4 * jnp.pi * t_expanded / period + jnp.pi / 4)
        )

        data = seasonal_component + harmonics

        return data

    def _generate_mixed_pattern(self, t: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        """Generate mixed patterns combining multiple components.

        Args:
            t: Time indices
            key: Random key

        Returns:
            Mixed pattern timeseries
        """
        keys = jax.random.split(key, 4)

        # Combine different patterns
        sinusoidal = self._generate_sinusoidal(t, keys[0]) * 0.4
        seasonal = self._generate_seasonal(t, keys[1]) * 0.3

        # Add some random walk component
        rw_steps = jax.random.normal(keys[2], (self.sequence_length, self.num_features)) * 0.1
        random_walk = jnp.cumsum(rw_steps, axis=0) * 0.3

        data = sinusoidal + seasonal + random_walk

        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return self.num_samples

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over the dataset."""
        for i in range(self.num_samples):
            yield {
                "timeseries": self._data[i],
                "sequence_length": jnp.array(self.sequence_length),
                "num_features": jnp.array(self.num_features),
            }

    def __getitem__(self, idx: int) -> jnp.ndarray:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Timeseries sample of shape (sequence_length, num_features)
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        return self._data[idx]

    def batch_iterator(self, batch_size: int) -> Iterator[jnp.ndarray]:
        """Create batched iterator over the dataset.

        Args:
            batch_size: Size of each batch

        Yields:
            Batches of timeseries data of shape (batch_size, sequence_length, num_features)
        """
        for i in range(0, self.num_samples, batch_size):
            end_idx = min(i + batch_size, self.num_samples)
            yield self._data[i:end_idx]

    def get_statistics(self) -> dict[str, jnp.ndarray]:
        """Get statistical information about the dataset.

        Returns:
            Dictionary with statistical measures
        """
        return {
            "mean": jnp.mean(self._data, axis=(0, 1)),
            "std": jnp.std(self._data, axis=(0, 1)),
            "min": jnp.min(self._data, axis=(0, 1)),
            "max": jnp.max(self._data, axis=(0, 1)),
            "sequence_length": self.sequence_length,
            "num_features": self.num_features,
            "num_samples": self.num_samples,
        }


def create_synthetic_timeseries_dataset(
    sequence_length: int = 100,
    num_features: int = 1,
    num_samples: int = 1000,
    pattern_type: str = "sinusoidal",
    noise_level: float = 0.1,
    **kwargs,
) -> SyntheticTimeseriesDataset:
    """Factory function to create a synthetic timeseries dataset.

    Args:
        sequence_length: Length of each time series
        num_features: Number of features per timestep
        num_samples: Number of time series to generate
        pattern_type: Type of pattern to generate
        noise_level: Level of noise to add
        **kwargs: Additional arguments passed to the dataset

    Returns:
        Synthetic timeseries dataset
    """
    rngs = nnx.Rngs(42)  # Default seed

    return SyntheticTimeseriesDataset(
        sequence_length=sequence_length,
        num_features=num_features,
        num_samples=num_samples,
        pattern_type=pattern_type,
        noise_level=noise_level,
        rngs=rngs,
        **kwargs,
    )


def create_simple_timeseries_dataset(
    sequence_length: int = 50,
    num_samples: int = 100,
    **kwargs,
) -> SyntheticTimeseriesDataset:
    """Factory function to create a simple timeseries dataset for testing.

    Args:
        sequence_length: Length of each time series
        num_samples: Number of time series to generate
        **kwargs: Additional arguments

    Returns:
        Simple synthetic timeseries dataset
    """
    return create_synthetic_timeseries_dataset(
        sequence_length=sequence_length,
        num_features=1,
        num_samples=num_samples,
        pattern_type="sinusoidal",
        noise_level=0.05,
        **kwargs,
    )
