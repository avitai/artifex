"""Temporal representations and processors for timeseries modality."""

import jax.numpy as jnp
from flax import nnx

from ..base import BaseProcessor
from .base import TimeseriesModalityConfig


class TimeseriesProcessor(BaseProcessor):
    """Base processor for timeseries data.

    Provides common functionality for temporal data processing
    including normalization, windowing, and basic transformations.
    """

    def __init__(
        self,
        config: TimeseriesModalityConfig,
        sequence_length: int,
        num_features: int,
        normalize: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the timeseries processor.

        Args:
            config: Timeseries modality configuration
            sequence_length: Length of the time series
            num_features: Number of features per timestep
            normalize: Whether to apply normalization
            rngs: Random number generator keys
        """
        super().__init__(config, rngs=rngs)
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.normalize = normalize

    def process(self, data: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Process input data.

        Args:
            data: Input timeseries data of shape (batch, sequence, features)
            **kwargs: Additional processing parameters

        Returns:
            Processed timeseries data
        """
        if data.ndim != 3:
            raise ValueError(f"Expected 3D input, got {data.ndim}D")

        batch_size, seq_len, num_feat = data.shape

        if seq_len != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {seq_len}")

        if num_feat != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {num_feat}")

        # Apply normalization if enabled
        if self.normalize:
            # Compute statistics across the sequence dimension
            mean = jnp.mean(data, axis=1, keepdims=True)
            std = jnp.std(data, axis=1, keepdims=True)
            # Avoid division by zero
            std = jnp.where(std == 0, 1.0, std)
            data = (data - mean) / std

        return data

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        """Process timeseries data.

        Args:
            x: Input timeseries data of shape (batch, sequence, features)
            deterministic: Whether to use deterministic processing

        Returns:
            Processed timeseries data
        """
        return self.process(x, deterministic=deterministic)

    def reverse(self, x: jnp.ndarray) -> jnp.ndarray:
        """Reverse the processing (for generation tasks).

        Args:
            x: Processed timeseries data

        Returns:
            Original-scale timeseries data
        """
        # Base implementation just returns the input
        # Subclasses can override for more complex reverse transformations
        return x


class FourierProcessor(nnx.Module):
    """Fourier feature processor for timeseries data.

    Computes Fourier features to capture frequency domain information
    in time series data, useful for models that need to understand
    periodic patterns and frequency content.
    """

    def __init__(
        self,
        num_frequencies: int = 64,
        max_frequency: float = 100.0,
        include_original: bool = True,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the Fourier processor.

        Args:
            num_frequencies: Number of frequency components to include
            max_frequency: Maximum frequency to include
            include_original: Whether to include original time domain data
            rngs: Random number generator keys
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.max_frequency = max_frequency
        self.include_original = include_original

        # Create frequency basis
        frequencies = jnp.linspace(0, max_frequency, num_frequencies)
        self.frequencies = nnx.Param(frequencies, trainable=False)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        """Compute Fourier features for timeseries data.

        Args:
            x: Input timeseries data of shape (batch, sequence, features)
            deterministic: Whether to use deterministic processing

        Returns:
            Fourier-transformed timeseries data
        """
        batch_size, seq_len, num_features = x.shape

        # Create time indices
        t = jnp.arange(seq_len, dtype=jnp.float32)
        t = t[None, :, None]  # Shape: (1, sequence, 1)

        # Compute Fourier features for each frequency
        frequencies = self.frequencies.value[:, None, None]  # Shape: (num_freq, 1, 1)
        phase = 2 * jnp.pi * frequencies * t / seq_len  # Broadcasting

        # Compute sine and cosine features
        sin_features = jnp.sin(phase)  # Shape: (num_freq, sequence, 1)
        cos_features = jnp.cos(phase)  # Shape: (num_freq, sequence, 1)

        # Repeat for all features
        sin_features = jnp.tile(sin_features, (1, 1, num_features))
        cos_features = jnp.tile(cos_features, (1, 1, num_features))

        # Combine sine and cosine features
        fourier_features = jnp.concatenate([sin_features, cos_features], axis=0)
        # Shape: (2*num_freq, sequence, num_features)

        # Transpose to (sequence, 2*num_freq*num_features)
        fourier_features = jnp.transpose(fourier_features, (1, 0, 2))
        fourier_features = fourier_features.reshape(seq_len, -1)

        # Expand for batch dimension
        fourier_features = jnp.tile(fourier_features[None, :, :], (batch_size, 1, 1))

        if self.include_original:
            # Concatenate with original data
            return jnp.concatenate([x, fourier_features], axis=-1)
        else:
            return fourier_features


class MultiScaleProcessor(nnx.Module):
    """Multi-scale temporal representation processor.

    Creates multiple temporal resolutions of the input data
    to capture patterns at different time scales.
    """

    def __init__(
        self,
        scale_factors: list[int] | None = None,
        aggregation_method: str = "mean",
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the multi-scale processor.

        Args:
            scale_factors: List of downsampling factors for different scales
            aggregation_method: Method for aggregating ('mean', 'max', 'sum')
            rngs: Random number generator keys
        """
        super().__init__()
        if scale_factors is None:
            scale_factors = [1, 2, 4, 8]
        self.scale_factors = scale_factors
        self.aggregation_method = aggregation_method

        if aggregation_method not in ["mean", "max", "sum"]:
            raise ValueError(f"Invalid aggregation method: {aggregation_method}")

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool = False,
    ) -> dict[str, jnp.ndarray]:
        """Create multi-scale representations.

        Args:
            x: Input timeseries data of shape (batch, sequence, features)
            deterministic: Whether to use deterministic processing

        Returns:
            Dictionary mapping scale names to downsampled representations
        """
        batch_size, seq_len, num_features = x.shape
        representations = {}

        for scale in self.scale_factors:
            if scale == 1:
                # Original resolution
                representations[f"scale_{scale}"] = x
            else:
                # Downsample by the scale factor
                if seq_len % scale != 0:
                    # Pad sequence to make it divisible by scale
                    pad_size = scale - (seq_len % scale)
                    x_padded = jnp.pad(x, ((0, 0), (0, pad_size), (0, 0)), mode="edge")
                else:
                    x_padded = x

                padded_len = x_padded.shape[1]
                target_len = padded_len // scale

                # Reshape for aggregation
                x_reshaped = x_padded.reshape(batch_size, target_len, scale, num_features)

                # Apply aggregation
                if self.aggregation_method == "mean":
                    x_downsampled = jnp.mean(x_reshaped, axis=2)
                elif self.aggregation_method == "max":
                    x_downsampled = jnp.max(x_reshaped, axis=2)
                elif self.aggregation_method == "sum":
                    x_downsampled = jnp.sum(x_reshaped, axis=2)

                representations[f"scale_{scale}"] = x_downsampled

        return representations

    def reconstruct(self, representations: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Reconstruct original timeseries from multi-scale representations.

        Args:
            representations: Multi-scale representations

        Returns:
            Reconstructed timeseries data
        """
        # Use the finest scale (scale_1) as the base reconstruction
        if "scale_1" in representations:
            return representations["scale_1"]

        # If scale_1 is not available, use the finest available scale
        available_scales = sorted([int(k.split("_")[1]) for k in representations.keys()])
        finest_scale = available_scales[0]
        finest_repr = representations[f"scale_{finest_scale}"]

        if finest_scale == 1:
            return finest_repr

        # # Upsample the finest available representation
        # batch_size, seq_len, num_features = finest_repr.shape
        # target_len = seq_len * finest_scale

        # Simple upsampling by repetition
        upsampled = jnp.repeat(finest_repr, finest_scale, axis=1)

        return upsampled


class TrendDecompositionProcessor(nnx.Module):
    """Trend and seasonal decomposition processor.

    Decomposes time series into trend, seasonal, and residual components
    using moving averages and seasonal patterns.
    """

    def __init__(
        self,
        period: int = 24,
        method: str = "seasonal",
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the trend decomposition processor.

        Args:
            period: Period for seasonal decomposition
            method: Decomposition method ('seasonal', 'moving_average')
            rngs: Random number generator keys
        """
        super().__init__()
        self.period = period
        self.method = method

        if method not in ["seasonal", "moving_average"]:
            raise ValueError(f"Invalid decomposition method: {method}")

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool = False,
    ) -> dict[str, jnp.ndarray]:
        """Decompose timeseries into components.

        Args:
            x: Input timeseries data of shape (batch, sequence, features)
            deterministic: Whether to use deterministic processing

        Returns:
            Dictionary with 'trend', 'seasonal', and 'residual' components
        """
        batch_size, seq_len, num_features = x.shape

        if self.method == "moving_average":
            # Simple moving average for trend
            trend = self._moving_average(x, window_size=self.period)
            detrended = x - trend

            # Simple seasonal pattern extraction
            seasonal = self._extract_seasonal_pattern(detrended)
            residual = detrended - seasonal

        elif self.method == "seasonal":
            # Seasonal decomposition using period-based averaging
            trend = self._moving_average(x, window_size=self.period)
            detrended = x - trend
            seasonal = self._seasonal_decomposition(detrended)
            residual = detrended - seasonal

        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "original": x,
        }

    def _moving_average(self, x: jnp.ndarray, window_size: int) -> jnp.ndarray:
        """Compute moving average for trend extraction.

        Args:
            x: Input data
            window_size: Size of the moving window

        Returns:
            Moving average (trend component)
        """
        if window_size >= x.shape[1]:
            # If window is too large, return global mean
            return jnp.full_like(x, jnp.mean(x, axis=1, keepdims=True))

        batch_size, seq_len, num_features = x.shape

        # Simple moving average with proper shape preservation
        trend = jnp.zeros_like(x)
        half_window = window_size // 2

        for i in range(seq_len):
            start_idx = max(0, i - half_window)
            end_idx = min(seq_len, i + half_window + 1)
            trend = trend.at[:, i, :].set(jnp.mean(x[:, start_idx:end_idx, :], axis=1))

        return trend

    def _extract_seasonal_pattern(self, x: jnp.ndarray) -> jnp.ndarray:
        """Extract seasonal pattern from detrended data.

        Args:
            x: Detrended data

        Returns:
            Seasonal component
        """
        batch_size, seq_len, num_features = x.shape

        if self.period >= seq_len:
            # No seasonal pattern if period is too large
            return jnp.zeros_like(x)

        # Create seasonal indices
        seasonal_indices = jnp.arange(seq_len) % self.period

        # Compute seasonal pattern by averaging over cycles
        seasonal_pattern = jnp.zeros((batch_size, self.period, num_features))

        for i in range(self.period):
            mask = seasonal_indices == i
            if jnp.sum(mask) > 0:
                seasonal_pattern = seasonal_pattern.at[:, i, :].set(jnp.mean(x[:, mask, :], axis=1))

        # Repeat seasonal pattern to match sequence length
        num_cycles = seq_len // self.period
        remainder = seq_len % self.period

        seasonal = jnp.tile(seasonal_pattern, (1, num_cycles, 1))
        if remainder > 0:
            seasonal = jnp.concatenate([seasonal, seasonal_pattern[:, :remainder, :]], axis=1)

        return seasonal

    def _seasonal_decomposition(self, x: jnp.ndarray) -> jnp.ndarray:
        """Perform seasonal decomposition.

        Args:
            x: Input data

        Returns:
            Seasonal component
        """
        # For now, use the same method as extract_seasonal_pattern
        # In a full implementation, this could use more sophisticated methods
        return self._extract_seasonal_pattern(x)

    def reconstruct(self, components: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Reconstruct timeseries from decomposed components.

        Args:
            components: Dictionary with trend, seasonal, and residual components

        Returns:
            Reconstructed timeseries
        """
        if "original" in components:
            return components["original"]

        # Reconstruct by adding components
        reconstructed = jnp.zeros_like(components["trend"])

        if "trend" in components:
            reconstructed += components["trend"]
        if "seasonal" in components:
            reconstructed += components["seasonal"]
        if "residual" in components:
            reconstructed += components["residual"]

        return reconstructed
