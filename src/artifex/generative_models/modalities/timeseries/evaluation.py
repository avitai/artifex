"""Evaluation metrics for timeseries modality."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.evaluation.metrics.statistical import (
    _compute_autocorrelation,
    _compute_skewness,
)

from ..base import BaseEvaluationSuite
from .base import TimeseriesModalityConfig


class TimeseriesEvaluationSuite(BaseEvaluationSuite):
    """Comprehensive evaluation suite for timeseries generation.

    Provides temporal-specific metrics including:
    - Dynamic Time Warping (DTW) distance
    - Autocorrelation analysis
    - Spectral analysis
    - Trend preservation
    - Temporal consistency metrics
    """

    def __init__(
        self,
        config: TimeseriesModalityConfig,
        sequence_length: int,
        num_features: int = 1,
        max_lag: int | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the timeseries evaluation suite.

        Args:
            config: Evaluation configuration
            sequence_length: Length of the time series
            num_features: Number of features per timestep
            max_lag: Maximum lag for autocorrelation analysis
            rngs: Random number generator keys
        """
        if rngs is None:
            rngs = nnx.Rngs(42)
        super().__init__(config, rngs=rngs)
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.max_lag = max_lag if max_lag is not None else min(50, sequence_length // 4)

    def evaluate_batch(
        self,
        generated_data: jax.Array,
        reference_data: jax.Array | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate a batch of generated data.

        Args:
            generated_data: Generated data to evaluate
            reference_data: Reference data for comparison (optional)
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary of evaluation metrics
        """
        if reference_data is not None:
            return self.compute_metrics(reference_data, generated_data, **kwargs)
        else:
            # Compute quality metrics without reference
            return self.compute_quality_metrics(generated_data)

    def compute_metrics(
        self,
        real_data: jnp.ndarray,
        generated_data: jnp.ndarray,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute comprehensive timeseries evaluation metrics.

        Args:
            real_data: Real timeseries data of shape (batch, sequence, features)
            generated_data: Generated timeseries data of same shape
            **kwargs: Additional arguments

        Returns:
            Dictionary of evaluation metrics
        """
        if real_data.shape != generated_data.shape:
            raise ValueError(
                f"Shape mismatch: real_data {real_data.shape} "
                f"vs generated_data {generated_data.shape}"
            )

        metrics = {}

        # Basic distance metrics
        metrics.update(self._compute_basic_metrics(real_data, generated_data))

        # Temporal-specific metrics
        metrics.update(self._compute_temporal_metrics(real_data, generated_data))

        # Spectral analysis
        metrics.update(self._compute_spectral_metrics(real_data, generated_data))

        # Statistical distribution metrics
        metrics.update(self._compute_statistical_metrics(real_data, generated_data))

        # Autocorrelation analysis
        metrics.update(self._compute_autocorrelation_metrics(real_data, generated_data))

        return metrics

    def _compute_basic_metrics(
        self, real_data: jnp.ndarray, generated_data: jnp.ndarray
    ) -> dict[str, float]:
        """Compute basic distance metrics.

        Args:
            real_data: Real timeseries data
            generated_data: Generated timeseries data

        Returns:
            Dictionary of basic metrics
        """
        # Mean Squared Error
        mse = float(jnp.mean((real_data - generated_data) ** 2))

        # Mean Absolute Error
        mae = float(jnp.mean(jnp.abs(real_data - generated_data)))

        # Root Mean Squared Error
        rmse = float(jnp.sqrt(mse))

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
        }

    def _compute_temporal_metrics(
        self, real_data: jnp.ndarray, generated_data: jnp.ndarray
    ) -> dict[str, float]:
        """Compute temporal-specific metrics.

        Args:
            real_data: Real timeseries data
            generated_data: Generated timeseries data

        Returns:
            Dictionary of temporal metrics
        """
        # Dynamic Time Warping distance (simplified)
        dtw_distance = self._compute_dtw_distance(real_data, generated_data)

        # Trend preservation
        trend_correlation = self._compute_trend_correlation(real_data, generated_data)

        # Temporal consistency (smoothness)
        temporal_consistency = self._compute_temporal_consistency(real_data, generated_data)

        return {
            "dtw_distance": dtw_distance,
            "trend_correlation": trend_correlation,
            "temporal_consistency": temporal_consistency,
        }

    def _compute_spectral_metrics(
        self, real_data: jnp.ndarray, generated_data: jnp.ndarray
    ) -> dict[str, float]:
        """Compute spectral analysis metrics.

        Args:
            real_data: Real timeseries data
            generated_data: Generated timeseries data

        Returns:
            Dictionary of spectral metrics
        """
        # Compute FFT for both datasets
        real_fft = jnp.fft.fft(real_data, axis=1)
        generated_fft = jnp.fft.fft(generated_data, axis=1)

        # Power spectral density
        real_psd = jnp.abs(real_fft) ** 2
        generated_psd = jnp.abs(generated_fft) ** 2

        # Spectral distance
        spectral_distance = float(jnp.mean(jnp.abs(real_psd - generated_psd) / (real_psd + 1e-8)))

        # Dominant frequency correlation
        real_dominant_freq = jnp.argmax(real_psd, axis=1)
        generated_dominant_freq = jnp.argmax(generated_psd, axis=1)

        freq_correlation = float(
            jnp.corrcoef(
                real_dominant_freq.flatten(),
                generated_dominant_freq.flatten(),
                rowvar=False,
            )[0, 1]
        )

        # Handle NaN case
        if jnp.isnan(freq_correlation):
            freq_correlation = 0.0

        return {
            "spectral_distance": spectral_distance,
            "frequency_correlation": freq_correlation,
        }

    def _compute_statistical_metrics(
        self, real_data: jnp.ndarray, generated_data: jnp.ndarray
    ) -> dict[str, float]:
        """Compute statistical distribution metrics.

        Args:
            real_data: Real timeseries data
            generated_data: Generated timeseries data

        Returns:
            Dictionary of statistical metrics
        """
        # Mean and variance preservation
        real_mean = jnp.mean(real_data, axis=(0, 1))
        generated_mean = jnp.mean(generated_data, axis=(0, 1))
        mean_error = float(jnp.mean(jnp.abs(real_mean - generated_mean)))

        real_var = jnp.var(real_data, axis=(0, 1))
        generated_var = jnp.var(generated_data, axis=(0, 1))
        variance_error = float(jnp.mean(jnp.abs(real_var - generated_var)))

        # Skewness and kurtosis preservation
        real_skewness = _compute_skewness(real_data)
        generated_skewness = _compute_skewness(generated_data)
        skewness_error = float(jnp.abs(real_skewness - generated_skewness))

        return {
            "mean_error": mean_error,
            "variance_error": variance_error,
            "skewness_error": skewness_error,
        }

    def _compute_autocorrelation_metrics(
        self, real_data: jnp.ndarray, generated_data: jnp.ndarray
    ) -> dict[str, float]:
        """Compute autocorrelation analysis metrics.

        Args:
            real_data: Real timeseries data
            generated_data: Generated timeseries data

        Returns:
            Dictionary of autocorrelation metrics
        """
        # Compute autocorrelation functions
        real_acf = _compute_autocorrelation(real_data, self.max_lag)
        generated_acf = _compute_autocorrelation(generated_data, self.max_lag)

        # Autocorrelation distance
        acf_distance = float(jnp.mean(jnp.abs(real_acf - generated_acf)))

        # Autocorrelation correlation
        acf_correlation = float(
            jnp.corrcoef(real_acf.flatten(), generated_acf.flatten(), rowvar=False)[0, 1]
        )

        # Handle NaN case
        if jnp.isnan(acf_correlation):
            acf_correlation = 0.0

        return {
            "autocorr_distance": acf_distance,
            "autocorr_correlation": acf_correlation,
        }

    def _compute_dtw_distance(self, real_data: jnp.ndarray, generated_data: jnp.ndarray) -> float:
        """Compute simplified Dynamic Time Warping distance.

        Args:
            real_data: Real timeseries data
            generated_data: Generated timeseries data

        Returns:
            Average DTW distance
        """
        # Simplified DTW using Euclidean distance matrix
        # For computational efficiency, we compute DTW for first few samples
        max_samples = min(10, real_data.shape[0])

        dtw_distances = []

        for i in range(max_samples):
            # Get single time series
            real_series = real_data[i]  # Shape: (sequence, features)
            generated_series = generated_data[i]  # Shape: (sequence, features)

            # Compute distance matrix
            dist_matrix = self._compute_distance_matrix(real_series, generated_series)

            # Simplified DTW path (diagonal approximation)
            dtw_dist = jnp.mean(jnp.diag(dist_matrix))
            dtw_distances.append(dtw_dist)

        return float(jnp.mean(jnp.array(dtw_distances)))

    def _compute_distance_matrix(self, series1: jnp.ndarray, series2: jnp.ndarray) -> jnp.ndarray:
        """Compute pairwise distance matrix between two time series.

        Args:
            series1: First time series of shape (sequence, features)
            series2: Second time series of shape (sequence, features)

        Returns:
            Distance matrix of shape (sequence, sequence)
        """
        # Expand dimensions for broadcasting
        s1_expanded = series1[:, None, :]  # Shape: (seq1, 1, features)
        s2_expanded = series2[None, :, :]  # Shape: (1, seq2, features)

        # Compute Euclidean distances
        distances = jnp.sqrt(jnp.sum((s1_expanded - s2_expanded) ** 2, axis=-1))

        return distances

    def _compute_trend_correlation(
        self, real_data: jnp.ndarray, generated_data: jnp.ndarray
    ) -> float:
        """Compute trend preservation correlation.

        Args:
            real_data: Real timeseries data
            generated_data: Generated timeseries data

        Returns:
            Trend correlation coefficient
        """
        # Extract trends using simple moving average
        window_size = max(5, self.sequence_length // 10)

        real_trends = self._extract_trend(real_data, window_size)
        generated_trends = self._extract_trend(generated_data, window_size)

        # Compute correlation between trends
        correlation = jnp.corrcoef(real_trends.flatten(), generated_trends.flatten(), rowvar=False)[
            0, 1
        ]

        # Handle NaN case
        if jnp.isnan(correlation):
            return 0.0

        return float(correlation)

    def _extract_trend(self, data: jnp.ndarray, window_size: int) -> jnp.ndarray:
        """Extract trend using moving average.

        Args:
            data: Input timeseries data
            window_size: Size of the moving window

        Returns:
            Trend component
        """
        # Simple moving average for trend extraction
        if window_size >= data.shape[1]:
            # If window is too large, return global mean
            return jnp.mean(data, axis=1, keepdims=True)

        # Pad the data
        pad_size = window_size // 2
        padded_data = jnp.pad(data, ((0, 0), (pad_size, pad_size), (0, 0)), mode="edge")

        # Compute moving average
        kernel = jnp.ones(window_size) / window_size
        trend = jnp.apply_along_axis(
            lambda x: jnp.convolve(x, kernel, mode="valid"), axis=1, arr=padded_data
        )

        return trend

    def _compute_temporal_consistency(
        self, real_data: jnp.ndarray, generated_data: jnp.ndarray
    ) -> float:
        """Compute temporal consistency (smoothness) metric.

        Args:
            real_data: Real timeseries data
            generated_data: Generated timeseries data

        Returns:
            Temporal consistency score
        """
        # Compute first differences (derivatives)
        real_diff = jnp.diff(real_data, axis=1)
        generated_diff = jnp.diff(generated_data, axis=1)

        # Compute variance of differences (measure of smoothness)
        real_smoothness = jnp.var(real_diff, axis=1)
        generated_smoothness = jnp.var(generated_diff, axis=1)

        # Consistency is the negative absolute difference in smoothness
        consistency = -jnp.mean(jnp.abs(real_smoothness - generated_smoothness))

        return float(consistency)


def compute_timeseries_metrics(
    real_data: jnp.ndarray,
    generated_data: jnp.ndarray,
    **kwargs: Any,
) -> dict[str, float]:
    """Factory function to compute timeseries evaluation metrics.

    Args:
        real_data: Real timeseries data
        generated_data: Generated timeseries data
        **kwargs: Additional arguments for evaluation suite

    Returns:
        Dictionary of evaluation metrics
    """

    # Extract dimensions from data
    sequence_length = real_data.shape[1]
    num_features = real_data.shape[2] if real_data.ndim > 2 else 1

    # Create configuration
    config = TimeseriesModalityConfig(
        sequence_length=sequence_length,
        num_features=num_features,
        univariate=(num_features == 1),
    )

    # Create evaluation suite
    evaluator = TimeseriesEvaluationSuite(
        config=config,
        sequence_length=sequence_length,
        num_features=num_features,
        rngs=kwargs.pop("rngs", nnx.Rngs(42)),
        **kwargs,
    )

    # Compute metrics
    return evaluator.compute_metrics(real_data, generated_data)
