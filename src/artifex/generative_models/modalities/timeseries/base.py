"""Base timeseries modality implementation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.extensions.base import ModelExtension

from ..base import BaseModalityConfig, BaseModalityImplementation


class TimeseriesRepresentation(Enum):
    """Types of timeseries representations."""

    RAW = "raw"
    FOURIER = "fourier"
    WAVELET = "wavelet"
    MULTI_SCALE = "multi_scale"
    TREND_SEASONAL = "trend_seasonal"


class DecompositionMethod(Enum):
    """Methods for trend/seasonal decomposition."""

    SEASONAL = "seasonal"
    STL = "stl"
    X13 = "x13"
    LOESS = "loess"


@dataclass
class TimeseriesModalityConfig(BaseModalityConfig):
    """Configuration for timeseries modality.

    Attributes:
        sequence_length: Length of the time series
        num_features: Number of features per timestep
        sampling_rate: Sampling rate of the time series
        representation: Type of representation to use
        use_fourier_features: Whether to include Fourier features
        num_frequencies: Number of Fourier frequencies to use
        max_frequency: Maximum frequency for Fourier features
        use_trend_decomposition: Whether to use trend decomposition
        decomposition_method: Method for trend decomposition
        decomposition_period: Period for seasonal decomposition
        multi_scale_factors: Scaling factors for multi-scale representation
        feature_names: Names of the features
        univariate: Whether this is a univariate time series
        stationary: Whether the time series is stationary
        seasonal_period: Known seasonal period if any
    """

    sequence_length: int = 100
    num_features: int = 1
    sampling_rate: float = 1.0
    representation: TimeseriesRepresentation = TimeseriesRepresentation.RAW
    use_fourier_features: bool = False
    num_frequencies: int = 64
    max_frequency: float = 100.0
    use_trend_decomposition: bool = False
    decomposition_method: DecompositionMethod = DecompositionMethod.SEASONAL
    decomposition_period: int = 24
    multi_scale_factors: list[int] = field(default_factory=lambda: [1, 2, 4])
    feature_names: list[str] = field(default_factory=list)
    univariate: bool = True
    stationary: bool = False
    seasonal_period: int | None = None

    def __post_init__(self):
        """Validate configuration parameters."""
        super().__post_init__()

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        if self.num_features <= 0:
            raise ValueError("num_features must be positive")

        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        if self.use_fourier_features:
            if self.num_frequencies <= 0:
                raise ValueError("num_frequencies must be positive when using Fourier features")
            if self.max_frequency <= 0:
                raise ValueError("max_frequency must be positive when using Fourier features")

        if self.use_trend_decomposition:
            if self.decomposition_period <= 0:
                raise ValueError("decomposition_period must be positive")

        if not self.multi_scale_factors:
            raise ValueError("multi_scale_factors cannot be empty")

        if any(factor <= 0 for factor in self.multi_scale_factors):
            raise ValueError("all multi_scale_factors must be positive")

        # Validate univariate setting first
        if self.univariate and self.num_features != 1:
            raise ValueError("num_features must be 1 for univariate time series")

        # Set default feature names if not provided
        if not self.feature_names:
            if self.univariate:
                self.feature_names = ["value"]
            else:
                self.feature_names = [f"feature_{i}" for i in range(self.num_features)]

        if len(self.feature_names) != self.num_features:
            raise ValueError(
                f"Length of feature_names ({len(self.feature_names)}) "
                f"must match num_features ({self.num_features})"
            )

        # Validate seasonal period
        if self.seasonal_period is not None:
            if self.seasonal_period <= 0:
                raise ValueError("seasonal_period must be positive")
            if self.seasonal_period >= self.sequence_length:
                raise ValueError("seasonal_period must be less than sequence_length")

    @property
    def is_multivariate(self) -> bool:
        """Check if this is a multivariate time series."""
        return not self.univariate

    @property
    def expected_shape(self) -> tuple[int, ...]:
        """Get the expected shape of input data."""
        return (self.sequence_length, self.num_features)


class TimeseriesModality(BaseModalityImplementation):
    """Modality for temporal sequence generation and processing.

    This modality provides support for various types of time series data,
    including univariate and multivariate sequences, with multiple
    representation options and temporal-specific processing capabilities.
    """

    def __init__(
        self,
        config: TimeseriesModalityConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the timeseries modality.

        Args:
            config: Configuration for the timeseries modality
            rngs: Random number generator keys
        """
        super().__init__(config, rngs=rngs)
        self.config = config
        self.name = "timeseries"

        # Store core parameters
        self.sequence_length = config.sequence_length
        self.num_features = config.num_features
        self.sampling_rate = config.sampling_rate

    def get_extensions(
        self, config: dict[str, Any], *, rngs: nnx.Rngs
    ) -> dict[str, ModelExtension]:
        """Get timeseries-specific extensions.

        Args:
            config: Extension configuration
            rngs: Random number generator keys

        Returns:
            Dictionary mapping extension names to extension instances
        """
        extensions = {}

        # For now, return empty dict as extensions don't exist yet
        # TODO: Implement temporal extensions
        return extensions

    def get_adapter(self, model_cls: type) -> Any:
        """Get an adapter for the specified model class.

        Args:
            model_cls: The model class to adapt

        Returns:
            A model adapter for the specified model class
        """
        from .adapters import get_timeseries_adapter

        return get_timeseries_adapter(model_cls, self.config)

    def preprocess(self, data: jnp.ndarray) -> jnp.ndarray:
        """Preprocess timeseries data.

        Args:
            data: Raw timeseries data of shape (batch_size, sequence_length, num_features)

        Returns:
            Preprocessed timeseries data
        """
        if data.ndim != 3:
            raise ValueError(f"Expected 3D input (batch, sequence, features), got {data.ndim}D")

        batch_size, seq_len, num_feat = data.shape

        if seq_len != self.sequence_length:
            raise ValueError(f"Expected sequence_length {self.sequence_length}, got {seq_len}")

        if num_feat != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {num_feat}")

        # Apply normalization if enabled
        if self.config.normalize:
            # Compute statistics across batch and sequence dimensions
            mean = jnp.mean(data, axis=(0, 1), keepdims=True)
            std = jnp.std(data, axis=(0, 1), keepdims=True)
            # Avoid division by zero
            std = jnp.where(std == 0, 1.0, std)
            data = (data - mean) / std

        return data

    def postprocess(self, data: jnp.ndarray) -> jnp.ndarray:
        """Postprocess generated timeseries data.

        Args:
            data: Generated timeseries data

        Returns:
            Postprocessed timeseries data
        """
        # Ensure finite values
        data = jnp.where(jnp.isfinite(data), data, 0.0)

        # Clip extreme values if needed
        if self.config.stationary:
            # For stationary series, clip to reasonable bounds
            data = jnp.clip(data, -5.0, 5.0)

        return data

    def validate_data(self, data: jnp.ndarray) -> bool:
        """Validate timeseries data format and content.

        Args:
            data: Timeseries data to validate

        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check basic shape requirements
            if data.ndim != 3:
                return False

            batch_size, seq_len, num_feat = data.shape

            if seq_len != self.sequence_length:
                return False

            if num_feat != self.num_features:
                return False

            # Check for finite values
            if not jnp.all(jnp.isfinite(data)):
                return False

            # Check for reasonable value ranges
            if jnp.any(jnp.abs(data) > 1e6):
                return False

            return True

        except Exception:
            return False

    def get_feature_info(self) -> dict[str, Any]:
        """Get information about the features in this timeseries.

        Returns:
            Dictionary with feature information
        """
        return {
            "feature_names": self.config.feature_names,
            "num_features": self.num_features,
            "sequence_length": self.sequence_length,
            "sampling_rate": self.sampling_rate,
            "is_univariate": self.config.univariate,
            "is_stationary": self.config.stationary,
            "seasonal_period": self.config.seasonal_period,
            "representation": self.config.representation.value,
        }
