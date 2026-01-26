"""Model adapters for timeseries modality."""

import dataclasses
from typing import Any

from flax import nnx

from artifex.generative_models.core.base import GenerativeModel

from ..base import ModelAdapter


@dataclasses.dataclass(frozen=True)
class TimeseriesAdapterConfig:
    """Configuration for timeseries adapters.

    Attributes:
        name: Name of the adapter
        sequence_length: Length of the time series
        num_features: Number of features per time step
        sampling_rate: Sampling rate in Hz
        use_temporal_position_encoding: Whether to use temporal position encoding
        causal_masking: Whether to use causal masking
    """

    name: str = "timeseries_adapter"
    sequence_length: int = 128
    num_features: int = 1
    sampling_rate: float = 1.0
    use_temporal_position_encoding: bool = True
    causal_masking: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")
        if self.num_features <= 0:
            raise ValueError(f"num_features must be positive, got {self.num_features}")
        if self.sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {self.sampling_rate}")


class TimeseriesTransformerAdapter(ModelAdapter):
    """Adapter for transformer models with timeseries data.

    Adapts generic transformer architectures to work with temporal
    sequence data by adding appropriate position encodings and
    temporal-specific extensions.
    """

    def __init__(self, config: TimeseriesAdapterConfig | None = None):
        """Initialize the timeseries transformer adapter.

        Args:
            config: Timeseries adapter configuration
        """
        self.config = config or TimeseriesAdapterConfig()

    def create(self, config: Any, *, rngs: nnx.Rngs, **kwargs: Any) -> GenerativeModel:
        """Create a transformer model adapted for timeseries.

        Args:
            config: Model configuration (dataclass config)
            rngs: Random number generator keys
            **kwargs: Additional keyword arguments

        Returns:
            Adapted transformer model for timeseries
        """
        # Import here to avoid circular imports
        from artifex.generative_models.models.transformers import TransformerModel

        # Create the base transformer model
        model = TransformerModel(
            config=config,
            rngs=rngs,
        )

        # Add timeseries-specific extensions
        extensions = self._get_timeseries_extensions(config, rngs=rngs)
        for name, extension in extensions.items():
            model.add_extension(name, extension)

        return model

    def _get_timeseries_extensions(
        self,
        config: Any,
        *,
        rngs: nnx.Rngs,
    ) -> dict[str, Any]:
        """Get timeseries-specific extensions for transformers.

        Args:
            config: Model configuration (dataclass config)
            rngs: Random number generator keys

        Returns:
            Dictionary of extensions
        """
        extensions = {}

        # Check config attributes or adapter config
        use_temporal_encoding = getattr(
            config, "use_temporal_position_encoding", self.config.use_temporal_position_encoding
        )
        use_causal = getattr(config, "causal_masking", self.config.causal_masking)

        # Add temporal position encoding
        if use_temporal_encoding:
            from artifex.generative_models.extensions.temporal import TemporalPositionExtension

            extensions["temporal_position"] = TemporalPositionExtension(
                sequence_length=self.config.sequence_length,
                sampling_rate=self.config.sampling_rate,
                rngs=rngs,
            )

        # Add attention mask for causal modeling
        if use_causal:
            from artifex.generative_models.extensions.attention import CausalMaskExtension

            extensions["causal_mask"] = CausalMaskExtension(
                sequence_length=self.config.sequence_length,
                rngs=rngs,
            )

        return extensions


class TimeseriesRNNAdapter(ModelAdapter):
    """Adapter for RNN/LSTM models with timeseries data.

    Adapts recurrent neural networks to work with timeseries by
    adding appropriate state handling and temporal processing.
    """

    def __init__(self, config: TimeseriesAdapterConfig | None = None):
        """Initialize the timeseries RNN adapter.

        Args:
            config: Timeseries adapter configuration
        """
        self.config = config or TimeseriesAdapterConfig()

    def create(self, config: Any, *, rngs: nnx.Rngs, **kwargs: Any) -> GenerativeModel:
        """Create an RNN model adapted for timeseries.

        Args:
            config: Model configuration (dataclass config)
            rngs: Random number generator keys
            **kwargs: Additional keyword arguments

        Returns:
            Adapted RNN model for timeseries
        """
        # Import here to avoid circular imports
        from artifex.generative_models.models.rnn import RNNModel

        # Create the base RNN model
        model = RNNModel(
            config=config,
            rngs=rngs,
        )

        # Add timeseries-specific extensions
        extensions = self._get_timeseries_extensions(config, rngs=rngs)
        for name, extension in extensions.items():
            model.add_extension(name, extension)

        return model

    def _get_timeseries_extensions(
        self,
        config: Any,
        *,
        rngs: nnx.Rngs,
    ) -> dict[str, Any]:
        """Get timeseries-specific extensions for RNNs.

        Args:
            config: Model configuration
            rngs: Random number generator keys

        Returns:
            Dictionary of extensions
        """
        extensions = {}

        # Check config attributes
        use_seq_dropout = getattr(config, "use_sequence_dropout", False)
        use_grad_clip = getattr(config, "use_gradient_clipping", True)

        # Add sequence dropout for regularization
        if use_seq_dropout:
            from artifex.generative_models.extensions.regularization import (
                SequenceDropoutExtension,
            )

            extensions["sequence_dropout"] = SequenceDropoutExtension(
                dropout_rate=getattr(config, "sequence_dropout_rate", 0.1),
                rngs=rngs,
            )

        # Add gradient clipping for stability
        if use_grad_clip:
            from artifex.generative_models.extensions.optimization import GradientClippingExtension

            extensions["gradient_clipping"] = GradientClippingExtension(
                max_norm=getattr(config, "gradient_clip_norm", 1.0),
                rngs=rngs,
            )

        return extensions


class TimeseriesDiffusionAdapter(ModelAdapter):
    """Adapter for diffusion models with timeseries data.

    Adapts diffusion models to work with temporal sequences by
    adding appropriate noise scheduling and temporal conditioning.
    """

    def __init__(self, config: TimeseriesAdapterConfig | None = None):
        """Initialize the timeseries diffusion adapter.

        Args:
            config: Timeseries adapter configuration
        """
        self.config = config or TimeseriesAdapterConfig()

    def create(self, config: Any, *, rngs: nnx.Rngs, **kwargs: Any) -> GenerativeModel:
        """Create a diffusion model adapted for timeseries.

        Args:
            config: Model configuration (dataclass config)
            rngs: Random number generator keys
            **kwargs: Additional keyword arguments

        Returns:
            Adapted diffusion model for timeseries
        """
        # Import here to avoid circular imports
        from artifex.generative_models.models.diffusion import DiffusionModel

        # Create the base diffusion model
        model = DiffusionModel(
            config=config,
            rngs=rngs,
        )

        # Add timeseries-specific extensions
        extensions = self._get_timeseries_extensions(config, rngs=rngs)
        for name, extension in extensions.items():
            model.add_extension(name, extension)

        return model

    def _get_timeseries_extensions(
        self,
        config: Any,
        *,
        rngs: nnx.Rngs,
    ) -> dict[str, Any]:
        """Get timeseries-specific extensions for diffusion models.

        Args:
            config: Model configuration
            rngs: Random number generator keys

        Returns:
            Dictionary of extensions
        """
        extensions = {}

        # Check config attributes
        use_temporal_cond = getattr(config, "use_temporal_conditioning", True)
        use_temporal_noise = getattr(config, "use_temporal_noise_schedule", False)

        # Add temporal conditioning
        if use_temporal_cond:
            from artifex.generative_models.extensions.temporal import TemporalConditioningExtension

            extensions["temporal_conditioning"] = TemporalConditioningExtension(
                sequence_length=self.config.sequence_length,
                conditioning_dim=getattr(config, "conditioning_dim", 64),
                rngs=rngs,
            )

        # Add noise scheduling for temporal data
        if use_temporal_noise:
            from artifex.generative_models.extensions.diffusion import (
                TemporalNoiseScheduleExtension,
            )

            extensions["temporal_noise"] = TemporalNoiseScheduleExtension(
                sequence_length=self.config.sequence_length,
                schedule_type=getattr(config, "temporal_schedule_type", "position_dependent"),
                rngs=rngs,
            )

        return extensions


class TimeseriesVAEAdapter(ModelAdapter):
    """Adapter for VAE models with timeseries data.

    Adapts Variational Autoencoders to work with temporal sequences
    by adding appropriate encoder/decoder architectures and regularization.
    """

    def __init__(self, config: TimeseriesAdapterConfig | None = None):
        """Initialize the timeseries VAE adapter.

        Args:
            config: Timeseries adapter configuration
        """
        self.config = config or TimeseriesAdapterConfig()

    def create(self, config: Any, *, rngs: nnx.Rngs, **kwargs: Any) -> GenerativeModel:
        """Create a VAE model adapted for timeseries.

        Args:
            config: Model configuration (dataclass config)
            rngs: Random number generator keys
            **kwargs: Additional keyword arguments

        Returns:
            Adapted VAE model for timeseries
        """
        # Create the base VAE model using factory
        from artifex.generative_models.factory import create_model

        model = create_model(config, rngs=rngs)

        # Add timeseries-specific extensions
        extensions = self._get_timeseries_extensions(config, rngs=rngs)
        for name, extension in extensions.items():
            model.add_extension(name, extension)

        return model

    def _get_timeseries_extensions(
        self,
        config: Any,
        *,
        rngs: nnx.Rngs,
    ) -> dict[str, Any]:
        """Get timeseries-specific extensions for VAEs.

        Args:
            config: Model configuration
            rngs: Random number generator keys

        Returns:
            Dictionary of extensions
        """
        extensions = {}

        # Check config attributes
        use_kl_anneal = getattr(config, "use_kl_annealing", True)
        use_recon_weight = getattr(config, "use_reconstruction_weighting", False)

        # Add KL annealing for better training
        if use_kl_anneal:
            from artifex.generative_models.extensions.vae import KLAnnealingExtension

            extensions["kl_annealing"] = KLAnnealingExtension(
                total_steps=getattr(config, "total_training_steps", 10000),
                annealing_type=getattr(config, "kl_annealing_type", "linear"),
                rngs=rngs,
            )

        # Add reconstruction loss weighting
        if use_recon_weight:
            from artifex.generative_models.extensions.vae import ReconstructionWeightingExtension

            extensions["reconstruction_weighting"] = ReconstructionWeightingExtension(
                weighting_type=getattr(config, "reconstruction_weighting_type", "temporal"),
                sequence_length=self.config.sequence_length,
                rngs=rngs,
            )

        return extensions


def get_timeseries_adapter(
    model_cls: type, config: TimeseriesAdapterConfig | None = None
) -> ModelAdapter:
    """Get the appropriate adapter for a given model class and timeseries configuration.

    Args:
        model_cls: The model class to adapt
        config: Timeseries adapter configuration

    Returns:
        Appropriate model adapter for the given model class
    """
    # Map model classes to their adapters
    adapter_mapping = {
        "TransformerModel": TimeseriesTransformerAdapter,
        "RNNModel": TimeseriesRNNAdapter,
        "DiffusionModel": TimeseriesDiffusionAdapter,
        "VAEModel": TimeseriesVAEAdapter,
    }

    # Get adapter by class name to avoid import issues
    model_name = model_cls.__name__

    if model_name in adapter_mapping:
        return adapter_mapping[model_name](config)

    # Default adapter for unknown model types
    return TimeseriesTransformerAdapter(config)
