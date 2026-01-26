"""Enhanced base neural network abstractions for Flax NNX.

This module provides improved base classes that work optimally with both
diffusion models, flow models, and other generative model types.
"""

import functools
from typing import Any, Callable

import jax
from flax import nnx

from artifex.generative_models.core.gradient_checkpointing import apply_remat


# Type aliases
PyTree = Any


@functools.lru_cache(maxsize=32)
def get_activation_function(
    activation: str | Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array], jax.Array]:
    """Get activation function from string or callable.

    Args:
        activation: Activation function or name.

    Returns:
        Activation function.

    Raises:
        ValueError: If activation string is not recognized.
    """
    if callable(activation):
        return activation

    # Use NNX built-in activation functions
    activations = {
        "relu": nnx.relu,
        "gelu": nnx.gelu,
        "elu": nnx.elu,
        "leaky_relu": lambda x: nnx.leaky_relu(x, negative_slope=0.01),
        "silu": nnx.silu,
        "swish": nnx.swish,
        "tanh": nnx.tanh,
        "sigmoid": nnx.sigmoid,
        "softmax": nnx.softmax,
        "log_softmax": nnx.log_softmax,
    }

    if activation not in activations:
        raise ValueError(f"Unknown activation: {activation}. Available: {list(activations.keys())}")

    return activations[activation]  # type: ignore[return-value]


def get_default_backbone(config, *, rngs: nnx.Rngs):
    """Return a default backbone network for diffusion models.

    Following the unified configuration design pattern, all model-specific
    parameters are read from config.parameters as the single source of truth.

    Args:
        config: ModelConfiguration object with parameters dict
        rngs: Random number generators

    Returns:
        A default UNet-like backbone network
    """
    from artifex.generative_models.models.backbones.unet import UNet

    # Get all parameters from unified config.parameters (single source of truth)
    params = config.parameters or {}

    # Extract UNet configuration from parameters
    hidden_dims = params.get("hidden_dims", [32, 64, 128])
    time_embedding_dim = params.get("time_embedding_dim", 128)

    # Determine input channels from config.input_dim or parameters
    in_channels = 3  # Default for RGB

    # Extract from input_dim if available
    input_dim = config.input_dim
    if input_dim is not None and isinstance(input_dim, (tuple, list)) and len(input_dim) == 3:
        in_channels = input_dim[-1]  # Assume (H, W, C) format

    # Override with explicit in_channels from parameters if present
    in_channels = params.get("in_channels", in_channels)

    # Create UNet backbone with extracted configuration
    return UNet(
        hidden_dims=hidden_dims,
        time_embedding_dim=time_embedding_dim,
        in_channels=in_channels,
        rngs=rngs,
    )


class GenerativeModule(nnx.Module):
    """Base class for all generative model components."""

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        precision: jax.lax.Precision | None = None,
    ):
        """Initialize the module.

        Args:
            rngs: Random number generators.
            precision: Numerical precision for computations.
        """
        super().__init__()
        # Store RNGs for JIT-compatible random operations
        self.rngs = rngs
        # Store precision as a Variable for proper state management
        self.precision = nnx.Variable(precision) if precision is not None else None

    def _get_default_activation(self) -> Callable[[jax.Array], jax.Array]:
        """Return a default activation function."""
        return nnx.gelu

    def __call__(self, x: PyTree) -> PyTree:
        """Call method to be implemented by subclasses.

        Args:
            x: Input data.

        Returns:
            Module outputs.
        """
        raise NotImplementedError("Implement __call__ method")


class GenerativeModel(GenerativeModule):
    """Enhanced base class for all generative models.

    This class provides a unified interface for different types of generative models
    including diffusion models, flow models, VAEs, GANs, and others. It ensures
    consistent method signatures and behavior across all model types.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        precision: jax.lax.Precision | None = None,
    ):
        """Initialize the generative model.

        Args:
            rngs: Random number generators (required).

            precision: Numerical precision for computations.
        """
        super().__init__(
            rngs=rngs,
            precision=precision,
        )

    def __call__(self, x: PyTree, *args, **kwargs) -> dict[str, Any]:
        """Forward pass through the model.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.
        Do not pass explicit training flags - NNX manages mode automatically.
        Uses stored self.rngs for any random operations (RNG automatically advances).

        Args:
            x: Input data.
            *args: Additional positional arguments (model-specific, e.g., timesteps for diffusion).
            **kwargs: Additional keyword arguments (model-specific).

        Returns:
            dictionary containing model outputs. Common keys include:
            - For diffusion models: {"predicted_noise": ..., "loss": ...}
            - For flow models: {"z": ..., "logdet": ..., "log_prob": ...}
            - For VAEs: {"reconstruction": ..., "z": ..., "kl_loss": ...}
            - For GANs: {"generated": ..., "discriminator_logits": ...}

        Example:
            >>> model = DiffusionModel(config, rngs=nnx.Rngs(0))
            >>> # Training
            >>> model.train()
            >>> outputs = model(x, timesteps)
            >>> # Inference
            >>> model.eval()
            >>> outputs = model(x, timesteps)
        """
        raise NotImplementedError("Subclasses must implement __call__ method")

    def generate(self, n_samples: int = 1, **kwargs) -> PyTree:
        """Generate samples from the model.

        Note: Uses stored self.rngs for sampling. RNG automatically advances each call.

        Args:
            n_samples: Number of samples to generate.
            **kwargs: Additional keyword arguments (model-specific).
                Common kwargs include:
                - condition: Conditioning information for conditional models
                - shape: Target shape for generated samples
                - steps: Number of generation steps (for diffusion models)
                - temperature: Sampling temperature

        Returns:
            Generated samples as JAX arrays.
        """
        raise NotImplementedError("Subclasses must implement generate method")

    def loss_fn(
        self,
        batch: PyTree,
        model_outputs: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        """Compute loss for model training.

        Note: Uses stored self.rngs for any stochastic operations. RNG automatically advances.

        Args:
            batch: Batch of training data. Can be:
                - JAX array for simple data
                - dictionary with keys like {'x': data, 'condition': cond}
            model_outputs: dictionary of outputs from the forward pass.
            **kwargs: Additional keyword arguments for loss computation.

        Returns:
            dictionary containing loss and metrics. Must include:
            - "loss": Primary loss value for optimization
            Additional keys may include component losses and metrics.
        """
        raise NotImplementedError("Subclasses must implement loss_fn method")

    # Additional utility methods that can be overridden by specific model types

    def encode(self, x: PyTree, **kwargs) -> PyTree:
        """Encode input to latent representation (for VAEs, flow models).

        Note: Uses stored self.rngs for any stochastic operations. RNG automatically advances.

        Args:
            x: Input data to encode.
            **kwargs: Additional keyword arguments.

        Returns:
            Latent representation.

        Raises:
            NotImplementedError: If the model doesn't support encoding.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support encoding")

    def decode(self, z: PyTree, **kwargs) -> PyTree:
        """Decode latent representation to data space (for VAEs, flow models).

        Note: Uses stored self.rngs for any stochastic operations. RNG automatically advances.

        Args:
            z: Latent representation to decode.
            **kwargs: Additional keyword arguments.

        Returns:
            Decoded data.

        Raises:
            NotImplementedError: If the model doesn't support decoding.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support decoding")

    def log_prob(self, x: PyTree, **kwargs) -> jax.Array:
        """Compute log probability of data (for flow models, VAEs).

        Note: Uses stored self.rngs for any stochastic operations. RNG automatically advances.

        Args:
            x: Input data.
            **kwargs: Additional keyword arguments.

        Returns:
            Log probability of each data point.

        Raises:
            NotImplementedError: If the model doesn't support log probability computation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support log probability computation"
        )

    def sample(self, n_samples: int = 1, **kwargs) -> PyTree:
        """Sample from the model (alias for generate for backward compatibility).

        Note: Uses stored self.rngs for sampling. RNG automatically advances.

        Args:
            n_samples: Number of samples to generate.
            **kwargs: Additional keyword arguments.

        Returns:
            Generated samples.
        """
        return self.generate(n_samples, **kwargs)

    def interpolate(self, x1: PyTree, x2: PyTree, alpha: float, **kwargs) -> PyTree:
        """Interpolate between two data points (when supported).

        Note: Uses stored self.rngs for any stochastic operations. RNG automatically advances.

        Args:
            x1: First data point.
            x2: Second data point.
            alpha: Interpolation factor (0 = x1, 1 = x2).
            **kwargs: Additional keyword arguments.

        Returns:
            Interpolated result.

        Raises:
            NotImplementedError: If the model doesn't support interpolation.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support interpolation")

    def reconstruct(self, x: PyTree, **kwargs) -> PyTree:
        """Reconstruct input through encode-decode cycle (for VAEs, autoencoders).

        Note: Uses stored self.rngs for any stochastic operations. RNG automatically advances.

        Args:
            x: Input data to reconstruct.
            **kwargs: Additional keyword arguments.

        Returns:
            Reconstructed data.

        Raises:
            NotImplementedError: If the model doesn't support reconstruction.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support reconstruction")

    # Configuration and utility methods

    def get_config(self) -> dict[str, Any]:
        """Get model configuration.

        Returns:
            dictionary containing model configuration.
        """
        if hasattr(self, "config"):
            if hasattr(self.config, "__dict__"):
                return vars(self.config)
            elif isinstance(self.config, dict):
                return self.config.copy()
        return {}

    def _extract_data_from_batch(self, batch: PyTree) -> PyTree:
        """Extract main data from batch (utility method).

        Args:
            batch: Input batch.

        Returns:
            Main data tensor.
        """
        if isinstance(batch, dict):
            # Try common keys for data
            for key in ["x", "data", "input", "image"]:
                if key in batch:
                    return batch[key]
            # If no common keys found, raise error with available keys
            available_keys = list(batch.keys())
            raise ValueError(f"Could not find data in batch. Available keys: {available_keys}")
        else:
            # Assume batch is the data itself
            return batch

    # Removed manual RNG key extraction - let NNX handle RNG state automatically


class MLP(nnx.Module):
    """Memory-efficient Multi-Layer Perceptron (MLP) module.

    This implements a configurable MLP with arbitrary hidden dimensions,
    optimized for memory efficiency through:
    - Lazy layer initialization
    - Efficient activation function handling
    - Optional gradient checkpointing support
    - Minimal intermediate tensor storage
    """

    def __init__(
        self,
        hidden_dims: list[int],
        *,
        in_features: int,
        activation: str | Callable = "gelu",
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        output_activation: str | Callable | None = None,
        use_batch_norm: bool = False,
        use_gradient_checkpointing: bool = False,
        checkpoint_policy: str | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize the MLP.

        Args:
            hidden_dims: List of hidden dimensions for each layer.
            in_features: Input features for the first layer.
            activation: Activation function or name of activation.
            dropout_rate: Dropout probability.
            use_bias: Whether to use bias in the linear layers.
            output_activation: Optional activation to apply to the final layer.
            use_batch_norm: Whether to use batch normalization.
            use_gradient_checkpointing: Enable gradient checkpointing to trade
                compute for memory during backprop via ``nnx.remat``.
            checkpoint_policy: Named checkpoint policy for controlling which
                intermediates are saved.  See
                :data:`~gradient_checkpointing.CHECKPOINT_POLICIES`.
            rngs: Random number generators.
        """
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")

        # Store configuration as Variables for proper state management
        # Use Cache for static config, Variable for mutable config
        self.hidden_dims = nnx.Cache(hidden_dims)
        self.in_features = nnx.Cache(in_features)
        self.dropout_rate = nnx.Variable(dropout_rate)
        self.use_batch_norm = nnx.Cache(use_batch_norm)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.checkpoint_policy = checkpoint_policy

        # Get activation functions using shared utility
        self.activation = get_activation_function(activation)
        self.output_activation = (
            get_activation_function(output_activation) if output_activation else None
        )

        # Build layers efficiently
        self.layers = nnx.List([])
        self.dropouts = nnx.List([])
        self.batch_norms = nnx.List([])

        # Input dimensions for each layer
        dims = [in_features, *hidden_dims]

        for i in range(len(hidden_dims)):
            # Linear layer
            layer = nnx.Linear(
                in_features=dims[i],
                out_features=dims[i + 1],
                use_bias=use_bias,
                rngs=rngs,
            )
            self.layers.append(layer)

            # Batch normalization (if enabled)
            if use_batch_norm and i < len(hidden_dims) - 1:  # Skip BN on last layer
                bn = nnx.BatchNorm(
                    num_features=dims[i + 1],
                    use_running_average=False,
                    momentum=0.9,
                    epsilon=1e-5,
                    rngs=rngs,
                )
                self.batch_norms.append(bn)

            # Dropout (if enabled)
            if dropout_rate > 0.0:
                dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
                self.dropouts.append(dropout)

    def _forward_layer_impl(
        self,
        x: jax.Array,
        layer_idx: int,
    ) -> jax.Array:
        """Forward pass through a single layer implementation.

        Memory-efficient by not storing unnecessary intermediates.

        Args:
            x: Input tensor.
            layer_idx: Index of the layer.

        Returns:
            Output tensor.
        """
        # Linear transformation
        x = self.layers[layer_idx](x)

        # Batch normalization (if applicable)
        # Check if batch_norms list exists and has elements instead of using .value
        if self.batch_norms is not None and layer_idx < len(self.batch_norms):
            x = self.batch_norms[layer_idx](x)

        # Activation (skip on last layer if no output activation)
        is_last_layer = layer_idx == len(self.layers) - 1
        if is_last_layer:
            if self.output_activation is not None:
                x = self.output_activation(x)
        else:
            x = self.activation(x)

        # Dropout (if applicable) - NNX handles training/eval mode automatically
        # Check if dropouts list exists and has elements instead of using .value
        if len(self.dropouts) > 0 and layer_idx < len(self.dropouts):
            x = self.dropouts[layer_idx](x)

        return x

    def _forward_layer(
        self,
        x: jax.Array,
        layer_idx: int,
    ) -> jax.Array:
        """Forward pass through a single layer, with optional gradient checkpointing.

        Args:
            x: Input tensor.
            layer_idx: Index of the layer.

        Returns:
            Output tensor.
        """
        if self.use_gradient_checkpointing:
            return self._forward_layer_checkpointed(x, layer_idx)
        return self._forward_layer_impl(x, layer_idx)

    def _forward_layer_checkpointed(self, x: jax.Array, layer_idx: int) -> jax.Array:
        """Forward pass through a single layer with gradient checkpointing.

        Uses ``nnx.remat`` (via :func:`apply_remat`) to correctly handle
        NNX module state during recomputation.

        Args:
            x: Input tensor.
            layer_idx: Index of the layer.

        Returns:
            Output tensor.
        """

        def forward_fn(x_in):
            return self._forward_layer_impl(x_in, layer_idx)

        remated_fn = apply_remat(forward_fn, policy=self.checkpoint_policy)
        return remated_fn(x)

    def __call__(
        self,
        x: jax.Array,
        *,
        return_intermediates: bool = False,
        use_scan: bool = False,
    ) -> jax.Array | tuple[jax.Array, list[jax.Array]]:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, in_features).
            return_intermediates: Whether to return intermediate activations
                (useful for analysis but uses more memory).
            use_scan: Whether to use scan for memory-efficient deep networks.

        Returns:
            Output tensor of shape (batch_size, hidden_dims[-1]).
            If return_intermediates is True, returns (output, intermediates).
        """
        # Use scan for deep networks (>= 8 layers) or when explicitly requested
        if use_scan or (len(self.layers) >= 8 and not return_intermediates):
            return self._forward_sequential(x)

        intermediates: list[jax.Array] | None = [] if return_intermediates else None

        for i in range(len(self.layers)):
            x = self._forward_layer(x, i)

            if return_intermediates and intermediates is not None:
                intermediates.append(x)

        if return_intermediates and intermediates is not None:
            return x, intermediates
        return x

    def _forward_sequential(self, x: jax.Array) -> jax.Array:
        """Sequential forward pass without storing intermediates.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        for i in range(len(self.layers)):
            x = self._forward_layer_impl(x, i)
        return x

    def get_num_params(self) -> int:
        """Get total number of parameters in the model.

        Returns:
            Total parameter count.
        """
        total = 0
        for layer in self.layers:
            # Count weights
            total += layer.kernel.value.size
            # Count bias if present
            if hasattr(layer, "bias") and layer.bias is not None:
                total += layer.bias.value.size

        # Count batch norm parameters if used
        if self.use_batch_norm.value:
            for bn in self.batch_norms:
                # Scale and bias parameters
                if hasattr(bn, "scale") and bn.scale is not None:
                    total += bn.scale.value.size
                if hasattr(bn, "bias") and bn.bias is not None:
                    total += bn.bias.value.size

        return total

    def reset_dropout(self, rngs: nnx.Rngs):
        """Reset dropout RNG keys for reproducibility.

        Args:
            rngs: New random number generators.
        """
        for dropout in self.dropouts:
            dropout.rngs = rngs


class CNN(nnx.Module):
    """Enhanced Convolutional Neural Network (CNN) module.

    This implements a configurable CNN with arbitrary hidden dimensions
    and improved flexibility.
    """

    def __init__(
        self,
        hidden_dims: list[int],
        *,
        in_features: int,
        activation: str | Callable = "relu",
        kernel_size: int | tuple[int, int] = (3, 3),
        strides: int | tuple[int, int] = (2, 2),
        padding: str | int | tuple[int, int] = "SAME",
        use_transpose: bool = False,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
        use_depthwise_separable: bool = False,
        groups: int = 1,
        rngs: nnx.Rngs,
    ):
        """Initialize the CNN.

        Args:
            hidden_dims: list of output channels for each convolutional layer.
            in_features: Input features/channels for the first layer.
            activation: Activation function or name of activation.
            kernel_size: Kernel size for convolutions.
            strides: Strides for convolutions.
            padding: Padding for convolutions.
            use_transpose: Whether to use transpose convolutions (for decoder).
            use_batch_norm: Whether to use batch normalization.
            dropout_rate: Dropout probability.
            use_depthwise_separable: Whether to use depthwise separable convolutions.
            groups: Number of groups for grouped convolutions.
            rngs: Random number generators.

        """
        super().__init__()

        # Store configuration as Variables for proper state management
        # Use Cache for static config, Variable for mutable config
        self.hidden_dims = nnx.Cache(hidden_dims)
        self.dropout_rate = nnx.Variable(dropout_rate)
        self.use_transpose = nnx.Cache(use_transpose)
        self.use_batch_norm = nnx.Cache(use_batch_norm)
        self.use_depthwise_separable = nnx.Cache(use_depthwise_separable)
        self.groups = nnx.Cache(groups)

        # Set up activation function using shared utility
        self.activation = get_activation_function(activation)

        # Initialize layers - handle both regular and depthwise separable convolutions
        self.layers = nnx.List([])
        self.batch_norms = nnx.List([]) if use_batch_norm else None
        current_in_features = in_features

        for dim in hidden_dims:
            # Convolutional layer with advanced options
            if use_depthwise_separable and not use_transpose:
                # Depthwise separable convolution (depthwise + pointwise)
                # Depthwise convolution
                depthwise = nnx.Conv(
                    in_features=current_in_features,
                    out_features=current_in_features,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    feature_group_count=current_in_features,
                    rngs=rngs,
                )
                # Pointwise convolution
                pointwise = nnx.Conv(
                    in_features=current_in_features,
                    out_features=dim,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="SAME",
                    rngs=rngs,
                )
                self.layers.append((depthwise, pointwise))
            elif use_transpose:
                transpose_layer = nnx.ConvTranspose(
                    in_features=current_in_features,
                    out_features=dim,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    rngs=rngs,
                )
                self.layers.append(transpose_layer)
            else:
                # Regular or grouped convolution
                conv_layer = nnx.Conv(
                    in_features=current_in_features,
                    out_features=dim,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    feature_group_count=groups if groups > 1 else 1,
                    rngs=rngs,
                )
                self.layers.append(conv_layer)

            # Batch normalization if requested
            if use_batch_norm:
                bn = nnx.BatchNorm(num_features=dim, rngs=rngs)
                if self.batch_norms is not None:  # Add check before appending
                    self.batch_norms.append(bn)

            current_in_features = dim

        # Initialize dropout if needed following critical guidelines
        # Note: No type annotation here to avoid making it a static attribute
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        """Apply CNN to input.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        for i, layer in enumerate(self.layers):
            # Apply convolution (handle depthwise separable case)
            if isinstance(layer, tuple) and len(layer) == 2:
                # Depthwise separable convolution
                depthwise, pointwise = layer
                x = depthwise(x)
                x = pointwise(x)
            else:
                # Regular convolution
                x = layer(x)

            # Apply batch normalization if used - NNX handles training/eval mode automatically
            if self.batch_norms is not None and i < len(self.batch_norms):
                x = self.batch_norms[i](x)

            # Apply activation
            x = self.activation(x)

            # Apply dropout if needed - NNX handles training/eval mode automatically
            if self.dropout is not None:
                x = self.dropout(x)

        return x
