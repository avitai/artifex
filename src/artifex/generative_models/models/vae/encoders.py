"""Encoder implementations for VAE models."""

from flax import nnx

from artifex.generative_models.core.base import CNN, MLP
from artifex.generative_models.core.configuration.network_configs import EncoderConfig


class Flatten(nnx.Module):
    """Flatten module that reshapes the input to a 2D tensor."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize the Flatten module.

        Args:
            rngs: Random number generators (required for nnx.Module subclasses)
        """
        super().__init__()

    def __call__(self, x):
        """Flatten the input tensor to 2D."""
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)


class MLPEncoder(nnx.Module):
    """Simple MLP encoder for VAE."""

    def __init__(
        self,
        config: EncoderConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the MLP encoder.

        Args:
            config: EncoderConfig with hidden_dims, latent_dim, activation, input_shape
            rngs: Random number generator
        """
        super().__init__()

        hidden_dims = list(config.hidden_dims)
        latent_dim = config.latent_dim
        activation = config.activation
        input_shape = config.input_shape
        use_batch_norm = config.use_batch_norm

        # Calculate input features from input_shape
        input_features = None
        if input_shape and len(input_shape) > 0:
            input_features = 1
            for dim in input_shape:
                input_features *= dim

        self.latent_dim = latent_dim

        self.backbone = MLP(
            hidden_dims=hidden_dims,
            activation=activation,
            in_features=input_features,
            use_batch_norm=use_batch_norm,
            rngs=rngs,
        )
        self.mean_proj = nnx.Linear(in_features=hidden_dims[-1], out_features=latent_dim, rngs=rngs)
        self.logvar_proj = nnx.Linear(
            in_features=hidden_dims[-1], out_features=latent_dim, rngs=rngs
        )

    def __call__(self, x):
        """Forward pass of the encoder.

        Args:
            x: Input tensor

        Returns:
            Tuple of (mean, log_var)
        """
        # Flatten input if needed
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)

        features = self.backbone(x)
        mean = self.mean_proj(features)
        log_var = self.logvar_proj(features)
        return mean, log_var


class CNNEncoder(nnx.Module):
    """CNN-based encoder for VAE."""

    def __init__(
        self,
        config: EncoderConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the CNN encoder.

        Args:
            config: EncoderConfig with hidden_dims, latent_dim, activation, input_shape
            rngs: Random number generator
        """
        super().__init__()

        hidden_dims = list(config.hidden_dims)
        latent_dim = config.latent_dim
        activation = config.activation
        input_shape = config.input_shape
        use_batch_norm = config.use_batch_norm

        self.latent_dim = latent_dim

        # Default to 3 channels (RGB) if not specified
        in_channels = 3
        if input_shape and len(input_shape) >= 3:
            # Use the last dimension as number of channels
            in_channels = input_shape[-1]

        self.cnn = CNN(
            hidden_dims=hidden_dims,
            activation=activation,
            in_features=in_channels,
            use_batch_norm=use_batch_norm,
            rngs=rngs,
        )
        self.flatten = Flatten(rngs=rngs)

        # Calculate expected flattened size after CNN
        # With SAME padding and stride 2, output = ceil(input/stride)
        num_layers = len(hidden_dims)
        if input_shape and len(input_shape) >= 2:
            h, w = input_shape[:2]
            # Compute output size for each layer (ceil division for SAME padding)
            for _ in range(num_layers):
                h = (h + 1) // 2  # ceil(h / 2)
                w = (w + 1) // 2  # ceil(w / 2)
            final_h = max(1, h)
            final_w = max(1, w)
            flattened_size = final_h * final_w * hidden_dims[-1]
        else:
            # Fallback: assume some reasonable size
            flattened_size = hidden_dims[-1] * 4 * 4

        self.mean_proj = nnx.Linear(in_features=flattened_size, out_features=latent_dim, rngs=rngs)
        self.logvar_proj = nnx.Linear(
            in_features=flattened_size, out_features=latent_dim, rngs=rngs
        )

    def __call__(self, x):
        """Forward pass of the encoder.

        Args:
            x: Input tensor

        Returns:
            Tuple of (mean, log_var)
        """
        features = self.cnn(x)
        features = self.flatten(features)
        mean = self.mean_proj(features)
        log_var = self.logvar_proj(features)
        return mean, log_var


class ResNetEncoder(nnx.Module):
    """ResNet-based encoder for VAE.

    Note: This is a placeholder implementation that uses CNN internally.
    A full ResNet implementation would include proper residual blocks.
    """

    def __init__(
        self,
        config: EncoderConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize ResNet encoder.

        Args:
            config: EncoderConfig with hidden_dims, latent_dim, activation, input_shape
            rngs: Random number generators
        """
        super().__init__()

        self.latent_dim = config.latent_dim
        # For now, use CNN encoder as a placeholder
        # A real ResNet would have residual blocks
        self.cnn_encoder = CNNEncoder(config=config, rngs=rngs)

    def __call__(self, x):
        """Forward pass through ResNet encoder.

        Args:
            x: Input tensor

        Returns:
            Tuple of (mean, log_var) for the latent distribution
        """
        return self.cnn_encoder(x)


class ConditionalEncoder(nnx.Module):
    """Wrapper that adds conditioning to any encoder.

    This wrapper adds class conditioning to an existing encoder by
    converting labels to one-hot and concatenating them with the input.
    This follows the standard CVAE pattern from popular implementations.
    """

    def __init__(
        self,
        encoder: nnx.Module,
        num_classes: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize conditional encoder wrapper.

        Args:
            encoder: Base encoder to wrap
            num_classes: Number of classes for conditioning
            rngs: Random number generators (for API consistency)
        """
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes

    def __call__(self, x, condition=None):
        """Forward pass with conditioning.

        Args:
            x: Input data
            condition: Class labels (integers) or one-hot encoded conditions

        Returns:
            Tuple of (mean, log_var) from the encoder
        """
        import jax
        import jax.numpy as jnp

        if condition is not None:
            # Convert integer labels to one-hot if needed
            if condition.dtype in [jnp.int32, jnp.int64]:
                condition = jax.nn.one_hot(condition, self.num_classes)

            # Concatenate conditioning with input based on input shape
            if x.ndim == 4:  # (batch, height, width, channels)
                batch_size = x.shape[0]
                # Broadcast one-hot to spatial dimensions
                cond_spatial = condition.reshape(batch_size, 1, 1, -1)
                cond_spatial = jnp.broadcast_to(
                    cond_spatial, (batch_size, x.shape[1], x.shape[2], self.num_classes)
                )
                x = jnp.concatenate([x, cond_spatial], axis=-1)
            elif x.ndim == 2:  # (batch, features)
                # For flat inputs, just concatenate
                x = jnp.concatenate([x, condition], axis=-1)
            else:
                # For other shapes, flatten and concatenate
                batch_size = x.shape[0]
                x_flat = x.reshape(batch_size, -1)
                x = jnp.concatenate([x_flat, condition], axis=-1)

        # Pass through the base encoder
        return self.encoder(x)


def create_encoder(
    config: EncoderConfig,
    encoder_type: str = "dense",
    *,
    conditional: bool = False,
    num_classes: int | None = None,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Create an encoder from EncoderConfig.

    Args:
        config: EncoderConfig with network architecture settings
        encoder_type: Type of encoder (dense, cnn, resnet)
        conditional: Whether to create conditional encoder
        num_classes: Number of classes for conditional encoder (one-hot dimension)
        rngs: Random number generators

    Returns:
        Encoder module
    """
    # If conditional, adjust input_shape to account for concatenated one-hot condition
    if conditional:
        if num_classes is None:
            raise ValueError("num_classes must be provided for conditional encoder")
        # Calculate flattened input size for any input shape and add condition dimensions
        if config.input_shape:
            # Compute flattened size from input_shape (works for any dimensionality)
            flattened_size = 1
            for dim in config.input_shape:
                flattened_size *= dim
            # Adjust for conditioning - one-hot is concatenated to flattened input
            adjusted_input_shape = (flattened_size + num_classes,)
            config = EncoderConfig(
                name=config.name,
                hidden_dims=config.hidden_dims,
                activation=config.activation,
                input_shape=adjusted_input_shape,
                latent_dim=config.latent_dim,
                use_batch_norm=config.use_batch_norm,
            )

    if encoder_type == "dense":
        encoder = MLPEncoder(config=config, rngs=rngs)
    elif encoder_type == "cnn":
        encoder = CNNEncoder(config=config, rngs=rngs)
    elif encoder_type == "resnet":
        encoder = ResNetEncoder(config=config, rngs=rngs)
    elif encoder_type == "spatial":
        from artifex.generative_models.models.vae.spatial_autoencoder import SpatialEncoder

        encoder = SpatialEncoder(config=config, rngs=rngs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # Wrap in conditional encoder if needed
    if conditional:
        encoder = ConditionalEncoder(
            encoder=encoder,
            num_classes=num_classes,
            rngs=rngs,
        )

    return encoder
