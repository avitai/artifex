"""Base Generative Adversarial Network (GAN) implementation."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel
from artifex.generative_models.core.configuration.gan_config import GANConfig
from artifex.generative_models.core.configuration.network_configs import (
    DiscriminatorConfig,
    GeneratorConfig,
)
from artifex.generative_models.core.losses import adversarial


class Generator(nnx.Module):
    """Generator network for GAN.

    Base generator using fully-connected (Dense) layers.
    For convolutional generators, use DCGANGenerator or other specialized subclasses.
    """

    def __init__(
        self,
        config: GeneratorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize generator network.

        Args:
            config: GeneratorConfig with network architecture parameters
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not GeneratorConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for Generator")

        if not isinstance(config, GeneratorConfig):
            raise TypeError(f"config must be GeneratorConfig, got {type(config).__name__}")

        super().__init__()

        # Store config for reference
        self.config = config

        # Extract configuration values
        hidden_dims = list(config.hidden_dims)
        self.hidden_dims = hidden_dims
        self.output_shape = config.output_shape
        self.latent_dim = config.latent_dim
        self.batch_norm = config.batch_norm
        self.dropout_rate = config.dropout_rate

        # Store activation function from nnx (not jax.nn)
        self.activation_fn = self._get_activation_fn(config.activation)

        # Create the dense layers using nnx.List
        layers_list = []
        input_dim = self.latent_dim
        for dim in hidden_dims:
            layers_list.append(nnx.Linear(in_features=input_dim, out_features=dim, rngs=rngs))
            input_dim = dim
        self.layers = nnx.List(layers_list)

        # Output layer
        output_size = int(jnp.prod(jnp.array(self.output_shape[1:])))
        last_dim = hidden_dims[-1] if hidden_dims else self.latent_dim
        self.output_layer = nnx.Linear(in_features=last_dim, out_features=output_size, rngs=rngs)

        # Batch norm layers if needed using nnx.List
        if self.batch_norm:
            self.bn_layers = nnx.List([nnx.BatchNorm(dim, rngs=rngs) for dim in hidden_dims])
        else:
            self.bn_layers = None

        # Create dropout layer if needed
        if self.dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=self.dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def _get_activation_fn(self, activation: str):
        """Get activation function from nnx or jax.nn.

        Args:
            activation: Name of activation function

        Returns:
            Activation function
        """
        # Try nnx first (preferred)
        if hasattr(nnx, activation):
            return getattr(nnx, activation)
        # Fall back to jax.nn if not found in nnx
        elif hasattr(jax.nn, activation):
            return getattr(jax.nn, activation)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def __call__(self, z: jax.Array) -> jax.Array:
        """Forward pass through generator.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.
        Training/eval modes automatically control BatchNorm and Dropout behavior.

        Args:
            z: Latent vector of shape (batch_size, latent_dim)

        Returns:
            Generated output of shape (batch_size, *output_shape[1:])
        """
        x = z

        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply batch normalization if enabled
            if self.bn_layers is not None:
                x = self.bn_layers[i](x)

            # Apply activation function
            x = self.activation_fn(x)

            # Apply dropout if enabled
            if self.dropout is not None:
                x = self.dropout(x)

        # Output layer with tanh activation
        x = self.output_layer(x)
        x = jnp.reshape(x, (-1, *self.output_shape[1:]))

        # Apply tanh activation for bounded outputs (common in GANs)
        return jnp.tanh(x)


class Discriminator(nnx.Module):
    """Discriminator network for GAN.

    Base discriminator using fully-connected (Dense) layers.
    For convolutional discriminators, use DCGANDiscriminator or other specialized subclasses.
    """

    def __init__(
        self,
        config: DiscriminatorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize discriminator network.

        Args:
            config: DiscriminatorConfig with network architecture parameters
            rngs: Random number generators

        Raises:
            ValueError: If rngs is None
            TypeError: If config is not DiscriminatorConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for Discriminator")

        if not isinstance(config, DiscriminatorConfig):
            raise TypeError(f"config must be DiscriminatorConfig, got {type(config).__name__}")

        super().__init__()

        # Store config for reference
        self.config = config

        # Extract configuration values
        hidden_dims = list(config.hidden_dims)
        self.hidden_dims = hidden_dims
        self.input_shape = config.input_shape
        self.activation = config.activation
        self.leaky_relu_slope = config.leaky_relu_slope
        self.batch_norm = config.batch_norm
        self.dropout_rate = config.dropout_rate
        self.use_spectral_norm = config.use_spectral_norm

        # Store activation function
        self.activation_fn = self._get_activation_fn(config.activation, config.leaky_relu_slope)

        # Compute input dimension from shape
        input_dim = int(jnp.prod(jnp.array(config.input_shape[1:])))

        # Pre-allocate all layers (no lazy initialization)
        layers_list = []
        curr_dim = input_dim
        for dim in hidden_dims:
            layer = nnx.Linear(in_features=curr_dim, out_features=dim, rngs=rngs)
            if self.use_spectral_norm:
                # Note: Spectral normalization would be implemented here
                # but is not currently available in flax.nnx
                pass
            layers_list.append(layer)
            curr_dim = dim
        self.layers = nnx.List(layers_list)

        # Output layer (single output for discrimination)
        self.output_layer = nnx.Linear(in_features=curr_dim, out_features=1, rngs=rngs)

        # Batch norm layers if needed using nnx.List
        if self.batch_norm:
            self.bn_layers = nnx.List([nnx.BatchNorm(dim, rngs=rngs) for dim in hidden_dims])
        else:
            self.bn_layers = None

        # Create dropout layer if needed
        if self.dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=self.dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def _get_activation_fn(self, activation: str, leaky_relu_slope: float):
        """Get activation function.

        Args:
            activation: Name of activation function
            leaky_relu_slope: Negative slope for leaky ReLU. Must be explicit.

        Returns:
            Activation function
        """
        if activation == "leaky_relu":
            # Create a closure that captures the slope
            return lambda x: jax.nn.leaky_relu(x, negative_slope=leaky_relu_slope)
        # Try nnx first (preferred)
        elif hasattr(nnx, activation):
            return getattr(nnx, activation)
        # Fall back to jax.nn if not found in nnx
        elif hasattr(jax.nn, activation):
            return getattr(jax.nn, activation)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through discriminator.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.
        Training/eval modes automatically control BatchNorm and Dropout behavior.

        Args:
            x: Input data of shape (batch_size, *features)

        Returns:
            Discriminator output of shape (batch_size, 1) in range [0, 1]
        """
        # Flatten input
        x = jnp.reshape(x, (x.shape[0], -1))

        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply batch normalization if enabled
            if self.bn_layers is not None:
                x = self.bn_layers[i](x)

            # Apply activation function
            x = self.activation_fn(x)

            # Apply dropout if enabled
            if self.dropout is not None:
                x = self.dropout(x)

        # Output layer with sigmoid activation for [0,1] range
        x = self.output_layer(x)
        return jax.nn.sigmoid(x)


class GAN(GenerativeModel):
    """Generative Adversarial Network (GAN) implementation.

    Base GAN using fully-connected Generator and Discriminator.
    For convolutional GANs, use DCGAN or other specialized subclasses.
    """

    def __init__(
        self,
        config: GANConfig,
        *,
        rngs: nnx.Rngs,
        precision: jax.lax.Precision | None = None,
    ):
        """Initialize GAN model.

        Args:
            config: GANConfig with nested GeneratorConfig and DiscriminatorConfig
            rngs: Random number generators
            precision: Numerical precision for computations

        Raises:
            ValueError: If rngs is None or missing 'sample' stream
            TypeError: If config is not GANConfig
        """
        if rngs is None:
            raise ValueError("rngs must be provided for GAN")

        if not isinstance(config, GANConfig):
            raise TypeError(f"config must be GANConfig, got {type(config).__name__}")

        super().__init__(
            rngs=rngs,
            precision=precision,
        )

        # Store config
        self.config = config

        # Store RNG for dynamic use in generate() and loss_fn()
        # Validate that 'sample' stream exists
        if "sample" not in rngs:
            raise ValueError(
                "rngs must contain 'sample' stream for GAN. "
                "Initialize with: nnx.Rngs(params=0, dropout=1, sample=2)"
            )
        self.rngs = rngs

        # Extract GAN-level hyperparameters from config
        self.latent_dim = config.generator.latent_dim
        self.loss_type = config.loss_type
        self.gradient_penalty_weight = config.gradient_penalty_weight

        # Create generator using nested GeneratorConfig
        self.generator = Generator(
            config=config.generator,
            rngs=rngs,
        )

        # Create discriminator using nested DiscriminatorConfig
        self.discriminator = Discriminator(
            config=config.discriminator,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, **kwargs: Any) -> dict[str, Any]:
        """Forward pass through the GAN model.

        For a GAN, this runs data through the discriminator.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            x: Input data of shape (batch_size, *data_shape)
            **kwargs: Additional keyword arguments

        Returns:
            Dict with model outputs including discriminator scores
        """
        real_scores = self.discriminator(x)

        # In regular forward pass, we don't generate fake samples
        # Those are only created in specific methods like generate() or loss_fn()
        fake_samples = None
        fake_scores = None

        return {
            "real_scores": real_scores,
            "fake_scores": fake_scores,
            "fake_samples": fake_samples,
        }

    def generate(
        self,
        n_samples: int = 1,
        *,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Generate samples from the generator.

        Note: Uses stored self.rngs for sampling. RNG automatically advances each call.

        Args:
            n_samples: Number of samples to generate
            batch_size: Alternative way to specify number of samples (for compatibility)
            **kwargs: Additional keyword arguments

        Returns:
            Generated samples of shape (num_samples, *output_shape[1:])
        """
        # Use batch_size if provided, otherwise use n_samples
        num_samples = batch_size if batch_size is not None else n_samples

        # Sample from latent space using stored RNG (validated at init)
        # RNG automatically advances each call
        sample_rng = self.rngs.sample()

        # Sample latent vectors from standard normal distribution
        z = jax.random.normal(sample_rng, (num_samples, self.latent_dim))

        # Generate samples through generator network
        return self.generator(z)

    def loss_fn(
        self,
        batch: dict[str, Any],
        model_outputs: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute GAN loss for training.

        Note: Uses stored self.rngs for sampling. RNG automatically advances each call.

        Args:
            batch: Input batch containing real data (dict with 'x' or 'data' key, or raw array)
            model_outputs: Model outputs (unused for GAN loss computation)
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary containing:
                - loss: Total combined loss (generator + discriminator)
                - generator_loss: Generator loss component
                - discriminator_loss: Discriminator loss component
                - real_scores_mean: Mean discriminator score on real data
                - fake_scores_mean: Mean discriminator score on fake data
        """
        # Extract real data from batch
        if isinstance(batch, dict):
            real_data = batch.get("x", batch.get("data"))
            if real_data is None:
                raise ValueError("Batch must contain 'x' or 'data' key")
        else:
            real_data = batch

        # Sample RNG for generating latent vectors using stored RNG (validated at init)
        # RNG automatically advances each call
        sample_rng = self.rngs.sample()

        # Sample latent vectors from standard normal distribution
        batch_size = real_data.shape[0]
        z = jax.random.normal(sample_rng, (batch_size, self.latent_dim))

        # Generate fake samples
        fake_samples = self.generator(z)

        # Get discriminator outputs for real and fake samples
        real_scores = self.discriminator(real_data)
        fake_scores = self.discriminator(fake_samples)

        # Compute loss based on specified loss type
        if self.loss_type == "vanilla":
            generator_loss = adversarial.vanilla_generator_loss(fake_scores)
            discriminator_loss = adversarial.vanilla_discriminator_loss(real_scores, fake_scores)
        elif self.loss_type == "least_squares":
            generator_loss = adversarial.least_squares_generator_loss(fake_scores)
            discriminator_loss = adversarial.least_squares_discriminator_loss(
                real_scores, fake_scores
            )
        elif self.loss_type == "wasserstein":
            generator_loss = adversarial.wasserstein_generator_loss(fake_scores)
            discriminator_loss = adversarial.wasserstein_discriminator_loss(
                real_scores, fake_scores
            )
        elif self.loss_type == "hinge":
            generator_loss = adversarial.hinge_generator_loss(fake_scores)
            discriminator_loss = adversarial.hinge_discriminator_loss(real_scores, fake_scores)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        # Calculate total combined loss
        total_loss = generator_loss + discriminator_loss

        # Return metrics dictionary
        return {
            "loss": total_loss,
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_loss,
            "real_scores_mean": jnp.mean(real_scores),
            "fake_scores_mean": jnp.mean(fake_scores),
        }
