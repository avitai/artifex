"""Base Generative Adversarial Network (GAN) implementation."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel, get_activation_function
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

        # Store activation function using canonical resolver
        self.activation_fn = get_activation_function(config.activation)

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

        # Store activation function — use custom slope for leaky_relu, canonical resolver otherwise
        if config.activation == "leaky_relu":
            slope = config.leaky_relu_slope
            self.activation_fn = lambda x: nnx.leaky_relu(x, negative_slope=slope)
        else:
            self.activation_fn = get_activation_function(config.activation)

        # Compute input dimension from shape
        input_dim = int(jnp.prod(jnp.array(config.input_shape[1:])))

        # Pre-allocate all layers (no lazy initialization)
        layers_list = []
        curr_dim = input_dim
        for dim in hidden_dims:
            layer = nnx.Linear(in_features=curr_dim, out_features=dim, rngs=rngs)
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
        return nnx.sigmoid(x)


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

        # Store RNG for dynamic use in generation and objective evaluation
        # Validate that 'sample' stream exists
        if "sample" not in rngs:
            raise ValueError(
                "rngs must contain 'sample' stream for GAN. "
                "Initialize with: nnx.Rngs(params=0, dropout=1, sample=2)"
            )
        self.rngs = rngs

        generator_config = config.generator
        discriminator_config = config.discriminator
        if not isinstance(generator_config, GeneratorConfig):
            raise TypeError(
                f"config.generator must be GeneratorConfig, got {type(generator_config).__name__}"
            )
        if not isinstance(discriminator_config, DiscriminatorConfig):
            raise TypeError(
                "config.discriminator must be DiscriminatorConfig, "
                f"got {type(discriminator_config).__name__}"
            )

        # Extract GAN-level hyperparameters from config
        self.latent_dim = generator_config.latent_dim
        self.loss_type = config.loss_type
        self.gradient_penalty_weight = config.gradient_penalty_weight

        # Create generator using nested GeneratorConfig
        self.generator = Generator(
            config=generator_config,
            rngs=rngs,
        )

        # Create discriminator using nested DiscriminatorConfig
        self.discriminator = Discriminator(
            config=discriminator_config,
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
        # Those are only created in dedicated objective methods.
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

    def _extract_real_data(self, batch: dict[str, Any] | jax.Array) -> jax.Array:
        """Extract real samples from a GAN batch."""
        if isinstance(batch, dict):
            real_data = batch.get("x", batch.get("data"))
            if real_data is None:
                raise ValueError("Batch must contain 'x' or 'data' key")
        else:
            real_data = batch
        return real_data

    def _sample_latents(self, batch_size: int) -> jax.Array:
        """Sample latent vectors using the model-managed RNG stream."""
        sample_rng = self.rngs.sample()
        return jax.random.normal(sample_rng, (batch_size, self.latent_dim))

    def _compute_generator_loss(self, fake_scores: jax.Array) -> jax.Array:
        """Compute generator loss for the configured adversarial objective."""
        return adversarial.generator_loss(fake_scores, loss_type=self.loss_type)

    def _compute_discriminator_loss(
        self,
        real_scores: jax.Array,
        fake_scores: jax.Array,
    ) -> jax.Array:
        """Compute discriminator loss for the configured adversarial objective."""
        return adversarial.discriminator_loss(
            real_scores,
            fake_scores,
            loss_type=self.loss_type,
        )

    def generator_objective(
        self,
        batch: dict[str, Any] | jax.Array,
    ) -> dict[str, jax.Array]:
        """Compute the generator-only objective.

        GAN training has separate generator and discriminator optimization
        targets. This method exposes the generator target explicitly.
        """
        real_data = self._extract_real_data(batch)
        z = self._sample_latents(real_data.shape[0])
        fake_samples = self.generator(z)
        fake_scores = self.discriminator(fake_samples)
        generator_loss = self._compute_generator_loss(fake_scores)

        return {
            "total_loss": generator_loss,
            "generator_loss": generator_loss,
            "fake_scores_mean": jnp.mean(fake_scores),
        }

    def discriminator_objective(
        self,
        batch: dict[str, Any] | jax.Array,
    ) -> dict[str, jax.Array]:
        """Compute the discriminator-only objective."""
        real_data = self._extract_real_data(batch)
        z = self._sample_latents(real_data.shape[0])
        fake_samples = jax.lax.stop_gradient(self.generator(z))
        real_scores = self.discriminator(real_data)
        fake_scores = self.discriminator(fake_samples)
        discriminator_loss = self._compute_discriminator_loss(real_scores, fake_scores)

        return {
            "total_loss": discriminator_loss,
            "discriminator_loss": discriminator_loss,
            "real_scores_mean": jnp.mean(real_scores),
            "fake_scores_mean": jnp.mean(fake_scores),
        }

    def loss_fn(
        self,
        batch: dict[str, Any],
        model_outputs: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """This class does not expose a combined single-objective loss surface."""
        del batch, model_outputs, kwargs
        raise NotImplementedError(
            "GAN training requires separate generator and discriminator objectives. "
            "Use generator_objective(), discriminator_objective(), or GANTrainer."
        )
