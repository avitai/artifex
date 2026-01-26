"""Base Normalizing Flow model implementation."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel
from artifex.generative_models.core.configuration import FlowConfig


class FlowLayer(nnx.Module):
    """Base class for normalizing flow layers.

    A flow layer implements a bijective transformation with a tractable
    Jacobian determinant.
    """

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize flow layer.

        Args:
            rngs: Random number generators
        """
        super().__init__()

    def forward(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Forward transformation of the flow layer.

        Args:
            x: Input tensor
            rngs: Optional random number generators

        Returns:
            Tuple of (transformed_x, log_det_jacobian)
        """
        raise NotImplementedError("Forward transformation not implemented")

    def inverse(self, y: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Inverse transformation of the flow layer.

        Args:
            y: Input tensor
            rngs: Optional random number generators

        Returns:
            Tuple of (transformed_y, log_det_jacobian)
        """
        raise NotImplementedError("Inverse transformation not implemented")


class NormalizingFlow(GenerativeModel):
    """Base class for normalizing flow models.

    Normalizing flows transform a simple base distribution into a complex
    target distribution through a sequence of invertible transformations,
    allowing both sampling and density estimation.
    """

    def __init__(
        self,
        config: FlowConfig,
        *,
        rngs: nnx.Rngs,
        precision: jax.lax.Precision | None = None,
    ):
        """Initialize normalizing flow model.

        Args:
            config: Flow configuration (must be FlowConfig or subclass).
            rngs: Random number generators.
            precision: Optional precision for JAX operations.

        Raises:
            TypeError: If config is not a FlowConfig
        """
        if not isinstance(config, FlowConfig):
            raise TypeError(f"config must be FlowConfig or a subclass, got {type(config).__name__}")

        super().__init__(rngs=rngs, precision=precision)
        self.config = config

        # Configuration parameters from FlowConfig
        self.input_dim = config.input_dim
        self.latent_dim = config.latent_dim

        # Extract flow parameters from the config
        self.base_distribution_type = config.base_distribution
        self.base_distribution_params = dict(config.base_distribution_params)

        # Store coupling network config for subclasses to use
        self.coupling_network_config = config.coupling_network

        # Initialize flow layers as nnx.List (required for Flax NNX 0.12.0+)
        # Subclasses can append layers to this list
        self.flow_layers = nnx.List([])

        # Set up base distribution
        self._setup_base_distribution()

    def _setup_base_distribution(self) -> None:
        """Set up the base distribution for the normalizing flow.

        Sets up log_prob_fn and sample_fn based on the specified base distribution.
        """
        if self.base_distribution_type == "normal":
            # Normal distribution
            loc = self.base_distribution_params.get("loc", 0.0)
            scale = self.base_distribution_params.get("scale", 1.0)

            def log_prob_normal(x: jax.Array) -> jax.Array:
                """Compute log probability for normal distribution."""
                # Get the shape of the input
                input_shape = x.shape
                batch_size = input_shape[0]

                # Handle different input shapes
                if len(input_shape) == 2:
                    x_flat = x
                else:
                    # Flatten all but batch dimension
                    x_flat = jnp.reshape(x, (batch_size, -1))

                # Compute log probability
                return -0.5 * jnp.sum(
                    jnp.log(2 * jnp.pi * scale**2) + ((x_flat - loc) / scale) ** 2, axis=-1
                )

            self.log_prob_fn = log_prob_normal

            def normal_sample_fn(key: jax.typing.ArrayLike, n: int) -> jax.Array:
                """Sample from normal distribution."""
                # Determine sample shape
                if isinstance(self.latent_dim, (tuple, list)):
                    sample_shape = (n, *self.latent_dim)
                else:
                    sample_shape = (n, self.latent_dim)

                # Generate samples
                samples = loc + scale * jax.random.normal(key, shape=sample_shape)
                return samples

            self.sample_fn = normal_sample_fn

        elif self.base_distribution_type == "uniform":
            low = self.base_distribution_params.get("low", -1.0)
            high = self.base_distribution_params.get("high", 1.0)

            def log_prob_uniform(x: jax.Array) -> jax.Array:
                """Compute log probability for uniform distribution."""
                # Get the shape of the input
                input_shape = x.shape
                batch_size = input_shape[0]

                # Calculate total elements per sample
                if len(input_shape) == 2:
                    elements_per_sample = self.latent_dim
                    if isinstance(elements_per_sample, (tuple, list)):
                        elements_per_sample = jnp.prod(jnp.array(elements_per_sample))
                else:
                    elements_per_sample = jnp.prod(jnp.array(input_shape[1:]))

                # Log probability is constant for uniform distribution
                log_prob = jnp.ones(batch_size) * (-jnp.log(high - low) * elements_per_sample)
                return log_prob

            self.log_prob_fn = log_prob_uniform

            def uniform_sample_fn(key: jax.typing.ArrayLike, n: int) -> jax.Array:
                """Sample from uniform distribution."""
                # Determine sample shape
                if isinstance(self.latent_dim, (tuple, list)):
                    sample_shape = (n, *self.latent_dim)
                else:
                    sample_shape = (n, self.latent_dim)

                # Generate samples
                samples = low + (high - low) * jax.random.uniform(key, shape=sample_shape)
                return samples

            self.sample_fn = uniform_sample_fn

        else:
            raise ValueError(f"Unsupported base distribution: {self.base_distribution_type}")

    def forward(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Forward transformation through all flow layers.

        Args:
            x: Input tensor from data space
            rngs: Optional random number generators

        Returns:
            Tuple of (transformed_x, total_log_det_jacobian)
        """
        y = x
        total_log_det = jnp.zeros(x.shape[0])

        # Pass through each flow layer
        for layer in self.flow_layers:
            y, log_det = layer.forward(y, rngs=rngs)
            total_log_det += log_det

        return y, total_log_det

    def inverse(self, z: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Inverse transformation through all flow layers.

        Args:
            z: Input tensor from latent space
            rngs: Optional random number generators

        Returns:
            Tuple of (reconstructed_data, total_log_det_jacobian)
        """
        x = z
        total_log_det = jnp.zeros(z.shape[0])

        # Pass through each flow layer in reverse order
        for layer in reversed(self.flow_layers):
            x, log_det = layer.inverse(x, rngs=rngs)
            total_log_det += log_det

        return x, total_log_det

    def log_prob(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Compute log probability of data points.

        Args:
            x: Input data points
            rngs: Optional random number generators

        Returns:
            Log probability of each data point
        """
        # Transform x to z (latent space)
        z, logdet = self.forward(x, rngs=rngs)

        # Compute log probability of z under base distribution
        log_prob_z = self.log_prob_fn(z)

        # Log probability of x is log probability of z plus log determinant of Jacobian
        return log_prob_z + logdet

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs | None = None, **kwargs) -> dict[str, Any]:
        """Forward pass through the flow model.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            x: Input data
            rngs: Optional random number generators
            **kwargs: Additional keyword arguments

        Returns:
            dictionary with model outputs
        """
        z, logdet = self.forward(x, rngs=rngs)
        log_prob = self.log_prob_fn(z) + logdet

        return {
            "z": z,
            "logdet": logdet,
            "log_prob": log_prob,
            "log_prob_x": log_prob,  # Alias for convenience
        }

    def generate(self, n_samples: int = 1, *, rngs: nnx.Rngs | None = None, **kwargs) -> jax.Array:
        """Generate samples from the flow model.

        Args:
            n_samples: Number of samples to generate
            rngs: Optional random number generators
            **kwargs: Additional keyword arguments

        Returns:
            Generated samples
        """
        sample_key = (rngs or self.rngs).sample()

        # Sample from base distribution
        z = self.sample_fn(sample_key, n_samples)

        # Transform through inverse flow
        x, _ = self.inverse(z, rngs=rngs)

        return x

    def loss_fn(
        self,
        batch: Any,
        model_outputs: dict[str, Any],
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Compute loss for the flow model.

        Args:
            batch: Input batch data
            model_outputs: Model outputs from forward pass
            rngs: Optional random number generators
            **kwargs: Additional keyword arguments

        Returns:
            dictionary containing loss and metrics
        """
        # Extract data from batch
        if isinstance(batch, dict):
            x = batch.get("x", batch.get("data", batch))
        else:
            x = batch

        # Compute log probability of the data
        log_prob = self.log_prob(x, rngs=rngs)

        # Loss is negative log likelihood
        loss = -jnp.mean(log_prob)

        return {
            "loss": loss,
            "nll_loss": loss,
            "log_prob": jnp.mean(log_prob),
            "avg_log_prob": jnp.mean(log_prob),
        }

    def sample(self, n_samples: int = 1, *, rngs: nnx.Rngs | None = None, **kwargs) -> jax.Array:
        """Sample from the flow model (alias for generate).

        Args:
            n_samples: Number of samples to generate
            rngs: Optional random number generators
            **kwargs: Additional keyword arguments

        Returns:
            Generated samples
        """
        return self.generate(n_samples, rngs=rngs, **kwargs)

    def log_likelihood(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Compute log likelihood of data points (alias for log_prob).

        Args:
            x: Input data points
            rngs: Optional random number generators

        Returns:
            Log likelihood of each data point
        """
        return self.log_prob(x, rngs=rngs)
