"""Inverse Autoregressive Flow (IAF) implementation.

Based on the paper "Improved Variational Inference with Inverse Autoregressive Flow"
by Kingma et al. Reference implementations from benchmark_VAE and other sources are used.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import IAFConfig
from artifex.generative_models.models.flow.base import FlowLayer, NormalizingFlow
from artifex.generative_models.models.flow.made import MADE


class IAFLayer(FlowLayer):
    """A single IAF layer using MADE for autoregressive transformation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        rngs: nnx.Rngs,
        order: jax.Array | None = None,
    ):
        """Initialize IAF layer.

        Args:
            input_dim: Input dimension
            hidden_dims: Hidden dimensions for the autoregressive network
            rngs: Random number generators
            order: Variable ordering for autoregressive connections
        """
        super().__init__(rngs=rngs)
        self.input_dim = input_dim

        # Create MADE network for IAF
        self.made = MADE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_multiplier=2,  # For mean and log_scale
            rngs=rngs,
            order=order,
        )

    def forward(self, z: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Forward transformation (z -> x).

        For IAF, this is the efficient direction since we can compute
        all transformations in parallel. Based on benchmark_VAE reference.

        Args:
            z: Input latent tensor of shape (batch_size, input_dim)
            rngs: Optional random number generators

        Returns:
            Tuple of (transformed_output, log_det_jacobian)
        """
        # Get autoregressive parameters
        mean, log_scale = self.made(z)

        # Apply transformation: x = mean + z * exp(log_scale)
        # Following benchmark_VAE: y = y * s.exp() + m
        x = z * jnp.exp(log_scale) + mean

        # Log determinant is sum of log_scale
        log_det = jnp.sum(log_scale, axis=-1)

        # Flip dimensions as in benchmark_VAE reference
        x = jnp.flip(x, axis=-1)

        return x, log_det

    def inverse(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Inverse transformation (x -> z).

        For IAF, this is the slow direction requiring sequential computation.
        Based on the reference implementation from benchmark_VAE.

        Args:
            x: Input data tensor of shape (batch_size, input_dim)
            rngs: Optional random number generators

        Returns:
            Tuple of (latent_output, log_det_jacobian)
        """
        # Reverse the dimension flip from forward pass
        x = jnp.flip(x, axis=-1)

        batch_size = x.shape[0]
        z = jnp.zeros_like(x)
        log_det_jac = jnp.zeros(batch_size)

        # Sequential computation for inverse (autoregressive property)
        # Each z_i depends on previously computed z_{1...i-1}
        for i in range(self.input_dim):
            # Get parameters from MADE using current partial z
            # This is key: we use the current state of z (with zeros for future variables)
            mean, log_scale = self.made(z)

            # Inverse transformation: z_i = (x_i - mean_i) / exp(log_scale_i)
            # Following benchmark_VAE: y[:, i] = (x[:, i] - m[:, i]) * (-s[:, i]).exp()
            # Which is equivalent to: (x[:, i] - m[:, i]) / exp(s[:, i])
            z_i = (x[:, i] - mean[:, i]) * jnp.exp(-log_scale[:, i])

            # Update z at position i
            z = z.at[:, i].set(z_i)

            # Accumulate log determinant
            # For inverse: log|det(J^-1)| = -log|det(J)| = -sum(log_scale)
            log_det_jac -= log_scale[:, i]

        return z, log_det_jac


class IAF(NormalizingFlow):
    """Inverse Autoregressive Flow model.

    A normalizing flow that uses inverse autoregressive transformations
    to model complex posterior distributions in variational inference.
    """

    def __init__(
        self,
        config: IAFConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize IAF model.

        Args:
            config: IAF configuration.
            rngs: Random number generators.

        Raises:
            TypeError: If config is not a IAFConfig
        """
        if not isinstance(config, IAFConfig):
            raise TypeError(f"config must be IAFConfig, got {type(config).__name__}")

        super().__init__(config, rngs=rngs)

        # Extract configuration from IAFConfig
        input_dim = config.input_dim
        if isinstance(input_dim, (tuple, list)):
            self.iaf_input_dim = int(jnp.prod(jnp.array(input_dim)))
            self.original_shape = input_dim
            # Set input_dim for compatibility with base class and tests
            self.input_dim = self.iaf_input_dim
        else:
            self.iaf_input_dim = input_dim
            self.original_shape = (input_dim,)
            self.input_dim = input_dim

        # Extract configuration from IAFConfig
        self.hidden_dims = list(config.coupling_network.hidden_dims)
        self.num_layers = config.num_layers
        self.reverse_ordering = config.reverse_ordering

        # Build IAF layers
        self.flow_layers = nnx.List([])
        for i in range(self.num_layers):
            # Alternate ordering between layers if specified
            if self.reverse_ordering and i % 2 == 1:
                order = jnp.arange(self.iaf_input_dim)[::-1]
            else:
                order = None

            layer = IAFLayer(
                input_dim=self.iaf_input_dim,
                hidden_dims=self.hidden_dims,
                rngs=rngs,
                order=order,
            )
            self.flow_layers.append(layer)

    def forward(self, z: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Forward transformation (z -> x).

        Args:
            z: Input latent tensor
            rngs: Optional random number generators

        Returns:
            Tuple of (transformed_output, log_det_jacobian)
        """
        # Flatten input if needed
        batch_size = z.shape[0]
        if len(z.shape) > 2:
            z_flat = z.reshape(batch_size, -1)
        else:
            z_flat = z

        x = z_flat
        total_logdet = jnp.zeros(batch_size)

        # Forward through all layers
        for layer in self.flow_layers:
            x, logdet = layer.forward(x)
            total_logdet += logdet

        return x, total_logdet

    def inverse(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, jax.Array]:
        """Inverse transformation (x -> z).

        Args:
            x: Input data tensor
            rngs: Optional random number generators

        Returns:
            Tuple of (latent_output, log_det_jacobian)
        """
        # Flatten input if needed
        batch_size = x.shape[0]
        if len(x.shape) > 2:
            x_flat = x.reshape(batch_size, -1)
        else:
            x_flat = x

        z = x_flat
        total_logdet = jnp.zeros(batch_size)

        # Inverse through all layers (in reverse order)
        for layer in reversed(self.flow_layers):
            z, logdet = layer.inverse(z)
            total_logdet += logdet

        return z, total_logdet

    def __call__(
        self,
        x: jax.Array,
        *args,
        rngs: nnx.Rngs | None = None,
        training: bool = False,
        **kwargs,
    ) -> dict[str, jax.Array]:
        """Forward pass through IAF.

        Args:
            x: Input latent tensor (using x for compatibility with base class)
            rngs: Optional random number generators
            training: Whether in training mode
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with model outputs including x, logdet
        """
        # Forward through the flow
        output_x, total_logdet = self.forward(x, rngs=rngs)

        # Reshape output back to original shape if needed
        batch_size = output_x.shape[0]
        if len(self.original_shape) > 1:
            output_x = output_x.reshape(batch_size, *self.original_shape)

        return {
            "x": output_x,
            "logdet": total_logdet,
        }

    def sample(self, n_samples: int = 1, *, rngs: nnx.Rngs | None = None, **kwargs) -> jax.Array:
        """Sample from the IAF model.

        Args:
            n_samples: Number of samples to generate
            rngs: Optional random number generators

        Returns:
            Generated samples
        """
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(42))

        # Sample from base distribution (standard Gaussian)
        if hasattr(rngs, "params") and callable(rngs.params):
            key = rngs.params()
        else:
            # Fallback to a default key if rngs is not properly set up
            key = jax.random.PRNGKey(42)

        z = jax.random.normal(key, (n_samples, self.iaf_input_dim))

        # Transform through flow
        result = self(z, rngs=rngs, training=False)
        return result["x"]

    def log_prob(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
        training: bool = False,
        **kwargs,
    ) -> jax.Array:
        """Compute log probability of data.

        Args:
            x: Input data
            rngs: Optional random number generators
            training: Whether in training mode

        Returns:
            Log probabilities
        """
        # For IAF, we need to do inverse transformation first
        # This is computationally expensive for IAF
        z, total_logdet = self.inverse(x, rngs=rngs)

        # Log probability of base distribution (standard Gaussian)
        log_base = -0.5 * jnp.sum(z**2, axis=-1) - 0.5 * self.iaf_input_dim * jnp.log(2 * jnp.pi)

        # Total log probability
        return log_base + total_logdet
