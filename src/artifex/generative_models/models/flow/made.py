"""Masked Autoencoder for Distribution Estimation (MADE).

This implementation is shared between MAF and IAF models and is based on
the reference implementations from benchmark_VAE and other sources.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class MADE(nnx.Module):
    """Masked Autoencoder for Distribution Estimation.

    This is a core building block for autoregressive flows like MAF and IAF.
    It implements an autoregressive neural network with masked weights to
    ensure the autoregressive property.

    Based on the paper "MADE: Masked Autoencoder for Distribution Estimation"
    by Germain et al.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_multiplier: int = 2,
        *,
        rngs: nnx.Rngs,
        order: jax.Array | None = None,
        activation: str = "relu",
    ):
        """Initialize MADE.

        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            output_multiplier: Output dimension multiplier (2 for mean/log_scale)
            rngs: Random number generators
            order: Variable ordering (if None, uses natural ordering)
            activation: Activation function name
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_multiplier = output_multiplier
        self.output_dim = input_dim * output_multiplier

        # Set up variable ordering
        if order is None:
            self.order = jnp.arange(input_dim)
        else:
            self.order = jnp.asarray(order)

        # Build the network with masking
        self._build_network(rngs, activation)

    def _build_network(self, rngs: nnx.Rngs, activation: str):
        """Build the masked network."""
        # Create layer dimensions
        dims = (self.input_dim, *self.hidden_dims, self.output_dim)

        # Assign degrees to each unit
        self.degrees = self._assign_degrees(dims)

        # Create masked layers
        self.layers = nnx.List([])
        for i in range(len(dims) - 1):
            layer = nnx.Linear(
                dims[i],
                dims[i + 1],
                rngs=rngs,
                use_bias=True,
            )
            self.layers.append(layer)

        # Create masks for each layer
        self.masks = self._create_masks(dims)

        # Set activation function
        self.activation_name = activation

    def _assign_degrees(self, dims: list[int]) -> nnx.List:
        """Assign degrees to each unit in the network."""
        degrees = nnx.List([])

        # Input layer degrees (based on ordering)
        degrees.append(self.order)

        # Hidden layer degrees
        for dim in dims[1:-1]:
            # Sample degrees from [1, input_dim-1] for hidden units
            min_degree = jnp.min(self.order)
            max_degree = jnp.max(self.order) - 1
            degree_range = max_degree - min_degree + 1

            # Create uniform distribution of degrees
            hidden_degrees = jnp.tile(
                jnp.arange(min_degree, max_degree + 1), (dim // degree_range + 1)
            )[:dim]
            degrees.append(hidden_degrees)

        # Output layer degrees (repeated for each output multiplier)
        output_degrees = jnp.tile(self.order, self.output_multiplier)
        degrees.append(output_degrees)

        return degrees

    def _create_masks(self, dims: list[int]) -> nnx.List:
        """Create masks for each layer to enforce autoregressive property."""
        masks = nnx.List([])

        for i in range(len(dims) - 1):
            # Create mask: connection exists if input_degree < output_degree
            input_degrees = self.degrees[i]
            output_degrees = self.degrees[i + 1]

            # Broadcast to create mask matrix
            mask = input_degrees[:, None] < output_degrees[None, :]
            masks.append(mask.astype(jnp.float32))

        return masks

    def __call__(self, x: jax.Array, **kwargs) -> tuple[jax.Array, jax.Array]:
        """Forward pass through the masked network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of (mu, log_alpha) each of shape (batch_size, input_dim)
        """
        h = x

        # Forward through masked layers
        for i, layer in enumerate(self.layers[:-1]):
            # Apply masked linear transformation directly
            masked_weight = layer.kernel * self.masks[i]
            h = h @ masked_weight + layer.bias
            # Apply activation
            if self.activation_name == "relu":
                h = nnx.relu(h)
            elif self.activation_name == "tanh":
                h = jnp.tanh(h)
            elif self.activation_name == "elu":
                h = nnx.elu(h)
            else:
                raise ValueError(f"Unknown activation: {self.activation_name}")

        # Final layer (no activation)
        final_layer = self.layers[-1]
        masked_weight = final_layer.kernel * self.masks[-1]
        h = h @ masked_weight + final_layer.bias

        # Split output into mu and log_alpha
        mu, log_alpha = jnp.split(h, 2, axis=-1)
        return mu, log_alpha


def create_made(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_multiplier: int = 2,
    rngs: nnx.Rngs | None = None,
    **kwargs,
) -> MADE:
    """Factory function to create MADE models.

    Args:
        input_dim: Input dimension
        hidden_dims: Hidden layer dimensions
        output_multiplier: Output dimension multiplier
        rngs: Random number generators
        **kwargs: Additional arguments for MADE

    Returns:
        MADE instance
    """
    if rngs is None:
        rngs = nnx.Rngs(42)

    return MADE(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_multiplier=output_multiplier,
        rngs=rngs,
        **kwargs,
    )
