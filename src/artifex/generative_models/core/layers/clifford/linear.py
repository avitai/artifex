"""CliffordLinear — Clifford algebra linear layer.

Ported from ``microsoft/cliffordlayers`` (MIT license).
Uses a single stacked ``nnx.Param`` instead of ``nn.ParameterList``.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.layers.clifford.kernels import (
    get_1d_clifford_kernel,
    get_2d_clifford_kernel,
    get_3d_clifford_kernel,
)


_KERNEL_DISPATCH: dict[int, type] = {
    1: get_1d_clifford_kernel,
    2: get_2d_clifford_kernel,
    3: get_3d_clifford_kernel,
}


class CliffordLinear(nnx.Module):
    """Clifford algebra linear layer.

    Applies a linear transformation in Clifford space via the geometric
    product structure encoded in the kernel construction.

    Input shape:  ``(B, C_in, n_blades)``
    Output shape: ``(B, C_out, n_blades)``

    Args:
        metric: Diagonal metric entries, e.g. ``(1, 1)`` for Cl(2,0).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        use_bias: Whether to add a learnable bias.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        metric: tuple[int, ...],
        in_channels: int,
        out_channels: int,
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize CliffordLinear."""
        super().__init__()
        self.metric = jnp.asarray(metric, dtype=jnp.float32)
        self.dim = len(metric)
        self.n_blades = 2**self.dim

        if self.dim not in _KERNEL_DISPATCH:
            raise ValueError(f"CliffordLinear supports dim 1-3, got {self.dim}.")

        self._get_kernel = _KERNEL_DISPATCH[self.dim]
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Kaiming uniform initialization
        fan_in = in_channels * self.n_blades
        bound = 1.0 / math.sqrt(fan_in)
        key = rngs.params()
        self.weight = nnx.Param(
            jax.random.uniform(
                key, (self.n_blades, out_channels, in_channels), minval=-bound, maxval=bound
            )
        )

        if use_bias:
            key_b = rngs.params()
            self.bias = nnx.Param(
                jax.random.uniform(
                    key_b, (self.n_blades, out_channels), minval=-bound, maxval=bound
                )
            )
        else:
            self.bias = None

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        """Forward pass.

        Args:
            x: Input ``(B, C_in, n_blades)``.
            deterministic: Unused, kept for API consistency.

        Returns:
            Output ``(B, C_out, n_blades)``.
        """
        batch = x.shape[0]
        n_blades = x.shape[-1]
        if n_blades != self.n_blades:
            raise ValueError(f"Input has {n_blades} blades, expected {self.n_blades}.")

        # (B, C, I) -> (B, I, C) -> (B, I*C)
        x_perm = jnp.moveaxis(x, -1, 1)
        x_flat = x_perm.reshape(batch, -1)

        # Construct Clifford kernel and apply
        _, kernel = self._get_kernel(self.weight.value, self.metric)
        # kernel: (I*O, I*C)
        output = x_flat @ kernel.T

        if self.bias is not None:
            output = output + self.bias.value.reshape(-1)

        # (B, I*O) -> (B, I, O) -> (B, O, I)
        output = output.reshape(batch, self.n_blades, -1)
        output = jnp.moveaxis(output, 1, -1)
        return output
