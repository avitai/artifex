"""Clifford normalization and multi-vector activation layers.

Ported from ``microsoft/cliffordlayers`` (MIT license).
Provides unified CliffordBatchNorm (no separate 1d/2d/3d),
CliffordGroupNorm, and MultiVectorActivation.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.layers.clifford._functional import whiten_data


class CliffordBatchNorm(nnx.Module):
    """Clifford batch normalization.

    Unified for all spatial dimensionalities. Jointly whitens the
    ``I``-dimensional blade vectors per channel via Cholesky decomposition,
    then applies an optional learnable affine transformation.

    Input:  ``(B, *D, C, I)``
    Output: ``(B, *D, C, I)``

    Args:
        metric: Diagonal metric entries, e.g. ``(1, 1)`` for Cl(2,0).
        channels: Number of channels.
        epsilon: Regularization for Cholesky decomposition.
        momentum: EMA momentum for running statistics.
        use_affine: Whether to apply learnable affine transformation.
        use_running_stats: Whether to maintain running statistics.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        metric: tuple[int, ...],
        channels: int,
        epsilon: float = 1e-5,
        momentum: float = 0.1,
        use_affine: bool = True,
        use_running_stats: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize CliffordBatchNorm."""
        super().__init__()
        self.n_blades = 2 ** len(metric)
        self.channels = channels
        self.epsilon = epsilon
        self.momentum = momentum

        if use_affine:
            # Identity transform at init — weight: (I, I, C), bias: (I, C)
            eye = jnp.eye(self.n_blades, dtype=jnp.float32)
            self.weight = nnx.Param(
                jnp.broadcast_to(eye[..., None], (self.n_blades, self.n_blades, channels)).copy()
            )
            self.bias = nnx.Param(jnp.zeros((self.n_blades, channels)))
        else:
            self.weight = None
            self.bias = None

        if use_running_stats:
            self.running_mean = nnx.BatchStat(jnp.zeros((self.n_blades, channels)))
            eye = jnp.eye(self.n_blades, dtype=jnp.float32)
            self.running_cov = nnx.BatchStat(
                jnp.broadcast_to(eye[..., None], (self.n_blades, self.n_blades, channels)).copy()
            )
        else:
            self.running_mean = None
            self.running_cov = None

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        """Forward pass.

        Args:
            x: Input ``(B, *D, C, I)``.
            deterministic: If True, use running stats; if False, compute from batch.

        Returns:
            Normalized output ``(B, *D, C, I)``.
        """
        n_blades = x.shape[-1]
        if n_blades != self.n_blades:
            raise ValueError(f"Input has {n_blades} blades, expected {self.n_blades}.")

        training = not deterministic
        rm = self.running_mean.value if self.running_mean is not None else None
        rc = self.running_cov.value if self.running_cov is not None else None

        x_norm, new_rm, new_rc = whiten_data(
            x,
            training=training,
            running_mean=rm,
            running_cov=rc,
            momentum=self.momentum,
            eps=self.epsilon,
        )

        # Update running stats in training mode
        if training and self.running_mean is not None and new_rm is not None:
            self.running_mean.value = new_rm
        if training and self.running_cov is not None and new_rc is not None:
            self.running_cov.value = new_rc

        # Affine transform
        if self.weight is not None and self.bias is not None:
            x_norm = self._apply_affine(x_norm)

        return x_norm

    def _apply_affine(self, x: jax.Array) -> jax.Array:
        """Apply learnable ``(I, I)`` matrix multiply + bias per channel."""
        n_spatial = x.ndim - 3  # exclude B, C, I

        # weight: (I, I, C) -> (C, I, I) -> (1, 1..., C, I, I)
        w = jnp.transpose(self.weight.value, (2, 0, 1))
        w = w.reshape((1,) * (1 + n_spatial) + (self.channels, self.n_blades, self.n_blades))

        # bias: (I, C) -> (C, I) -> (1, 1..., C, I)
        b = self.bias.value.T
        b = b.reshape((1,) * (1 + n_spatial) + (self.channels, self.n_blades))

        # (..., C, I, I) @ (..., C, I, 1) -> squeeze -> (..., C, I)
        return jnp.matmul(w, x[..., None]).squeeze(-1) + b


class CliffordGroupNorm(nnx.Module):
    """Clifford group normalization.

    Divides channels into groups and whitens within each group.
    No running statistics — always computes from the input.

    Input:  ``(B, *D, C, I)``
    Output: ``(B, *D, C, I)``

    Args:
        metric: Diagonal metric entries.
        num_groups: Number of channel groups.
        channels: Total number of channels (must be divisible by ``num_groups``).
        epsilon: Regularization for Cholesky decomposition.
        use_affine: Whether to apply learnable affine transformation.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        metric: tuple[int, ...],
        num_groups: int,
        channels: int,
        epsilon: float = 1e-5,
        use_affine: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize CliffordGroupNorm."""
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_groups ({num_groups})."
            )

        self.n_blades = 2 ** len(metric)
        self.num_groups = num_groups
        self.channels = channels
        self.channels_per_group = channels // num_groups
        self.epsilon = epsilon

        if use_affine:
            eye = jnp.eye(self.n_blades, dtype=jnp.float32)
            self.weight = nnx.Param(
                jnp.broadcast_to(
                    eye[..., None], (self.n_blades, self.n_blades, self.channels_per_group)
                ).copy()
            )
            self.bias = nnx.Param(jnp.zeros((self.n_blades, self.channels_per_group)))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: Input ``(B, *D, C, I)``.

        Returns:
            Normalized output ``(B, *D, C, I)``.
        """
        batch = x.shape[0]
        spatial = x.shape[1:-2]
        n_spatial = len(spatial)
        n_blades = x.shape[-1]

        # Split channels into groups: (B, *D, C, I) -> (B, *D, G, C_pg, I)
        x = x.reshape(batch, *spatial, self.num_groups, self.channels_per_group, n_blades)

        # Move G after B: (B, *D, G, C_pg, I) -> (B, G, *D, C_pg, I) -> (B*G, *D, C_pg, I)
        perm_fwd = [0, 1 + n_spatial, *range(1, 1 + n_spatial), 2 + n_spatial, 3 + n_spatial]
        x = jnp.transpose(x, perm_fwd)
        x = x.reshape(batch * self.num_groups, *spatial, self.channels_per_group, n_blades)

        # Whiten (always from data, no running stats)
        x_norm, _, _ = whiten_data(
            x,
            training=True,
            running_mean=None,
            running_cov=None,
            momentum=0.0,
            eps=self.epsilon,
        )

        # Reshape back: (B*G, *D, C_pg, I) -> (B, G, *D, C_pg, I) -> (B, *D, G, C_pg, I)
        x_norm = x_norm.reshape(batch, self.num_groups, *spatial, self.channels_per_group, n_blades)
        perm_bwd = [0, *range(2, 2 + n_spatial), 1, 2 + n_spatial, 3 + n_spatial]
        x_norm = jnp.transpose(x_norm, perm_bwd)

        # (B, *D, G, C_pg, I) -> (B, *D, C, I)
        x_norm = x_norm.reshape(batch, *spatial, self.channels, n_blades)

        # Affine: per-group weight/bias tiled across groups
        if self.weight is not None and self.bias is not None:
            # Tile to full channel count: (I, I, C_pg) -> (I, I, C)
            w = jnp.tile(self.weight.value, (1, 1, self.num_groups))
            b = jnp.tile(self.bias.value, (1, self.num_groups))

            # (I, I, C) -> (C, I, I) -> (1, 1..., C, I, I)
            w = jnp.transpose(w, (2, 0, 1))
            w = w.reshape((1,) * (1 + n_spatial) + (self.channels, self.n_blades, self.n_blades))

            # (I, C) -> (C, I) -> (1, 1..., C, I)
            b = b.T
            b = b.reshape((1,) * (1 + n_spatial) + (self.channels, self.n_blades))

            x_norm = jnp.matmul(w, x_norm[..., None]).squeeze(-1) + b

        return x_norm


class MultiVectorActivation(nnx.Module):
    """Multi-vector activation for Clifford algebra layers.

    Applies a gated activation based on selected blade components.
    Does NOT store a ``CliffordAlgebra`` — uses direct index operations
    for embed/get.

    Input:  ``(B, *D, C, n_blades)``
    Output: ``(B, *D, C, n_blades)``

    Args:
        channels: Number of channels.
        n_blades: Total number of algebra blades.
        input_blades: Indices of active blades in the input.
        kernel_blades: Blade indices for computing the gate. Defaults to
            ``input_blades`` if ``None``.
        aggregation: ``"linear"``, ``"sum"``, or ``"mean"``.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        channels: int,
        n_blades: int,
        input_blades: tuple[int, ...],
        kernel_blades: tuple[int, ...] | None = None,
        aggregation: str = "linear",
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize MultiVectorActivation."""
        super().__init__()
        self.channels = channels
        self.n_blades = n_blades
        self.input_blades = tuple(input_blades)
        self.kernel_blades = (
            tuple(kernel_blades) if kernel_blades is not None else self.input_blades
        )
        self.aggregation = aggregation

        if aggregation == "linear":
            fan_in = len(self.kernel_blades)
            bound = 1.0 / fan_in**0.5
            key = rngs.params()
            self.agg_weight = nnx.Param(
                jax.random.uniform(key, (channels, fan_in), minval=-bound, maxval=bound)
            )
            key_b = rngs.params()
            self.agg_bias = nnx.Param(
                jax.random.uniform(key_b, (channels,), minval=-bound, maxval=bound)
            )
        elif aggregation not in ("sum", "mean"):
            raise ValueError(f"Unknown aggregation: {aggregation!r}")

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: Input ``(B, *D, C, n_blades)``.

        Returns:
            Gated output ``(B, *D, C, n_blades)``.
        """
        # Embed: scatter active blades into full multivector
        v = self._embed(x)

        # Extract kernel blades for gating
        kb = jnp.asarray(self.kernel_blades)
        v_kb = v[..., kb]  # (..., C, len_kb)

        # Compute gate
        if self.aggregation == "linear":
            gate = jnp.einsum("...ck,ck->...c", v_kb, self.agg_weight.value) + self.agg_bias.value
            gate = jax.nn.sigmoid(gate[..., None])  # (..., C, 1)
        elif self.aggregation == "sum":
            gate = jax.nn.sigmoid(jnp.sum(v_kb, axis=-1, keepdims=True))
        elif self.aggregation == "mean":
            gate = jax.nn.sigmoid(jnp.mean(v_kb, axis=-1, keepdims=True))
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation!r}")

        # Apply gate and extract output blades
        return self._get(v * gate)

    def _embed(self, x: jax.Array) -> jax.Array:
        """Scatter input blades into full multivector."""
        if len(self.input_blades) == self.n_blades:
            return x
        indices = jnp.asarray(self.input_blades)
        shape = (*x.shape[:-1], self.n_blades)
        return jnp.zeros(shape, dtype=x.dtype).at[..., indices].set(x)

    def _get(self, v: jax.Array) -> jax.Array:
        """Extract active blades from full multivector."""
        if len(self.input_blades) == self.n_blades:
            return v
        indices = jnp.asarray(self.input_blades)
        return v[..., indices]
