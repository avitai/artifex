"""Spline-based KAN layers (DenseKAN and EfficientKAN).

Adapted from jaxKAN BaseLayer and SplineLayer (MIT license).

Reference:
    Liu et al., "KAN: Kolmogorov-Arnold Networks" (arXiv:2404.19756)
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.layers.kan._utils import (
    _initialize_kan_params,
    _solve_full_lstsq,
)
from artifex.generative_models.core.layers.kan.grids import (
    BSplineBasis,
    DenseKANGrid,
    EfficientKANGrid,
)


class DenseKANLayer(nnx.Module):
    """Original spline-based KAN layer (dense grid: per-edge knots).

    Each edge (i, j) has its own knot vector and B-spline basis.
    Computes: y_j = sum_i [ c_spl_ji * (c_basis . B(x) + c_res * res(x)) ] + b

    Attributes:
        n_in: Number of input features.
        n_out: Number of output features.
        k: Spline order (polynomial degree).
        residual: Residual activation function, or None.
        grid: DenseKANGrid storing per-edge knot positions.
        bspline: BSplineBasis evaluator.
        c_spl: External edge weights (n_out, n_in), or None.
        c_basis: Spline coefficients (n_in*n_out, G+k).
        c_res: Residual weights (n_out, n_in), or None.
        bias: Bias (n_out,), or None.
    """

    def __init__(
        self,
        n_in: int = 2,
        n_out: int = 5,
        k: int = 3,
        grid_intervals: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        grid_e: float = 0.05,
        residual: Callable | None = nnx.silu,
        external_weights: bool = True,
        init_scheme: dict | None = None,
        add_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the dense KAN layer.

        Args:
            n_in: Number of input features.
            n_out: Number of output features.
            k: Spline order.
            grid_intervals: Number of grid intervals.
            grid_range: Initial grid range.
            grid_e: Grid mixing parameter (1.0=uniform, 0.0=adaptive).
            residual: Residual activation function, or None.
            external_weights: Whether to apply external edge weights.
            init_scheme: Initialization config dict.
            add_bias: Whether to include bias.
            rngs: Random number generators.
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.residual = residual

        self.grid = DenseKANGrid(
            n_in=n_in,
            n_out=n_out,
            k=k,
            grid_intervals=grid_intervals,
            grid_range=grid_range,
            grid_e=grid_e,
        )
        self.bspline = BSplineBasis(k=k)

        # External weights
        if external_weights:
            self.c_spl = nnx.Param(jnp.ones((n_out, n_in)))
        else:
            self.c_spl = None

        # Initialize params via shared utility
        basis_dim = grid_intervals + k
        c_res, c_basis = _initialize_kan_params(
            n_in,
            n_out,
            basis_dim,
            rngs,
            init_scheme=init_scheme,
            residual_fn=residual,
            basis_fn=self.basis,
            param_shape="dense",
        )
        self.c_basis = nnx.Param(c_basis)
        self.c_res = nnx.Param(c_res) if residual is not None else None

        # Bias
        self.bias = nnx.Param(jnp.zeros((n_out,))) if add_bias else None

    def basis(self, x: jax.Array) -> jax.Array:
        """Evaluate B-spline basis functions.

        Args:
            x: Input, shape (batch, n_in).

        Returns:
            Basis values, shape (n_in*n_out, G+k, batch).
        """
        batch = x.shape[0]
        # Extend to (batch, n_in*n_out)
        x_ext = jnp.einsum("ij,k->ikj", x, jnp.ones(self.n_out)).reshape(
            (batch, self.n_in * self.n_out)
        )
        # Transpose to (n_in*n_out, batch)
        x_ext = x_ext.T

        grid = jnp.expand_dims(self.grid.knots.value, axis=2)
        x_in = jnp.expand_dims(x_ext, axis=1)

        # k=0: indicator functions
        basis_splines = ((x_in >= grid[:, :-1]) & (x_in < grid[:, 1:])).astype(jnp.float32)

        for order in range(1, self.k + 1):
            left = (x_in - grid[:, : -(order + 1)]) / (grid[:, order:-1] - grid[:, : -(order + 1)])
            right = (grid[:, order + 1 :] - x_in) / (grid[:, order + 1 :] - grid[:, 1:(-order)])
            basis_splines = left * basis_splines[:, :-1] + right * basis_splines[:, 1:]
        return basis_splines

    def update_grid(
        self,
        x: jax.Array,
        new_intervals: int,
    ) -> None:
        """Adaptively refine the grid and refit spline coefficients.

        Args:
            x: Input data, shape (batch, n_in).
            new_intervals: New number of grid intervals.
        """
        # Get current activations
        bi = self.basis(x)  # (n_in*n_out, G+k, batch)
        ci = self.c_basis.value  # (n_in*n_out, G+k)
        ci_bi = jnp.einsum("ij,ijk->ik", ci, bi)  # (n_in*n_out, batch)

        # Update grid
        self.grid.update(x, new_intervals)

        # Compute new basis
        a = self.basis(x)  # (n_in*n_out, G_new+k, batch)
        bj = jnp.transpose(a, (0, 2, 1))  # (n_in*n_out, batch, G_new+k)
        ci_bi = jnp.expand_dims(ci_bi, axis=-1)  # (n_in*n_out, batch, 1)

        # Solve for new coefficients
        cj = _solve_full_lstsq(bj, ci_bi)
        self.c_basis = nnx.Param(jnp.squeeze(cj, axis=-1))

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Forward pass.

        Args:
            x: Input, shape (batch, n_in).
            deterministic: Unused (KAN forward is deterministic).

        Returns:
            Output, shape (batch, n_out).
        """
        batch = x.shape[0]

        # B-spline activation
        bi = self.basis(x)  # (n_in*n_out, G+k, batch)
        ci = self.c_basis.value  # (n_in*n_out, G+k)
        spl = jnp.einsum("ij,ijk->ik", ci, bi)  # (n_in*n_out, batch)
        spl = spl.T  # (batch, n_in*n_out)

        # Apply external weights
        if self.c_spl is not None:
            cnst = self.c_spl.value.reshape(1, self.n_in * self.n_out)
            y = cnst * spl
        else:
            y = spl

        # Residual activation
        if self.residual is not None and self.c_res is not None:
            x_ext = jnp.einsum("ij,k->ikj", x, jnp.ones(self.n_out)).reshape(
                (batch, self.n_in * self.n_out)
            )
            res = self.residual(x_ext.T).T  # (batch, n_in*n_out)
            cnst_res = self.c_res.value.reshape(1, self.n_in * self.n_out)
            y = y + cnst_res * res

        # Sum over inputs
        y = jnp.sum(y.reshape(batch, self.n_out, self.n_in), axis=2)

        if self.bias is not None:
            y = y + self.bias.value

        return y


class EfficientKANLayer(nnx.Module):
    """Efficient (matrix-based) spline KAN layer.

    Uses a shared grid per input dimension with matrix-based B-spline
    evaluation for better batched performance.

    Attributes:
        n_in: Number of input features.
        n_out: Number of output features.
        k: Spline order.
        residual: Residual activation function, or None.
        grid: EfficientKANGrid storing per-input knot positions.
        c_spl: External edge weights (n_out, n_in), or None.
        c_basis: Spline coefficients (n_out, n_in, G+k).
        c_res: Residual weights (n_out, n_in), or None.
        bias: Bias (n_out,), or None.
    """

    def __init__(
        self,
        n_in: int = 2,
        n_out: int = 5,
        k: int = 3,
        grid_intervals: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        grid_e: float = 0.05,
        residual: Callable | None = nnx.silu,
        external_weights: bool = True,
        init_scheme: dict | None = None,
        add_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the efficient KAN layer.

        Args:
            n_in: Number of input features.
            n_out: Number of output features.
            k: Spline order.
            grid_intervals: Number of grid intervals.
            grid_range: Initial grid range.
            grid_e: Grid mixing parameter.
            residual: Residual activation function, or None.
            external_weights: Whether to apply external edge weights.
            init_scheme: Initialization config dict.
            add_bias: Whether to include bias.
            rngs: Random number generators.
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.residual = residual

        self.grid = EfficientKANGrid(
            n_nodes=n_in,
            k=k,
            grid_intervals=grid_intervals,
            grid_range=grid_range,
            grid_e=grid_e,
        )

        if external_weights:
            self.c_spl = nnx.Param(jnp.ones((n_out, n_in)))
        else:
            self.c_spl = None

        basis_dim = grid_intervals + k
        c_res, c_basis = _initialize_kan_params(
            n_in,
            n_out,
            basis_dim,
            rngs,
            init_scheme=init_scheme,
            residual_fn=residual,
            basis_fn=self.basis,
            param_shape="efficient",
        )
        self.c_basis = nnx.Param(c_basis)
        self.c_res = nnx.Param(c_res) if residual is not None else None
        self.bias = nnx.Param(jnp.zeros((n_out,))) if add_bias else None

    def basis(self, x: jax.Array) -> jax.Array:
        """Evaluate B-spline basis functions.

        Args:
            x: Input, shape (batch, n_in).

        Returns:
            Basis values, shape (batch, n_in, G+k).
        """
        grid = self.grid.knots.value  # (n_in, G+2k+1)
        x_exp = jnp.expand_dims(x, axis=-1)  # (batch, n_in, 1)

        basis_splines = ((x_exp >= grid[:, :-1]) & (x_exp < grid[:, 1:])).astype(jnp.float32)

        for order in range(1, self.k + 1):
            left = (x_exp - grid[:, : -(order + 1)]) / (grid[:, order:-1] - grid[:, : -(order + 1)])
            right = (grid[:, order + 1 :] - x_exp) / (grid[:, order + 1 :] - grid[:, 1:(-order)])
            basis_splines = left * basis_splines[:, :, :-1] + right * basis_splines[:, :, 1:]
        return basis_splines

    def update_grid(
        self,
        x: jax.Array,
        new_intervals: int,
    ) -> None:
        """Adaptively refine the grid and refit spline coefficients.

        Args:
            x: Input data, shape (batch, n_in).
            new_intervals: New number of grid intervals.
        """
        bi = self.basis(x).transpose(1, 0, 2)  # (n_in, batch, G+k)
        ci = self.c_basis.value.transpose(1, 2, 0)  # (n_in, G+k, n_out)
        ci_bi = jnp.einsum("ijk,ikm->ijm", bi, ci)  # (n_in, batch, n_out)

        self.grid.update(x, new_intervals)

        bj = self.basis(x).transpose(1, 0, 2)  # (n_in, batch, G_new+k)
        cj = _solve_full_lstsq(bj, ci_bi)  # (n_in, G_new+k, n_out)
        self.c_basis = nnx.Param(cj.transpose(2, 0, 1))

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Forward pass.

        Args:
            x: Input, shape (batch, n_in).
            deterministic: Unused (KAN forward is deterministic).

        Returns:
            Output, shape (batch, n_out).
        """
        batch = x.shape[0]

        bi = self.basis(x)  # (batch, n_in, G+k)
        spl = bi.reshape(batch, -1)  # (batch, n_in*(G+k))

        if self.c_spl is not None:
            spl_w = self.c_basis.value * self.c_spl.value[..., None]
        else:
            spl_w = self.c_basis.value

        spl_w = spl_w.reshape(self.n_out, -1)  # (n_out, n_in*(G+k))
        y = jnp.matmul(spl, spl_w.T)  # (batch, n_out)

        if self.residual is not None and self.c_res is not None:
            res = self.residual(x)  # (batch, n_in)
            y = y + jnp.matmul(res, self.c_res.value.T)

        if self.bias is not None:
            y = y + self.bias.value

        return y
