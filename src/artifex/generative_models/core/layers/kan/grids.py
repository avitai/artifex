"""Grid modules for KAN spline layers.

Provides NNX-tracked grid state for B-spline KAN layers. Each grid
stores knot positions as ``nnx.Variable`` for proper state management
under JAX transforms.

Adapted from jaxKAN (MIT license).
"""

import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.layers.kan._utils import (
    _solve_full_lstsq,
)


class BSplineBasis(nnx.Module):
    """Reusable B-spline basis evaluator using Cox-de Boor recursion.

    Given a grid of knot positions, evaluates B-spline basis functions
    of order ``k`` at the given input points. JIT-compatible.

    Attributes:
        k: Spline order (polynomial degree).
    """

    def __init__(self, k: int = 3) -> None:
        """Initialize B-spline basis evaluator.

        Args:
            k: Spline order (polynomial degree).
        """
        super().__init__()
        self.k = k

    def __call__(
        self,
        x: jnp.ndarray,
        grid: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate B-spline basis functions.

        Args:
            x: Input values, shape (..., n_knot_sets).
            grid: Knot positions, shape (n_knot_sets, G+2k+1).

        Returns:
            Basis values, shape (..., n_knot_sets, G+k).
        """
        x_expanded = jnp.expand_dims(x, axis=-1)
        grid_expanded = grid

        # k=0: indicator functions
        basis = (
            (x_expanded >= grid_expanded[..., :-1]) & (x_expanded < grid_expanded[..., 1:])
        ).astype(jnp.float32)

        # Cox-de Boor recursion
        for order in range(1, self.k + 1):
            left_num = x_expanded - grid_expanded[..., : -(order + 1)]
            left_den = grid_expanded[..., order:-1] - grid_expanded[..., : -(order + 1)]
            right_num = grid_expanded[..., order + 1 :] - x_expanded
            right_den = grid_expanded[..., order + 1 :] - grid_expanded[..., 1:(-order)]

            left_term = left_num / left_den
            right_term = right_num / right_den

            basis = left_term * basis[..., :-1] + right_term * basis[..., 1:]

        return basis


class DenseKANGrid(nnx.Module):
    """Grid for the original (dense) KAN layer.

    Stores knots with shape ``(n_in * n_out, G + 2k + 1)`` so each
    input-output edge has its own knot vector.

    Attributes:
        n_in: Number of input features.
        n_out: Number of output features.
        k: Spline order.
        grid_intervals: Number of grid intervals.
        grid_range: Initial range for grid endpoints.
        grid_e: Grid mixing parameter.
        knots: NNX Variable storing knot positions.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        k: int = 3,
        grid_intervals: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        grid_e: float = 0.05,
    ) -> None:
        """Initialize the dense KAN grid.

        Args:
            n_in: Number of input features.
            n_out: Number of output features.
            k: Spline order.
            grid_intervals: Number of grid intervals.
            grid_range: Initial range for grid endpoints.
            grid_e: Grid mixing parameter.
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.grid_intervals = grid_intervals
        self.grid_range = grid_range
        self.grid_e = grid_e

        self.knots = nnx.Variable(self._build_knots(grid_intervals))

    def _build_knots(self, num_intervals: int) -> jnp.ndarray:
        """Build uniform knot vector.

        Args:
            num_intervals: Number of grid intervals.

        Returns:
            Knot array of shape (n_in*n_out, G+2k+1).
        """
        h = (self.grid_range[1] - self.grid_range[0]) / num_intervals
        grid = (
            jnp.arange(-self.k, num_intervals + self.k + 1, dtype=jnp.float32) * h
            + self.grid_range[0]
        )
        return jnp.tile(grid[None, :], (self.n_in * self.n_out, 1))

    def update(
        self,
        x: jnp.ndarray,
        new_intervals: int,
    ) -> None:
        """Adapt grid to data distribution.

        Redistributes knots based on input data percentiles, mixing
        adaptive and uniform grids according to ``grid_e``.

        Args:
            x: Input data, shape (batch, n_in).
            new_intervals: New number of grid intervals.
        """
        batch = x.shape[0]
        n_edges = self.n_in * self.n_out

        # Extend to (batch, n_in*n_out) then transpose
        x_ext = jnp.einsum("ij,k->ikj", x, jnp.ones(self.n_out)).reshape((batch, n_edges))
        x_ext = x_ext.T
        x_sorted = jnp.sort(x_ext, axis=1)

        # Adaptive grid: sample from data quantiles
        ids = jnp.concatenate(
            (
                jnp.floor(batch / new_intervals * jnp.arange(new_intervals)).astype(int),
                jnp.array([-1]),
            )
        )
        grid_adaptive = x_sorted[:, ids]

        # Uniform grid: span min to max
        margin = 0.01
        step = (x_sorted[:, -1] - x_sorted[:, 0] + 2 * margin) / new_intervals
        grid_uniform = (
            jnp.arange(new_intervals + 1, dtype=jnp.float32) * step[:, None]
            + x_sorted[:, 0][:, None]
            - margin
        )

        # Mix adaptive and uniform
        grid = self.grid_e * grid_uniform + (1.0 - self.grid_e) * grid_adaptive

        # Augment with k extra knots on each side
        h = (grid[:, [-1]] - grid[:, [0]]) / new_intervals
        left = jnp.squeeze(jnp.arange(self.k, 0, -1) * h[:, None], axis=1)
        right = jnp.squeeze(jnp.arange(1, self.k + 1) * h[:, None], axis=1)
        grid = jnp.concatenate([grid[:, [0]] - left, grid, grid[:, [-1]] + right], axis=1)

        self.knots = nnx.Variable(grid)
        self.grid_intervals = new_intervals


class EfficientKANGrid(nnx.Module):
    """Grid for the efficient (matrix-based) KAN layer.

    Stores knots with shape ``(n_in, G + 2k + 1)`` — one knot vector
    shared per input dimension (not per edge).

    Attributes:
        n_nodes: Number of input nodes.
        k: Spline order.
        grid_intervals: Number of grid intervals.
        grid_range: Initial range for grid endpoints.
        grid_e: Grid mixing parameter.
        knots: NNX Variable storing knot positions.
    """

    def __init__(
        self,
        n_nodes: int,
        k: int = 3,
        grid_intervals: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        grid_e: float = 0.05,
    ) -> None:
        """Initialize the efficient KAN grid.

        Args:
            n_nodes: Number of input nodes.
            k: Spline order.
            grid_intervals: Number of grid intervals.
            grid_range: Initial range for grid endpoints.
            grid_e: Grid mixing parameter.
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.k = k
        self.grid_intervals = grid_intervals
        self.grid_range = grid_range
        self.grid_e = grid_e

        self.knots = nnx.Variable(self._build_knots(grid_intervals))

    def _build_knots(self, num_intervals: int) -> jnp.ndarray:
        """Build uniform knot vector.

        Args:
            num_intervals: Number of grid intervals.

        Returns:
            Knot array of shape (n_nodes, G+2k+1).
        """
        h = (self.grid_range[1] - self.grid_range[0]) / num_intervals
        grid = (
            jnp.arange(-self.k, num_intervals + self.k + 1, dtype=jnp.float32) * h
            + self.grid_range[0]
        )
        return jnp.tile(grid[None, :], (self.n_nodes, 1))

    def update(
        self,
        x: jnp.ndarray,
        new_intervals: int,
    ) -> None:
        """Adapt grid to data distribution.

        Args:
            x: Input data, shape (batch, n_nodes).
            new_intervals: New number of grid intervals.
        """
        batch = x.shape[0]
        x_sorted = jnp.sort(x, axis=0)

        ids = jnp.concatenate(
            (
                jnp.floor(batch / new_intervals * jnp.arange(new_intervals)).astype(int),
                jnp.array([-1]),
            )
        )
        grid_adaptive = x_sorted[ids]

        margin = 0.01
        step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / new_intervals
        grid_uniform = (
            jnp.arange(new_intervals + 1, dtype=jnp.float32)[:, None] * step + x_sorted[0] - margin
        )

        grid = self.grid_e * grid_uniform + (1.0 - self.grid_e) * grid_adaptive

        h = (grid[-1] - grid[0]) / new_intervals
        left = h * jnp.arange(self.k, 0, -1)[:, None]
        right = h * jnp.arange(1, self.k + 1)[:, None]
        grid = jnp.concatenate([grid[:1] - left, grid, grid[-1:] + right], axis=0)

        self.knots = nnx.Variable(grid.T)
        self.grid_intervals = new_intervals


class RBFKANGrid(nnx.Module):
    """Grid for RBF KAN layers storing center positions.

    Attributes:
        n_nodes: Number of input nodes.
        num_centers: Number of RBF centers (D).
        grid_range: Range for center placement.
        grid_e: Grid mixing parameter.
        knots: NNX Variable storing center positions.
    """

    def __init__(
        self,
        n_nodes: int,
        num_centers: int = 5,
        grid_range: tuple[float, float] = (-2.0, 2.0),
        grid_e: float = 1.0,
    ) -> None:
        """Initialize the RBF grid.

        Args:
            n_nodes: Number of input nodes.
            num_centers: Number of RBF centers.
            grid_range: Range for center placement.
            grid_e: Grid mixing parameter.
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.num_centers = num_centers
        self.grid_range = grid_range
        self.grid_e = grid_e

        centers = jnp.linspace(grid_range[0], grid_range[1], num_centers)
        self.knots = nnx.Variable(jnp.tile(centers[None, :], (n_nodes, 1)))

    def update(
        self,
        x: jnp.ndarray,
        new_num_centers: int,
    ) -> None:
        """Adapt centers to data distribution.

        Args:
            x: Input data, shape (batch, n_nodes).
            new_num_centers: New number of centers.
        """
        batch = x.shape[0]
        x_sorted = jnp.sort(x, axis=0)

        ids = jnp.concatenate(
            (
                jnp.floor(batch / new_num_centers * jnp.arange(new_num_centers)).astype(int),
                jnp.array([-1]),
            )
        )
        grid_adaptive = x_sorted[ids].T

        margin = 0.01
        centers_uniform = jnp.linspace(
            float(x_sorted[0].min()) - margin,
            float(x_sorted[-1].max()) + margin,
            new_num_centers,
        )
        grid_uniform = jnp.tile(centers_uniform[None, :], (self.n_nodes, 1))

        self.knots = nnx.Variable(
            self.grid_e * grid_uniform + (1.0 - self.grid_e) * grid_adaptive[:, :new_num_centers]
        )
        self.num_centers = new_num_centers


# Re-export _solve_full_lstsq for use in layer update_grid methods
solve_full_lstsq = _solve_full_lstsq
