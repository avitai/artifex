"""Non-spline KAN layer variants using polynomial/function basis.

Includes Chebyshev, Fourier, Legendre, RBF, and Sine KAN layers.
All adapted from jaxKAN (MIT license) to Artifex conventions.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.layers.kan._utils import (
    _initialize_kan_params,
    _solve_full_lstsq,
)
from artifex.generative_models.core.layers.kan.grids import RBFKANGrid


# Chebyshev polynomials up to degree 20 (closed-form)
_CHEBYSHEV_POLYS: dict[int, Callable[[jax.Array], jax.Array]] = {
    0: lambda x: jnp.ones_like(x),
    1: lambda x: x,
    2: lambda x: 2 * x**2 - 1,
    3: lambda x: 4 * x**3 - 3 * x,
    4: lambda x: 8 * x**4 - 8 * x**2 + 1,
    5: lambda x: 16 * x**5 - 20 * x**3 + 5 * x,
    6: lambda x: 32 * x**6 - 48 * x**4 + 18 * x**2 - 1,
    7: lambda x: 64 * x**7 - 112 * x**5 + 56 * x**3 - 7 * x,
    8: lambda x: 128 * x**8 - 256 * x**6 + 160 * x**4 - 32 * x**2 + 1,
    9: lambda x: 256 * x**9 - 576 * x**7 + 432 * x**5 - 120 * x**3 + 9 * x,
    10: lambda x: 512 * x**10 - 1280 * x**8 + 1120 * x**6 - 400 * x**4 + 50 * x**2 - 1,
    11: lambda x: 1024 * x**11 - 2816 * x**9 + 2816 * x**7 - 1232 * x**5 + 220 * x**3 - 11 * x,
    12: lambda x: (
        2048 * x**12 - 6144 * x**10 + 6912 * x**8 - 3584 * x**6 + 840 * x**4 - 72 * x**2 + 1
    ),
    13: lambda x: (
        4096 * x**13
        - 13312 * x**11
        + 16640 * x**9
        - 9984 * x**7
        + 2912 * x**5
        - 364 * x**3
        + 13 * x
    ),
    14: lambda x: (
        8192 * x**14
        - 28672 * x**12
        + 39424 * x**10
        - 26880 * x**8
        + 9408 * x**6
        - 1568 * x**4
        + 98 * x**2
        - 1
    ),
    15: lambda x: (
        16384 * x**15
        - 61440 * x**13
        + 92160 * x**11
        - 70400 * x**9
        + 28800 * x**7
        - 6048 * x**5
        + 560 * x**3
        - 15 * x
    ),
    16: lambda x: (
        32768 * x**16
        - 131072 * x**14
        + 212992 * x**12
        - 180224 * x**10
        + 84480 * x**8
        - 21504 * x**6
        + 2688 * x**4
        - 128 * x**2
        + 1
    ),
    17: lambda x: (
        65536 * x**17
        - 278528 * x**15
        + 487424 * x**13
        - 452608 * x**11
        + 239360 * x**9
        - 71808 * x**7
        + 11424 * x**5
        - 816 * x**3
        + 17 * x
    ),
    18: lambda x: (
        131072 * x**18
        - 589824 * x**16
        + 1105920 * x**14
        - 1118208 * x**12
        + 658944 * x**10
        - 228096 * x**8
        + 44352 * x**6
        - 4320 * x**4
        + 162 * x**2
        - 1
    ),
    19: lambda x: (
        262144 * x**19
        - 1245184 * x**17
        + 2490368 * x**15
        - 2723840 * x**13
        + 1770496 * x**11
        - 695552 * x**9
        + 160512 * x**7
        - 20064 * x**5
        + 1140 * x**3
        - 19 * x
    ),
    20: lambda x: (
        524288 * x**20
        - 2621440 * x**18
        + 5570560 * x**16
        - 6553600 * x**14
        + 4659200 * x**12
        - 2050048 * x**10
        + 549120 * x**8
        - 84480 * x**6
        + 6600 * x**4
        - 200 * x**2
        + 1
    ),
}

# Legendre polynomials up to degree 20
_LEGENDRE_POLYS: dict[int, Callable[[jax.Array], jax.Array]] = {
    0: lambda x: jnp.ones_like(x),
    1: lambda x: x,
    2: lambda x: (3 * x**2 - 1) / 2,
    3: lambda x: (5 * x**3 - 3 * x) / 2,
    4: lambda x: (35 * x**4 - 30 * x**2 + 3) / 8,
    5: lambda x: (63 * x**5 - 70 * x**3 + 15 * x) / 8,
    6: lambda x: (231 * x**6 - 315 * x**4 + 105 * x**2 - 5) / 16,
    7: lambda x: (429 * x**7 - 693 * x**5 + 315 * x**3 - 35 * x) / 16,
    8: lambda x: (6435 * x**8 - 12012 * x**6 + 6930 * x**4 - 1260 * x**2 + 35) / 128,
    9: lambda x: (12155 * x**9 - 25740 * x**7 + 18018 * x**5 - 4620 * x**3 + 315 * x) / 128,
    10: lambda x: (
        (46189 * x**10 - 109395 * x**8 + 90090 * x**6 - 30030 * x**4 + 3465 * x**2 - 63) / 256
    ),
}


def _standard_forward(
    x: jax.Array,
    basis_fn: Callable,
    c_basis: nnx.Param,
    n_out: int,
    *,
    c_ext: nnx.Param | None = None,
    residual: Callable | None = None,
    c_res: nnx.Param | None = None,
    bias: nnx.Param | None = None,
) -> jax.Array:
    """Shared forward pass for non-spline basis layers (DRY).

    Args:
        x: Input, shape (batch, n_in).
        basis_fn: Callable returning (batch, n_in, D).
        c_basis: Basis coefficients (n_out, n_in, D).
        n_out: Output dimension.
        c_ext: External weights (n_out, n_in), or None.
        residual: Residual activation function, or None.
        c_res: Residual weights (n_out, n_in), or None.
        bias: Bias (n_out,), or None.

    Returns:
        Output, shape (batch, n_out).
    """
    batch = x.shape[0]

    bi = basis_fn(x)  # (batch, n_in, D)
    act = bi.reshape(batch, -1)  # (batch, n_in*D)

    if c_ext is not None:
        act_w = c_basis.value * c_ext.value[..., None]
    else:
        act_w = c_basis.value

    act_w = act_w.reshape(n_out, -1)  # (n_out, n_in*D)
    y = jnp.matmul(act, act_w.T)  # (batch, n_out)

    if residual is not None and c_res is not None:
        res = residual(x)
        y = y + jnp.matmul(res, c_res.value.T)

    if bias is not None:
        y = y + bias.value

    return y


def _standard_update_grid(
    x: jax.Array,
    basis_fn: Callable,
    c_basis: nnx.Param,
    degree_attr: str,
    layer: nnx.Module,
    new_degree: int,
) -> None:
    """Shared update_grid for non-spline basis layers (DRY).

    Updates degree, recomputes basis, solves for new coefficients.

    Args:
        x: Input data, shape (batch, n_in).
        basis_fn: Callable returning basis values.
        c_basis: Current basis coefficients parameter.
        degree_attr: Name of degree attribute on layer (e.g. 'D').
        layer: The layer module to update.
        new_degree: New degree/D value.
    """
    bi = basis_fn(x).transpose(1, 0, 2)
    ci = c_basis.value.transpose(1, 2, 0)
    ci_bi = jnp.einsum("ijk,ikm->ijm", bi, ci)

    setattr(layer, degree_attr, new_degree)

    bj = basis_fn(x).transpose(1, 0, 2)
    cj = _solve_full_lstsq(bj, ci_bi)
    layer.c_basis = nnx.Param(cj.transpose(2, 0, 1))


class ChebyshevKANLayer(nnx.Module):
    """Chebyshev polynomial KAN layer.

    Three evaluation flavors:
        - ``"default"``: arccos-based (arXiv:2405.07200)
        - ``"modified"``: recurrence relation
        - ``"exact"``: closed-form polynomials (max degree 20)

    Attributes:
        n_in: Number of input features.
        n_out: Number of output features.
        D: Polynomial degree.
        flavor: Evaluation method.
    """

    def __init__(
        self,
        n_in: int = 2,
        n_out: int = 5,
        D: int = 5,
        flavor: str = "default",
        residual: Callable | None = None,
        external_weights: bool = False,
        init_scheme: dict | None = None,
        add_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the Chebyshev KAN layer.

        Args:
            n_in: Number of input features.
            n_out: Number of output features.
            D: Chebyshev polynomial degree.
            flavor: One of 'default', 'modified', 'exact'.
            residual: Residual activation function, or None.
            external_weights: Whether to apply external edge weights.
            init_scheme: Initialization config dict.
            add_bias: Whether to include bias.
            rngs: Random number generators.

        Raises:
            ValueError: If flavor is 'exact' and D exceeds max degree.
        """
        super().__init__()

        if flavor == "exact" and D > max(_CHEBYSHEV_POLYS):
            raise ValueError(f"'exact' flavor max degree is {max(_CHEBYSHEV_POLYS)}, got {D}")

        self.n_in = n_in
        self.n_out = n_out
        self.D = D
        self.flavor = flavor
        self.residual = residual

        self.bias = nnx.Param(jnp.zeros((n_out,))) if add_bias else None

        if external_weights:
            self.c_ext = nnx.Param(jnp.ones((n_out, n_in)))
        else:
            self.c_ext = None

        ext_dim = D if self.bias is not None else D + 1
        c_res, c_basis = _initialize_kan_params(
            n_in,
            n_out,
            ext_dim,
            rngs,
            init_scheme=init_scheme,
            residual_fn=residual,
            basis_fn=self.basis,
            param_shape="efficient",
        )
        self.c_basis = nnx.Param(c_basis)
        self.c_res = nnx.Param(c_res) if residual is not None else None

    def basis(self, x: jax.Array) -> jax.Array:
        """Evaluate Chebyshev basis functions.

        Args:
            x: Input, shape (batch, n_in).

        Returns:
            Basis values, shape (batch, n_in, D) or (batch, n_in, D+1).
        """
        batch = x.shape[0]
        x = jnp.tanh(x)

        if self.flavor == "default":
            x_exp = jnp.expand_dims(x, axis=-1)
            x_exp = jnp.tile(x_exp, (1, 1, self.D + 1))
            x_exp = jnp.arccos(x_exp)
            x_exp = x_exp * jnp.arange(self.D + 1)
            cheb = jnp.cos(x_exp)

        elif self.flavor == "modified":
            cheb = jnp.ones((batch, self.n_in, self.D + 1))
            cheb = cheb.at[:, :, 1].set(x)
            for k in range(2, self.D + 1):
                cheb = cheb.at[:, :, k].set(2 * x * cheb[:, :, k - 1] - cheb[:, :, k - 2])

        elif self.flavor == "exact":
            cheb = jnp.stack(
                [_CHEBYSHEV_POLYS[i](x) for i in range(self.D + 1)],
                axis=-1,
            )

        else:
            raise ValueError(f"Unknown flavor: {self.flavor}")

        # Exclude constant term if bias is used
        if self.bias is not None:
            return cheb[:, :, 1:]
        return cheb

    def update_grid(self, x: jax.Array, d_new: int) -> None:
        """Increase polynomial degree and refit coefficients.

        Args:
            x: Input data, shape (batch, n_in).
            d_new: New polynomial degree.
        """
        _standard_update_grid(x, self.basis, self.c_basis, "D", self, d_new)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Forward pass.

        Args:
            x: Input, shape (batch, n_in).
            deterministic: Unused (deterministic forward pass).

        Returns:
            Output, shape (batch, n_out).
        """
        return _standard_forward(
            x,
            self.basis,
            self.c_basis,
            self.n_out,
            c_ext=self.c_ext,
            residual=self.residual,
            c_res=self.c_res,
            bias=self.bias,
        )


class FourierKANLayer(nnx.Module):
    """Fourier series KAN layer.

    Uses sin/cos basis functions. Simpler API than spline variants:
    no residual or external_weights parameters.

    Attributes:
        n_in: Number of input features.
        n_out: Number of output features.
        D: Fourier series order.
        c_cos: Cosine coefficients (n_out, n_in, D).
        c_sin: Sine coefficients (n_out, n_in, D).
    """

    def __init__(
        self,
        n_in: int = 2,
        n_out: int = 5,
        D: int = 5,
        smooth_init: bool = True,
        add_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the Fourier KAN layer.

        Args:
            n_in: Number of input features.
            n_out: Number of output features.
            D: Fourier series order (number of sin/cos terms).
            smooth_init: Whether to apply frequency-dependent smoothing.
            add_bias: Whether to include bias.
            rngs: Random number generators.
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.D = D

        self.bias = nnx.Param(jnp.zeros((n_out,))) if add_bias else None

        norm_factor = jnp.arange(1, D + 1) ** 2 if smooth_init else jnp.sqrt(float(D))

        inits = nnx.initializers.normal(stddev=1.0 / jnp.sqrt(float(n_in)))(
            rngs.params(), (2, n_out, n_in, D), jnp.float32
        )
        inits = inits / norm_factor

        self.c_cos = nnx.Param(inits[0])
        self.c_sin = nnx.Param(inits[1])

    def basis(
        self,
        x: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate Fourier basis functions.

        Args:
            x: Input, shape (batch, n_in).

        Returns:
            Tuple of (cosines, sines) each with shape (batch, n_in, D).
        """
        x_exp = jnp.expand_dims(x, axis=-1)
        d_array = jnp.arange(1, self.D + 1).reshape(1, 1, self.D)
        dx = d_array * x_exp
        return jnp.cos(dx), jnp.sin(dx)

    def update_grid(self, x: jax.Array, d_new: int) -> None:
        """Increase Fourier order and refit coefficients.

        Args:
            x: Input data, shape (batch, n_in).
            d_new: New Fourier order.
        """
        ci, si = self.basis(x)
        ci = ci.transpose(1, 0, 2)
        si = si.transpose(1, 0, 2)
        cos_w = self.c_cos.value.transpose(1, 2, 0)
        sin_w = self.c_sin.value.transpose(1, 2, 0)

        cosines = jnp.einsum("ijk,ikm->ijm", ci, cos_w)
        sines = jnp.einsum("ijk,ikm->ijm", si, sin_w)

        self.D = d_new

        cj, sj = self.basis(x)
        cj = cj.transpose(1, 0, 2)
        sj = sj.transpose(1, 0, 2)

        new_cos_w = _solve_full_lstsq(cj, cosines).transpose(2, 0, 1)
        new_sin_w = _solve_full_lstsq(sj, sines).transpose(2, 0, 1)

        self.c_cos = nnx.Param(new_cos_w)
        self.c_sin = nnx.Param(new_sin_w)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Forward pass.

        Args:
            x: Input, shape (batch, n_in).
            deterministic: Unused (deterministic forward pass).

        Returns:
            Output, shape (batch, n_out).
        """
        batch = x.shape[0]
        ci, si = self.basis(x)

        cosines = ci.reshape(batch, -1)
        sines = si.reshape(batch, -1)

        cos_w = self.c_cos.value.reshape(self.n_out, -1)
        sin_w = self.c_sin.value.reshape(self.n_out, -1)

        y = jnp.matmul(cosines, cos_w.T) + jnp.matmul(sines, sin_w.T)

        if self.bias is not None:
            y = y + self.bias.value

        return y


class LegendreKANLayer(nnx.Module):
    """Legendre polynomial KAN layer.

    Two evaluation flavors:
        - ``"default"``: three-term recurrence
        - ``"exact"``: closed-form polynomials (max degree 10)

    Attributes:
        n_in: Number of input features.
        n_out: Number of output features.
        D: Polynomial degree.
        flavor: Evaluation method.
    """

    def __init__(
        self,
        n_in: int = 2,
        n_out: int = 5,
        D: int = 5,
        flavor: str = "default",
        residual: Callable | None = None,
        external_weights: bool = False,
        init_scheme: dict | None = None,
        add_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the Legendre KAN layer.

        Args:
            n_in: Number of input features.
            n_out: Number of output features.
            D: Legendre polynomial degree.
            flavor: One of 'default' or 'exact'.
            residual: Residual activation function, or None.
            external_weights: Whether to apply external edge weights.
            init_scheme: Initialization config dict.
            add_bias: Whether to include bias.
            rngs: Random number generators.

        Raises:
            ValueError: If flavor is 'exact' and D exceeds max degree.
        """
        super().__init__()

        if flavor == "exact" and D > max(_LEGENDRE_POLYS):
            raise ValueError(f"'exact' flavor max degree is {max(_LEGENDRE_POLYS)}, got {D}")

        self.n_in = n_in
        self.n_out = n_out
        self.D = D
        self.flavor = flavor
        self.residual = residual

        self.bias = nnx.Param(jnp.zeros((n_out,))) if add_bias else None

        if external_weights:
            self.c_ext = nnx.Param(jnp.ones((n_out, n_in)))
        else:
            self.c_ext = None

        ext_dim = D if self.bias is not None else D + 1
        c_res, c_basis = _initialize_kan_params(
            n_in,
            n_out,
            ext_dim,
            rngs,
            init_scheme=init_scheme,
            residual_fn=residual,
            basis_fn=self.basis,
            param_shape="efficient",
        )
        self.c_basis = nnx.Param(c_basis)
        self.c_res = nnx.Param(c_res) if residual is not None else None

    def basis(self, x: jax.Array) -> jax.Array:
        """Evaluate Legendre basis functions.

        Args:
            x: Input, shape (batch, n_in).

        Returns:
            Basis values, shape (batch, n_in, D) or (batch, n_in, D+1).
        """
        batch = x.shape[0]
        x = jnp.tanh(x)

        if self.flavor == "default":
            leg = jnp.ones((batch, self.n_in, self.D + 1))
            leg = leg.at[:, :, 1].set(x)
            for k in range(2, self.D + 1):
                leg = leg.at[:, :, k].set(
                    ((2 * k - 1) * x * leg[:, :, k - 1] - (k - 1) * leg[:, :, k - 2]) / k
                )

        elif self.flavor == "exact":
            leg = jnp.stack(
                [_LEGENDRE_POLYS[i](x) for i in range(self.D + 1)],
                axis=-1,
            )

        else:
            raise ValueError(f"Unknown flavor: {self.flavor}")

        if self.bias is not None:
            return leg[:, :, 1:]
        return leg

    def update_grid(self, x: jax.Array, d_new: int) -> None:
        """Increase polynomial degree and refit coefficients.

        Args:
            x: Input data, shape (batch, n_in).
            d_new: New polynomial degree.
        """
        _standard_update_grid(x, self.basis, self.c_basis, "D", self, d_new)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Forward pass.

        Args:
            x: Input, shape (batch, n_in).
            deterministic: Unused (deterministic forward pass).

        Returns:
            Output, shape (batch, n_out).
        """
        return _standard_forward(
            x,
            self.basis,
            self.c_basis,
            self.n_out,
            c_ext=self.c_ext,
            residual=self.residual,
            c_res=self.c_res,
            bias=self.bias,
        )


class RBFKANLayer(nnx.Module):
    """Radial basis function KAN layer.

    Uses configurable RBF kernels (currently Gaussian).

    Attributes:
        n_in: Number of input features.
        n_out: Number of output features.
        D: Number of basis functions.
        kernel: Kernel configuration dict.
        grid: RBFKANGrid storing center positions.
    """

    def __init__(
        self,
        n_in: int = 2,
        n_out: int = 5,
        D: int = 5,
        kernel: dict | None = None,
        grid_range: tuple[float, float] = (-2.0, 2.0),
        grid_e: float = 1.0,
        residual: Callable | None = None,
        external_weights: bool = False,
        init_scheme: dict | None = None,
        add_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the RBF KAN layer.

        Args:
            n_in: Number of input features.
            n_out: Number of output features.
            D: Number of basis functions (centers).
            kernel: Kernel configuration dict (default: gaussian).
            grid_range: Range for center placement.
            grid_e: Grid mixing parameter.
            residual: Residual activation function, or None.
            external_weights: Whether to apply external edge weights.
            init_scheme: Initialization config dict.
            add_bias: Whether to include bias.
            rngs: Random number generators.

        Raises:
            ValueError: If kernel type is unknown.
        """
        super().__init__()
        self.kernel = kernel or {"type": "gaussian"}
        self.n_in = n_in
        self.n_out = n_out
        self.D = D
        self.residual = residual

        self.grid = RBFKANGrid(
            n_nodes=n_in,
            num_centers=D,
            grid_range=grid_range,
            grid_e=grid_e,
        )

        self.bias = nnx.Param(jnp.zeros((n_out,))) if add_bias else None

        if external_weights:
            self.c_ext = nnx.Param(jnp.ones((n_out, n_in)))
        else:
            self.c_ext = None

        c_res, c_basis = _initialize_kan_params(
            n_in,
            n_out,
            D,
            rngs,
            init_scheme=init_scheme,
            residual_fn=residual,
            basis_fn=self.basis,
            param_shape="efficient",
        )
        self.c_basis = nnx.Param(c_basis)
        self.c_res = nnx.Param(c_res) if residual is not None else None

    def basis(self, x: jax.Array) -> jax.Array:
        """Evaluate RBF basis functions.

        Args:
            x: Input, shape (batch, n_in).

        Returns:
            Basis values, shape (batch, n_in, D).

        Raises:
            ValueError: If kernel type is unknown.
        """
        kernel_type = self.kernel.get("type", "gaussian")
        grid_item = self.grid.knots.value

        if kernel_type == "gaussian":
            std = self.kernel.get("std", 1.0)
            return jnp.exp(-0.5 * ((x[..., :, None] - grid_item[None, ...]) / std) ** 2)

        raise ValueError(f"Unknown kernel type: {kernel_type}")

    def update_grid(self, x: jax.Array, d_new: int) -> None:
        """Update grid centers and refit coefficients.

        Args:
            x: Input data, shape (batch, n_in).
            d_new: New number of centers.
        """
        bi = self.basis(x).transpose(1, 0, 2)
        ci = self.c_basis.value.transpose(1, 2, 0)
        ci_bi = jnp.einsum("ijk,ikm->ijm", bi, ci)

        self.grid.update(x, d_new)
        self.D = d_new

        bj = self.basis(x).transpose(1, 0, 2)
        cj = _solve_full_lstsq(bj, ci_bi)
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
            deterministic: Unused (deterministic forward pass).

        Returns:
            Output, shape (batch, n_out).
        """
        return _standard_forward(
            x,
            self.basis,
            self.c_basis,
            self.n_out,
            c_ext=self.c_ext,
            residual=self.residual,
            c_res=self.c_res,
            bias=self.bias,
        )


class SineKANLayer(nnx.Module):
    """Sine-based KAN layer with learnable frequency and phase.

    Uses sin(omega * x + phase) basis functions with normalization,
    producing SIREN-like representations.

    Attributes:
        n_in: Number of input features.
        n_out: Number of output features.
        D: Number of basis functions.
        omega: Learnable frequency parameters (D, 1).
        phase: Learnable phase parameters (D, 1).
    """

    def __init__(
        self,
        n_in: int = 2,
        n_out: int = 5,
        D: int = 5,
        residual: Callable | None = None,
        external_weights: bool = False,
        init_scheme: dict | None = None,
        add_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the Sine KAN layer.

        Args:
            n_in: Number of input features.
            n_out: Number of output features.
            D: Number of sine basis functions.
            residual: Residual activation function, or None.
            external_weights: Whether to apply external edge weights.
            init_scheme: Initialization config dict.
            add_bias: Whether to include bias.
            rngs: Random number generators.
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.D = D
        self.residual = residual

        self.bias = nnx.Param(jnp.zeros((n_out,))) if add_bias else None

        self.omega = nnx.Param(
            nnx.initializers.normal(stddev=1.0)(rngs.params(), (D, 1), jnp.float32)
        )
        self.phase = nnx.Param(jnp.zeros((D, 1)))

        if external_weights:
            self.c_ext = nnx.Param(jnp.ones((n_out, n_in)))
        else:
            self.c_ext = None

        c_res, c_basis = _initialize_kan_params(
            n_in,
            n_out,
            D,
            rngs,
            init_scheme=init_scheme,
            residual_fn=residual,
            basis_fn=self.basis,
            param_shape="efficient",
        )
        self.c_basis = nnx.Param(c_basis)
        self.c_res = nnx.Param(c_res) if residual is not None else None

    def basis(self, x: jax.Array) -> jax.Array:
        """Evaluate sine basis functions with normalization.

        Args:
            x: Input, shape (batch, n_in).

        Returns:
            Normalized basis values, shape (batch, n_in, D).
        """
        x_exp = jnp.expand_dims(x, axis=-1)  # (batch, n_in, 1)
        omegas = self.omega.value.reshape(1, 1, self.D)
        p = self.phase.value.reshape(1, 1, self.D)

        wx = omegas * x_exp
        s = jnp.sin(wx + p)

        mu = jnp.exp(-0.5 * omegas**2) * jnp.sin(p)
        std = jnp.sqrt(0.5 * (1.0 - jnp.exp(-2.0 * omegas**2) * jnp.cos(2.0 * p)) - mu**2)

        return (s - mu) / (std + 1e-8)

    def update_grid(self, x: jax.Array, d_new: int) -> None:
        """Increase number of basis functions and refit.

        Args:
            x: Input data, shape (batch, n_in).
            d_new: New number of basis functions.
        """
        bi = self.basis(x).transpose(1, 0, 2)
        ci = self.c_basis.value.transpose(1, 2, 0)
        ci_bi = jnp.einsum("ijk,ikm->ijm", bi, ci)

        self.D = d_new
        self.omega = nnx.Param(
            nnx.initializers.normal(stddev=1.0)(nnx.Rngs(0).params(), (d_new, 1), jnp.float32)
        )
        self.phase = nnx.Param(jnp.zeros((d_new, 1)))

        bj = self.basis(x).transpose(1, 0, 2)
        cj = _solve_full_lstsq(bj, ci_bi)
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
            deterministic: Unused (deterministic forward pass).

        Returns:
            Output, shape (batch, n_out).
        """
        return _standard_forward(
            x,
            self.basis,
            self.c_basis,
            self.n_out,
            c_ext=self.c_ext,
            residual=self.residual,
            c_res=self.c_res,
            bias=self.bias,
        )
