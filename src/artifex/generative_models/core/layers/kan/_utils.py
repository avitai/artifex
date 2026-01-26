"""Private utilities for KAN layers.

Provides shared initialization logic and least-squares solvers used
across all KAN layer variants (DRY extraction from jaxKAN).
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx


@jax.jit
def _solve_single_lstsq(
    a_matrix: jax.Array,
    b_matrix: jax.Array,
) -> jax.Array:
    """Solve AX = B via normal equations (A^T A) X = A^T B.

    Faster than jnp.linalg.lstsq for well-conditioned systems.

    Args:
        a_matrix: Shape (M, N).
        b_matrix: Shape (M, K).

    Returns:
        Solution X with shape (N, K).
    """
    ata = jnp.dot(a_matrix.T, a_matrix)
    atb = jnp.dot(a_matrix.T, b_matrix)
    return jax.scipy.linalg.solve(ata, atb, assume_a="pos")


@jax.jit
def _solve_full_lstsq(
    a_full: jax.Array,
    b_full: jax.Array,
) -> jax.Array:
    """Batched least-squares solve via vmap over leading dimension.

    Args:
        a_full: Shape (batch, M, N).
        b_full: Shape (batch, M, K).

    Returns:
        Solution X with shape (batch, N, K).
    """
    return jax.vmap(_solve_single_lstsq)(a_full, b_full)


def _initialize_kan_params(
    n_in: int,
    n_out: int,
    basis_dim: int,
    rngs: nnx.Rngs,
    *,
    init_scheme: dict | None = None,
    residual_fn: Callable | None = None,
    basis_fn: Callable | None = None,
    param_shape: str = "efficient",
) -> tuple[jax.Array | None, jax.Array]:
    """Shared parameter initialization for all KAN layer variants.

    Extracts the duplicated _initialize_params logic from jaxKAN into a
    single DRY function used by all 7+ layer types.

    Args:
        n_in: Number of input features.
        n_out: Number of output features.
        basis_dim: Number of basis functions (G+k for spline, D for others).
        rngs: NNX random number generators.
        init_scheme: Initialization config dict with 'type' key.
            Supported: 'default', 'power', 'lecun', 'glorot',
            'glorot_fine', 'custom'.
        residual_fn: Residual activation function, or None.
        basis_fn: Callable that evaluates basis on a sample batch
            (needed for lecun/glorot/glorot_fine). Should accept
            shape (sample_size, n_in) and return (sample_size, n_in, D).
        param_shape: 'dense' for (n_in*n_out, D), 'efficient' for
            (n_out, n_in, D).

    Returns:
        Tuple of (c_res, c_basis) where c_res is None if no residual.
    """
    if init_scheme is None:
        init_scheme = {"type": "default"}

    init_type = init_scheme.get("type", "default")
    c_res = None

    if init_type == "default":
        if residual_fn is not None:
            c_res = nnx.initializers.glorot_uniform(in_axis=-1, out_axis=-2)(
                rngs.params(), (n_out, n_in), jnp.float32
            )

        std = init_scheme.get("std", 0.1)
        if param_shape == "dense":
            shape = (n_in * n_out, basis_dim)
        else:
            shape = (n_out, n_in, basis_dim)
        c_basis = nnx.initializers.normal(stddev=std)(rngs.params(), shape, jnp.float32)

    elif init_type == "power":
        c_res, c_basis = _init_power(
            n_in,
            n_out,
            basis_dim,
            rngs,
            init_scheme,
            residual_fn,
            param_shape,
        )

    elif init_type == "lecun":
        c_res, c_basis = _init_lecun(
            n_in,
            n_out,
            basis_dim,
            rngs,
            init_scheme,
            residual_fn,
            basis_fn,
            param_shape,
        )

    elif init_type in ("glorot", "glorot_fine"):
        c_res, c_basis = _init_glorot(
            n_in,
            n_out,
            basis_dim,
            rngs,
            init_scheme,
            residual_fn,
            basis_fn,
            param_shape,
            fine_grained=(init_type == "glorot_fine"),
        )

    elif init_type == "custom":
        if residual_fn is not None:
            c_res = init_scheme.get("c_res")
        c_basis = init_scheme.get("c_basis")

    else:
        raise ValueError(f"Unknown initialization method: {init_type}")

    return c_res, c_basis


def _init_power(
    n_in: int,
    n_out: int,
    basis_dim: int,
    rngs: nnx.Rngs,
    scheme: dict,
    residual_fn: Callable | None,
    param_shape: str,
) -> tuple[jax.Array | None, jax.Array]:
    """Power-law initialization."""
    const_b = scheme.get("const_b", 1.0)
    pow_b1 = scheme.get("pow_b1", 0.5)
    pow_b2 = scheme.get("pow_b2", 0.5)
    c_res = None

    if residual_fn is not None:
        basis_term = basis_dim + 1
        const_r = scheme.get("const_r", 1.0)
        pow_r1 = scheme.get("pow_r1", 0.5)
        pow_r2 = scheme.get("pow_r2", 0.5)
        std_res = const_r / ((basis_term**pow_r1) * (n_in**pow_r2))
        c_res = nnx.initializers.normal(stddev=std_res)(rngs.params(), (n_out, n_in), jnp.float32)
    else:
        basis_term = basis_dim

    std_b = const_b / ((basis_term**pow_b1) * (n_in**pow_b2))
    shape = (n_in * n_out, basis_dim) if param_shape == "dense" else (n_out, n_in, basis_dim)
    c_basis = nnx.initializers.normal(stddev=std_b)(rngs.params(), shape, jnp.float32)
    return c_res, c_basis


def _generate_sample(
    scheme: dict,
    sample_key: jax.Array,
) -> jax.Array:
    """Generate a sample of points for init calibration."""
    distrib = scheme.get("distribution", "uniform") or "uniform"
    sample_size = scheme.get("sample_size", 10000) or 10000
    if distrib == "uniform":
        return jax.random.uniform(sample_key, shape=(sample_size,), minval=-1.0, maxval=1.0)
    return jax.random.normal(sample_key, shape=(sample_size,))


def _init_lecun(
    n_in: int,
    n_out: int,
    basis_dim: int,
    rngs: nnx.Rngs,
    scheme: dict,
    residual_fn: Callable | None,
    basis_fn: Callable | None,
    param_shape: str,
) -> tuple[jax.Array | None, jax.Array]:
    """LeCun-like initialization (Var[in] = Var[out])."""
    sample_key = rngs.params()
    sample = _generate_sample(scheme, sample_key)
    gain = scheme.get("gain") or float(sample.std())
    c_res = None

    if basis_fn is not None:
        sample_ext = jnp.tile(sample[:, None], (1, n_in))
        y_b = basis_fn(sample_ext)
        y_b_sq_mean = float((y_b**2).mean())
    else:
        y_b_sq_mean = 1.0

    if residual_fn is not None:
        scale = n_in * (basis_dim + 1)
        y_res = residual_fn(sample)
        y_res_sq_mean = float((y_res**2).mean())
        std_res = gain / jnp.sqrt(scale * y_res_sq_mean)
        c_res = nnx.initializers.normal(stddev=std_res)(rngs.params(), (n_out, n_in), jnp.float32)
    else:
        scale = n_in * basis_dim

    std_b = gain / jnp.sqrt(scale * y_b_sq_mean)
    shape = (n_in * n_out, basis_dim) if param_shape == "dense" else (n_out, n_in, basis_dim)
    c_basis = nnx.initializers.normal(stddev=std_b)(rngs.params(), shape, jnp.float32)
    return c_res, c_basis


def _init_glorot(
    n_in: int,
    n_out: int,
    basis_dim: int,
    rngs: nnx.Rngs,
    scheme: dict,
    residual_fn: Callable | None,
    basis_fn: Callable | None,
    param_shape: str,
    *,
    fine_grained: bool = False,
) -> tuple[jax.Array | None, jax.Array]:
    """Glorot-like initialization balancing forward/backward variance."""
    sample_key = rngs.params()
    sample = _generate_sample(scheme, sample_key)
    gain = scheme.get("gain") or float(sample.std())
    c_res = None

    if basis_fn is not None:
        sample_ext = jnp.tile(sample[:, None], (1, n_in))

        if fine_grained:
            # Per-mode variance
            y_b = basis_fn(sample_ext)
            mu0 = (y_b**2).mean(axis=(0, 1))

            def basis_scalar(x: jax.Array) -> jax.Array:
                return basis_fn(jnp.array([[x]]))[0, 0, :]

            jac_fn = jax.jacrev(basis_scalar)
            mu1 = (jax.vmap(jac_fn)(sample) ** 2).mean(axis=0)
        else:
            # Aggregated variance
            y_b = basis_fn(sample_ext)
            y_b_sq_mean = float((y_b**2).mean())

            def basis_scalar(x: jax.Array) -> jax.Array:
                return basis_fn(jnp.array([[x]]))[0, 0, :]

            jac_fn = jax.jacobian(basis_scalar)
            num_batches = 20
            batch_size = len(sample) // num_batches
            grad_sq_accum = 0.0
            for i in range(num_batches):
                batch = sample[i * batch_size : (i + 1) * batch_size]
                grad_batch = jax.vmap(jac_fn)(batch)
                grad_sq_accum += (grad_batch**2).sum()
            grad_b_sq_mean = grad_sq_accum / (len(sample) * basis_dim)
    else:
        y_b_sq_mean = 1.0
        grad_b_sq_mean = 1.0

    # Residual handling (same for both glorot variants)
    if residual_fn is not None:
        scale_in = n_in * (basis_dim + 1)
        scale_out = n_out * (basis_dim + 1)
        y_res = residual_fn(sample)
        y_res_sq_mean = float((y_res**2).mean())

        def r_fn(x: jax.Array) -> jax.Array:
            return residual_fn(x)

        grad_res = jax.vmap(jax.jacobian(r_fn))(sample)
        grad_res_sq_mean = float((grad_res**2).mean())

        std_res = gain * jnp.sqrt(2.0 / (scale_in * y_res_sq_mean + scale_out * grad_res_sq_mean))
        c_res = nnx.initializers.normal(stddev=std_res)(rngs.params(), (n_out, n_in), jnp.float32)
    else:
        scale_in = n_in * basis_dim
        scale_out = n_out * basis_dim

    shape = (n_in * n_out, basis_dim) if param_shape == "dense" else (n_out, n_in, basis_dim)

    if fine_grained and basis_fn is not None:
        sigma_vec = gain * jnp.sqrt(1.0 / (scale_in * mu0 + scale_out * mu1))
        noise = nnx.initializers.normal(stddev=1.0)(rngs.params(), shape, jnp.float32)
        c_basis = noise * sigma_vec
    else:
        std_b = gain * jnp.sqrt(2.0 / (scale_in * y_b_sq_mean + scale_out * grad_b_sq_mean))
        c_basis = nnx.initializers.normal(stddev=std_b)(rngs.params(), shape, jnp.float32)

    return c_res, c_basis
