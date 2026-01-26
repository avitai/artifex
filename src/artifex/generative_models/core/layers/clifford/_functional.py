"""Private functional utilities for Clifford layers.

Provides reshape-based convolution dispatch, batch multiplications,
and the Cholesky-whitening used by Clifford normalization layers.
"""

import jax
import jax.numpy as jnp
from jax import lax


def clifford_convnd(
    x: jax.Array,
    n_blades: int,
    kernel: jax.Array,
    bias: jax.Array | None,
    strides: tuple[int, ...],
    padding: str | list[tuple[int, int]],
    dilation: tuple[int, ...],
) -> jax.Array:
    """Apply a Clifford convolution by reshaping blades into the channel dimension.

    The input layout is ``(B, *D, C, I)`` (channels-last, blades last).
    Internally, blades and channels are merged for ``lax.conv_general_dilated``
    and then separated again.

    Args:
        x: Input tensor ``(B, *D, C, I)``.
        n_blades: Number of algebra blades.
        kernel: Clifford kernel ``(I*O, I*C, *K)``.
        bias: Optional bias ``(I*O,)`` or ``None``.
        strides: Convolution strides.
        padding: ``'SAME'``, ``'VALID'``, or explicit padding pairs.
        dilation: Kernel dilation factors.

    Returns:
        Output tensor ``(B, *D', C_out, I)``.
    """
    batch = x.shape[0]
    spatial = x.shape[1:-2]
    n_spatial = len(spatial)

    # (B, *D, C, I) -> (B, *D, I, C) -> (B, *D, I*C)
    x = jnp.moveaxis(x, -1, -2)
    x = x.reshape(batch, *spatial, -1)

    # Build dimension_numbers: "N...C" for channels-last
    in_spec = "N" + "".join(chr(ord("D") + i) for i in range(n_spatial)) + "C"
    # Kernel format: (O, I, *K) — JAX expects OI...
    kern_spec = "OI" + "".join(chr(ord("D") + i) for i in range(n_spatial))
    out_spec = in_spec
    dn = lax.conv_dimension_numbers(x.shape, kernel.shape, (in_spec, kern_spec, out_spec))

    output = lax.conv_general_dilated(
        x,
        kernel,
        window_strides=strides,
        padding=padding,
        rhs_dilation=dilation,
        dimension_numbers=dn,
    )

    if bias is not None:
        output = output + bias

    # (B, *D', I*O) -> (B, *D', I, O) -> swap -> (B, *D', O, I) = (B, *D', C_out, I)
    out_spatial = output.shape[1:-1]
    output = output.reshape(batch, *out_spatial, n_blades, -1)
    output = jnp.moveaxis(output, -2, -1)
    return output


def batchmul1d(x: jax.Array, weights: jax.Array) -> jax.Array:
    """Batch multiplication: ``(B, I, X), (O, I, X) -> (B, O, X)``."""
    return jnp.einsum("bix,oix->box", x, weights)


def batchmul2d(x: jax.Array, weights: jax.Array) -> jax.Array:
    """Batch multiplication: ``(B, I, X, Y), (O, I, X, Y) -> (B, O, X, Y)``."""
    return jnp.einsum("bixy,oixy->boxy", x, weights)


def batchmul3d(x: jax.Array, weights: jax.Array) -> jax.Array:
    """Batch multiplication: ``(B, I, X, Y, Z), (O, I, X, Y, Z) -> (B, O, X, Y, Z)``."""
    return jnp.einsum("bixyz,oixyz->boxyz", x, weights)


def whiten_data(
    x: jax.Array,
    training: bool,
    running_mean: jax.Array | None,
    running_cov: jax.Array | None,
    momentum: float,
    eps: float,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    """Jointly whiten features in ``(B, *D, C, I)`` tensors.

    Takes ``I``-dimensional vectors and whitens individually for each channel
    ``C`` over ``(B, *D)``.

    Unlike the PyTorch reference, this function **returns** updated running
    statistics instead of mutating them in-place (JAX immutability).

    Args:
        x: Input ``(B, *D, C, I)``.
        training: Whether to compute stats from the batch.
        running_mean: Running mean ``(I, C)`` or ``None``.
        running_cov: Running covariance ``(I, I, C)`` or ``None``.
        momentum: EMA momentum.
        eps: Regularization for Cholesky decomposition.

    Returns:
        ``(whitened_x, new_running_mean, new_running_cov)``.
    """
    ndim = x.ndim
    if ndim < 3:
        raise ValueError(f"Expected at least 3D input, got {ndim}D.")

    channels = x.shape[-2]
    n_blades = x.shape[-1]
    spatial_dims = x.shape[1:-2]

    # Axes to reduce over: batch + spatial
    reduce_axes = (0, *range(1, 1 + len(spatial_dims)))

    # -- Mean --
    new_running_mean = running_mean
    if training or running_mean is None:
        # mean over (B, *D) -> (C, I)
        mean = jnp.mean(x, axis=reduce_axes)  # (C, I)
        if running_mean is not None:
            new_running_mean = running_mean + momentum * (mean.T - running_mean)
    else:
        mean = running_mean.T  # (I, C) -> (C, I)

    # Center
    shape = (1,) * (1 + len(spatial_dims)) + (channels, n_blades)
    x_centered = x - mean.reshape(shape)

    # -- Covariance --
    new_running_cov = running_cov
    if training or running_cov is None:
        # Reshape to (C, I, N) where N = B * prod(D)
        # x_centered: (B, *D, C, I)
        perm = list(range(ndim))
        # Move C to front, I second, rest after
        c_idx = ndim - 2
        i_idx = ndim - 1
        perm = [c_idx, i_idx] + [j for j in range(ndim) if j not in (c_idx, i_idx)]
        x_perm = jnp.transpose(x_centered, perm)  # (C, I, B, *D)
        x_flat = x_perm.reshape(channels, n_blades, -1)  # (C, I, N)
        n_samples = x_flat.shape[-1]
        cov = jnp.matmul(x_flat, jnp.swapaxes(x_flat, -1, -2)) / n_samples  # (C, I, I)
        if running_cov is not None:
            # cov: (C, I, I), running_cov: (I, I, C)
            cov_perm = jnp.transpose(cov, (1, 2, 0))  # (I, I, C)
            new_running_cov = running_cov + momentum * (cov_perm - running_cov)
    else:
        cov = jnp.transpose(running_cov, (2, 0, 1))  # (I, I, C) -> (C, I, I)

    # -- Cholesky whitening --
    # Scale eps by max eigenvalue magnitude to avoid numerical issues
    max_vals = jnp.amax(cov, axis=(1, 2))  # (C,)
    eye = jnp.eye(n_blades, dtype=cov.dtype)
    cov_reg = cov + eps * jnp.einsum("ij,c->cij", eye, jnp.maximum(max_vals, 1e-8))

    # Upper Cholesky: U^T U = cov_reg
    chol_lower = jax.scipy.linalg.cholesky(cov_reg, lower=True)
    chol_upper = jnp.swapaxes(chol_lower, -1, -2)  # (C, I, I)

    # Solve U @ z = x_centered for each spatial location
    # Broadcast chol_upper to match x_centered batch/spatial dims
    u_shape = (1,) * (1 + len(spatial_dims)) + (channels, n_blades, n_blades)
    u_bcast = jnp.broadcast_to(
        chol_upper.reshape(u_shape),
        (*x_centered.shape, n_blades),
    )

    x_whitened = jax.scipy.linalg.solve_triangular(
        u_bcast,
        x_centered[..., None],
        lower=False,
    ).squeeze(-1)

    return x_whitened, new_running_mean, new_running_cov
