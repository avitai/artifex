"""Clifford kernel construction for 1D, 2D, and 3D Clifford algebras.

Ported from ``microsoft/cliffordlayers`` (MIT license). Each function takes
a stacked weight tensor ``w`` of shape ``(n_blades, out_ch, in_ch, *K)``
and a metric ``g`` and returns the full Clifford-structured weight matrix.
"""

import jax.numpy as jnp


def get_1d_clifford_kernel(
    w: jnp.ndarray,
    g: jnp.ndarray,
) -> tuple[int, jnp.ndarray]:
    """Clifford kernel for Cl(1): ``g = (-1,)`` yields a complex kernel.

    Args:
        w: Weight tensor ``(2, out_ch, in_ch, ...)``.
        g: Metric signature ``(1,)``.

    Returns:
        ``(n_blades=2, kernel)`` where kernel has shape
        ``(2*out_ch, 2*in_ch, ...)``.
    """
    if g.size != 1:
        raise ValueError(f"1D kernel expects metric of size 1, got {g.size}.")
    if w.shape[0] != 2:
        raise ValueError(f"1D kernel expects 2 blades, got {w.shape[0]}.")

    k0 = jnp.concatenate([w[0], g[0] * w[1]], axis=1)
    k1 = jnp.concatenate([w[1], w[0]], axis=1)
    return 2, jnp.concatenate([k0, k1], axis=0)


def get_2d_clifford_kernel(
    w: jnp.ndarray,
    g: jnp.ndarray,
) -> tuple[int, jnp.ndarray]:
    """Clifford kernel for Cl(2): ``g = (-1,-1)`` yields a quaternion kernel.

    Args:
        w: Weight tensor ``(4, out_ch, in_ch, ...)``.
        g: Metric signature ``(2,)``.

    Returns:
        ``(n_blades=4, kernel)`` where kernel has shape
        ``(4*out_ch, 4*in_ch, ...)``.
    """
    if g.size != 2:
        raise ValueError(f"2D kernel expects metric of size 2, got {g.size}.")
    if w.shape[0] != 4:
        raise ValueError(f"2D kernel expects 4 blades, got {w.shape[0]}.")

    k0 = jnp.concatenate([w[0], g[0] * w[1], g[1] * w[2], -g[0] * g[1] * w[3]], axis=1)
    k1 = jnp.concatenate([w[1], w[0], -g[1] * w[3], g[1] * w[2]], axis=1)
    k2 = jnp.concatenate([w[2], g[0] * w[3], w[0], -g[0] * w[1]], axis=1)
    k3 = jnp.concatenate([w[3], w[2], -w[1], w[0]], axis=1)
    return 4, jnp.concatenate([k0, k1, k2, k3], axis=0)


def get_2d_clifford_rotation_kernel(
    w: jnp.ndarray,
    g: jnp.ndarray,
) -> tuple[int, jnp.ndarray]:
    """Rotational Clifford kernel for Cl(2) with ``g = (-1, -1)``.

    The weight tensor has 6 slices: 4 Clifford weights, 1 scale, 1 zero kernel.

    Args:
        w: Weight tensor ``(6, out_ch, in_ch, ...)``.
        g: Metric signature ``(2,)`` — must be ``(-1, -1)``.

    Returns:
        ``(n_blades=4, kernel)`` where kernel has shape
        ``(4*out_ch, 4*in_ch, ...)``.
    """
    if g.size != 2:
        raise ValueError(f"2D rotation kernel expects metric of size 2, got {g.size}.")
    if not (float(g[0]) == -1.0 and float(g[1]) == -1.0):
        raise ValueError("Rotation kernel requires signature (-1, -1).")
    if w.shape[0] != 6:
        raise ValueError(f"2D rotation kernel expects 6 weight slices, got {w.shape[0]}.")

    # Scalar output kernel
    k0 = jnp.concatenate([w[0], -w[1], -w[2], -w[3]], axis=1)

    # Quaternion rotation
    s0 = w[0] * w[0]
    s1 = w[1] * w[1]
    s2 = w[2] * w[2]
    s3 = w[3] * w[3]
    norm = jnp.sqrt(s0 + s1 + s2 + s3 + 1e-4)
    w0_n = w[0] / norm
    w1_n = w[1] / norm
    w2_n = w[2] / norm
    w3_n = w[3] / norm

    nf = 2.0
    sq1 = nf * (w1_n * w1_n)
    sq2 = nf * (w2_n * w2_n)
    sq3 = nf * (w3_n * w3_n)
    r01 = nf * w0_n * w1_n
    r02 = nf * w0_n * w2_n
    r03 = nf * w0_n * w3_n
    r12 = nf * w1_n * w2_n
    r13 = nf * w1_n * w3_n
    r23 = nf * w2_n * w3_n

    scale = w[4]
    zero = w[5]

    k1 = jnp.concatenate(
        [zero, scale * (1.0 - (sq2 + sq3)), scale * (r12 - r03), scale * (r13 + r02)],
        axis=1,
    )
    k2 = jnp.concatenate(
        [zero, scale * (r12 + r03), scale * (1.0 - (sq1 + sq3)), scale * (r23 - r01)],
        axis=1,
    )
    k3 = jnp.concatenate(
        [zero, scale * (r13 - r02), scale * (r23 + r01), scale * (1.0 - (sq1 + sq2))],
        axis=1,
    )
    return 4, jnp.concatenate([k0, k1, k2, k3], axis=0)


def get_3d_clifford_kernel(
    w: jnp.ndarray,
    g: jnp.ndarray,
) -> tuple[int, jnp.ndarray]:
    """Clifford kernel for Cl(3): 8-blade algebra.

    Args:
        w: Weight tensor ``(8, out_ch, in_ch, ...)``.
        g: Metric signature ``(3,)``.

    Returns:
        ``(n_blades=8, kernel)`` where kernel has shape
        ``(8*out_ch, 8*in_ch, ...)``.
    """
    if g.size != 3:
        raise ValueError(f"3D kernel expects metric of size 3, got {g.size}.")
    if w.shape[0] != 8:
        raise ValueError(f"3D kernel expects 8 blades, got {w.shape[0]}.")

    k0 = jnp.concatenate(
        [
            w[0],
            w[1] * g[0],
            w[2] * g[1],
            w[3] * g[2],
            -w[4] * g[0] * g[1],
            -w[5] * g[0] * g[2],
            -w[6] * g[1] * g[2],
            -w[7] * g[0] * g[1] * g[2],
        ],
        axis=1,
    )
    k1 = jnp.concatenate(
        [
            w[1],
            w[0],
            -w[4] * g[1],
            -w[5] * g[2],
            w[2] * g[1],
            w[3] * g[2],
            -w[7] * g[1] * g[2],
            -w[6] * g[2] * g[1],
        ],
        axis=1,
    )
    k2 = jnp.concatenate(
        [
            w[2],
            w[4] * g[0],
            w[0],
            -w[6] * g[2],
            -w[1] * g[0],
            w[7] * g[0] * g[2],
            w[3] * g[2],
            w[5] * g[2] * g[0],
        ],
        axis=1,
    )
    k3 = jnp.concatenate(
        [
            w[3],
            w[5] * g[0],
            w[6] * g[1],
            w[0],
            -w[7] * g[0] * g[1],
            -w[1] * g[0],
            -w[2] * g[1],
            -w[4] * g[0] * g[1],
        ],
        axis=1,
    )
    k4 = jnp.concatenate(
        [
            w[4],
            w[2],
            -w[1],
            g[2] * w[7],
            w[0],
            -w[6] * g[2],
            w[5] * g[2],
            w[3] * g[2],
        ],
        axis=1,
    )
    k5 = jnp.concatenate(
        [
            w[5],
            w[3],
            -w[7] * g[1],
            -w[1],
            w[6] * g[1],
            w[0],
            -w[4] * g[1],
            -w[2] * g[1],
        ],
        axis=1,
    )
    k6 = jnp.concatenate(
        [
            w[6],
            w[7] * g[0],
            w[3],
            -w[2],
            -w[5] * g[0],
            w[4] * g[0],
            w[0],
            w[1] * g[0],
        ],
        axis=1,
    )
    k7 = jnp.concatenate(
        [
            w[7],
            w[6],
            -w[5],
            w[4],
            w[3],
            -w[2],
            w[1],
            w[0],
        ],
        axis=1,
    )
    return 8, jnp.concatenate([k0, k1, k2, k3, k4, k5, k6, k7], axis=0)
