"""Clifford algebra foundation: basis blade ordering, Cayley tables, and algebra operations.

Ported from ``microsoft/cliffordlayers`` (MIT license). All tensor operations use JAX.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp


class BasisBladeOrder:
    """Short-lex ordered basis blades for a Clifford algebra Cl(p, q).

    For *n* basis vectors the algebra has ``2^n`` basis blades.
    This class builds the index-to-bitmap and bitmap-to-index lookup tables
    together with the grade of each blade.

    Attributes:
        n_vectors: Number of generating vectors.
        index_to_bitmap: Maps blade index -> bitmap representation.
        bitmap_to_index: Maps bitmap -> blade index.
        grades: Grade (number of vectors in the wedge) for each blade.
    """

    def __init__(self, n_vectors: int) -> None:
        """Initialize BasisBladeOrder."""
        n_blades = 2**n_vectors

        # Short-lex: blade index IS the bitmap. Grade = popcount(bitmap).
        self.n_vectors = n_vectors
        self.index_to_bitmap = jnp.arange(n_blades, dtype=jnp.int32)
        self.bitmap_to_index = jnp.arange(n_blades, dtype=jnp.int32)
        self.grades = jnp.array(
            [_count_set_bits(i) for i in range(n_blades)],
            dtype=jnp.int32,
        )


class CliffordAlgebra:
    """Clifford algebra Cl(g) over a diagonal metric ``g``.

    This is a pure mathematical utility (NOT an ``nnx.Module``) that stores
    pre-computed lookup tables as JAX arrays.  It supports dimensions 1, 2, 3
    (2, 4, 8 blades respectively).

    Attributes:
        metric: Diagonal entries of the metric tensor, shape ``(dim,)``.
        dim: Number of basis vectors.
        n_blades: Total number of basis blades (``2^dim``).
        cayley: Dense geometric multiplication table, ``(n_blades, n_blades, n_blades)``.
        grades: Grade of each basis blade, ``(n_blades,)``.
    """

    def __init__(self, metric: Sequence[int] | jax.Array) -> None:
        """Initialize CliffordAlgebra."""
        self.metric = jnp.asarray(metric, dtype=jnp.float32)
        self.dim = int(self.metric.shape[0])
        if self.dim not in (1, 2, 3):
            raise ValueError(f"Only dimensions 1, 2, 3 are supported, got {self.dim}.")
        self.bbo = BasisBladeOrder(self.dim)
        self.n_blades = int(2**self.dim)
        self.cayley = _construct_cayley_table(self.bbo, self.metric)
        self.grades = self.bbo.grades

    # ------------------------------------------------------------------
    # Core algebra operations
    # ------------------------------------------------------------------

    def geometric_product(self, a: jax.Array, b: jax.Array) -> jax.Array:
        """Compute the geometric product of two multivectors.

        Args:
            a: Multivector batch ``(..., n_blades)``.
            b: Multivector batch ``(..., n_blades)``.

        Returns:
            Geometric product ``a * b``, same shape as inputs.
        """
        original_shape = a.shape
        a_flat = a.reshape(-1, self.n_blades)
        b_flat = b.reshape(-1, self.n_blades)
        result = jnp.einsum("bi,ijk,bk->bj", a_flat, self.cayley, b_flat)
        return result.reshape(original_shape)

    def reverse(
        self,
        mv: jax.Array,
        blades: Sequence[int] | jax.Array | None = None,
    ) -> jax.Array:
        """Reverse a multivector (reversion anti-automorphism).

        The reverse of a *k*-blade picks up a sign ``(-1)^{k(k-1)/2}``.

        Args:
            mv: Multivector batch ``(..., n_blades)`` or ``(..., len(blades))``.
            blades: Optional subset of blade indices present in *mv*.

        Returns:
            Reversed multivector.
        """
        grades = self.grades
        if blades is not None:
            grades = grades[jnp.asarray(blades, dtype=jnp.int32)]
        signs = jnp.pow(-1.0, jnp.floor(grades * (grades - 1) / 2.0))
        return signs * mv

    def embed(
        self,
        tensor: jax.Array,
        tensor_index: Sequence[int] | jax.Array,
    ) -> jax.Array:
        """Embed a tensor into a full multivector.

        Places the components of *tensor* (last-axis size ``len(tensor_index)``)
        into the blade positions given by *tensor_index*.

        Args:
            tensor: Input ``(..., len(tensor_index))``.
            tensor_index: Blade indices to fill.

        Returns:
            Multivector ``(..., n_blades)`` with zeros elsewhere.
        """
        indices = jnp.asarray(tensor_index, dtype=jnp.int32)
        mv = jnp.zeros((*tensor.shape[:-1], self.n_blades), dtype=tensor.dtype)
        return mv.at[..., indices].set(tensor)

    def get(
        self,
        mv: jax.Array,
        blade_index: Sequence[int] | jax.Array,
    ) -> jax.Array:
        """Extract components from a multivector.

        Args:
            mv: Multivector ``(..., n_blades)``.
            blade_index: Blade indices to extract.

        Returns:
            Tensor ``(..., len(blade_index))``.
        """
        return mv[..., jnp.asarray(blade_index, dtype=jnp.int32)]

    def mag2(self, mv: jax.Array) -> jax.Array:
        """Squared magnitude: ``reverse(mv) * mv``."""
        return self.geometric_product(self.reverse(mv), mv)

    def norm(self, mv: jax.Array) -> jax.Array:
        """Multivector norm: ``sqrt(|mag2(mv)[..., :1]|)``."""
        return jnp.sqrt(jnp.abs(self.mag2(mv)[..., :1]))

    def sandwich(self, a: jax.Array, b: jax.Array) -> jax.Array:
        """Sandwich product ``a * b * reverse(a)``."""
        return self.geometric_product(
            self.geometric_product(a, b),
            self.reverse(a),
        )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _count_set_bits(bitmap: int) -> int:
    """Count the number of bits set to 1 in *bitmap*."""
    count = 0
    while bitmap > 0:
        count += bitmap & 1
        bitmap >>= 1
    return count


def _canonical_reordering_sign(bitmap_a: int, bitmap_b: int, metric: Sequence[float]) -> float:
    """Sign and scale factor when multiplying two basis blades.

    Accounts for the euclidean reordering sign *and* the diagonal metric entries
    for shared basis vectors.
    """
    # Euclidean reordering sign
    a = bitmap_a >> 1
    swaps = 0
    while a:
        swaps += _count_set_bits(a & bitmap_b)
        a >>= 1
    sign = 1 if (swaps & 1) == 0 else -1

    # Metric contribution: shared basis vectors
    shared = bitmap_a & bitmap_b
    i = 0
    while shared:
        if shared & 1:
            sign *= int(metric[i])
        shared >>= 1
        i += 1

    return float(sign)


def _construct_cayley_table(
    bbo: BasisBladeOrder,
    metric: jax.Array,
) -> jax.Array:
    """Build the dense geometric multiplication table.

    Returns:
        ``jnp.array`` of shape ``(n_blades, n_blades, n_blades)`` where
        ``cayley[i, v, j]`` is the coefficient contributed to blade *v*
        when multiplying blade *i* by blade *j*.
    """
    n = 2**bbo.n_vectors
    # Build as a Python list first (construction is done once at init time).
    i2b = [int(bbo.index_to_bitmap[k]) for k in range(n)]
    b2i = [int(bbo.bitmap_to_index[k]) for k in range(n)]
    metric_list = [float(metric[k]) for k in range(len(metric))]

    table = [[[0.0] * n for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            bitmap_v = i2b[i] ^ i2b[j]
            v = b2i[bitmap_v]
            sign = _canonical_reordering_sign(i2b[i], i2b[j], metric_list)
            table[i][v][j] = sign

    return jnp.array(table, dtype=jnp.float32)
