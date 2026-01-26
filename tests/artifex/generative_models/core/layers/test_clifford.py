"""Tests for Clifford algebra layers.

Written FIRST (TDD) — implementation follows to make these pass.
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from flax import nnx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rngs() -> nnx.Rngs:
    """Shared RNG fixture."""
    return nnx.Rngs(0)


@pytest.fixture
def algebra_1d():
    """Cl(1,0) — 2 blades."""
    from artifex.generative_models.core.layers.clifford.algebra import CliffordAlgebra

    return CliffordAlgebra((1,))


@pytest.fixture
def algebra_2d():
    """Cl(2,0) — 4 blades."""
    from artifex.generative_models.core.layers.clifford.algebra import CliffordAlgebra

    return CliffordAlgebra((1, 1))


@pytest.fixture
def algebra_3d():
    """Cl(3,0) — 8 blades."""
    from artifex.generative_models.core.layers.clifford.algebra import CliffordAlgebra

    return CliffordAlgebra((1, 1, 1))


# ===========================================================================
# TestBasisBladeOrder
# ===========================================================================


class TestBasisBladeOrder:
    """Tests for BasisBladeOrder short-lex ordering."""

    def test_1d_blades(self) -> None:
        """1D algebra has 2 blades: scalar, e1."""
        from artifex.generative_models.core.layers.clifford.algebra import BasisBladeOrder

        bbo = BasisBladeOrder(1)
        assert bbo.index_to_bitmap.shape == (2,)
        npt.assert_array_equal(bbo.grades, [0, 1])

    def test_2d_blades(self) -> None:
        """2D algebra has 4 blades: 1, e1, e2, e12."""
        from artifex.generative_models.core.layers.clifford.algebra import BasisBladeOrder

        bbo = BasisBladeOrder(2)
        assert bbo.index_to_bitmap.shape == (4,)
        npt.assert_array_equal(bbo.grades, [0, 1, 1, 2])

    def test_3d_blades(self) -> None:
        """3D algebra has 8 blades."""
        from artifex.generative_models.core.layers.clifford.algebra import BasisBladeOrder

        bbo = BasisBladeOrder(3)
        assert bbo.index_to_bitmap.shape == (8,)
        npt.assert_array_equal(bbo.grades, [0, 1, 1, 2, 1, 2, 2, 3])

    def test_bitmap_roundtrip(self) -> None:
        """index -> bitmap -> index is identity."""
        from artifex.generative_models.core.layers.clifford.algebra import BasisBladeOrder

        bbo = BasisBladeOrder(3)
        for i in range(8):
            bitmap = int(bbo.index_to_bitmap[i])
            assert int(bbo.bitmap_to_index[bitmap]) == i

    def test_grades_values(self) -> None:
        """Grade equals number of basis vectors in each blade."""
        from artifex.generative_models.core.layers.clifford.algebra import BasisBladeOrder

        bbo = BasisBladeOrder(2)
        # 1 (grade 0), e1 (grade 1), e2 (grade 1), e12 (grade 2)
        assert int(bbo.grades[0]) == 0
        assert int(bbo.grades[1]) == 1
        assert int(bbo.grades[2]) == 1
        assert int(bbo.grades[3]) == 2


# ===========================================================================
# TestCliffordAlgebra
# ===========================================================================


class TestCliffordAlgebra:
    """Tests for CliffordAlgebra core operations."""

    def test_init_1d(self, algebra_1d) -> None:
        """Cl(1,0) has dim=1, n_blades=2."""
        assert algebra_1d.dim == 1
        assert algebra_1d.n_blades == 2
        assert algebra_1d.cayley.shape == (2, 2, 2)

    def test_init_2d(self, algebra_2d) -> None:
        """Cl(2,0) has dim=2, n_blades=4."""
        assert algebra_2d.dim == 2
        assert algebra_2d.n_blades == 4
        assert algebra_2d.cayley.shape == (4, 4, 4)

    def test_init_3d(self, algebra_3d) -> None:
        """Cl(3,0) has dim=3, n_blades=8."""
        assert algebra_3d.dim == 3
        assert algebra_3d.n_blades == 8
        assert algebra_3d.cayley.shape == (8, 8, 8)

    def test_init_unsupported_dim(self) -> None:
        """Dimensions > 3 raise ValueError."""
        from artifex.generative_models.core.layers.clifford.algebra import CliffordAlgebra

        with pytest.raises(ValueError, match="Only dimensions 1, 2, 3"):
            CliffordAlgebra((1, 1, 1, 1))

    def test_cayley_1d_euclidean(self, algebra_1d) -> None:
        """In Cl(1,0): e1*e1 = g[0] = 1."""
        # e1 is blade index 1
        e1 = jnp.array([[0.0, 1.0]])  # batch=1, blades=2
        result = algebra_1d.geometric_product(e1, e1)
        # Should give scalar = 1 (metric is +1)
        npt.assert_allclose(float(result[0, 0]), 1.0, atol=1e-6)
        npt.assert_allclose(float(result[0, 1]), 0.0, atol=1e-6)

    def test_cayley_2d_anticommutation(self, algebra_2d) -> None:
        """In Cl(2,0): e1*e2 = e12, e2*e1 = -e12."""
        e1 = jnp.array([[0.0, 1.0, 0.0, 0.0]])
        e2 = jnp.array([[0.0, 0.0, 1.0, 0.0]])
        e1e2 = algebra_2d.geometric_product(e1, e2)
        e2e1 = algebra_2d.geometric_product(e2, e1)
        # e1*e2 should have e12 component = +1
        npt.assert_allclose(float(e1e2[0, 3]), 1.0, atol=1e-6)
        # e2*e1 should have e12 component = -1
        npt.assert_allclose(float(e2e1[0, 3]), -1.0, atol=1e-6)

    def test_geometric_product_associative(self, algebra_2d) -> None:
        """(a*b)*c == a*(b*c) for random multivectors."""
        key = jax.random.key(42)
        a = jax.random.normal(key, (3, 4))
        b = jax.random.normal(jax.random.key(43), (3, 4))
        c = jax.random.normal(jax.random.key(44), (3, 4))
        lhs = algebra_2d.geometric_product(algebra_2d.geometric_product(a, b), c)
        rhs = algebra_2d.geometric_product(a, algebra_2d.geometric_product(b, c))
        npt.assert_allclose(lhs, rhs, atol=1e-5)

    def test_geometric_product_scalar_identity(self, algebra_2d) -> None:
        """Scalar 1 is the identity: 1*a == a."""
        key = jax.random.key(99)
        a = jax.random.normal(key, (2, 4))
        one = jnp.array([[1.0, 0.0, 0.0, 0.0]] * 2)
        result = algebra_2d.geometric_product(one, a)
        npt.assert_allclose(result, a, atol=1e-6)

    def test_reverse_involution(self, algebra_2d) -> None:
        """reverse(reverse(x)) == x."""
        key = jax.random.key(77)
        x = jax.random.normal(key, (5, 4))
        result = algebra_2d.reverse(algebra_2d.reverse(x))
        npt.assert_allclose(result, x, atol=1e-6)

    def test_reverse_scalar_unchanged(self, algebra_2d) -> None:
        """Scalars are not affected by reversion."""
        scalar = jnp.array([[3.0, 0.0, 0.0, 0.0]])
        result = algebra_2d.reverse(scalar)
        npt.assert_allclose(result, scalar, atol=1e-6)

    def test_embed_get_roundtrip(self, algebra_2d) -> None:
        """embed then get recovers original tensor."""
        t = jnp.array([[1.0, 2.0]])  # 2 components
        indices = (1, 2)  # e1, e2
        mv = algebra_2d.embed(t, indices)
        recovered = algebra_2d.get(mv, indices)
        npt.assert_allclose(recovered, t, atol=1e-6)

    def test_embed_shape(self, algebra_3d) -> None:
        """Embed produces correct output dimensions."""
        t = jnp.ones((4, 5, 3))  # last dim = 3 components
        mv = algebra_3d.embed(t, (0, 1, 2))
        assert mv.shape == (4, 5, 8)

    def test_norm_positive(self, algebra_2d) -> None:
        """Norm is non-negative for random multivectors."""
        key = jax.random.key(55)
        mv = jax.random.normal(key, (10, 4))
        n = algebra_2d.norm(mv)
        assert jnp.all(n >= 0)

    def test_mag2_scalar_part(self, algebra_2d) -> None:
        """mag2 result has most energy in the scalar blade."""
        e1 = jnp.array([[0.0, 1.0, 0.0, 0.0]])
        m = algebra_2d.mag2(e1)
        # e1~ * e1 = e1*e1 = g[0] = 1 (scalar)
        npt.assert_allclose(float(m[0, 0]), 1.0, atol=1e-6)

    def test_sandwich_rotation_2d(self) -> None:
        """Sandwich with a known rotor produces expected rotation in Cl(0,2)."""
        from artifex.generative_models.core.layers.clifford.algebra import CliffordAlgebra

        # Cl(0,2) with metric (-1,-1)
        alg = CliffordAlgebra((-1, -1))
        # Rotor for 90-degree rotation: R = cos(pi/4) + sin(pi/4)*e12
        angle = jnp.pi / 4.0
        rotor = jnp.array([[jnp.cos(angle), 0.0, 0.0, jnp.sin(angle)]])
        # Rotate e1 vector
        e1 = jnp.array([[0.0, 1.0, 0.0, 0.0]])
        result = alg.sandwich(rotor, e1)
        # Should produce e2 (90-degree rotation)
        npt.assert_allclose(float(result[0, 1]), 0.0, atol=1e-5)
        npt.assert_allclose(float(result[0, 2]), 1.0, atol=1e-5)


# ===========================================================================
# TestCliffordKernels
# ===========================================================================


class TestCliffordKernels:
    """Tests for Clifford kernel construction functions."""

    def test_1d_kernel_shape(self) -> None:
        """1D kernel: (2, O, I, K) -> (2*O, 2*I, K)."""
        from artifex.generative_models.core.layers.clifford.kernels import get_1d_clifford_kernel

        w = jnp.ones((2, 4, 3, 5))
        g = jnp.array([1.0])
        n_blades, k = get_1d_clifford_kernel(w, g)
        assert n_blades == 2
        assert k.shape == (8, 6, 5)

    def test_2d_kernel_shape(self) -> None:
        """2D kernel: (4, O, I, H, W) -> (4*O, 4*I, H, W)."""
        from artifex.generative_models.core.layers.clifford.kernels import get_2d_clifford_kernel

        w = jnp.ones((4, 4, 3, 5, 5))
        g = jnp.array([1.0, 1.0])
        n_blades, k = get_2d_clifford_kernel(w, g)
        assert n_blades == 4
        assert k.shape == (16, 12, 5, 5)

    def test_2d_rotation_kernel_shape(self) -> None:
        """2D rotation kernel: 6 slices -> (4*O, 4*I, H, W)."""
        from artifex.generative_models.core.layers.clifford.kernels import (
            get_2d_clifford_rotation_kernel,
        )

        w = jnp.ones((6, 4, 3, 5, 5))
        g = jnp.array([-1.0, -1.0])
        n_blades, k = get_2d_clifford_rotation_kernel(w, g)
        assert n_blades == 4
        assert k.shape == (16, 12, 5, 5)

    def test_3d_kernel_shape(self) -> None:
        """3D kernel: (8, O, I, D, H, W) -> (8*O, 8*I, D, H, W)."""
        from artifex.generative_models.core.layers.clifford.kernels import get_3d_clifford_kernel

        w = jnp.ones((8, 2, 3, 3, 3, 3))
        g = jnp.array([1.0, 1.0, 1.0])
        n_blades, k = get_3d_clifford_kernel(w, g)
        assert n_blades == 8
        assert k.shape == (16, 24, 3, 3, 3)

    def test_kernel_wrong_blade_count_raises(self) -> None:
        """Wrong number of weight blades raises ValueError."""
        from artifex.generative_models.core.layers.clifford.kernels import get_2d_clifford_kernel

        w = jnp.ones((3, 4, 4, 3, 3))  # 3 blades, should be 4
        g = jnp.array([1.0, 1.0])
        with pytest.raises(ValueError, match="4 blades"):
            get_2d_clifford_kernel(w, g)

    def test_kernel_wrong_metric_dim_raises(self) -> None:
        """Wrong metric size raises ValueError."""
        from artifex.generative_models.core.layers.clifford.kernels import get_1d_clifford_kernel

        w = jnp.ones((2, 4, 4, 3))
        g = jnp.array([1.0, 1.0])  # size 2, should be 1
        with pytest.raises(ValueError, match="metric of size 1"):
            get_1d_clifford_kernel(w, g)

    def test_kernel_algebraic_consistency(self, algebra_2d) -> None:
        """Kernel applied to basis vector matches geometric product."""
        from artifex.generative_models.core.layers.clifford.kernels import get_2d_clifford_kernel

        # Identity-like weight: w[i] = delta_{i0} * I_{1x1}
        w = jnp.zeros((4, 1, 1))
        w = w.at[0].set(1.0)
        g = jnp.array([1.0, 1.0])
        _, kernel = get_2d_clifford_kernel(w, g)
        # kernel should act as identity on 4-blade vectors
        assert kernel.shape == (4, 4)
        npt.assert_allclose(kernel, jnp.eye(4), atol=1e-6)

    def test_rotation_kernel_wrong_signature_raises(self) -> None:
        """Rotation kernel with wrong signature raises ValueError."""
        from artifex.generative_models.core.layers.clifford.kernels import (
            get_2d_clifford_rotation_kernel,
        )

        w = jnp.ones((6, 4, 3, 5, 5))
        g = jnp.array([1.0, 1.0])  # should be (-1, -1)
        with pytest.raises(ValueError, match="signature"):
            get_2d_clifford_rotation_kernel(w, g)


# ===========================================================================
# TestFunctional
# ===========================================================================


class TestFunctional:
    """Tests for private functional utilities."""

    def test_batchmul1d(self) -> None:
        """batchmul1d contracts correctly."""
        from artifex.generative_models.core.layers.clifford._functional import batchmul1d

        x = jnp.ones((2, 3, 5))
        w = jnp.ones((4, 3, 5))
        out = batchmul1d(x, w)
        assert out.shape == (2, 4, 5)

    def test_batchmul2d(self) -> None:
        """batchmul2d contracts correctly."""
        from artifex.generative_models.core.layers.clifford._functional import batchmul2d

        x = jnp.ones((2, 3, 5, 5))
        w = jnp.ones((4, 3, 5, 5))
        out = batchmul2d(x, w)
        assert out.shape == (2, 4, 5, 5)

    def test_batchmul3d(self) -> None:
        """batchmul3d contracts correctly."""
        from artifex.generative_models.core.layers.clifford._functional import batchmul3d

        x = jnp.ones((2, 3, 4, 4, 4))
        w = jnp.ones((4, 3, 4, 4, 4))
        out = batchmul3d(x, w)
        assert out.shape == (2, 4, 4, 4, 4)

    def test_whiten_data_shape(self) -> None:
        """whiten_data returns correct shape and updated stats."""
        from artifex.generative_models.core.layers.clifford._functional import whiten_data

        # Channels-last: (B, H, W, C=3, I=2)
        x = jax.random.normal(jax.random.key(0), (4, 8, 8, 3, 2))
        r_mean = jnp.zeros((2, 3))  # (I, C)
        r_cov = jnp.broadcast_to(jnp.eye(2)[..., None], (2, 2, 3)).copy()  # (I, I, C)
        whitened, new_mean, new_cov = whiten_data(
            x,
            training=True,
            running_mean=r_mean,
            running_cov=r_cov,
            momentum=0.1,
            eps=1e-5,
        )
        assert whitened.shape == x.shape
        assert new_mean.shape == (2, 3)
        assert new_cov.shape == (2, 2, 3)


# ===========================================================================
# TestCliffordLinear
# ===========================================================================


class TestCliffordLinear:
    """Tests for CliffordLinear module."""

    def test_init(self, rngs) -> None:
        """Attributes and param shapes are correct."""
        from artifex.generative_models.core.layers.clifford.linear import CliffordLinear

        layer = CliffordLinear(metric=(1, 1), in_channels=8, out_channels=16, rngs=rngs)
        assert layer.in_channels == 8
        assert layer.out_channels == 16
        assert layer.n_blades == 4
        assert layer.weight.value.shape == (4, 16, 8)

    def test_forward_shape_2d(self, rngs) -> None:
        """Forward (B, C_in, I) -> (B, C_out, I)."""
        from artifex.generative_models.core.layers.clifford.linear import CliffordLinear

        layer = CliffordLinear(metric=(1, 1), in_channels=8, out_channels=16, rngs=rngs)
        x = jnp.ones((2, 8, 4))
        out = layer(x)
        assert out.shape == (2, 16, 4)

    def test_forward_shape_1d(self, rngs) -> None:
        """Forward for 1D algebra."""
        from artifex.generative_models.core.layers.clifford.linear import CliffordLinear

        layer = CliffordLinear(metric=(1,), in_channels=4, out_channels=8, rngs=rngs)
        x = jnp.ones((3, 4, 2))
        out = layer(x)
        assert out.shape == (3, 8, 2)

    def test_forward_shape_3d(self, rngs) -> None:
        """Forward for 3D algebra."""
        from artifex.generative_models.core.layers.clifford.linear import CliffordLinear

        layer = CliffordLinear(metric=(1, 1, 1), in_channels=4, out_channels=8, rngs=rngs)
        x = jnp.ones((2, 4, 8))
        out = layer(x)
        assert out.shape == (2, 8, 8)

    def test_no_bias(self, rngs) -> None:
        """No bias parameter when use_bias=False."""
        from artifex.generative_models.core.layers.clifford.linear import CliffordLinear

        layer = CliffordLinear(
            metric=(1,), in_channels=4, out_channels=8, use_bias=False, rngs=rngs
        )
        assert not hasattr(layer, "bias") or layer.bias is None
        x = jnp.ones((2, 4, 2))
        out = layer(x)
        assert out.shape == (2, 8, 2)

    def test_jit_compatible(self, rngs) -> None:
        """Layer works under jax.jit."""
        from artifex.generative_models.core.layers.clifford.linear import CliffordLinear

        layer = CliffordLinear(metric=(1, 1), in_channels=4, out_channels=8, rngs=rngs)
        x = jnp.ones((2, 4, 4))

        @nnx.jit
        def forward(model, x):
            return model(x)

        out = forward(layer, x)
        assert out.shape == (2, 8, 4)

    def test_gradient_flow(self, rngs) -> None:
        """Gradients flow through the layer."""
        from artifex.generative_models.core.layers.clifford.linear import CliffordLinear

        layer = CliffordLinear(metric=(1, 1), in_channels=4, out_channels=8, rngs=rngs)
        x = jnp.ones((2, 4, 4))

        def loss_fn(model):
            return jnp.sum(model(x))

        grads = nnx.grad(loss_fn)(layer)
        assert grads.weight.value.shape == layer.weight.value.shape


# ===========================================================================
# TestCliffordConv
# ===========================================================================


class TestCliffordConv1d:
    """Tests for CliffordConv1d."""

    def test_init(self, rngs) -> None:
        """Param shapes are correct."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv1d

        layer = CliffordConv1d(metric=(1,), in_channels=4, out_channels=8, kernel_size=3, rngs=rngs)
        assert layer.weight.value.shape == (2, 8, 4, 3)

    def test_forward_shape(self, rngs) -> None:
        """Output spatial dim with SAME padding."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv1d

        layer = CliffordConv1d(
            metric=(1,),
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((2, 16, 4, 2))  # (B, L, C, I)
        out = layer(x)
        assert out.shape == (2, 16, 8, 2)

    def test_stride(self, rngs) -> None:
        """Stride reduces spatial dimension."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv1d

        layer = CliffordConv1d(
            metric=(1,),
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((2, 16, 4, 2))
        out = layer(x)
        assert out.shape == (2, 8, 8, 2)

    def test_jit_compatible(self, rngs) -> None:
        """CliffordConv1d works under jit."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv1d

        layer = CliffordConv1d(
            metric=(1,),
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((2, 8, 4, 2))

        @nnx.jit
        def forward(m, x):
            return m(x)

        assert forward(layer, x).shape == (2, 8, 4, 2)

    def test_gradient_flow(self, rngs) -> None:
        """Gradients flow through CliffordConv1d."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv1d

        layer = CliffordConv1d(
            metric=(1,),
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((2, 8, 4, 2))

        def loss_fn(m):
            return jnp.sum(m(x))

        grads = nnx.grad(loss_fn)(layer)
        assert grads.weight.value.shape == layer.weight.value.shape


class TestCliffordConv2d:
    """Tests for CliffordConv2d."""

    def test_init(self, rngs) -> None:
        """Param shapes are correct."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv2d

        layer = CliffordConv2d(
            metric=(1, 1),
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            rngs=rngs,
        )
        assert layer.weight.value.shape == (4, 8, 4, 3, 3)

    def test_forward_shape(self, rngs) -> None:
        """Output spatial dims with SAME padding."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv2d

        layer = CliffordConv2d(
            metric=(1, 1),
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((2, 8, 8, 4, 4))  # (B, H, W, C, I)
        out = layer(x)
        assert out.shape == (2, 8, 8, 8, 4)

    def test_stride(self, rngs) -> None:
        """Stride reduces spatial dimensions."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv2d

        layer = CliffordConv2d(
            metric=(1, 1),
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((2, 8, 8, 4, 4))
        out = layer(x)
        assert out.shape == (2, 4, 4, 8, 4)

    def test_rotation_kernel(self, rngs) -> None:
        """Rotation mode works for Cl(0,2)."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv2d

        layer = CliffordConv2d(
            metric=(-1, -1),
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            padding="SAME",
            rotation=True,
            rngs=rngs,
        )
        x = jnp.ones((2, 8, 8, 4, 4))
        out = layer(x)
        assert out.shape == (2, 8, 8, 8, 4)

    def test_jit_compatible(self, rngs) -> None:
        """CliffordConv2d works under jit."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv2d

        layer = CliffordConv2d(
            metric=(1, 1),
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((2, 8, 8, 4, 4))

        @nnx.jit
        def forward(m, x):
            return m(x)

        assert forward(layer, x).shape == (2, 8, 8, 4, 4)


class TestCliffordConv3d:
    """Tests for CliffordConv3d."""

    def test_init(self, rngs) -> None:
        """Param shapes are correct."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv3d

        layer = CliffordConv3d(
            metric=(1, 1, 1),
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            rngs=rngs,
        )
        assert layer.weight.value.shape == (8, 4, 2, 3, 3, 3)

    def test_forward_shape(self, rngs) -> None:
        """Output spatial dims with SAME padding."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv3d

        layer = CliffordConv3d(
            metric=(1, 1, 1),
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((1, 4, 4, 4, 2, 8))  # (B, D, H, W, C, I)
        out = layer(x)
        assert out.shape == (1, 4, 4, 4, 4, 8)

    def test_stride(self, rngs) -> None:
        """Stride reduces spatial dimensions."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv3d

        layer = CliffordConv3d(
            metric=(1, 1, 1),
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((1, 4, 4, 4, 2, 8))
        out = layer(x)
        assert out.shape == (1, 2, 2, 2, 4, 8)

    def test_jit_compatible(self, rngs) -> None:
        """CliffordConv3d works under jit."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv3d

        layer = CliffordConv3d(
            metric=(1, 1, 1),
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((1, 4, 4, 4, 2, 8))

        @nnx.jit
        def forward(m, x):
            return m(x)

        assert forward(layer, x).shape == (1, 4, 4, 4, 2, 8)

    def test_gradient_flow(self, rngs) -> None:
        """Gradients flow through CliffordConv3d."""
        from artifex.generative_models.core.layers.clifford.conv import CliffordConv3d

        layer = CliffordConv3d(
            metric=(1, 1, 1),
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((1, 4, 4, 4, 2, 8))

        def loss_fn(m):
            return jnp.sum(m(x))

        grads = nnx.grad(loss_fn)(layer)
        assert grads.weight.value.shape == layer.weight.value.shape


# ===========================================================================
# TestCliffordSpectralConv
# ===========================================================================


class TestCliffordSpectralConv2d:
    """Tests for CliffordSpectralConv2d."""

    def test_init(self, rngs) -> None:
        """Attributes set correctly."""
        from artifex.generative_models.core.layers.clifford.spectral import CliffordSpectralConv2d

        layer = CliffordSpectralConv2d(
            metric=(1, 1),
            in_channels=4,
            out_channels=8,
            modes1=4,
            modes2=4,
            rngs=rngs,
        )
        assert layer.n_blades == 4
        assert layer.modes1 == 4
        assert layer.modes2 == 4

    def test_forward_shape(self, rngs) -> None:
        """Output shape matches input spatial dims."""
        from artifex.generative_models.core.layers.clifford.spectral import CliffordSpectralConv2d

        layer = CliffordSpectralConv2d(
            metric=(1, 1),
            in_channels=4,
            out_channels=8,
            modes1=4,
            modes2=4,
            rngs=rngs,
        )
        x = jnp.ones((2, 16, 16, 4, 4))  # (B, H, W, C, I)
        out = layer(x)
        assert out.shape == (2, 16, 16, 8, 4)

    def test_modes_truncation(self, rngs) -> None:
        """Fewer modes than spatial dims still works."""
        from artifex.generative_models.core.layers.clifford.spectral import CliffordSpectralConv2d

        layer = CliffordSpectralConv2d(
            metric=(1, 1),
            in_channels=4,
            out_channels=4,
            modes1=2,
            modes2=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 16, 16, 4, 4))
        out = layer(x)
        assert out.shape == (2, 16, 16, 4, 4)

    def test_no_multiply(self, rngs) -> None:
        """multiply=False mode truncates without kernel."""
        from artifex.generative_models.core.layers.clifford.spectral import CliffordSpectralConv2d

        layer = CliffordSpectralConv2d(
            metric=(1, 1),
            in_channels=4,
            out_channels=4,
            modes1=4,
            modes2=4,
            multiply=False,
            rngs=rngs,
        )
        x = jnp.ones((2, 16, 16, 4, 4))
        out = layer(x)
        assert out.shape == (2, 16, 16, 4, 4)

    def test_jit_compatible(self, rngs) -> None:
        """CliffordSpectralConv2d works under jit."""
        from artifex.generative_models.core.layers.clifford.spectral import CliffordSpectralConv2d

        layer = CliffordSpectralConv2d(
            metric=(1, 1),
            in_channels=4,
            out_channels=4,
            modes1=4,
            modes2=4,
            rngs=rngs,
        )
        x = jnp.ones((2, 16, 16, 4, 4))

        @nnx.jit
        def forward(m, x):
            return m(x)

        assert forward(layer, x).shape == (2, 16, 16, 4, 4)

    def test_gradient_flow(self, rngs) -> None:
        """Gradients flow through CliffordSpectralConv2d."""
        from artifex.generative_models.core.layers.clifford.spectral import CliffordSpectralConv2d

        layer = CliffordSpectralConv2d(
            metric=(1, 1),
            in_channels=4,
            out_channels=4,
            modes1=4,
            modes2=4,
            rngs=rngs,
        )
        x = jnp.ones((2, 16, 16, 4, 4))

        def loss_fn(m):
            return jnp.sum(m(x))

        grads = nnx.grad(loss_fn)(layer)
        assert grads.weights.value.shape == layer.weights.value.shape


class TestCliffordSpectralConv3d:
    """Tests for CliffordSpectralConv3d."""

    def test_init(self, rngs) -> None:
        """Attributes set correctly."""
        from artifex.generative_models.core.layers.clifford.spectral import CliffordSpectralConv3d

        layer = CliffordSpectralConv3d(
            metric=(1, 1, 1),
            in_channels=2,
            out_channels=4,
            modes1=2,
            modes2=2,
            modes3=2,
            rngs=rngs,
        )
        assert layer.n_blades == 8

    def test_forward_shape(self, rngs) -> None:
        """Output shape matches input spatial dims."""
        from artifex.generative_models.core.layers.clifford.spectral import CliffordSpectralConv3d

        layer = CliffordSpectralConv3d(
            metric=(1, 1, 1),
            in_channels=2,
            out_channels=4,
            modes1=2,
            modes2=2,
            modes3=2,
            rngs=rngs,
        )
        x = jnp.ones((1, 8, 8, 8, 2, 8))  # (B, D, H, W, C, I)
        out = layer(x)
        assert out.shape == (1, 8, 8, 8, 4, 8)

    def test_no_multiply(self, rngs) -> None:
        """multiply=False mode."""
        from artifex.generative_models.core.layers.clifford.spectral import CliffordSpectralConv3d

        layer = CliffordSpectralConv3d(
            metric=(1, 1, 1),
            in_channels=2,
            out_channels=2,
            modes1=2,
            modes2=2,
            modes3=2,
            multiply=False,
            rngs=rngs,
        )
        x = jnp.ones((1, 8, 8, 8, 2, 8))
        out = layer(x)
        assert out.shape == (1, 8, 8, 8, 2, 8)

    def test_jit_compatible(self, rngs) -> None:
        """CliffordSpectralConv3d works under jit."""
        from artifex.generative_models.core.layers.clifford.spectral import CliffordSpectralConv3d

        layer = CliffordSpectralConv3d(
            metric=(1, 1, 1),
            in_channels=2,
            out_channels=2,
            modes1=2,
            modes2=2,
            modes3=2,
            rngs=rngs,
        )
        x = jnp.ones((1, 8, 8, 8, 2, 8))

        @nnx.jit
        def forward(m, x):
            return m(x)

        assert forward(layer, x).shape == (1, 8, 8, 8, 2, 8)

    def test_gradient_flow(self, rngs) -> None:
        """Gradients flow through CliffordSpectralConv3d."""
        from artifex.generative_models.core.layers.clifford.spectral import CliffordSpectralConv3d

        layer = CliffordSpectralConv3d(
            metric=(1, 1, 1),
            in_channels=2,
            out_channels=2,
            modes1=2,
            modes2=2,
            modes3=2,
            rngs=rngs,
        )
        x = jnp.ones((1, 8, 8, 8, 2, 8))

        def loss_fn(m):
            return jnp.sum(m(x))

        grads = nnx.grad(loss_fn)(layer)
        assert grads.weights.value.shape == layer.weights.value.shape

    def test_modes_truncation(self, rngs) -> None:
        """Fewer modes than spatial dims still works."""
        from artifex.generative_models.core.layers.clifford.spectral import CliffordSpectralConv3d

        layer = CliffordSpectralConv3d(
            metric=(1, 1, 1),
            in_channels=2,
            out_channels=2,
            modes1=2,
            modes2=2,
            modes3=2,
            rngs=rngs,
        )
        x = jnp.ones((1, 16, 16, 16, 2, 8))
        out = layer(x)
        assert out.shape == (1, 16, 16, 16, 2, 8)


# ===========================================================================
# TestCliffordBatchNorm
# ===========================================================================


class TestCliffordBatchNorm:
    """Tests for CliffordBatchNorm."""

    def test_init(self, rngs) -> None:
        """Param and stat shapes are correct."""
        from artifex.generative_models.core.layers.clifford.norm import CliffordBatchNorm

        layer = CliffordBatchNorm(metric=(1, 1), channels=8, rngs=rngs)
        assert layer.weight.value.shape == (4, 4, 8)
        assert layer.bias.value.shape == (4, 8)
        assert layer.running_mean.value.shape == (4, 8)
        assert layer.running_cov.value.shape == (4, 4, 8)

    def test_forward_shape(self, rngs) -> None:
        """Output shape matches input."""
        from artifex.generative_models.core.layers.clifford.norm import CliffordBatchNorm

        layer = CliffordBatchNorm(metric=(1, 1), channels=4, rngs=rngs)
        x = jnp.ones((2, 8, 8, 4, 4))  # (B, H, W, C, I)
        out = layer(x, deterministic=False)
        assert out.shape == x.shape

    def test_deterministic_mode(self, rngs) -> None:
        """Deterministic mode uses running stats and produces consistent output."""
        from artifex.generative_models.core.layers.clifford.norm import CliffordBatchNorm

        layer = CliffordBatchNorm(metric=(1,), channels=4, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (4, 8, 4, 2))
        out1 = layer(x, deterministic=True)
        out2 = layer(x, deterministic=True)
        npt.assert_allclose(out1, out2, atol=1e-6)

    def test_training_updates_running_stats(self, rngs) -> None:
        """Training mode updates running stats."""
        from artifex.generative_models.core.layers.clifford.norm import CliffordBatchNorm

        layer = CliffordBatchNorm(metric=(1,), channels=4, rngs=rngs)
        old_mean = layer.running_mean.value.copy()
        x = jax.random.normal(jax.random.key(0), (4, 8, 4, 2))
        layer(x, deterministic=False)
        # Running mean should have changed
        assert not jnp.allclose(layer.running_mean.value, old_mean)

    def test_no_affine(self, rngs) -> None:
        """No affine parameters when use_affine=False."""
        from artifex.generative_models.core.layers.clifford.norm import CliffordBatchNorm

        layer = CliffordBatchNorm(metric=(1,), channels=4, use_affine=False, rngs=rngs)
        assert layer.weight is None
        assert layer.bias is None
        x = jnp.ones((2, 8, 4, 2))
        out = layer(x, deterministic=False)
        assert out.shape == x.shape

    def test_no_running_stats(self, rngs) -> None:
        """No running stats when use_running_stats=False."""
        from artifex.generative_models.core.layers.clifford.norm import CliffordBatchNorm

        layer = CliffordBatchNorm(
            metric=(1,),
            channels=4,
            use_running_stats=False,
            rngs=rngs,
        )
        assert layer.running_mean is None
        assert layer.running_cov is None

    def test_jit_compatible(self, rngs) -> None:
        """CliffordBatchNorm works under jit."""
        from artifex.generative_models.core.layers.clifford.norm import CliffordBatchNorm

        layer = CliffordBatchNorm(metric=(1,), channels=4, rngs=rngs)
        x = jnp.ones((2, 8, 4, 2))

        @nnx.jit
        def forward(m, x):
            return m(x, deterministic=True)

        assert forward(layer, x).shape == (2, 8, 4, 2)

    def test_gradient_flow(self, rngs) -> None:
        """Gradients flow through CliffordBatchNorm."""
        from artifex.generative_models.core.layers.clifford.norm import CliffordBatchNorm

        layer = CliffordBatchNorm(metric=(1, 1), channels=4, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (4, 8, 8, 4, 4))

        def loss_fn(m):
            return jnp.sum(m(x, deterministic=False))

        grads = nnx.grad(loss_fn)(layer)
        assert grads.weight.value.shape == layer.weight.value.shape


# ===========================================================================
# TestCliffordGroupNorm
# ===========================================================================


class TestCliffordGroupNorm:
    """Tests for CliffordGroupNorm."""

    def test_init(self, rngs) -> None:
        """Param shapes are correct."""
        from artifex.generative_models.core.layers.clifford.norm import CliffordGroupNorm

        layer = CliffordGroupNorm(metric=(1, 1), num_groups=2, channels=8, rngs=rngs)
        assert layer.num_groups == 2
        # Weight/bias use channels_per_group
        assert layer.weight.value.shape == (4, 4, 4)  # n_blades, n_blades, ch/groups
        assert layer.bias.value.shape == (4, 4)

    def test_forward_shape(self, rngs) -> None:
        """Output shape matches input."""
        from artifex.generative_models.core.layers.clifford.norm import CliffordGroupNorm

        layer = CliffordGroupNorm(metric=(1, 1), num_groups=2, channels=8, rngs=rngs)
        x = jnp.ones((2, 8, 8, 8, 4))  # (B, H, W, C, I)
        out = layer(x)
        assert out.shape == x.shape

    def test_jit_compatible(self, rngs) -> None:
        """CliffordGroupNorm works under jit."""
        from artifex.generative_models.core.layers.clifford.norm import CliffordGroupNorm

        layer = CliffordGroupNorm(metric=(1,), num_groups=2, channels=4, rngs=rngs)
        x = jnp.ones((2, 8, 4, 2))

        @nnx.jit
        def forward(m, x):
            return m(x)

        assert forward(layer, x).shape == (2, 8, 4, 2)

    def test_gradient_flow(self, rngs) -> None:
        """Gradients flow through CliffordGroupNorm."""
        from artifex.generative_models.core.layers.clifford.norm import CliffordGroupNorm

        layer = CliffordGroupNorm(metric=(1,), num_groups=2, channels=4, rngs=rngs)
        x = jax.random.normal(jax.random.key(0), (4, 8, 4, 2))

        def loss_fn(m):
            return jnp.sum(m(x))

        grads = nnx.grad(loss_fn)(layer)
        assert grads.weight.value.shape == layer.weight.value.shape

    def test_indivisible_channels_raises(self, rngs) -> None:
        """Channels not divisible by groups raises ValueError."""
        from artifex.generative_models.core.layers.clifford.norm import CliffordGroupNorm

        with pytest.raises(ValueError, match="divisible"):
            CliffordGroupNorm(metric=(1,), num_groups=3, channels=8, rngs=rngs)

    def test_no_affine(self, rngs) -> None:
        """No affine parameters when use_affine=False."""
        from artifex.generative_models.core.layers.clifford.norm import CliffordGroupNorm

        layer = CliffordGroupNorm(
            metric=(1,),
            num_groups=2,
            channels=4,
            use_affine=False,
            rngs=rngs,
        )
        assert layer.weight is None
        assert layer.bias is None


# ===========================================================================
# TestMultiVectorActivation
# ===========================================================================


class TestMultiVectorActivation:
    """Tests for MultiVectorActivation."""

    def test_init(self, rngs) -> None:
        """Attributes set correctly."""
        from artifex.generative_models.core.layers.clifford.norm import MultiVectorActivation

        layer = MultiVectorActivation(
            channels=8,
            n_blades=4,
            input_blades=(0, 1, 2, 3),
            rngs=rngs,
        )
        assert layer.n_blades == 4
        assert layer.channels == 8

    def test_forward_shape(self, rngs) -> None:
        """Output shape matches input."""
        from artifex.generative_models.core.layers.clifford.norm import MultiVectorActivation

        layer = MultiVectorActivation(
            channels=4,
            n_blades=4,
            input_blades=(0, 1, 2, 3),
            rngs=rngs,
        )
        x = jnp.ones((2, 8, 8, 4, 4))  # (B, H, W, C, I)
        out = layer(x)
        assert out.shape == x.shape

    def test_sum_aggregation(self, rngs) -> None:
        """Sum aggregation mode works."""
        from artifex.generative_models.core.layers.clifford.norm import MultiVectorActivation

        layer = MultiVectorActivation(
            channels=4,
            n_blades=4,
            input_blades=(0, 1, 2, 3),
            aggregation="sum",
            rngs=rngs,
        )
        x = jnp.ones((2, 8, 8, 4, 4))
        out = layer(x)
        assert out.shape == x.shape

    def test_mean_aggregation(self, rngs) -> None:
        """Mean aggregation mode works."""
        from artifex.generative_models.core.layers.clifford.norm import MultiVectorActivation

        layer = MultiVectorActivation(
            channels=4,
            n_blades=4,
            input_blades=(0, 1, 2, 3),
            aggregation="mean",
            rngs=rngs,
        )
        x = jnp.ones((2, 8, 8, 4, 4))
        out = layer(x)
        assert out.shape == x.shape

    def test_linear_aggregation(self, rngs) -> None:
        """Linear aggregation mode works."""
        from artifex.generative_models.core.layers.clifford.norm import MultiVectorActivation

        layer = MultiVectorActivation(
            channels=4,
            n_blades=4,
            input_blades=(0, 1, 2, 3),
            aggregation="linear",
            rngs=rngs,
        )
        x = jnp.ones((2, 8, 8, 4, 4))
        out = layer(x)
        assert out.shape == x.shape

    def test_jit_compatible(self, rngs) -> None:
        """MultiVectorActivation works under jit."""
        from artifex.generative_models.core.layers.clifford.norm import MultiVectorActivation

        layer = MultiVectorActivation(
            channels=4,
            n_blades=4,
            input_blades=(0, 1, 2, 3),
            rngs=rngs,
        )
        x = jnp.ones((2, 8, 8, 4, 4))

        @nnx.jit
        def forward(m, x):
            return m(x)

        assert forward(layer, x).shape == x.shape
