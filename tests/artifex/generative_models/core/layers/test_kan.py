"""Tests for Kolmogorov-Arnold Network (KAN) layers.

Covers: KANConfig, BSplineBasis, all 7 layer variants, ConvKANLayer,
create_kan_layer factory, grid update, JIT, gradient flow.
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from flax import nnx

from artifex.generative_models.core.layers.kan import (
    BSplineBasis,
    ChebyshevKANLayer,
    ConvKANLayer,
    create_kan_layer,
    DenseKANGrid,
    DenseKANLayer,
    EfficientKANGrid,
    EfficientKANLayer,
    FourierKANLayer,
    KANConfig,
    LegendreKANLayer,
    RBFKANGrid,
    RBFKANLayer,
    SineKANLayer,
)


N_IN = 4
N_OUT = 3
BATCH = 8


@pytest.fixture
def rngs():
    """Fixture providing NNX Rngs for tests."""
    return nnx.Rngs(42)


@pytest.fixture
def sample_x():
    """Fixture providing a standard sample input."""
    return jax.random.normal(jax.random.key(0), (BATCH, N_IN))


# ---------------------------------------------------------------------------
# KANConfig tests
# ---------------------------------------------------------------------------
class TestKANConfig:
    """Tests for KANConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        cfg = KANConfig()
        assert cfg.k == 3
        assert cfg.grid_intervals == 3
        assert cfg.grid_range == (-1.0, 1.0)
        assert cfg.degree == 5
        assert cfg.residual is True
        assert cfg.init_scheme == "default"

    def test_frozen(self):
        """Test config is immutable."""
        cfg = KANConfig()
        with pytest.raises(AttributeError):
            cfg.k = 5  # type: ignore[misc]

    def test_validation_k(self):
        """Test k must be >= 0."""
        with pytest.raises(ValueError, match="k must be >= 0"):
            KANConfig(k=-1)

    def test_validation_grid_intervals(self):
        """Test grid_intervals must be >= 1."""
        with pytest.raises(ValueError, match="grid_intervals must be >= 1"):
            KANConfig(grid_intervals=0)

    def test_validation_grid_range(self):
        """Test grid_range[0] < grid_range[1]."""
        with pytest.raises(ValueError, match="grid_range"):
            KANConfig(grid_range=(1.0, -1.0))

    def test_validation_grid_e(self):
        """Test grid_e in [0, 1]."""
        with pytest.raises(ValueError, match="grid_e"):
            KANConfig(grid_e=1.5)

    def test_validation_degree(self):
        """Test degree must be >= 1."""
        with pytest.raises(ValueError, match="degree must be >= 1"):
            KANConfig(degree=0)

    def test_validation_init_scheme(self):
        """Test unknown init_scheme raises."""
        with pytest.raises(ValueError, match="Unknown init_scheme"):
            KANConfig(init_scheme="bogus")


# ---------------------------------------------------------------------------
# BSplineBasis tests
# ---------------------------------------------------------------------------
class TestBSplineBasis:
    """Tests for the BSplineBasis evaluator."""

    def test_output_shape(self):
        """Test basis output shape."""
        bspline = BSplineBasis(k=3)
        grid = EfficientKANGrid(n_nodes=N_IN, k=3, grid_intervals=5)
        x = jnp.ones((BATCH, N_IN))
        result = bspline(x, grid.knots.value)
        assert result.shape == (BATCH, N_IN, 5 + 3)  # G + k

    def test_jit(self):
        """Test JIT compilation."""
        bspline = BSplineBasis(k=3)
        grid = EfficientKANGrid(n_nodes=N_IN, k=3, grid_intervals=5)
        x = jnp.ones((BATCH, N_IN))

        @jax.jit
        def evaluate(x_):
            return bspline(x_, grid.knots.value)

        result = evaluate(x)
        assert result.shape == (BATCH, N_IN, 8)

    def test_partition_of_unity(self):
        """B-splines of order k should sum to ~1 in the interior."""
        k = 3
        grid_intervals = 10
        bspline = BSplineBasis(k=k)
        grid = EfficientKANGrid(n_nodes=1, k=k, grid_intervals=grid_intervals)
        # Points in the interior of the grid
        x = jnp.linspace(-0.8, 0.8, 50).reshape(-1, 1)
        basis = bspline(x, grid.knots.value)  # (50, 1, G+k)
        sums = basis.sum(axis=-1)  # (50, 1)
        npt.assert_allclose(sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Grid tests
# ---------------------------------------------------------------------------
class TestDenseKANGrid:
    """Tests for DenseKANGrid."""

    def test_knot_shape(self):
        """Test initial knot shape."""
        grid = DenseKANGrid(n_in=N_IN, n_out=N_OUT, k=3, grid_intervals=5)
        assert grid.knots.value.shape == (N_IN * N_OUT, 5 + 2 * 3 + 1)

    def test_update(self):
        """Test grid update changes knot shape."""
        grid = DenseKANGrid(n_in=N_IN, n_out=N_OUT, k=3, grid_intervals=5)
        x = jax.random.normal(jax.random.key(0), (BATCH, N_IN))
        grid.update(x, new_intervals=8)
        assert grid.knots.value.shape == (N_IN * N_OUT, 8 + 2 * 3 + 1)
        assert grid.grid_intervals == 8


class TestEfficientKANGrid:
    """Tests for EfficientKANGrid."""

    def test_knot_shape(self):
        """Test initial knot shape."""
        grid = EfficientKANGrid(n_nodes=N_IN, k=3, grid_intervals=5)
        assert grid.knots.value.shape == (N_IN, 5 + 2 * 3 + 1)

    def test_update(self):
        """Test grid update changes knot shape."""
        grid = EfficientKANGrid(n_nodes=N_IN, k=3, grid_intervals=5)
        x = jax.random.normal(jax.random.key(0), (BATCH, N_IN))
        grid.update(x, new_intervals=8)
        assert grid.knots.value.shape == (N_IN, 8 + 2 * 3 + 1)


class TestRBFKANGrid:
    """Tests for RBFKANGrid."""

    def test_knot_shape(self):
        """Test initial center shape."""
        grid = RBFKANGrid(n_nodes=N_IN, num_centers=7)
        assert grid.knots.value.shape == (N_IN, 7)

    def test_update(self):
        """Test grid update changes center shape."""
        grid = RBFKANGrid(n_nodes=N_IN, num_centers=7)
        x = jax.random.normal(jax.random.key(0), (BATCH, N_IN))
        grid.update(x, new_num_centers=10)
        assert grid.knots.value.shape == (N_IN, 10)


# ---------------------------------------------------------------------------
# Shared helpers for layer tests
# ---------------------------------------------------------------------------
def _check_forward_shape(layer, x):
    """Verify forward output shape is (batch, n_out)."""
    y = layer(x, deterministic=True)
    assert y.shape == (BATCH, N_OUT)
    assert jnp.isfinite(y).all()


def _check_jit(layer, x):
    """Verify JIT compiles without error."""

    @jax.jit
    def forward(x_):
        return layer(x_, deterministic=True)

    y = forward(x)
    assert y.shape == (BATCH, N_OUT)


def _check_gradient_flow(layer, x):
    """Verify gradients propagate to the input."""

    def loss_fn(x_):
        return jnp.sum(layer(x_, deterministic=True) ** 2)

    grads = jax.grad(loss_fn)(x)
    assert grads.shape == x.shape
    assert jnp.isfinite(grads).all()


# ---------------------------------------------------------------------------
# DenseKANLayer tests
# ---------------------------------------------------------------------------
class TestDenseKANLayer:
    """Tests for the original dense KAN layer."""

    def test_init(self, rngs):
        """Test initialisation."""
        layer = DenseKANLayer(n_in=N_IN, n_out=N_OUT, rngs=rngs)
        assert layer.n_in == N_IN
        assert layer.n_out == N_OUT
        assert layer.c_spl is not None
        assert layer.c_res is not None
        assert layer.bias is not None

    def test_forward_shape(self, rngs, sample_x):
        """Test output shape."""
        layer = DenseKANLayer(n_in=N_IN, n_out=N_OUT, rngs=rngs)
        _check_forward_shape(layer, sample_x)

    def test_no_residual(self, rngs, sample_x):
        """Test without residual activation."""
        layer = DenseKANLayer(n_in=N_IN, n_out=N_OUT, residual=None, rngs=rngs)
        assert layer.c_res is None
        _check_forward_shape(layer, sample_x)

    def test_no_external_weights(self, rngs, sample_x):
        """Test without external edge weights."""
        layer = DenseKANLayer(n_in=N_IN, n_out=N_OUT, external_weights=False, rngs=rngs)
        assert layer.c_spl is None
        _check_forward_shape(layer, sample_x)

    def test_no_bias(self, rngs, sample_x):
        """Test without bias."""
        layer = DenseKANLayer(n_in=N_IN, n_out=N_OUT, add_bias=False, rngs=rngs)
        assert layer.bias is None
        _check_forward_shape(layer, sample_x)

    def test_basis_shape(self, rngs, sample_x):
        """Test basis output shape."""
        layer = DenseKANLayer(n_in=N_IN, n_out=N_OUT, grid_intervals=5, k=3, rngs=rngs)
        bi = layer.basis(sample_x)
        assert bi.shape == (N_IN * N_OUT, 5 + 3, BATCH)

    def test_grid_update(self, rngs, sample_x):
        """Test grid update with new intervals."""
        layer = DenseKANLayer(n_in=N_IN, n_out=N_OUT, grid_intervals=3, rngs=rngs)
        layer.update_grid(sample_x, new_intervals=6)
        y = layer(sample_x, deterministic=True)
        assert y.shape == (BATCH, N_OUT)

    def test_jit(self, rngs, sample_x):
        """Test JIT compilation."""
        layer = DenseKANLayer(n_in=N_IN, n_out=N_OUT, rngs=rngs)
        _check_jit(layer, sample_x)

    def test_gradient_flow(self, rngs, sample_x):
        """Test gradients propagate."""
        layer = DenseKANLayer(n_in=N_IN, n_out=N_OUT, rngs=rngs)
        _check_gradient_flow(layer, sample_x)


# ---------------------------------------------------------------------------
# EfficientKANLayer tests
# ---------------------------------------------------------------------------
class TestEfficientKANLayer:
    """Tests for the efficient matrix-based KAN layer."""

    def test_init(self, rngs):
        """Test initialisation."""
        layer = EfficientKANLayer(n_in=N_IN, n_out=N_OUT, rngs=rngs)
        assert layer.n_in == N_IN
        assert layer.n_out == N_OUT

    def test_forward_shape(self, rngs, sample_x):
        """Test output shape."""
        layer = EfficientKANLayer(n_in=N_IN, n_out=N_OUT, rngs=rngs)
        _check_forward_shape(layer, sample_x)

    def test_no_residual(self, rngs, sample_x):
        """Test without residual."""
        layer = EfficientKANLayer(n_in=N_IN, n_out=N_OUT, residual=None, rngs=rngs)
        assert layer.c_res is None
        _check_forward_shape(layer, sample_x)

    def test_no_external_weights(self, rngs, sample_x):
        """Test without external weights."""
        layer = EfficientKANLayer(n_in=N_IN, n_out=N_OUT, external_weights=False, rngs=rngs)
        assert layer.c_spl is None
        _check_forward_shape(layer, sample_x)

    def test_basis_shape(self, rngs, sample_x):
        """Test basis output shape."""
        layer = EfficientKANLayer(n_in=N_IN, n_out=N_OUT, grid_intervals=5, k=3, rngs=rngs)
        bi = layer.basis(sample_x)
        assert bi.shape == (BATCH, N_IN, 5 + 3)

    def test_grid_update(self, rngs, sample_x):
        """Test grid update."""
        layer = EfficientKANLayer(n_in=N_IN, n_out=N_OUT, grid_intervals=3, rngs=rngs)
        layer.update_grid(sample_x, new_intervals=6)
        y = layer(sample_x, deterministic=True)
        assert y.shape == (BATCH, N_OUT)

    def test_jit(self, rngs, sample_x):
        """Test JIT compilation."""
        layer = EfficientKANLayer(n_in=N_IN, n_out=N_OUT, rngs=rngs)
        _check_jit(layer, sample_x)

    def test_gradient_flow(self, rngs, sample_x):
        """Test gradients propagate."""
        layer = EfficientKANLayer(n_in=N_IN, n_out=N_OUT, rngs=rngs)
        _check_gradient_flow(layer, sample_x)


# ---------------------------------------------------------------------------
# ChebyshevKANLayer tests
# ---------------------------------------------------------------------------
class TestChebyshevKANLayer:
    """Tests for the Chebyshev polynomial KAN layer."""

    @pytest.mark.parametrize("flavor", ["default", "modified", "exact"])
    def test_forward_shape(self, rngs, sample_x, flavor):
        """Test output shape for each flavor."""
        layer = ChebyshevKANLayer(n_in=N_IN, n_out=N_OUT, D=5, flavor=flavor, rngs=rngs)
        _check_forward_shape(layer, sample_x)

    def test_init(self, rngs):
        """Test initialisation."""
        layer = ChebyshevKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        assert layer.D == 5
        assert layer.flavor == "default"
        assert layer.c_res is None  # no residual by default

    def test_with_residual(self, rngs, sample_x):
        """Test with residual activation."""
        layer = ChebyshevKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=5,
            residual=nnx.silu,
            rngs=rngs,
        )
        assert layer.c_res is not None
        _check_forward_shape(layer, sample_x)

    def test_exact_max_degree(self, rngs):
        """Test exact flavor enforces max degree."""
        with pytest.raises(ValueError, match="max degree"):
            ChebyshevKANLayer(n_in=N_IN, n_out=N_OUT, D=25, flavor="exact", rngs=rngs)

    def test_basis_shape(self, rngs, sample_x):
        """Test basis output shape."""
        layer = ChebyshevKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        bi = layer.basis(sample_x)
        assert bi.shape == (BATCH, N_IN, 5)  # D, constant excluded

    def test_grid_update(self, rngs, sample_x):
        """Test degree refinement."""
        layer = ChebyshevKANLayer(n_in=N_IN, n_out=N_OUT, D=3, rngs=rngs)
        layer.update_grid(sample_x, d_new=6)
        assert layer.D == 6
        y = layer(sample_x, deterministic=True)
        assert y.shape == (BATCH, N_OUT)

    def test_jit(self, rngs, sample_x):
        """Test JIT compilation."""
        layer = ChebyshevKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        _check_jit(layer, sample_x)

    def test_gradient_flow(self, rngs, sample_x):
        """Test gradients propagate."""
        layer = ChebyshevKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        _check_gradient_flow(layer, sample_x)


# ---------------------------------------------------------------------------
# FourierKANLayer tests
# ---------------------------------------------------------------------------
class TestFourierKANLayer:
    """Tests for the Fourier KAN layer."""

    def test_init(self, rngs):
        """Test initialisation."""
        layer = FourierKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        assert layer.D == 5
        assert layer.c_cos.value.shape == (N_OUT, N_IN, 5)
        assert layer.c_sin.value.shape == (N_OUT, N_IN, 5)

    def test_forward_shape(self, rngs, sample_x):
        """Test output shape."""
        layer = FourierKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        _check_forward_shape(layer, sample_x)

    def test_basis_shape(self, rngs, sample_x):
        """Test basis returns cos/sin tuple."""
        layer = FourierKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        cosines, sines = layer.basis(sample_x)
        assert cosines.shape == (BATCH, N_IN, 5)
        assert sines.shape == (BATCH, N_IN, 5)

    def test_no_bias(self, rngs, sample_x):
        """Test without bias."""
        layer = FourierKANLayer(n_in=N_IN, n_out=N_OUT, D=5, add_bias=False, rngs=rngs)
        assert layer.bias is None
        _check_forward_shape(layer, sample_x)

    def test_grid_update(self, rngs, sample_x):
        """Test Fourier order increase."""
        layer = FourierKANLayer(n_in=N_IN, n_out=N_OUT, D=3, rngs=rngs)
        layer.update_grid(sample_x, d_new=6)
        assert layer.D == 6
        y = layer(sample_x, deterministic=True)
        assert y.shape == (BATCH, N_OUT)

    def test_jit(self, rngs, sample_x):
        """Test JIT compilation."""
        layer = FourierKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        _check_jit(layer, sample_x)

    def test_gradient_flow(self, rngs, sample_x):
        """Test gradients propagate."""
        layer = FourierKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        _check_gradient_flow(layer, sample_x)


# ---------------------------------------------------------------------------
# LegendreKANLayer tests
# ---------------------------------------------------------------------------
class TestLegendreKANLayer:
    """Tests for the Legendre polynomial KAN layer."""

    @pytest.mark.parametrize("flavor", ["default", "exact"])
    def test_forward_shape(self, rngs, sample_x, flavor):
        """Test output shape for each flavor."""
        layer = LegendreKANLayer(n_in=N_IN, n_out=N_OUT, D=5, flavor=flavor, rngs=rngs)
        _check_forward_shape(layer, sample_x)

    def test_init(self, rngs):
        """Test initialisation."""
        layer = LegendreKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        assert layer.D == 5
        assert layer.flavor == "default"

    def test_exact_max_degree(self, rngs):
        """Test exact flavor enforces max degree."""
        with pytest.raises(ValueError, match="max degree"):
            LegendreKANLayer(n_in=N_IN, n_out=N_OUT, D=15, flavor="exact", rngs=rngs)

    def test_with_residual(self, rngs, sample_x):
        """Test with residual activation."""
        layer = LegendreKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=5,
            residual=nnx.silu,
            rngs=rngs,
        )
        assert layer.c_res is not None
        _check_forward_shape(layer, sample_x)

    def test_grid_update(self, rngs, sample_x):
        """Test degree refinement."""
        layer = LegendreKANLayer(n_in=N_IN, n_out=N_OUT, D=3, rngs=rngs)
        layer.update_grid(sample_x, d_new=6)
        assert layer.D == 6

    def test_jit(self, rngs, sample_x):
        """Test JIT compilation."""
        layer = LegendreKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        _check_jit(layer, sample_x)

    def test_gradient_flow(self, rngs, sample_x):
        """Test gradients propagate."""
        layer = LegendreKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        _check_gradient_flow(layer, sample_x)


# ---------------------------------------------------------------------------
# RBFKANLayer tests
# ---------------------------------------------------------------------------
class TestRBFKANLayer:
    """Tests for the RBF KAN layer."""

    def test_init(self, rngs):
        """Test initialisation."""
        layer = RBFKANLayer(n_in=N_IN, n_out=N_OUT, D=7, rngs=rngs)
        assert layer.D == 7
        assert layer.grid.knots.value.shape == (N_IN, 7)

    def test_forward_shape(self, rngs, sample_x):
        """Test output shape."""
        layer = RBFKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        _check_forward_shape(layer, sample_x)

    def test_basis_shape(self, rngs, sample_x):
        """Test basis output shape."""
        layer = RBFKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        bi = layer.basis(sample_x)
        assert bi.shape == (BATCH, N_IN, 5)

    def test_unknown_kernel(self, rngs, sample_x):
        """Test unknown kernel raises."""
        layer = RBFKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=5,
            kernel={"type": "unknown"},
            rngs=rngs,
        )
        with pytest.raises(ValueError, match="Unknown kernel"):
            layer(sample_x, deterministic=True)

    def test_grid_update(self, rngs, sample_x):
        """Test grid update with new centers."""
        layer = RBFKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        layer.update_grid(sample_x, d_new=8)
        assert layer.D == 8
        y = layer(sample_x, deterministic=True)
        assert y.shape == (BATCH, N_OUT)

    def test_jit(self, rngs, sample_x):
        """Test JIT compilation."""
        layer = RBFKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        _check_jit(layer, sample_x)

    def test_gradient_flow(self, rngs, sample_x):
        """Test gradients propagate."""
        layer = RBFKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        _check_gradient_flow(layer, sample_x)


# ---------------------------------------------------------------------------
# SineKANLayer tests
# ---------------------------------------------------------------------------
class TestSineKANLayer:
    """Tests for the Sine-based KAN layer."""

    def test_init(self, rngs):
        """Test initialisation."""
        layer = SineKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        assert layer.D == 5
        assert layer.omega.value.shape == (5, 1)
        assert layer.phase.value.shape == (5, 1)

    def test_forward_shape(self, rngs, sample_x):
        """Test output shape."""
        layer = SineKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        _check_forward_shape(layer, sample_x)

    def test_basis_shape(self, rngs, sample_x):
        """Test basis output shape."""
        layer = SineKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        bi = layer.basis(sample_x)
        assert bi.shape == (BATCH, N_IN, 5)

    def test_with_residual(self, rngs, sample_x):
        """Test with residual activation."""
        layer = SineKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=5,
            residual=nnx.silu,
            rngs=rngs,
        )
        assert layer.c_res is not None
        _check_forward_shape(layer, sample_x)

    def test_grid_update(self, rngs, sample_x):
        """Test D expansion."""
        layer = SineKANLayer(n_in=N_IN, n_out=N_OUT, D=3, rngs=rngs)
        layer.update_grid(sample_x, d_new=6)
        assert layer.D == 6
        y = layer(sample_x, deterministic=True)
        assert y.shape == (BATCH, N_OUT)

    def test_jit(self, rngs, sample_x):
        """Test JIT compilation."""
        layer = SineKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        _check_jit(layer, sample_x)

    def test_gradient_flow(self, rngs, sample_x):
        """Test gradients propagate."""
        layer = SineKANLayer(n_in=N_IN, n_out=N_OUT, D=5, rngs=rngs)
        _check_gradient_flow(layer, sample_x)


# ---------------------------------------------------------------------------
# ConvKANLayer tests
# ---------------------------------------------------------------------------
class TestConvKANLayer:
    """Tests for the convolutional KAN layer."""

    def test_init(self, rngs):
        """Test initialisation."""
        layer = ConvKANLayer(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            spatial_ndim=2,
            rngs=rngs,
        )
        assert layer.in_channels == 3
        assert layer.out_channels == 8
        assert layer.spatial_ndim == 2

    def test_invalid_spatial_ndim(self, rngs):
        """Test invalid spatial_ndim raises."""
        with pytest.raises(ValueError, match="spatial_ndim must be 1, 2, or 3"):
            ConvKANLayer(
                in_channels=3,
                out_channels=8,
                spatial_ndim=4,
                rngs=rngs,
            )

    def test_forward_2d(self, rngs):
        """Test 2D conv forward pass."""
        layer = ConvKANLayer(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            spatial_ndim=2,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((2, 8, 8, 2))
        y = layer(x, deterministic=True)
        assert y.shape == (2, 8, 8, 4)

    def test_forward_1d(self, rngs):
        """Test 1D conv forward pass."""
        layer = ConvKANLayer(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            spatial_ndim=1,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((2, 16, 3))
        y = layer(x, deterministic=True)
        assert y.shape == (2, 16, 4)

    def test_forward_3d(self, rngs):
        """Test 3D conv forward pass."""
        layer = ConvKANLayer(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            spatial_ndim=3,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((1, 4, 4, 4, 1))
        y = layer(x, deterministic=True)
        assert y.shape == (1, 4, 4, 4, 2)

    def test_valid_padding(self, rngs):
        """Test VALID padding reduces spatial dims."""
        layer = ConvKANLayer(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            spatial_ndim=2,
            padding="VALID",
            rngs=rngs,
        )
        x = jnp.ones((2, 8, 8, 2))
        y = layer(x, deterministic=True)
        assert y.shape == (2, 6, 6, 4)

    def test_stride(self, rngs):
        """Test stride reduces spatial dims."""
        layer = ConvKANLayer(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            stride=2,
            spatial_ndim=2,
            padding="SAME",
            rngs=rngs,
        )
        x = jnp.ones((2, 8, 8, 2))
        y = layer(x, deterministic=True)
        assert y.shape == (2, 4, 4, 4)

    def test_jit(self, rngs):
        """Test JIT compilation."""
        layer = ConvKANLayer(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            spatial_ndim=2,
            rngs=rngs,
        )
        x = jnp.ones((1, 4, 4, 2))

        @jax.jit
        def forward(x_):
            return layer(x_, deterministic=True)

        y = forward(x)
        assert y.shape == (1, 4, 4, 4)


# ---------------------------------------------------------------------------
# create_kan_layer factory tests
# ---------------------------------------------------------------------------
class TestCreateKANLayer:
    """Tests for the create_kan_layer factory."""

    @pytest.mark.parametrize(
        "layer_type,cls",
        [
            ("dense", DenseKANLayer),
            ("efficient", EfficientKANLayer),
            ("chebyshev", ChebyshevKANLayer),
            ("fourier", FourierKANLayer),
            ("legendre", LegendreKANLayer),
            ("rbf", RBFKANLayer),
            ("sine", SineKANLayer),
        ],
    )
    def test_creates_correct_type(self, rngs, layer_type, cls):
        """Test factory creates the right layer type."""
        layer = create_kan_layer(layer_type, n_in=N_IN, n_out=N_OUT, rngs=rngs)
        assert isinstance(layer, cls)

    def test_creates_conv(self, rngs):
        """Test factory creates conv layer."""
        layer = create_kan_layer(
            "conv",
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            spatial_ndim=2,
            rngs=rngs,
        )
        assert isinstance(layer, ConvKANLayer)

    def test_unknown_type_raises(self):
        """Test unknown layer type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown KAN layer type"):
            create_kan_layer("unknown")

    def test_all_types_registered(self):
        """Test all 8 layer types are registered."""
        from artifex.generative_models.core.layers.kan import _KAN_LAYER_REGISTRY

        assert len(_KAN_LAYER_REGISTRY) == 8
        expected = {
            "dense",
            "efficient",
            "chebyshev",
            "fourier",
            "legendre",
            "rbf",
            "sine",
            "conv",
        }
        assert set(_KAN_LAYER_REGISTRY.keys()) == expected


# ---------------------------------------------------------------------------
# Init scheme tests
# ---------------------------------------------------------------------------
class TestInitSchemes:
    """Test different init schemes work across layer types."""

    @pytest.mark.parametrize("scheme", ["default", "power", "lecun"])
    def test_efficient_schemes(self, rngs, sample_x, scheme):
        """Test init schemes on EfficientKANLayer."""
        layer = EfficientKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            init_scheme={"type": scheme},
            rngs=rngs,
        )
        y = layer(sample_x, deterministic=True)
        assert y.shape == (BATCH, N_OUT)
        assert jnp.isfinite(y).all()

    @pytest.mark.parametrize("scheme", ["default", "power"])
    def test_chebyshev_schemes(self, rngs, sample_x, scheme):
        """Test init schemes on ChebyshevKANLayer."""
        layer = ChebyshevKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=5,
            init_scheme={"type": scheme},
            rngs=rngs,
        )
        y = layer(sample_x, deterministic=True)
        assert y.shape == (BATCH, N_OUT)
        assert jnp.isfinite(y).all()
