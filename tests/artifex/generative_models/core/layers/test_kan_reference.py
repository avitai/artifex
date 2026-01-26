"""Numerical validation tests comparing Artifex KAN layers against jaxKAN reference.

Each test instantiates both the Artifex layer and the jaxKAN reference layer with
matching parameters, copies identical weights between them, runs the same input
through both, and asserts the outputs are numerically close (atol=1e-5).

The jaxKAN repository (reference implementation) lives at:
    /media/mahdi/ssd23/Works/jaxKAN/

Tested pairs:
    - DenseKANLayer   vs jaxKAN BaseLayer
    - EfficientKANLayer vs jaxKAN SplineLayer
    - ChebyshevKANLayer vs jaxKAN ChebyshevLayer
    - FourierKANLayer vs jaxKAN FourierLayer
    - LegendreKANLayer vs jaxKAN LegendreLayer
    - RBFKANLayer     vs jaxKAN RBFLayer
    - SineKANLayer    vs jaxKAN SineLayer
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


# Add jaxKAN to sys.path so we can import the reference implementations
_JAXKAN_ROOT = Path("/media/mahdi/ssd23/Works/jaxKAN")
if str(_JAXKAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_JAXKAN_ROOT))

from jaxkan.layers.Chebyshev import ChebyshevLayer as RefChebyshevLayer
from jaxkan.layers.Fourier import FourierLayer as RefFourierLayer
from jaxkan.layers.Legendre import LegendreLayer as RefLegendreLayer
from jaxkan.layers.RBF import RBFLayer as RefRBFLayer
from jaxkan.layers.Sine import SineLayer as RefSineLayer
from jaxkan.layers.Spline import BaseLayer as RefBaseLayer, SplineLayer as RefSplineLayer

from artifex.generative_models.core.layers.kan import (
    ChebyshevKANLayer,
    DenseKANLayer,
    EfficientKANLayer,
    FourierKANLayer,
    LegendreKANLayer,
    RBFKANLayer,
    SineKANLayer,
)


# Common test parameters
N_IN = 3
N_OUT = 4
BATCH = 16
SEED = 42
ATOL = 1e-5


def _make_input(seed: int = SEED) -> jax.Array:
    """Generate a deterministic input batch in [-1, 1]."""
    key = jax.random.key(seed)
    return jax.random.uniform(key, shape=(BATCH, N_IN), minval=-0.9, maxval=0.9)


class TestDenseKANLayerVsRefBaseLayer:
    """DenseKANLayer (Artifex) vs BaseLayer (jaxKAN) numerical comparison."""

    def test_basis_evaluation_matches(self) -> None:
        """Basis function evaluation produces identical outputs for matching grids."""
        k, g = 3, 5

        ref = RefBaseLayer(
            n_in=N_IN,
            n_out=N_OUT,
            k=k,
            G=g,
            grid_range=(-1, 1),
            grid_e=0.05,
            residual=None,
            external_weights=False,
            add_bias=False,
            seed=SEED,
        )

        art = DenseKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            k=k,
            grid_intervals=g,
            grid_range=(-1.0, 1.0),
            grid_e=0.05,
            residual=None,
            external_weights=False,
            add_bias=False,
            rngs=nnx.Rngs(SEED),
        )

        # Copy grid from reference to artifex
        art.grid.knots = nnx.Variable(ref.grid.item)

        x = _make_input()
        ref_basis = ref.basis(x)
        art_basis = art.basis(x)

        assert ref_basis.shape == art_basis.shape, (
            f"Shape mismatch: ref={ref_basis.shape}, art={art_basis.shape}"
        )
        assert jnp.allclose(ref_basis, art_basis, atol=ATOL), (
            f"Basis max diff: {jnp.abs(ref_basis - art_basis).max()}"
        )

    def test_forward_pass_matches(self) -> None:
        """Forward pass produces identical outputs with matching weights."""
        k, g = 3, 5

        ref = RefBaseLayer(
            n_in=N_IN,
            n_out=N_OUT,
            k=k,
            G=g,
            grid_range=(-1, 1),
            grid_e=0.05,
            residual=nnx.silu,
            external_weights=True,
            add_bias=True,
            seed=SEED,
        )

        art = DenseKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            k=k,
            grid_intervals=g,
            grid_range=(-1.0, 1.0),
            grid_e=0.05,
            residual=nnx.silu,
            external_weights=True,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        # Copy all weights from reference to artifex
        art.grid.knots = nnx.Variable(ref.grid.item)
        art.c_basis = nnx.Param(ref.c_basis.value)
        art.c_res = nnx.Param(ref.c_res.value)
        art.c_spl = nnx.Param(ref.c_spl.value)
        art.bias = nnx.Param(ref.bias.value)

        x = _make_input()
        ref_out = ref(x)
        art_out = art(x)

        assert ref_out.shape == art_out.shape, (
            f"Shape mismatch: ref={ref_out.shape}, art={art_out.shape}"
        )
        assert jnp.allclose(ref_out, art_out, atol=ATOL), (
            f"Forward max diff: {jnp.abs(ref_out - art_out).max()}"
        )

    def test_forward_no_residual_no_bias(self) -> None:
        """Forward pass matches without residual and bias."""
        k, g = 3, 3

        ref = RefBaseLayer(
            n_in=N_IN,
            n_out=N_OUT,
            k=k,
            G=g,
            grid_range=(-1, 1),
            grid_e=0.05,
            residual=None,
            external_weights=False,
            add_bias=False,
            seed=SEED,
        )

        art = DenseKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            k=k,
            grid_intervals=g,
            grid_range=(-1.0, 1.0),
            grid_e=0.05,
            residual=None,
            external_weights=False,
            add_bias=False,
            rngs=nnx.Rngs(SEED),
        )

        art.grid.knots = nnx.Variable(ref.grid.item)
        art.c_basis = nnx.Param(ref.c_basis.value)

        x = _make_input()
        ref_out = ref(x)
        art_out = art(x)

        assert jnp.allclose(ref_out, art_out, atol=ATOL), (
            f"Forward max diff: {jnp.abs(ref_out - art_out).max()}"
        )


class TestEfficientKANLayerVsRefSplineLayer:
    """EfficientKANLayer (Artifex) vs SplineLayer (jaxKAN) numerical comparison."""

    def test_basis_evaluation_matches(self) -> None:
        """Basis function evaluation produces identical outputs."""
        k, g = 3, 5

        ref = RefSplineLayer(
            n_in=N_IN,
            n_out=N_OUT,
            k=k,
            G=g,
            grid_range=(-1, 1),
            grid_e=0.05,
            residual=None,
            external_weights=False,
            add_bias=False,
            seed=SEED,
        )

        art = EfficientKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            k=k,
            grid_intervals=g,
            grid_range=(-1.0, 1.0),
            grid_e=0.05,
            residual=None,
            external_weights=False,
            add_bias=False,
            rngs=nnx.Rngs(SEED),
        )

        # Copy grid: jaxKAN SplineGrid stores as .item (n_in, G+2k+1)
        # Artifex EfficientKANGrid stores as .knots.value (n_in, G+2k+1)
        art.grid.knots = nnx.Variable(ref.grid.item)

        x = _make_input()
        ref_basis = ref.basis(x)
        art_basis = art.basis(x)

        assert ref_basis.shape == art_basis.shape, (
            f"Shape mismatch: ref={ref_basis.shape}, art={art_basis.shape}"
        )
        assert jnp.allclose(ref_basis, art_basis, atol=ATOL), (
            f"Basis max diff: {jnp.abs(ref_basis - art_basis).max()}"
        )

    def test_forward_pass_matches(self) -> None:
        """Forward pass produces identical outputs."""
        k, g = 3, 5

        ref = RefSplineLayer(
            n_in=N_IN,
            n_out=N_OUT,
            k=k,
            G=g,
            grid_range=(-1, 1),
            grid_e=0.05,
            residual=nnx.silu,
            external_weights=True,
            add_bias=True,
            seed=SEED,
        )

        art = EfficientKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            k=k,
            grid_intervals=g,
            grid_range=(-1.0, 1.0),
            grid_e=0.05,
            residual=nnx.silu,
            external_weights=True,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        art.grid.knots = nnx.Variable(ref.grid.item)
        art.c_basis = nnx.Param(ref.c_basis.value)
        art.c_res = nnx.Param(ref.c_res.value)
        art.c_spl = nnx.Param(ref.c_spl.value)
        art.bias = nnx.Param(ref.bias.value)

        x = _make_input()
        ref_out = ref(x)
        art_out = art(x)

        assert ref_out.shape == art_out.shape
        assert jnp.allclose(ref_out, art_out, atol=ATOL), (
            f"Forward max diff: {jnp.abs(ref_out - art_out).max()}"
        )

    def test_forward_no_residual_no_bias(self) -> None:
        """Forward pass matches without residual and bias."""
        k, g = 3, 3

        ref = RefSplineLayer(
            n_in=N_IN,
            n_out=N_OUT,
            k=k,
            G=g,
            grid_range=(-1, 1),
            grid_e=0.05,
            residual=None,
            external_weights=False,
            add_bias=False,
            seed=SEED,
        )

        art = EfficientKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            k=k,
            grid_intervals=g,
            grid_range=(-1.0, 1.0),
            grid_e=0.05,
            residual=None,
            external_weights=False,
            add_bias=False,
            rngs=nnx.Rngs(SEED),
        )

        art.grid.knots = nnx.Variable(ref.grid.item)
        art.c_basis = nnx.Param(ref.c_basis.value)

        x = _make_input()
        ref_out = ref(x)
        art_out = art(x)

        assert jnp.allclose(ref_out, art_out, atol=ATOL), (
            f"Forward max diff: {jnp.abs(ref_out - art_out).max()}"
        )


class TestChebyshevKANLayerVsRef:
    """ChebyshevKANLayer (Artifex) vs ChebyshevLayer (jaxKAN)."""

    @pytest.mark.parametrize("flavor", ["default", "modified", "exact"])
    def test_basis_evaluation_matches(self, flavor: str) -> None:
        """Basis evaluation matches for all flavors."""
        degree = 5

        ref = RefChebyshevLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            flavor=flavor,
            residual=None,
            external_weights=False,
            add_bias=True,
            seed=SEED,
        )

        art = ChebyshevKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            flavor=flavor,
            residual=None,
            external_weights=False,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        x = _make_input()
        ref_basis = ref.basis(x)
        art_basis = art.basis(x)

        assert ref_basis.shape == art_basis.shape, (
            f"Shape mismatch: ref={ref_basis.shape}, art={art_basis.shape}"
        )
        assert jnp.allclose(ref_basis, art_basis, atol=ATOL), (
            f"Basis max diff ({flavor}): {jnp.abs(ref_basis - art_basis).max()}"
        )

    @pytest.mark.parametrize("flavor", ["default", "modified", "exact"])
    def test_forward_pass_matches(self, flavor: str) -> None:
        """Forward pass matches with copied weights."""
        degree = 5

        ref = RefChebyshevLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            flavor=flavor,
            residual=None,
            external_weights=False,
            add_bias=True,
            seed=SEED,
        )

        art = ChebyshevKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            flavor=flavor,
            residual=None,
            external_weights=False,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        art.c_basis = nnx.Param(ref.c_basis.value)
        art.bias = nnx.Param(ref.bias.value)

        x = _make_input()
        ref_out = ref(x)
        art_out = art(x)

        assert ref_out.shape == art_out.shape
        assert jnp.allclose(ref_out, art_out, atol=ATOL), (
            f"Forward max diff ({flavor}): {jnp.abs(ref_out - art_out).max()}"
        )

    def test_forward_with_residual(self) -> None:
        """Forward pass with residual and external weights matches."""
        degree = 5

        ref = RefChebyshevLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            flavor="default",
            residual=nnx.silu,
            external_weights=True,
            add_bias=True,
            seed=SEED,
        )

        art = ChebyshevKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            flavor="default",
            residual=nnx.silu,
            external_weights=True,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        art.c_basis = nnx.Param(ref.c_basis.value)
        art.c_res = nnx.Param(ref.c_res.value)
        art.c_ext = nnx.Param(ref.c_ext.value)
        art.bias = nnx.Param(ref.bias.value)

        x = _make_input()
        ref_out = ref(x)
        art_out = art(x)

        assert jnp.allclose(ref_out, art_out, atol=ATOL), (
            f"Forward max diff (with residual): {jnp.abs(ref_out - art_out).max()}"
        )


class TestFourierKANLayerVsRef:
    """FourierKANLayer (Artifex) vs FourierLayer (jaxKAN)."""

    def test_basis_evaluation_matches(self) -> None:
        """Basis (cos, sin) evaluation produces identical outputs."""
        degree = 7

        ref = RefFourierLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            smooth_init=True,
            add_bias=True,
            seed=SEED,
        )

        art = FourierKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            smooth_init=True,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        x = _make_input()
        ref_c, ref_s = ref.basis(x)
        art_c, art_s = art.basis(x)

        assert jnp.allclose(ref_c, art_c, atol=ATOL), (
            f"Cos basis max diff: {jnp.abs(ref_c - art_c).max()}"
        )
        assert jnp.allclose(ref_s, art_s, atol=ATOL), (
            f"Sin basis max diff: {jnp.abs(ref_s - art_s).max()}"
        )

    def test_forward_pass_matches(self) -> None:
        """Forward pass matches with copied cos/sin weights."""
        degree = 7

        ref = RefFourierLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            smooth_init=True,
            add_bias=True,
            seed=SEED,
        )

        art = FourierKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            smooth_init=True,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        art.c_cos = nnx.Param(ref.c_cos.value)
        art.c_sin = nnx.Param(ref.c_sin.value)
        art.bias = nnx.Param(ref.bias.value)

        x = _make_input()
        ref_out = ref(x)
        art_out = art(x)

        assert ref_out.shape == art_out.shape
        assert jnp.allclose(ref_out, art_out, atol=ATOL), (
            f"Forward max diff: {jnp.abs(ref_out - art_out).max()}"
        )

    def test_forward_no_smooth_init(self) -> None:
        """Forward pass matches with smooth_init=False."""
        degree = 5

        ref = RefFourierLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            smooth_init=False,
            add_bias=True,
            seed=SEED,
        )

        art = FourierKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            smooth_init=False,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        art.c_cos = nnx.Param(ref.c_cos.value)
        art.c_sin = nnx.Param(ref.c_sin.value)
        art.bias = nnx.Param(ref.bias.value)

        x = _make_input()
        ref_out = ref(x)
        art_out = art(x)

        assert jnp.allclose(ref_out, art_out, atol=ATOL), (
            f"Forward max diff: {jnp.abs(ref_out - art_out).max()}"
        )


class TestLegendreKANLayerVsRef:
    """LegendreKANLayer (Artifex) vs LegendreLayer (jaxKAN)."""

    @pytest.mark.parametrize("flavor", ["default", "exact"])
    def test_basis_evaluation_matches(self, flavor: str) -> None:
        """Basis evaluation matches for both flavors."""
        degree = 5

        ref = RefLegendreLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            flavor=flavor,
            residual=None,
            external_weights=False,
            add_bias=True,
            seed=SEED,
        )

        art = LegendreKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            flavor=flavor,
            residual=None,
            external_weights=False,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        x = _make_input()
        ref_basis = ref.basis(x)
        art_basis = art.basis(x)

        assert ref_basis.shape == art_basis.shape, (
            f"Shape mismatch: ref={ref_basis.shape}, art={art_basis.shape}"
        )
        assert jnp.allclose(ref_basis, art_basis, atol=ATOL), (
            f"Basis max diff ({flavor}): {jnp.abs(ref_basis - art_basis).max()}"
        )

    @pytest.mark.parametrize("flavor", ["default", "exact"])
    def test_forward_pass_matches(self, flavor: str) -> None:
        """Forward pass matches with copied weights."""
        degree = 5

        ref = RefLegendreLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            flavor=flavor,
            residual=None,
            external_weights=False,
            add_bias=True,
            seed=SEED,
        )

        art = LegendreKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            flavor=flavor,
            residual=None,
            external_weights=False,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        art.c_basis = nnx.Param(ref.c_basis.value)
        art.bias = nnx.Param(ref.bias.value)

        x = _make_input()
        ref_out = ref(x)
        art_out = art(x)

        assert ref_out.shape == art_out.shape
        assert jnp.allclose(ref_out, art_out, atol=ATOL), (
            f"Forward max diff ({flavor}): {jnp.abs(ref_out - art_out).max()}"
        )

    def test_forward_with_residual(self) -> None:
        """Forward pass with residual and external weights matches."""
        degree = 5

        ref = RefLegendreLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            flavor="default",
            residual=nnx.silu,
            external_weights=True,
            add_bias=True,
            seed=SEED,
        )

        art = LegendreKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=degree,
            flavor="default",
            residual=nnx.silu,
            external_weights=True,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        art.c_basis = nnx.Param(ref.c_basis.value)
        art.c_res = nnx.Param(ref.c_res.value)
        art.c_ext = nnx.Param(ref.c_ext.value)
        art.bias = nnx.Param(ref.bias.value)

        x = _make_input()
        ref_out = ref(x)
        art_out = art(x)

        assert jnp.allclose(ref_out, art_out, atol=ATOL), (
            f"Forward max diff: {jnp.abs(ref_out - art_out).max()}"
        )


class TestRBFKANLayerVsRef:
    """RBFKANLayer (Artifex) vs RBFLayer (jaxKAN)."""

    def test_basis_evaluation_matches(self) -> None:
        """RBF basis evaluation produces identical outputs."""
        num_centers = 6

        ref = RefRBFLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=num_centers,
            kernel={"type": "gaussian", "std": 1.0},
            grid_range=(-2.0, 2.0),
            grid_e=1.0,
            residual=None,
            external_weights=False,
            add_bias=True,
            seed=SEED,
        )

        art = RBFKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=num_centers,
            kernel={"type": "gaussian", "std": 1.0},
            grid_range=(-2.0, 2.0),
            grid_e=1.0,
            residual=None,
            external_weights=False,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        # Copy grid: jaxKAN RBFGrid stores as .item, Artifex stores as .knots.value
        art.grid.knots = nnx.Variable(ref.grid.item)

        x = _make_input()
        ref_basis = ref.basis(x)
        art_basis = art.basis(x)

        assert ref_basis.shape == art_basis.shape, (
            f"Shape mismatch: ref={ref_basis.shape}, art={art_basis.shape}"
        )
        assert jnp.allclose(ref_basis, art_basis, atol=ATOL), (
            f"Basis max diff: {jnp.abs(ref_basis - art_basis).max()}"
        )

    def test_forward_pass_matches(self) -> None:
        """Forward pass matches with copied weights and grid."""
        num_centers = 6

        ref = RefRBFLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=num_centers,
            kernel={"type": "gaussian", "std": 1.0},
            grid_range=(-2.0, 2.0),
            grid_e=1.0,
            residual=None,
            external_weights=False,
            add_bias=True,
            seed=SEED,
        )

        art = RBFKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=num_centers,
            kernel={"type": "gaussian", "std": 1.0},
            grid_range=(-2.0, 2.0),
            grid_e=1.0,
            residual=None,
            external_weights=False,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        art.grid.knots = nnx.Variable(ref.grid.item)
        art.c_basis = nnx.Param(ref.c_basis.value)
        art.bias = nnx.Param(ref.bias.value)

        x = _make_input()
        ref_out = ref(x)
        art_out = art(x)

        assert ref_out.shape == art_out.shape
        assert jnp.allclose(ref_out, art_out, atol=ATOL), (
            f"Forward max diff: {jnp.abs(ref_out - art_out).max()}"
        )

    def test_forward_with_residual(self) -> None:
        """Forward pass with residual and external weights matches."""
        num_centers = 6

        ref = RefRBFLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=num_centers,
            kernel={"type": "gaussian", "std": 1.0},
            grid_range=(-2.0, 2.0),
            grid_e=1.0,
            residual=nnx.silu,
            external_weights=True,
            add_bias=True,
            seed=SEED,
        )

        art = RBFKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=num_centers,
            kernel={"type": "gaussian", "std": 1.0},
            grid_range=(-2.0, 2.0),
            grid_e=1.0,
            residual=nnx.silu,
            external_weights=True,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        art.grid.knots = nnx.Variable(ref.grid.item)
        art.c_basis = nnx.Param(ref.c_basis.value)
        art.c_res = nnx.Param(ref.c_res.value)
        art.c_ext = nnx.Param(ref.c_ext.value)
        art.bias = nnx.Param(ref.bias.value)

        x = _make_input()
        ref_out = ref(x)
        art_out = art(x)

        assert jnp.allclose(ref_out, art_out, atol=ATOL), (
            f"Forward max diff: {jnp.abs(ref_out - art_out).max()}"
        )


class TestSineKANLayerVsRef:
    """SineKANLayer (Artifex) vs SineLayer (jaxKAN)."""

    def test_basis_evaluation_matches(self) -> None:
        """Sine basis evaluation produces identical outputs."""
        num_basis = 6

        ref = RefSineLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=num_basis,
            residual=None,
            external_weights=False,
            add_bias=True,
            seed=SEED,
        )

        art = SineKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=num_basis,
            residual=None,
            external_weights=False,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        # Copy omega and phase from reference
        art.omega = nnx.Param(ref.omega.value)
        art.phase = nnx.Param(ref.phase.value)

        x = _make_input()
        ref_basis = ref.basis(x)
        art_basis = art.basis(x)

        assert ref_basis.shape == art_basis.shape, (
            f"Shape mismatch: ref={ref_basis.shape}, art={art_basis.shape}"
        )
        assert jnp.allclose(ref_basis, art_basis, atol=ATOL), (
            f"Basis max diff: {jnp.abs(ref_basis - art_basis).max()}"
        )

    def test_forward_pass_matches(self) -> None:
        """Forward pass matches with copied weights."""
        num_basis = 6

        ref = RefSineLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=num_basis,
            residual=None,
            external_weights=False,
            add_bias=True,
            seed=SEED,
        )

        art = SineKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=num_basis,
            residual=None,
            external_weights=False,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        art.omega = nnx.Param(ref.omega.value)
        art.phase = nnx.Param(ref.phase.value)
        art.c_basis = nnx.Param(ref.c_basis.value)
        art.bias = nnx.Param(ref.bias.value)

        x = _make_input()
        ref_out = ref(x)
        art_out = art(x)

        assert ref_out.shape == art_out.shape
        assert jnp.allclose(ref_out, art_out, atol=ATOL), (
            f"Forward max diff: {jnp.abs(ref_out - art_out).max()}"
        )

    def test_forward_with_residual(self) -> None:
        """Forward pass with residual and external weights matches."""
        num_basis = 6

        ref = RefSineLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=num_basis,
            residual=nnx.silu,
            external_weights=True,
            add_bias=True,
            seed=SEED,
        )

        art = SineKANLayer(
            n_in=N_IN,
            n_out=N_OUT,
            D=num_basis,
            residual=nnx.silu,
            external_weights=True,
            add_bias=True,
            rngs=nnx.Rngs(SEED),
        )

        art.omega = nnx.Param(ref.omega.value)
        art.phase = nnx.Param(ref.phase.value)
        art.c_basis = nnx.Param(ref.c_basis.value)
        art.c_res = nnx.Param(ref.c_res.value)
        art.c_ext = nnx.Param(ref.c_ext.value)
        art.bias = nnx.Param(ref.bias.value)

        x = _make_input()
        ref_out = ref(x)
        art_out = art(x)

        assert jnp.allclose(ref_out, art_out, atol=ATOL), (
            f"Forward max diff: {jnp.abs(ref_out - art_out).max()}"
        )


class TestGridNumericalEquivalence:
    """Verify grid initialization produces identical knot positions."""

    def test_dense_grid_init_matches_base_grid(self) -> None:
        """DenseKANGrid produces the same knots as jaxKAN BaseGrid."""
        from jaxkan.grids.BaseGrid import BaseGrid as RefBaseGrid

        from artifex.generative_models.core.layers.kan.grids import DenseKANGrid

        k, g = 3, 5
        ref_grid = RefBaseGrid(
            n_in=N_IN,
            n_out=N_OUT,
            k=k,
            G=g,
            grid_range=(-1, 1),
            grid_e=0.05,
        )
        art_grid = DenseKANGrid(
            n_in=N_IN,
            n_out=N_OUT,
            k=k,
            grid_intervals=g,
            grid_range=(-1.0, 1.0),
            grid_e=0.05,
        )

        assert jnp.allclose(ref_grid.item, art_grid.knots.value, atol=1e-7), (
            f"Grid max diff: {jnp.abs(ref_grid.item - art_grid.knots.value).max()}"
        )

    def test_efficient_grid_init_matches_spline_grid(self) -> None:
        """EfficientKANGrid produces the same knots as jaxKAN SplineGrid."""
        from jaxkan.grids.SplineGrid import SplineGrid as RefSplineGrid

        from artifex.generative_models.core.layers.kan.grids import EfficientKANGrid

        k, g = 3, 5
        ref_grid = RefSplineGrid(n_nodes=N_IN, k=k, G=g, grid_range=(-1, 1), grid_e=0.05)
        art_grid = EfficientKANGrid(
            n_nodes=N_IN,
            k=k,
            grid_intervals=g,
            grid_range=(-1.0, 1.0),
            grid_e=0.05,
        )

        assert jnp.allclose(ref_grid.item, art_grid.knots.value, atol=1e-7), (
            f"Grid max diff: {jnp.abs(ref_grid.item - art_grid.knots.value).max()}"
        )

    def test_rbf_grid_init_matches(self) -> None:
        """RBFKANGrid produces the same centers as jaxKAN RBFGrid."""
        from jaxkan.grids.RBFGrid import RBFGrid as RefRBFGrid

        from artifex.generative_models.core.layers.kan.grids import RBFKANGrid

        d = 6
        ref_grid = RefRBFGrid(n_nodes=N_IN, D=d, grid_range=(-2.0, 2.0), grid_e=1.0)
        art_grid = RBFKANGrid(
            n_nodes=N_IN,
            num_centers=d,
            grid_range=(-2.0, 2.0),
            grid_e=1.0,
        )

        assert jnp.allclose(ref_grid.item, art_grid.knots.value, atol=1e-7), (
            f"Grid max diff: {jnp.abs(ref_grid.item - art_grid.knots.value).max()}"
        )


class TestLstSqEquivalence:
    """Verify the least-squares solver produces matching results."""

    def test_solve_full_lstsq_matches(self) -> None:
        """Both lstsq implementations give the same solution."""
        from jaxkan.layers.utils import solve_full_lstsq as ref_lstsq

        from artifex.generative_models.core.layers.kan._utils import (
            _solve_full_lstsq as art_lstsq,
        )

        key = jax.random.key(SEED)
        k1, k2 = jax.random.split(key)
        a = jax.random.normal(k1, shape=(4, 10, 6))
        b = jax.random.normal(k2, shape=(4, 10, 3))

        ref_sol = ref_lstsq(a, b)
        art_sol = art_lstsq(a, b)

        assert jnp.allclose(ref_sol, art_sol, atol=1e-5), (
            f"LstSq max diff: {jnp.abs(ref_sol - art_sol).max()}"
        )
