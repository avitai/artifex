"""Tests for E(n) Equivariant Graph Neural Network layers.

Verifies:
- EGNNBlock: init, forward shape, deterministic mode, dropout
- EGNNLayer: init, forward shapes, attention, residual, equivariance,
  JIT compatibility, gradient flow
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from flax import nnx

from artifex.generative_models.core.layers.egnn import EGNNBlock, EGNNLayer


@pytest.fixture
def rngs():
    """Fixture providing NNX Rngs for tests."""
    return nnx.Rngs(params=jax.random.key(42), dropout=jax.random.key(7))


# ---------------------------------------------------------------------------
# EGNNBlock tests
# ---------------------------------------------------------------------------
class TestEGNNBlock:
    """Tests for the EGNNBlock MLP module."""

    def test_init_default(self, rngs):
        """Test initialisation with default parameters."""
        block = EGNNBlock(
            in_features=16,
            out_features=8,
            hidden_features=32,
            rngs=rngs,
        )
        assert block.num_layers == 2
        assert len(block.layers) == 2
        assert block.dropout is None

    def test_init_with_dropout(self, rngs):
        """Test initialisation with dropout."""
        block = EGNNBlock(
            in_features=16,
            out_features=8,
            hidden_features=32,
            dropout_rate=0.1,
            rngs=rngs,
        )
        assert block.dropout is not None

    def test_forward_shape(self, rngs):
        """Test output shape."""
        block = EGNNBlock(
            in_features=16,
            out_features=8,
            hidden_features=32,
            rngs=rngs,
        )
        x = jnp.ones((4, 16))
        y = block(x, deterministic=True)
        assert y.shape == (4, 8)

    def test_forward_batched(self, rngs):
        """Test with higher-dimensional batched input."""
        block = EGNNBlock(
            in_features=16,
            out_features=8,
            hidden_features=32,
            rngs=rngs,
        )
        x = jnp.ones((2, 5, 5, 16))
        y = block(x, deterministic=True)
        assert y.shape == (2, 5, 5, 8)

    def test_deterministic_mode(self, rngs):
        """Test deterministic mode produces consistent output."""
        block = EGNNBlock(
            in_features=16,
            out_features=8,
            hidden_features=32,
            dropout_rate=0.5,
            rngs=rngs,
        )
        x = jnp.ones((4, 16))
        y1 = block(x, deterministic=True)
        y2 = block(x, deterministic=True)
        npt.assert_array_equal(y1, y2)

    def test_no_layer_norm(self, rngs):
        """Test block without layer norm."""
        block = EGNNBlock(
            in_features=16,
            out_features=8,
            hidden_features=32,
            use_layer_norm=False,
            rngs=rngs,
        )
        assert len(block.norms) == 0
        x = jnp.ones((4, 16))
        y = block(x, deterministic=True)
        assert y.shape == (4, 8)

    def test_single_layer(self, rngs):
        """Test block with a single linear layer (no hidden)."""
        block = EGNNBlock(
            in_features=16,
            out_features=8,
            hidden_features=32,
            num_layers=1,
            rngs=rngs,
        )
        assert len(block.layers) == 1
        x = jnp.ones((4, 16))
        y = block(x, deterministic=True)
        assert y.shape == (4, 8)


# ---------------------------------------------------------------------------
# EGNNLayer tests
# ---------------------------------------------------------------------------
class TestEGNNLayer:
    """Tests for the EGNNLayer equivariant module."""

    @pytest.fixture
    def layer(self, rngs):
        """Fixture providing a default EGNNLayer."""
        return EGNNLayer(
            node_dim=16,
            edge_dim=4,
            hidden_dim=16,
            num_mlp_layers=2,
            dropout_rate=0.0,
            use_attention=True,
            residual=True,
            rngs=rngs,
        )

    @pytest.fixture
    def inputs(self):
        """Fixture providing standard test inputs."""
        key = jax.random.key(0)
        k1, k2, k3 = jax.random.split(key, 3)
        batch, n_nodes, node_dim, edge_dim = 2, 5, 16, 4

        node_features = jax.random.normal(k1, (batch, n_nodes, node_dim))
        coordinates = jax.random.normal(k2, (batch, n_nodes, 3))
        # Symmetric adjacency (fully connected, no self loops)
        edge_index = jnp.ones((batch, n_nodes, n_nodes)) - jnp.eye(n_nodes)
        edge_features = jax.random.normal(k3, (batch, n_nodes, n_nodes, edge_dim))
        return node_features, coordinates, edge_index, edge_features

    def test_init(self, layer):
        """Test layer initialises correctly."""
        assert layer.node_dim == 16
        assert layer.edge_dim == 4
        assert layer.hidden_dim == 16
        assert layer.use_attention is True
        assert layer.residual is True

    def test_forward_shapes(self, layer, inputs):
        """Test output shapes match expected dimensions."""
        nf, coord, ei, ef = inputs
        new_nf, new_coord, new_ef = layer(nf, coord, ei, ef, deterministic=True)
        assert new_nf.shape == nf.shape
        assert new_coord.shape == coord.shape
        # Edge features are the edge messages
        assert new_ef.shape[:3] == (*nf.shape[:2], nf.shape[1])

    def test_no_edge_features(self, rngs):
        """Test forward pass without edge features."""
        layer = EGNNLayer(
            node_dim=8,
            edge_dim=0,
            hidden_dim=8,
            dropout_rate=0.0,
            rngs=rngs,
        )
        batch, n_nodes = 2, 4
        nf = jnp.ones((batch, n_nodes, 8))
        coord = jnp.ones((batch, n_nodes, 3))
        ei = jnp.ones((batch, n_nodes, n_nodes)) - jnp.eye(n_nodes)

        new_nf, new_coord, _ = layer(nf, coord, ei, deterministic=True)
        assert new_nf.shape == (batch, n_nodes, 8)
        assert new_coord.shape == (batch, n_nodes, 3)

    def test_no_attention(self, rngs, inputs):
        """Test forward pass without attention."""
        layer = EGNNLayer(
            node_dim=16,
            edge_dim=4,
            hidden_dim=16,
            use_attention=False,
            dropout_rate=0.0,
            rngs=rngs,
        )
        nf, coord, ei, ef = inputs
        new_nf, new_coord, _ = layer(nf, coord, ei, ef, deterministic=True)
        assert new_nf.shape == nf.shape
        assert new_coord.shape == coord.shape

    def test_no_residual(self, rngs, inputs):
        """Test forward pass without residual connection."""
        layer = EGNNLayer(
            node_dim=16,
            edge_dim=4,
            hidden_dim=16,
            residual=False,
            dropout_rate=0.0,
            rngs=rngs,
        )
        nf, coord, ei, ef = inputs
        new_nf, _, _ = layer(nf, coord, ei, ef, deterministic=True)
        assert new_nf.shape == nf.shape

    def test_with_mask(self, layer, inputs):
        """Test forward pass with node mask."""
        nf, coord, ei, ef = inputs
        batch, n_nodes = nf.shape[:2]
        mask = jnp.ones((batch, n_nodes))
        mask = mask.at[:, -1].set(0)  # Mask out last node

        new_nf, new_coord, _ = layer(nf, coord, ei, ef, mask=mask, deterministic=True)
        assert new_nf.shape == nf.shape
        assert new_coord.shape == coord.shape

    def test_translation_equivariance(self, layer, inputs):
        """Test that coordinates are equivariant under translation."""
        nf, coord, ei, ef = inputs
        translation = jnp.array([1.0, 2.0, 3.0])

        _, new_coord_orig, _ = layer(nf, coord, ei, ef, deterministic=True)
        _, new_coord_translated, _ = layer(nf, coord + translation, ei, ef, deterministic=True)

        # f(x + t) should equal f(x) + t
        npt.assert_allclose(
            new_coord_translated,
            new_coord_orig + translation,
            atol=1e-5,
        )

    def test_node_features_translation_invariant(self, layer, inputs):
        """Test that node features are invariant under translation."""
        nf, coord, ei, ef = inputs
        translation = jnp.array([5.0, -3.0, 2.0])

        new_nf_orig, _, _ = layer(nf, coord, ei, ef, deterministic=True)
        new_nf_translated, _, _ = layer(nf, coord + translation, ei, ef, deterministic=True)

        npt.assert_allclose(new_nf_orig, new_nf_translated, atol=1e-5)

    def test_jit_compatible(self, layer, inputs):
        """Test JIT compilation works."""
        nf, coord, ei, ef = inputs

        @jax.jit
        def forward(nf_, coord_, ei_, ef_):
            return layer(nf_, coord_, ei_, ef_, deterministic=True)

        new_nf, new_coord, new_ef = forward(nf, coord, ei, ef)
        assert new_nf.shape == nf.shape
        assert new_coord.shape == coord.shape

    def test_gradient_flow(self, layer, inputs):
        """Test gradients propagate through the layer."""
        nf, coord, ei, ef = inputs

        def loss_fn(nf_):
            new_nf, new_coord, _ = layer(nf_, coord, ei, ef, deterministic=True)
            return jnp.sum(new_nf**2) + jnp.sum(new_coord**2)

        grads = jax.grad(loss_fn)(nf)
        assert grads.shape == nf.shape
        assert jnp.isfinite(grads).all()

    def test_deterministic_mode(self, rngs, inputs):
        """Test deterministic flag disables randomness."""
        layer = EGNNLayer(
            node_dim=16,
            edge_dim=4,
            hidden_dim=16,
            dropout_rate=0.5,
            rngs=rngs,
        )
        nf, coord, ei, ef = inputs

        new_nf1, _, _ = layer(nf, coord, ei, ef, deterministic=True)
        new_nf2, _, _ = layer(nf, coord, ei, ef, deterministic=True)
        npt.assert_array_equal(new_nf1, new_nf2)
