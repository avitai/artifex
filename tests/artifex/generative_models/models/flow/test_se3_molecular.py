"""Focused tests for the simplified molecular flow loss contract."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.models.flow.se3_molecular import SE3MolecularFlow


@pytest.fixture
def rngs():
    """Create random number generators for tests."""
    return nnx.Rngs(42)


@pytest.fixture
def se3_flow_config():
    """Small molecular flow config for focused unit tests."""
    return {
        "hidden_dim": 16,
        "num_layers": 2,
        "num_coupling_layers": 2,
        "max_atoms": 8,
        "atom_types": 5,
        "use_attention": True,
        "equivariant_layers": True,
    }


@pytest.fixture
def molecular_batch(rngs, se3_flow_config):
    """Create a small synthetic molecular batch."""
    batch_size = 3
    max_atoms = se3_flow_config["max_atoms"]
    return {
        "coordinates": jax.random.normal(rngs.params(), (batch_size, max_atoms, 3)),
        "atom_types": jax.random.randint(rngs.params(), (batch_size, max_atoms), 0, 5),
        "atom_mask": jnp.ones((batch_size, max_atoms), dtype=jnp.bool_),
    }


def test_loss_fn_returns_canonical_total_loss_dict(rngs, se3_flow_config, molecular_batch):
    """Molecular flow loss must follow the canonical flow loss contract."""
    model = SE3MolecularFlow(**se3_flow_config, rngs=rngs)

    losses = model.loss_fn(molecular_batch, {})

    assert set(losses) == {"total_loss", "nll_loss", "log_prob", "avg_log_prob"}
    assert jnp.isfinite(losses["total_loss"])
    assert jnp.isfinite(losses["nll_loss"])
    assert jnp.isfinite(losses["log_prob"])
    assert jnp.isfinite(losses["avg_log_prob"])
    assert jnp.allclose(losses["total_loss"], losses["nll_loss"])
    assert jnp.allclose(losses["total_loss"], -losses["log_prob"])


def test_loss_fn_supports_nnx_jit(rngs, se3_flow_config, molecular_batch):
    """Molecular flow loss must stay jittable at the NNX model boundary."""
    model = SE3MolecularFlow(**se3_flow_config, rngs=rngs)

    @nnx.jit
    def total_loss(mod, batch):
        return mod.loss_fn(batch, {})["total_loss"]

    loss = total_loss(model, molecular_batch)
    assert jnp.isfinite(loss)


def test_loss_fn_supports_nnx_grad(rngs, se3_flow_config, molecular_batch):
    """Molecular flow loss must stay differentiable at the NNX model boundary."""
    model = SE3MolecularFlow(**se3_flow_config, rngs=rngs)

    def total_loss(mod):
        return mod.loss_fn(molecular_batch, {})["total_loss"]

    loss, grads = nnx.value_and_grad(total_loss)(model)

    assert jnp.isfinite(loss)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert grad_leaves
    assert all(jnp.isfinite(leaf).all() for leaf in grad_leaves)
