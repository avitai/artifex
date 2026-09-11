"""Public trainer state composition preserves native checkpoint contents."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from substrax.checkpoint import OrbaxCheckpointStore

from artifex.generative_models.core.configuration import (
    ExtensionConfig,
    OptimizerConfig,
    TrainingConfig,
)
from artifex.generative_models.extensions.base import Extension
from artifex.generative_models.training.trainer import Trainer


def objective(model, batch, rng, step):
    """A small differentiable objective exercising native optimizer state."""
    del rng, step
    loss = jnp.mean(jnp.square(model(batch["x"]) - batch["y"]))
    return loss, {"loss": loss}


def assert_same(left, right):
    """Compare full pytree structure and values, including typed RNG keys."""
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for first, second in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        assert jnp.array_equal(first, second)


@pytest.mark.parametrize("with_extension", [False, True])
def test_detached_restore_then_apply_preserves_next_training_step(tmp_path, with_extension):
    """Inspect metadata without mutation, then resume the exact optimizer trajectory."""
    extension = Extension(ExtensionConfig(name="test"), rngs=nnx.Rngs(7))
    trainer = Trainer(
        nnx.Linear(2, 1, rngs=nnx.Rngs(3)),
        TrainingConfig(
            name="test",
            optimizer=OptimizerConfig(name="adam", optimizer_type="adam", learning_rate=1e-3),
        ),
        loss_fn=objective,
        checkpoint_dir=str(tmp_path / "native"),
        extensions={"test": extension} if with_extension else {},
    )
    batch = {"x": jnp.ones((2, 2)), "y": jnp.ones((2, 1))}
    trainer.train_step(batch)
    saved = trainer.checkpoint_state()
    assert set(saved) == {"model", "opt_state", "rng", "extensions"}
    with OrbaxCheckpointStore(tmp_path / "composed") as store:
        store.save(saved, 1, additional_metadata={"request": "expected"})
        trainer.train_step(batch)
        expected_next = trainer.checkpoint_state()
        restored, metadata = store.restore(trainer.checkpoint_state(), 1)
        assert metadata["request"] == "expected"
        assert trainer.step == 2
        assert_same(trainer.checkpoint_state(), expected_next)
        trainer.apply_checkpoint_state(restored, step=1)
    assert trainer.step == 1
    assert_same(trainer.checkpoint_state(), saved)
    trainer.train_step(batch)
    assert_same(trainer.checkpoint_state(), expected_next)
    trainer.save_checkpoint()
    trainer.train_step(batch)
    trainer.load_checkpoint(step=2)
    assert trainer.step == 2
    assert_same(trainer.checkpoint_state(), expected_next)
