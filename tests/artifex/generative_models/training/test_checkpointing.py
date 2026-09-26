"""Checkpoints a run resumes from exactly, with the best model kept by the store's records.

One save writes the model, the optimizer, the run's base key, the data position and the
extensions' state at a step; a restore into objects built the same way continues where the
saved run stood. Retention keeps the newest checkpoints and the best one by a recorded metric,
read from the store's records, so a resumed run keeps its best model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from datarax import Pipeline
from flax import nnx
from substrax.callbacks import CallbackList, EarlyStoppingCallback, EarlyStoppingConfig
from substrax.checkpoint import CheckpointMetadata

from artifex.generative_models.training.checkpointing import (
    Checkpointer,
    CheckpointPolicy,
    RunState,
)


def _objects(seed: int) -> tuple[nnx.Linear, nnx.Optimizer[Any], Pipeline, CallbackList]:
    model = nnx.Linear(2, 1, rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    data = {"x": np.arange(20, dtype=np.float32).reshape(10, 2), "y": np.ones((10, 1), np.float32)}
    pipeline = Pipeline.from_arrays(data, batch_size=4, seed=seed, shuffle=True)
    callbacks = CallbackList([EarlyStoppingCallback(EarlyStoppingConfig(patience=3))])
    return model, optimizer, pipeline, callbacks


def _train(model: nnx.Linear, optimizer: nnx.Optimizer[Any], batch: dict[str, jax.Array]) -> None:
    def loss(m: nnx.Linear) -> jax.Array:
        return jnp.mean((m(batch["x"]) - batch["y"]) ** 2)

    optimizer.update(model, nnx.grad(loss)(model))


def _run_state(
    objects: tuple[nnx.Linear, nnx.Optimizer[Any], Pipeline, CallbackList], key: jax.Array
) -> RunState:
    model, optimizer, pipeline, callbacks = objects
    return RunState(
        model=model,
        optimizer=optimizer,
        base_key=key,
        epoch=1,
        data=pipeline,
        extensions={"callbacks": callbacks},
    )


def test_a_restored_run_continues_exactly(tmp_path: Path) -> None:
    saved = _objects(0)
    model, optimizer, pipeline, callbacks = saved
    session = pipeline.session()
    _train(model, optimizer, next(session))
    callbacks.on_epoch_end(None, 0, {"val_loss": 0.5})  # type: ignore[arg-type]
    checkpointer = Checkpointer(CheckpointPolicy(directory=tmp_path))
    checkpointer.save(1, _run_state(saved, jax.random.key(3)))

    restored_objects = _objects(1)
    fresh, fresh_optimizer, fresh_pipeline, fresh_callbacks = restored_objects
    restored = checkpointer.restore(_run_state(restored_objects, jax.random.key(9)))

    for _ in range(2):
        _train(model, optimizer, next(session))
    fresh_session = fresh_pipeline.session()
    for _ in range(2):
        _train(fresh, fresh_optimizer, next(fresh_session))

    assert restored.step == 1
    assert restored.epoch == 1
    assert jnp.array_equal(
        jax.random.key_data(restored.base_key), jax.random.key_data(jax.random.key(3))
    )
    assert jnp.array_equal(fresh.kernel[...], model.kernel[...])
    early, fresh_early = next(iter(callbacks)), next(iter(fresh_callbacks))
    assert isinstance(early, EarlyStoppingCallback)
    assert isinstance(fresh_early, EarlyStoppingCallback)
    assert fresh_early.best_score == early.best_score == 0.5


def test_retention_keeps_the_newest_and_the_best_across_a_resume(tmp_path: Path) -> None:
    policy = CheckpointPolicy(directory=tmp_path, keep=1, best=("val_loss", "min"))
    state = _run_state(_objects(0), jax.random.key(0))
    Checkpointer(policy).save(10, state, metrics={"val_loss": 1.0})
    Checkpointer(policy).save(20, state, metrics={"val_loss": 0.5})

    resumed = Checkpointer(policy)
    resumed.save(30, state, metrics={"val_loss": 0.9})
    resumed.save(40, state, metrics={"val_loss": 0.8})

    assert resumed.steps() == [20, 40]
    assert resumed.best_step() == 20


def test_a_refused_record_leaves_the_objects_untouched(tmp_path: Path) -> None:
    checkpointer = Checkpointer(CheckpointPolicy(directory=tmp_path))
    checkpointer.save(5, _run_state(_objects(0), jax.random.key(0)), extra={"arch": "v1"})
    fresh_objects = _objects(1)
    fresh = fresh_objects[0]
    kernel = fresh.kernel[...]

    def validate(metadata: CheckpointMetadata) -> None:
        if metadata.extra.get("arch") != "v2":
            raise ValueError("checkpoint written by another architecture")

    with pytest.raises(ValueError, match=r"architecture"):
        checkpointer.restore(_run_state(fresh_objects, jax.random.key(0)), validate=validate)

    assert jnp.array_equal(fresh.kernel[...], kernel)


def test_restore_without_a_checkpoint_says_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        Checkpointer(CheckpointPolicy(directory=tmp_path)).restore(
            _run_state(_objects(0), jax.random.key(0))
        )


def test_the_save_cadence_is_the_policys(tmp_path: Path) -> None:
    policy = CheckpointPolicy(directory=tmp_path, every_steps=100)

    assert [policy.is_due(step) for step in (0, 99, 100, 250, 300)] == [
        False,
        False,
        True,
        False,
        True,
    ]


@pytest.mark.parametrize(
    "kwargs",
    [{"every_steps": 0}, {"keep": 0}, {"best": ("val_loss", "sideways")}],
    ids=["cadence", "keep", "mode"],
)
def test_a_policy_that_can_keep_nothing_is_refused(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"every_steps|keep|mode"):
        CheckpointPolicy(directory=tmp_path, **kwargs)
