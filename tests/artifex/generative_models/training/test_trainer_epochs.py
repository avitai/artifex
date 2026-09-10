"""Trainer epochs are permutations served by one pipeline, and the step is compiled once.

``Trainer.train`` builds one datarax pipeline per call and starts each later epoch with
``Pipeline.reset()``, so every epoch visits each record once in a fresh order, and the
gradient step runs through a compiled function that is traced once per batch shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import OptimizerConfig, TrainingConfig
from artifex.generative_models.training import trainer as trainer_module
from artifex.generative_models.training.trainer import Trainer


_N = 64
_BATCH = 16


class _Linear(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.dense = nnx.Linear(in_features=4, out_features=2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dense(x)


def _config() -> TrainingConfig:
    return TrainingConfig(
        name="epochs",
        batch_size=_BATCH,
        num_epochs=2,
        optimizer=OptimizerConfig(name="adam", optimizer_type="adam", learning_rate=1e-3),
    )


def _data() -> dict[str, jax.Array]:
    return {
        "input": jax.random.normal(jax.random.key(1), (_N, 4)),
        "id": jnp.arange(_N, dtype=jnp.int32),
    }


def _loss_fn(model: nnx.Module, batch: dict[str, jax.Array], _rng: jax.Array, _step: jax.Array):
    prediction = model(batch["input"])
    return jnp.mean(prediction**2), {"ids": batch["id"]}


def _ids_per_epoch(trainer: Trainer, steps_per_epoch: int) -> list[list[int]]:
    ids = [np.asarray(m["ids"]).tolist() for m in trainer.train_metrics]
    return [
        [i for batch in ids[e * steps_per_epoch : (e + 1) * steps_per_epoch] for i in batch]
        for e in range(len(ids) // steps_per_epoch)
    ]


def test_every_epoch_visits_each_record_once_in_a_fresh_order(tmp_path) -> None:
    trainer = Trainer(
        model=_Linear(rngs=nnx.Rngs(0)),
        training_config=_config(),
        loss_fn=_loss_fn,
        checkpoint_dir=str(tmp_path),
    )

    trainer.train(_data(), num_epochs=3, batch_size=_BATCH)

    epochs = _ids_per_epoch(trainer, _N // _BATCH)
    assert len(epochs) == 3
    for epoch in epochs:
        assert sorted(epoch) == list(range(_N))
    assert len({tuple(epoch) for epoch in epochs}) == 3


def test_train_builds_one_pipeline_per_call(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls = {"n": 0}
    original = trainer_module.create_data_pipeline

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "create_data_pipeline", counted)
    trainer = Trainer(
        model=_Linear(rngs=nnx.Rngs(0)),
        training_config=_config(),
        loss_fn=_loss_fn,
        checkpoint_dir=str(tmp_path),
    )

    trainer.train(_data(), num_epochs=3, batch_size=_BATCH)

    assert calls["n"] == 1
    assert len(trainer.train_metrics) == 3 * (_N // _BATCH)


def test_train_step_is_traced_once_for_same_shape_batches(tmp_path) -> None:
    traces = {"n": 0}

    def counting_loss(model, batch, rng, step):
        traces["n"] += 1  # Python side effect: runs at trace time only
        return _loss_fn(model, batch, rng, step)

    trainer = Trainer(
        model=_Linear(rngs=nnx.Rngs(0)),
        training_config=_config(),
        loss_fn=counting_loss,
        checkpoint_dir=str(tmp_path),
    )
    batch = {"input": jnp.ones((_BATCH, 4)), "id": jnp.arange(_BATCH, dtype=jnp.int32)}

    losses = [trainer.train_step(batch)["loss"] for _ in range(5)]

    assert traces["n"] == 1
    assert losses[-1] < losses[0]
    assert trainer.step == 5
