"""Checkpoints a run resumes from exactly, with the best model kept by the store's records.

One save writes the model, the optimizer, the run's base key, the data position and the
extensions' state at a step; a restore into objects built the same way continues where the
saved run stood. Retention keeps the newest checkpoints and the best one by a recorded metric,
read from the store's records, so a resumed run keeps its best model.
"""

from __future__ import annotations

from dataclasses import replace
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
from substrax.checkpoint import CheckpointMetadata, OrbaxCheckpointStore
from substrax.typing import CheckpointState

from artifex.generative_models.training.checkpointing import (
    Checkpointer,
    CheckpointPolicy,
    RunState,
)
from artifex.generative_models.training.step import GradientStep, StepContext


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


def _objective(
    model: nnx.Linear, batch: dict[str, jax.Array], context: StepContext
) -> tuple[jax.Array, dict[str, jax.Array]]:
    return jnp.mean((model(batch["x"]) - batch["y"]) ** 2), {}


def _run_state(
    objects: tuple[nnx.Linear, nnx.Optimizer[Any], Pipeline, CallbackList], key: jax.Array
) -> RunState:
    model, optimizer, pipeline, callbacks = objects
    return RunState(
        model=model,
        step=GradientStep(optimizer, _objective),
        base_key=key,
        epoch=1,
        data=pipeline,
        extra_state={"callbacks": callbacks},
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


@pytest.mark.parametrize("keep", [None, 7], ids=["all", "seven"])
def test_declared_retention_exceeds_the_storage_default(tmp_path: Path, keep: int | None) -> None:
    policy = CheckpointPolicy(directory=tmp_path, keep=keep)
    run = _run_state(_objects(0), jax.random.key(0))
    for step in range(1, 8):
        Checkpointer(policy).save(step, run)

    assert Checkpointer(policy).steps() == list(range(1, 8))


def test_old_best_survives_a_full_recent_window_and_resume(tmp_path: Path) -> None:
    policy = CheckpointPolicy(directory=tmp_path, keep=5, best=("val_loss", "min"))
    run = _run_state(_objects(0), jax.random.key(0))
    for step in range(1, 8):
        Checkpointer(policy).save(step, run, metrics={"val_loss": float(step)})

    resumed = Checkpointer(policy)
    assert resumed.steps() == [1, 3, 4, 5, 6, 7]
    assert resumed.best_step() == 1
    resumed.save(8, run, metrics={"val_loss": 0.5})
    assert resumed.steps() == [4, 5, 6, 7, 8]
    assert resumed.best_step() == 8


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


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_invalid_parameters_cannot_replace_a_retained_checkpoint(
    tmp_path: Path, invalid: float
) -> None:
    objects = _objects(0)
    run = _run_state(objects, jax.random.key(0))
    checkpointer = Checkpointer(CheckpointPolicy(directory=tmp_path, keep=1))
    checkpointer.save(1, run)
    saved_kernel = objects[0].kernel[...]
    objects[0].kernel[...] = jnp.full_like(saved_kernel, invalid)

    with pytest.raises(FloatingPointError, match="non-finite"):
        checkpointer.save(2, run)

    assert checkpointer.steps() == [1]
    restored_objects = _objects(1)
    checkpointer.restore(_run_state(restored_objects, jax.random.key(1)))
    np.testing.assert_array_equal(restored_objects[0].kernel[...], saved_kernel)


class _StateHolder:
    """Checkpointable numerical state with ordinary host metadata."""

    def __init__(self, value: Any) -> None:
        self.state = {"value": value, "label": "pending", "best": float("inf")}

    def get_state(self) -> CheckpointState:
        return self.state

    def set_state(self, state: CheckpointState, /) -> None:
        self.state = state


@pytest.mark.parametrize("location", ["mutable", "optimizer", "extension", "data"])
def test_invalid_numerical_state_is_refused_before_creating_storage(
    tmp_path: Path, location: str
) -> None:
    objects = _objects(0)
    model, optimizer, _, _ = objects
    run = _run_state(objects, jax.random.key(0))
    if location == "mutable":
        model.statistic = nnx.Variable(jnp.asarray(complex(1.0, float("nan"))))
    elif location == "optimizer":
        # A finite gradient overflows Adam's second moment while parameters stay finite.
        optimizer.update(model, nnx.grad(lambda m: jnp.sum(m.kernel[...]) * 1e30)(model))
        assert bool(jnp.all(jnp.isfinite(model.kernel[...])))
    elif location == "extension":
        run = replace(run, extra_state={"ema": _StateHolder(jnp.asarray(float("inf")))})
    else:
        run = replace(run, data=_StateHolder(np.asarray([float("nan")], dtype=np.float32)))
    directory = tmp_path / "checkpoint"

    with pytest.raises(FloatingPointError, match="non-finite"):
        Checkpointer(CheckpointPolicy(directory=directory)).save(1, run)

    assert not directory.exists()


@pytest.mark.parametrize("device_array", [False, True], ids=["numpy", "jax"])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_finite_arrays_are_checked_without_narrowing_their_dtype(
    tmp_path: Path, device_array: bool, dtype: Any
) -> None:
    value = np.asarray([1e100], dtype=dtype)
    with jax.enable_x64():
        value = jnp.asarray(value) if device_array else value
    run = replace(
        _run_state(_objects(0), jax.random.key(0)),
        extra_state={"statistics": _StateHolder(value)},
    )
    # Finiteness is a property of the saved float64 array, even with JAX x64 disabled.
    with jax.enable_x64(False):
        path = Checkpointer(CheckpointPolicy(directory=tmp_path)).save(1, run)
        assert not jax.config.x64_enabled
    assert path.is_dir()
    assert value.dtype == dtype


def test_keys_counters_and_uninitialized_callback_metadata_can_be_saved(tmp_path: Path) -> None:
    model = nnx.Module()
    model.counter = nnx.Variable(jnp.asarray(7, dtype=jnp.int32))
    optimizer = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)
    extension = nnx.Module()
    extension.enabled = nnx.Variable(jnp.asarray(True))
    callbacks = CallbackList([EarlyStoppingCallback(EarlyStoppingConfig(patience=3))])
    before = callbacks.get_state()
    run = RunState(
        model=model,
        step=GradientStep(optimizer, _objective),
        base_key=jax.random.key(0),
        epoch=0,
        extra_state={"callbacks": callbacks, "extension": extension},
    )
    checkpointer = Checkpointer(CheckpointPolicy(directory=tmp_path))
    checkpointer.save(0, run)
    extension.enabled[...] = jnp.asarray(False)
    checkpointer.restore(run)
    assert bool(extension.enabled[...])
    assert int(model.counter[...]) == 7
    assert callbacks.get_state() == before


@pytest.mark.parametrize("location", ["model", "optimizer", "mutable", "extra"])
def test_invalid_restored_arrays_are_refused_before_live_state_changes(
    tmp_path: Path, location: str
) -> None:
    source = _objects(0)
    model, optimizer, pipeline, callbacks = source
    model.statistic = nnx.Variable(jnp.asarray(0.0))
    extension = _StateHolder(jnp.asarray(0.0))
    if location == "model":
        model.kernel[...] = jnp.full_like(model.kernel[...], jnp.nan)
    elif location == "optimizer":
        optimizer.update(model, nnx.grad(lambda m: jnp.sum(m.kernel[...]) * 1e30)(model))
    elif location == "mutable":
        model.statistic[...] = jnp.asarray(jnp.inf)
    else:
        extension.state["value"] = jnp.asarray(jnp.inf)
    # The generic storage layer can contain numerical failure evidence. The training
    # owner must refuse it as continuation even when its metadata and dtypes match.
    with OrbaxCheckpointStore(tmp_path, max_to_keep=None) as store:
        store.save(
            1,
            {
                "model": nnx.state(model),
                "rng": jax.random.key(3),
                "data_iterator": pipeline.get_state(),
                "extensions": {
                    "step": nnx.state(GradientStep(optimizer, _objective)),
                    "extra_state": {
                        "callbacks": callbacks.get_state(),
                        "statistics": extension.get_state(),
                    },
                },
            },
        )
    fresh = _objects(1)
    fresh[0].statistic = nnx.Variable(jnp.asarray(5.0))
    fresh_extension = _StateHolder(jnp.asarray(2.0))
    run = replace(
        _run_state(fresh, jax.random.key(7)),
        extra_state={"callbacks": fresh[3], "statistics": fresh_extension},
    )
    before = [np.array(x) for x in jax.tree.leaves(nnx.state((fresh[0], fresh[1])))]
    data_before, callbacks_before = fresh[2].get_state(), fresh[3].get_state()

    with pytest.raises(FloatingPointError, match="non-finite"):
        Checkpointer(CheckpointPolicy(directory=tmp_path)).restore(run)

    for actual, expected in zip(
        jax.tree.leaves(nnx.state((fresh[0], fresh[1]))), before, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert fresh[2].get_state() == data_before
    assert fresh[3].get_state() == callbacks_before
    np.testing.assert_array_equal(fresh_extension.state["value"], 2.0)


def test_saved_data_position_requires_a_data_object_before_applying_state(tmp_path: Path) -> None:
    checkpointer = Checkpointer(CheckpointPolicy(directory=tmp_path))
    checkpointer.save(1, _run_state(_objects(0), jax.random.key(0)))
    fresh = _objects(1)
    kernel = np.array(fresh[0].kernel[...])

    with pytest.raises(ValueError, match="data"):
        checkpointer.restore(replace(_run_state(fresh, jax.random.key(1)), data=None))

    np.testing.assert_array_equal(fresh[0].kernel[...], kernel)


class _CountingStep(nnx.Module):
    """A native step whose optimizer and non-optimizer state must both resume."""

    def __init__(self, optimizer: nnx.Optimizer[Any]) -> None:
        self.update = GradientStep(optimizer, _objective)
        self.calls = nnx.Variable(jnp.asarray(0, dtype=jnp.int32))

    def __call__(
        self, model: nnx.Linear, batch: dict[str, jax.Array], context: StepContext
    ) -> dict[str, jax.Array]:
        result = self.update(model, batch, context)
        self.calls[...] += 1
        return result


@pytest.mark.parametrize("additional_optimizer", [False, True])
def test_step_owned_state_resumes_with_the_model_and_next_update(
    tmp_path: Path, additional_optimizer: bool
) -> None:
    source, fresh = _objects(0), _objects(1)
    step, fresh_step = _CountingStep(source[1]), _CountingStep(fresh[1])
    key = jax.random.key(11)
    batch = {"x": jnp.ones((3, 2)), "y": jnp.zeros((3, 1))}
    if additional_optimizer:
        step.second = nnx.Optimizer(source[0], optax.adam(0.02), wrt=nnx.Param)
        fresh_step.second = nnx.Optimizer(fresh[0], optax.adam(0.02), wrt=nnx.Param)
        step.second.update(source[0], nnx.grad(lambda m: jnp.sum(m.kernel[...]))(source[0]))

    def update(
        bundle: tuple[nnx.Linear, _CountingStep],
        batch: dict[str, jax.Array],
        context: StepContext,
    ) -> dict[str, jax.Array]:
        model, owned_step = bundle
        return owned_step(model, batch, context)

    live = nnx.jit_partial(
        update, (source[0], step), graph=True, graph_updates=False, donate_argnums=0
    )
    resumed = nnx.jit_partial(
        update, (fresh[0], fresh_step), graph=True, graph_updates=False, donate_argnums=0
    )
    source_run = RunState(model=source[0], step=step, base_key=key, epoch=0)
    fresh_run = RunState(model=fresh[0], step=fresh_step, base_key=key, epoch=0)
    live(batch, StepContext.at(key, 0))
    writer = Checkpointer(CheckpointPolicy(directory=tmp_path))
    writer.save(1, source_run)

    writer.restore(fresh_run)
    assert int(fresh_step.calls[...]) == 1
    live(batch, StepContext.at(key, 1))
    resumed(batch, StepContext.at(key, 1))

    assert int(fresh_step.calls[...]) == 2
    for actual, expected in zip(
        jax.tree.leaves(nnx.state((fresh[0], fresh_step))),
        jax.tree.leaves(nnx.state((source[0], step))),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


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
