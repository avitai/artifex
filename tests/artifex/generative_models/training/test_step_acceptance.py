"""Numerical rejection preserves the next accepted optimizer trajectory."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import ExtensionConfig
from artifex.generative_models.core.configuration.optimizer_config import OptimizerConfig
from artifex.generative_models.core.configuration.training_config import TrainingConfig
from artifex.generative_models.extensions.base import Extension
from artifex.generative_models.training.callbacks import BaseCallback, CallbackList, TrainerLike
from artifex.generative_models.training.trainer import Trainer


class Scalar(nnx.Module):
    """Minimal native model whose objective has a known numerical failure."""

    def __init__(self, value: float) -> None:
        self.weight = nnx.Param(jnp.asarray(value, jnp.float32))


def config() -> TrainingConfig:
    """Use one explicit constant-rate optimizer configuration."""
    return TrainingConfig(
        name="finite-probe",
        optimizer=OptimizerConfig(name="probe", optimizer_type="sgd", learning_rate=0.01),
        batch_size=1,
        num_epochs=1,
    )


def assert_same(left: Any, right: Any) -> None:
    """Compare complete numerical state, including typed RNG keys."""
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for prior, current in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        assert jnp.array_equal(prior, current)


@pytest.mark.parametrize("case", ["loss", "gradient", "optimizer_state", "update", "parameter"])
def test_invalid_step_leaves_complete_state_unchanged(case: str) -> None:
    model = Scalar(3e38 if case == "parameter" else 1.0)

    def objective(
        model: Scalar, batch: dict[str, Any], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, Any]]:
        del batch, rng, step
        p = model.weight[...]
        if case == "loss":
            value = p + jnp.inf
        elif case == "gradient":
            value = jnp.sqrt(p - p)
        elif case == "optimizer_state":
            value = p * 1e30
        elif case == "update":
            value = p * 2
        else:
            value = -p
        return value, {}

    optimizer = (
        optax.adam(1e-3)
        if case == "optimizer_state"
        else optax.sgd(3e38 if case in ("update", "parameter") else 0.01)
    )
    trainer = Trainer(model, config(), optimizer=optimizer, loss_fn=objective)
    before = jax.tree.map(np.array, trainer.checkpoint_state())
    with pytest.raises(FloatingPointError):
        trainer.train_step({})
    assert trainer.step == 0
    assert trainer.train_metrics == []
    for prior, current in zip(
        jax.tree.leaves(before), jax.tree.leaves(trainer.checkpoint_state()), strict=True
    ):
        np.testing.assert_array_equal(prior, current)


class StatefulScalar(Scalar):
    """A model with persistent forward state and a typed random stream."""

    def __init__(self) -> None:
        super().__init__(1.0)
        self.seen = nnx.BatchStat(jnp.asarray(0.0))
        self.rngs = nnx.Rngs(5)

    def __call__(self) -> jax.Array:
        self.seen[...] += 1
        return self.weight[...] + jax.random.uniform(self.rngs())


class StatefulExtension(Extension):
    """An auxiliary loss that updates its own running state."""

    def __init__(self) -> None:
        super().__init__(ExtensionConfig(name="stateful"), rngs=nnx.Rngs(7))
        self.seen = nnx.Variable(jnp.asarray(0.0))

    def loss_fn(self, batch: dict[str, Any], outputs: Any) -> jax.Array:
        """Update running state without contributing a parameter gradient."""
        del outputs
        self.seen[...] += batch["state_increment"]
        return jax.random.uniform(self._rngs()) * 0.01


def stateful_trainer() -> Trainer:
    """Build a reproducible stochastic objective with mutable model and extension state."""

    def objective(
        model: StatefulScalar, batch: dict[str, Any], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, Any]]:
        del step
        value = model() + jax.random.uniform(rng)
        return jnp.square(value - batch["target"]), {}

    return Trainer(
        StatefulScalar(),
        config(),
        optimizer=optax.adam(0.01),
        loss_fn=objective,
        extensions={"stateful": StatefulExtension()},
    )


@pytest.mark.parametrize("failure", ["loss", "mutable_state"])
def test_repeated_rejection_preserves_mutable_state_and_next_update(failure: str) -> None:
    """Retries do not consume model or trainer randomness, moments or mutable statistics."""
    trainer = stateful_trainer()
    reference = stateful_trainer()
    good = {"target": jnp.asarray(0.0), "state_increment": jnp.asarray(1.0)}
    trainer.train_step(good)
    reference.train_step(good)
    before = jax.tree.map(jnp.array, trainer.checkpoint_state())
    bad = {**good, "target" if failure == "loss" else "state_increment": jnp.asarray(jnp.inf)}
    for _ in range(2):
        with pytest.raises(FloatingPointError):
            trainer.train_step(bad)
        assert_same(before, trainer.checkpoint_state())
        assert trainer.step == 1
        assert len(trainer.train_metrics) == 1
    assert trainer.train_step(good) == reference.train_step(good)
    assert_same(trainer.checkpoint_state(), reference.checkpoint_state())


class RecordingCallback(BaseCallback):
    """Record attempt and acceptance notifications without altering numerical state."""

    def __init__(self) -> None:
        self.begins: list[int] = []
        self.ends: list[int] = []

    def on_batch_begin(self, trainer: TrainerLike, batch: int) -> None:
        """Record each attempt, including rejected ones."""
        del trainer
        self.begins.append(batch)

    def on_batch_end(self, trainer: TrainerLike, batch: int, logs: dict[str, Any]) -> None:
        """Record accepted updates only."""
        del trainer, logs
        self.ends.append(batch)


def test_success_matches_known_sgd_and_only_accepted_steps_are_logged() -> None:
    """An invalid auxiliary metric is diagnostic; the numerical update must be finite."""

    def objective(
        model: Scalar, batch: dict[str, Any], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, Any]]:
        del rng, step
        return 2 * model.weight[...] + batch["offset"], {"undefined_metric": jnp.nan}

    callback = RecordingCallback()
    logs: list[int] = []

    def log(step: int, metrics: dict[str, Any], *, prefix: str) -> None:
        del metrics
        assert prefix == "train"
        logs.append(step)

    trainer = Trainer(
        Scalar(1.0),
        config(),
        optimizer=optax.sgd(0.01),
        loss_fn=objective,
        callbacks=CallbackList([callback]),
        log_callback=log,
    )
    with pytest.raises(FloatingPointError, match="loss"):
        trainer.train_step({"offset": jnp.asarray(jnp.inf)})
    assert callback.begins == [0]
    assert callback.ends == logs == []
    trainer.train_step({"offset": jnp.asarray(0.0)})
    np.testing.assert_allclose(trainer.model.weight[...], 0.98, rtol=1e-6)
    assert callback.begins == [0, 0]
    assert callback.ends == logs == [1]
    assert trainer.train_metrics[0]["loss"] == 2.0


def test_nonfinite_input_state_is_rejected_even_if_unused_by_loss() -> None:
    """A forward pass must not silently repair a corrupted checkpoint."""
    trainer = stateful_trainer()
    trainer.model.seen[...] = jnp.inf
    with pytest.raises(FloatingPointError, match="input_state"):
        trainer.train_step({"target": jnp.asarray(0.0), "state_increment": jnp.asarray(1.0)})
    assert trainer.step == 0
    assert jnp.isinf(trainer.model.seen[...])


def test_objective_cannot_silently_change_static_graph_structure() -> None:
    """A changed graph must fail before any live numerical state is committed."""

    def objective(
        model: Scalar, batch: dict[str, Any], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, Any]]:
        del batch, rng, step
        model.new_static = "changed"
        return model.weight[...], {}

    trainer = Trainer(Scalar(1.0), config(), loss_fn=objective)
    before = jax.tree.map(jnp.array, trainer.checkpoint_state())
    with pytest.raises(ValueError, match="graph structure"):
        trainer.train_step({})
    assert not hasattr(trainer.model, "new_static")
    assert_same(before, trainer.checkpoint_state())
