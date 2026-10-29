"""Public Trainer numerical admission and native component integration."""

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from substrax.callbacks import BaseCallback, CallbackList

from artifex.generative_models.core.configuration import OptimizerConfig, TrainingConfig
from artifex.generative_models.training.trainer import Trainer


class ScalarModel(nnx.Module):
    """A single parameter with an independently known SGD trajectory."""

    def __init__(self) -> None:
        """Start at one, so squared loss has gradient two."""
        self.weight = nnx.Param(jnp.asarray(1.0))


def test_nonfinite_loss_is_reported_before_the_following_update() -> None:
    """A finite-gradient infinite loss must survive deferred observation."""
    model = ScalarModel()

    def objective(
        model: ScalarModel, batch: dict[str, jax.Array], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del batch, rng
        loss = model.weight[...] ** 2 + jnp.where(step == 0, jnp.inf, 0.0)
        return loss, {}

    trainer = Trainer(
        model,
        TrainingConfig(
            name="finite-admission",
            optimizer=OptimizerConfig(name="sgd", optimizer_type="sgd", learning_rate=0.1),
        ),
        loss_fn=objective,
    )
    metrics = trainer.train_step({})
    assert isinstance(metrics["loss"], jax.Array)
    assert float(model.weight[...]) == pytest.approx(0.8)

    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.train_step({})
    assert trainer.step == 1
    assert float(model.weight[...]) == pytest.approx(0.8)


def test_native_optimizer_filter_and_checkpoint_resume_are_one_trajectory() -> None:
    """A caller's optimizer owns its filter and all restored update state."""
    model = ScalarModel()
    model.frozen = nnx.Param(jnp.asarray(3.0))
    optimizer = nnx.Optimizer(
        model, optax.adamw(0.1, weight_decay=0.1), wrt=nnx.PathContains("weight")
    )

    def objective(
        model: ScalarModel, batch: dict[str, jax.Array], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del batch, step
        return model.weight[...] ** 2 + model.frozen[...] + jax.random.uniform(rng), {}

    trainer = Trainer(
        model,
        TrainingConfig(
            name="native-resume",
            optimizer=OptimizerConfig(name="unused", optimizer_type="sgd", learning_rate=0.1),
        ),
        optimizer=optimizer,
        loss_fn=objective,
    )
    assert trainer.optimizer is optimizer
    with pytest.raises(AttributeError):
        trainer.optimizer = nnx.Optimizer(model, optax.sgd(0.9), wrt=nnx.Param)
    trainer.train_step({})
    saved = jax.tree.map(jnp.array, trainer.checkpoint_state())
    expected_metrics = trainer.train_step({})
    expected = jax.tree.map(jnp.array, trainer.checkpoint_state())
    trainer.apply_checkpoint_state(saved, step=1)
    actual_metrics = trainer.train_step({})
    actual = trainer.checkpoint_state()
    assert float(model.frozen[...]) == 3.0
    assert float(actual_metrics["loss"]) == float(expected_metrics["loss"])
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert bool(jnp.array_equal(left, right))


def test_async_checkpoint_captures_the_step_before_donation_and_restores_objective(
    tmp_path,
) -> None:
    """The public lifecycle retains the full native step and waits on shutdown."""

    class Objective(nnx.Module):
        def __init__(self) -> None:
            self.calls = nnx.Variable(jnp.asarray(0, jnp.int32))

        def __call__(
            self, model: ScalarModel, batch: dict[str, jax.Array], rng: jax.Array, step: jax.Array
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            del batch, rng, step
            self.calls[...] += 1
            return model.weight[...] ** 2, {}

    objective = Objective()
    with Trainer(
        ScalarModel(),
        TrainingConfig(
            name="async-native",
            optimizer=OptimizerConfig(name="adam", optimizer_type="adam", learning_rate=0.1),
            checkpoint_dir=tmp_path / "checkpoints",
        ),
        loss_fn=objective,
    ) as trainer:
        trainer.train_step({})
        path = trainer.save_async()
        trainer.train_step({})
        trainer.wait_until_finished()
        assert path.is_dir()
        trainer.load_checkpoint(step=1)
        assert int(objective.calls[...]) == 1
        assert trainer.step == 1
        trainer.train_step({})
        assert int(objective.calls[...]) == 2


def test_final_failed_attempt_is_drained_before_epoch_completion() -> None:
    """A one-attempt run must not succeed merely because there is no next step."""

    def objective(
        model: ScalarModel, batch: dict[str, jax.Array], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del batch, rng, step
        return model.weight[...] ** 2 + jnp.inf, {}

    trainer = Trainer(
        ScalarModel(),
        TrainingConfig(
            name="terminal-admission",
            optimizer=OptimizerConfig(name="sgd", optimizer_type="sgd", learning_rate=0.1),
        ),
        train_data_loader=lambda _: iter([{}]),
        loss_fn=objective,
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.train_epoch(steps=1)


def test_a_later_success_cannot_erase_a_skipped_update_in_a_window() -> None:
    """Failed gradients hold the optimizer clock but attempts continue until observation."""
    model = ScalarModel()
    model.weight[...] = jnp.asarray(0.0)

    def objective(
        model: ScalarModel, batch: dict[str, jax.Array], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del batch, rng
        # sqrt has an infinite derivative at zero. The other attempts have
        # zero gradient, so the parameter remains zero and the losses stay finite.
        return jax.lax.cond(step == 1, lambda x: jnp.sqrt(x), lambda x: x**2, model.weight[...]), {}

    trainer = Trainer(
        model,
        TrainingConfig(
            name="window-admission",
            optimizer=OptimizerConfig(name="sgd", optimizer_type="sgd", learning_rate=0.1),
        ),
        loss_fn=objective,
        check_finite_every=3,
    )
    for _ in range(3):
        assert bool(jnp.isfinite(trainer.train_step({})["loss"]))
    assert trainer.step == 3
    assert int(trainer.optimizer.step[...]) == 3
    assert int(trainer.optimizer.opt_state.inner_state.count[...]) == 2
    assert bool(trainer.optimizer.opt_state.last_finite[...])
    assert int(trainer.optimizer.opt_state.total_notfinite[...]) == 1
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.train_step({})
    assert trainer.step == 3


def test_overflowing_diagnostic_norm_does_not_reject_finite_gradients() -> None:
    """Native gradient admission and a float32 norm diagnostic are different predicates."""
    model = ScalarModel()
    model.weight[...] = jnp.asarray(1e-20, jnp.float32)

    def objective(
        model: ScalarModel, batch: dict[str, jax.Array], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del batch, rng, step
        return model.weight[...] * jnp.asarray(1e20, jnp.float32), {}

    trainer = Trainer(
        model,
        TrainingConfig(
            name="finite-large-gradient",
            optimizer=OptimizerConfig(name="sgd", optimizer_type="sgd", learning_rate=1e-30),
        ),
        loss_fn=objective,
    )
    metrics = trainer.train_step({})
    assert bool(jnp.isinf(metrics["grad_norm"]))
    assert bool(jnp.isfinite(metrics["loss"]))
    trainer.check_finite()
    assert float(model.weight[...]) == pytest.approx(-1e-10, rel=1e-5, abs=1e-15)


def test_recovery_clears_failed_observation_but_failed_saves_write_nothing(tmp_path: Path) -> None:
    """A finite parameter tree cannot authorize publishing an unobserved bad loss."""

    def objective(
        model: ScalarModel, batch: dict[str, jax.Array], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del rng, step
        return model.weight[...] ** 2 + jnp.where(batch["bad"], jnp.inf, 0.0), {}

    with Trainer(
        ScalarModel(),
        TrainingConfig(
            name="recover-admission",
            optimizer=OptimizerConfig(name="sgd", optimizer_type="sgd", learning_rate=0.1),
            checkpoint_dir=tmp_path / "checkpoints",
        ),
        loss_fn=objective,
    ) as trainer:
        trainer.train_step({"bad": jnp.asarray(False)})
        first = trainer.save_checkpoint()
        trainer.train_step({"bad": jnp.asarray(True)})
        with pytest.raises(FloatingPointError, match="non-finite"):
            trainer.save_checkpoint()
        assert not (first.parent / "2").exists()
        trainer.load_checkpoint(step=1)
        trainer.train_step({"bad": jnp.asarray(False)})
        trainer.check_finite()
        assert trainer.step == 2


def test_objective_can_share_rngs_and_parameters_without_training_its_reference() -> None:
    """One differentiated graph preserves aliases and excludes objective-only Params."""
    model = ScalarModel()
    model.rngs = nnx.Rngs(7)

    class Objective(nnx.Module):
        def __init__(self) -> None:
            self.rngs = model.rngs
            self.shared_weight = model.weight
            self.reference = nnx.Param(jnp.asarray(7.0))

        def __call__(
            self, current: ScalarModel, batch: dict[str, jax.Array], rng: jax.Array, step: jax.Array
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            del batch, rng, step
            self.rngs()
            current.rngs()
            return (current.weight[...] + self.shared_weight[...]) ** 2 + self.reference[...], {}

    objective = Objective()
    trainer = Trainer(
        model,
        TrainingConfig(
            name="shared-objective",
            optimizer=OptimizerConfig(name="sgd", optimizer_type="sgd", learning_rate=0.1),
        ),
        loss_fn=objective,
    )
    assert int(model.rngs.default.count[...]) == 0
    trainer.train_step({})
    assert float(model.weight[...]) == pytest.approx(0.2)
    assert float(objective.reference[...]) == 7.0
    assert objective.shared_weight is model.weight
    assert objective.rngs is model.rngs
    assert int(model.rngs.default.count[...]) == 2


def test_a_failed_training_run_closes_its_native_checkpoint_lifecycle(tmp_path: Path) -> None:
    """An exception exit must not retain a writable native checkpoint owner."""

    def objective(
        model: ScalarModel, batch: dict[str, jax.Array], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del batch, rng, step
        return model.weight[...] ** 2 + jnp.inf, {}

    trainer = Trainer(
        ScalarModel(),
        TrainingConfig(
            name="error-shutdown",
            optimizer=OptimizerConfig(name="sgd", optimizer_type="sgd", learning_rate=0.1),
            checkpoint_dir=tmp_path / "checkpoints",
        ),
        loss_fn=objective,
    )
    saved = trainer.save_async()
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.train({"x": jnp.ones((1, 1))}, num_epochs=1, batch_size=1)
    assert saved.is_dir()
    with pytest.raises(RuntimeError, match="closed"):
        trainer.train_step({})


def test_evaluation_does_not_change_the_next_training_key() -> None:
    """Validation uses its own persisted RNG stream rather than consuming training keys."""

    def objective(
        model: ScalarModel, batch: dict[str, jax.Array], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del batch, step
        return model.weight[...] ** 2 + model.weight[...] * jax.random.uniform(rng), {}

    config = TrainingConfig(
        name="rng-separation",
        optimizer=OptimizerConfig(name="sgd", optimizer_type="sgd", learning_rate=0.1),
    )
    first = Trainer(ScalarModel(), config, loss_fn=objective)
    control = Trainer(ScalarModel(), config, loss_fn=objective)
    first.train_step({})
    control.apply_checkpoint_state(jax.tree.map(jnp.array, first.checkpoint_state()), step=1)
    first.evaluate({"x": jnp.ones((3, 1))}, batch_size=2)
    first.validate_step({"x": jnp.ones((2, 1))})
    actual = first.train_step({})
    expected = control.train_step({})
    assert float(actual["loss"]) == float(expected["loss"])
    assert float(first.model.weight[...]) == float(control.model.weight[...])


@pytest.mark.parametrize("initial_deterministic", [False, True])
def test_native_mode_views_preserve_the_callers_flags(initial_deterministic: bool) -> None:
    """Training and evaluation modes share Variables without mutating caller settings."""

    class Model(ScalarModel):
        def __init__(self) -> None:
            super().__init__()
            self.dropout = nnx.Dropout(0.5, deterministic=initial_deterministic, rngs=nnx.Rngs(7))

        def __call__(self, inputs: jax.Array) -> jax.Array:
            return self.weight[...] * self.dropout(inputs)

    def objective(
        model: ScalarModel, batch: dict[str, jax.Array], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del rng, step
        prediction = model(batch["x"])
        return jnp.mean(prediction), {"stochastic": jnp.any(prediction != model.weight[...])}

    model = Model()
    trainer = Trainer(
        model,
        TrainingConfig(
            name="native-views",
            optimizer=OptimizerConfig(name="sgd", optimizer_type="sgd", learning_rate=0.1),
        ),
        loss_fn=objective,
    )
    data = {"x": jnp.ones((8, 1))}
    evaluated = trainer.evaluate(data, batch_size=3)
    assert evaluated == {"loss": 1.0, "stochastic": 0.0}
    trained = trainer.train_step(data)
    assert bool(trained["stochastic"])
    assert model.dropout.deterministic is initial_deterministic


def test_numerical_steps_do_not_emit_driver_callbacks() -> None:
    """The driver owns host callbacks, both addressed by the pre-update attempt index."""
    events: list[tuple[str, int]] = []

    class Callback(BaseCallback):
        def on_batch_begin(self, trainer: object, batch_idx: int) -> None:
            del trainer
            events.append(("begin", batch_idx))

        def on_batch_end(self, trainer: object, batch_idx: int, logs: dict) -> None:
            del trainer, logs
            events.append(("end", batch_idx))

    def objective(
        model: ScalarModel, batch: dict[str, jax.Array], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del batch, rng, step
        return model.weight[...] ** 2, {}

    trainer = Trainer(
        ScalarModel(),
        TrainingConfig(
            name="driver-callbacks",
            optimizer=OptimizerConfig(name="sgd", optimizer_type="sgd", learning_rate=0.1),
        ),
        loss_fn=objective,
        callbacks=CallbackList([Callback()]),
    )
    trainer.train_step({})
    assert events == []
    trainer.train({"x": jnp.ones((1, 1))}, num_epochs=1, batch_size=1)
    assert events == [("begin", 1), ("end", 1)]


def test_nonfinite_validation_is_refused_before_epoch_selection() -> None:
    """A healthy training step cannot authorize selection using an infinite validation loss."""
    epochs: list[int] = []

    class Callback(BaseCallback):
        def on_epoch_end(self, trainer: object, epoch: int, logs: dict) -> None:
            del trainer, logs
            epochs.append(epoch)

    def objective(
        model: ScalarModel, batch: dict[str, jax.Array], rng: jax.Array, step: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del rng, step
        return model.weight[...] ** 2 + jnp.where(jnp.any(batch["bad"]), jnp.inf, 0.0), {}

    trainer = Trainer(
        ScalarModel(),
        TrainingConfig(
            name="validation-admission",
            optimizer=OptimizerConfig(name="sgd", optimizer_type="sgd", learning_rate=0.1),
        ),
        loss_fn=objective,
        callbacks=CallbackList([Callback()]),
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.train(
            {"bad": jnp.zeros(2, jnp.bool_)},
            num_epochs=1,
            batch_size=2,
            val_data={"bad": jnp.ones(2, jnp.bool_)},
        )
    assert epochs == []
