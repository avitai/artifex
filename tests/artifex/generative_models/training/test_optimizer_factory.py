"""The optimizer factory maps ``OptimizerConfig`` onto the optimizer substrax builds.

``create_optimizer(model, config, schedule=None)`` returns the optax transformation
``substrax.optim.create_transformation`` builds for the model, with ``schedule`` (or the
configuration's rate) as the optimizer's learning rate, so a decaying schedule reaches the
update; what optax would silently misread is refused by the configuration or by substrax.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from substrax.optim import current_learning_rate

from artifex.generative_models.core.configuration import OptimizerConfig
from artifex.generative_models.training import create_optimizer as exported_create_optimizer
from artifex.generative_models.training.optimizers import create_optimizer


class _Model(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.linear = nnx.Linear(4, 4, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


def _model() -> _Model:
    return _Model(rngs=nnx.Rngs(0))


def _loss(model: _Model) -> jax.Array:
    return jnp.mean(model(jnp.ones((2, 4))) ** 2)


def _config(**overrides: object) -> OptimizerConfig:
    fields: dict[str, object] = {"name": "test", "optimizer_type": "adam", "learning_rate": 1e-3}
    fields.update(overrides)
    return OptimizerConfig(**fields)  # type: ignore[arg-type]


class TestEveryOptimizerBuilds:
    """Each optimizer type substrax offers builds and moves a parameter."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"optimizer_type": "adam"},
            {"optimizer_type": "adamw", "weight_decay": 0.01},
            {"optimizer_type": "sgd", "momentum": 0.9},
            {"optimizer_type": "rmsprop"},
            {"optimizer_type": "adagrad"},
            {"optimizer_type": "lamb", "weight_decay": 0.01},
            {"optimizer_type": "radam"},
            {"optimizer_type": "nadam"},
        ],
        ids=lambda o: o["optimizer_type"],
    )
    def test_moves_the_parameters(self, overrides: dict[str, object]) -> None:
        model = _model()
        transformation = create_optimizer(model, _config(learning_rate=0.1, **overrides))
        optimizer = nnx.Optimizer(model, transformation, wrt=nnx.Param)
        before = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

        optimizer.update(model, nnx.grad(_loss)(model))

        after = nnx.state(model, nnx.Param)
        assert not all(
            jnp.array_equal(a, b)
            for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)
        )


class TestScheduleIsTheLearningRate:
    """``schedule`` overrides the configuration's rate and reaches the update."""

    def test_schedule_overrides_the_configured_rate(self) -> None:
        model = _model()
        optimizer = nnx.Optimizer(
            model, create_optimizer(model, _config(), schedule=5e-4), wrt=nnx.Param
        )

        assert float(current_learning_rate(optimizer)) == pytest.approx(5e-4)

    def test_without_a_schedule_the_configured_rate_applies(self) -> None:
        model = _model()
        optimizer = nnx.Optimizer(
            model, create_optimizer(model, _config(learning_rate=2e-3)), wrt=nnx.Param
        )

        assert float(current_learning_rate(optimizer)) == pytest.approx(2e-3)

    def test_a_schedule_decayed_to_zero_stops_the_parameters(self) -> None:
        model = _model()
        schedule = optax.linear_schedule(init_value=0.1, end_value=0.0, transition_steps=4)
        optimizer = nnx.Optimizer(
            model, create_optimizer(model, _config(), schedule=schedule), wrt=nnx.Param
        )
        for _ in range(4):
            optimizer.update(model, nnx.grad(_loss)(model))
        before = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

        optimizer.update(model, nnx.grad(_loss)(model))

        after = nnx.state(model, nnx.Param)
        assert float(current_learning_rate(optimizer)) == pytest.approx(0.0)
        assert all(
            jnp.array_equal(a, b)
            for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)
        )


class TestClipping:
    """The clip fields reach the transformation."""

    def test_clip_by_global_norm_bounds_the_update(self) -> None:
        model = _model()
        transformation = create_optimizer(
            model, _config(optimizer_type="sgd", learning_rate=1.0, gradient_clip_norm=1.0)
        )
        params = nnx.to_pure_dict(nnx.state(model, nnx.Param))
        grads = jax.tree.map(lambda p: jnp.full_like(p, 100.0), params)

        updates, _ = transformation.update(grads, transformation.init(params), params)

        norm = jnp.sqrt(sum(jnp.sum(u**2) for u in jax.tree.leaves(updates)))
        assert float(norm) == pytest.approx(1.0, rel=1e-5)

    def test_clip_by_value_bounds_each_entry(self) -> None:
        model = _model()
        transformation = create_optimizer(
            model, _config(optimizer_type="sgd", learning_rate=1.0, gradient_clip_value=0.5)
        )
        params = nnx.to_pure_dict(nnx.state(model, nnx.Param))
        grads = jax.tree.map(lambda p: jnp.full_like(p, 100.0), params)

        updates, _ = transformation.update(grads, transformation.init(params), params)

        largest = max(float(jnp.max(jnp.abs(u))) for u in jax.tree.leaves(updates))
        assert largest == pytest.approx(0.5)


class TestRefusals:
    """What optax would silently misread is refused."""

    def test_weight_decay_on_adam_is_refused(self) -> None:
        with pytest.raises(ValueError, match="weight_decay"):
            create_optimizer(_model(), _config(weight_decay=0.01))

    def test_fields_substrax_does_not_carry_are_gone(self) -> None:
        with pytest.raises(TypeError):
            _config(optimizer_type="sgd", nesterov=True)
        with pytest.raises(TypeError):
            _config(optimizer_type="adagrad", initial_accumulator_value=0.5)


class TestOptimizerFactoryIntegrationWithTrainer:
    """A factory-built transformation drives the trainer."""

    def test_trainer_can_use_factory_created_optimizer(self) -> None:
        from artifex.generative_models.core.configuration import TrainingConfig
        from artifex.generative_models.training.trainer import Trainer

        model = _model()
        optimizer_config = _config(name="test_optimizer")
        training_config = TrainingConfig(
            name="test_training", optimizer=optimizer_config, batch_size=4, num_epochs=2
        )

        def loss_fn(model, batch, rng, step):
            del rng, step
            loss = jnp.mean(model(batch["x"]) ** 2)
            return loss, {"loss": loss}

        trainer = Trainer(
            model=model,
            training_config=training_config,
            optimizer=create_optimizer(model, optimizer_config),
            loss_fn=loss_fn,
        )

        metrics = trainer.train_step({"x": jax.random.normal(jax.random.key(0), (4, 4))})

        assert jnp.isfinite(metrics["loss"])


def test_create_optimizer_is_exported_from_the_training_package() -> None:
    assert exported_create_optimizer is create_optimizer
