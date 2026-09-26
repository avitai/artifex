"""The training step: one gradient update of the trained parameters, compiled once.

``GradientStep`` differentiates an objective with respect to the parameters its optimizer
trains, updates them, keeps any other state the objective changes (batch statistics), and
returns device metrics: the objective's, the loss, the global gradient norm and, for an
optimizer built through substrax, the learning rate it applied. ``StepContext`` carries the
step, a key derived from it and host-set scalars as device values, so every step runs one
compiled program and a resumed run draws the keys the uninterrupted run drew.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax import monitoring
from substrax.optim import create_optimizer, OptimizerConfig

from artifex.generative_models.training.step import GradientStep, StepContext


_BACKEND_COMPILE = "/jax/core/compile/backend_compile_duration"


class _CompileLog:
    """Names every backend compile, so a count is of programs built, not of Python calls."""

    def __init__(self) -> None:
        self.names: list[str | None] = []

    def __call__(self, event: str, _duration: float, **kwargs: Any) -> None:
        if event == _BACKEND_COMPILE:
            self.names.append(kwargs.get("fun_name"))


_COMPILES = _CompileLog()
monitoring.register_event_duration_secs_listener(_COMPILES)


class _Regressor(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(3, 1, rngs=rngs)
        self.frozen = nnx.Variable(jnp.ones((3,)))
        self.norm = nnx.BatchNorm(3, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(self.norm(x * self.frozen[...]))


def _batch(seed: int = 0) -> dict[str, jax.Array]:
    rng = np.random.default_rng(seed)
    return {
        "x": jnp.asarray(rng.normal(size=(8, 3)), dtype=jnp.float32),
        "y": jnp.asarray(rng.normal(size=(8, 1)), dtype=jnp.float32),
    }


def _mse(
    model: _Regressor, batch: dict[str, jax.Array], ctx: StepContext
) -> tuple[jax.Array, dict[str, jax.Array]]:
    error = model(batch["x"]) - batch["y"]
    loss = jnp.mean(error**2)
    return loss, {"max_error": jnp.max(jnp.abs(error))}


def _sgd_setup(rate: float = 0.1) -> tuple[_Regressor, nnx.Optimizer[Any]]:
    model = _Regressor(rngs=nnx.Rngs(0))
    return model, nnx.Optimizer(model, optax.sgd(rate), wrt=nnx.Param)


def _reference_grads(model: _Regressor, batch: dict[str, jax.Array], ctx: StepContext) -> nnx.State:
    """The trained parameters' gradients, taken on a copy so the model itself is untouched."""
    copy = nnx.clone(model)

    def loss(m: _Regressor) -> jax.Array:
        return _mse(m, batch, ctx)[0]

    return nnx.grad(loss, argnums=nnx.DiffState(0, nnx.Param))(copy)


def test_the_context_derives_its_key_from_the_step() -> None:
    base = jax.random.key(7)

    first = StepContext.at(base, 3)
    again = StepContext.at(base, 3)
    later = StepContext.at(base, 4)

    assert first.step.dtype == jnp.int32
    assert jnp.array_equal(jax.random.key_data(first.key), jax.random.key_data(again.key))
    assert not jnp.array_equal(jax.random.key_data(first.key), jax.random.key_data(later.key))
    assert jnp.array_equal(
        jax.random.key_data(first.key), jax.random.key_data(jax.random.fold_in(base, 3))
    )


def test_a_function_of_the_context_compiles_once_across_steps() -> None:
    base = jax.random.key(0)

    @jax.jit
    def read(ctx: StepContext) -> jax.Array:
        return ctx.step + ctx.scalars["weight"] + jax.random.uniform(ctx.key)

    mark = len(_COMPILES.names)
    for step in range(4):
        read(StepContext.at(base, step, scalars={"weight": 0.5}))

    assert _COMPILES.names[mark:].count("jit(read)") == 1


def test_the_step_applies_the_update_a_hand_written_one_applies() -> None:
    model, optimizer = _sgd_setup()
    batch, ctx = _batch(), StepContext.at(jax.random.key(0), 0)
    grads = _reference_grads(model, batch, ctx)
    expected_kernel = model.linear.kernel[...] - 0.1 * grads["linear"]["kernel"][...]

    metrics = GradientStep(_mse)(model, optimizer, batch, ctx)

    np.testing.assert_allclose(model.linear.kernel[...], expected_kernel, rtol=1e-6, atol=1e-7)
    assert set(metrics) >= {"loss", "max_error", "grad_norm"}


def test_the_grad_norm_is_the_global_norm_of_the_trained_parameters_gradients() -> None:
    model, optimizer = _sgd_setup()
    batch, ctx = _batch(), StepContext.at(jax.random.key(0), 0)
    expected = optax.tree.norm(_reference_grads(model, batch, ctx))

    metrics = GradientStep(_mse)(model, optimizer, batch, ctx)

    np.testing.assert_allclose(metrics["grad_norm"], expected, rtol=1e-6)


def test_only_the_trained_parameters_change_and_other_state_is_kept() -> None:
    model, optimizer = _sgd_setup()
    frozen = model.frozen[...]
    mean_before = model.norm.mean[...]

    GradientStep(_mse)(model, optimizer, _batch(), StepContext.at(jax.random.key(0), 0))

    assert jnp.array_equal(model.frozen[...], frozen)
    assert not jnp.array_equal(model.norm.mean[...], mean_before)


def test_the_learning_rate_is_reported_for_a_substrax_optimizer() -> None:
    model = _Regressor(rngs=nnx.Rngs(0))
    schedule = optax.linear_schedule(0.1, 0.0, transition_steps=10)
    optimizer = create_optimizer(
        model, OptimizerConfig(optimizer_type="sgd", learning_rate=schedule)
    )
    step = GradientStep(_mse)

    rates = [
        float(
            step(model, optimizer, _batch(), StepContext.at(jax.random.key(0), i))["learning_rate"]
        )
        for i in range(3)
    ]

    np.testing.assert_allclose(rates, [0.1, 0.09, 0.08], rtol=1e-6)


def test_a_plain_optax_optimizer_reports_no_learning_rate() -> None:
    model, optimizer = _sgd_setup()

    metrics = GradientStep(_mse)(model, optimizer, _batch(), StepContext.at(jax.random.key(0), 0))

    assert "learning_rate" not in metrics


def test_the_step_compiles_once_in_tree_mode_with_model_and_optimizer_donated() -> None:
    model, optimizer = _sgd_setup()
    eager_model, eager_optimizer = _sgd_setup()
    step = GradientStep(_mse)
    base = jax.random.key(0)

    @nnx.jit(graph=False, graph_updates=False, donate_argnums=(0, 1))
    def train_step(
        model: _Regressor,
        optimizer: nnx.Optimizer[Any],
        batch: dict[str, jax.Array],
        ctx: StepContext,
    ) -> dict[str, jax.Array]:
        return step(model, optimizer, batch, ctx)

    mark = len(_COMPILES.names)
    for i in range(3):
        train_step(model, optimizer, _batch(i), StepContext.at(base, i))
        step(eager_model, eager_optimizer, _batch(i), StepContext.at(base, i))

    assert _COMPILES.names[mark:].count("jit(train_step)") == 1
    _assert_float32_close(model.linear.kernel[...], eager_model.linear.kernel[...])


def _assert_float32_close(actual: jax.Array, expected: jax.Array) -> None:
    """Compiled and eager float32 results agree to a few ULP at the values' scale."""
    reference = np.asarray(expected, dtype=np.float64)
    atol = 8 * np.finfo(np.float32).eps * max(1.0, float(np.abs(reference).max()))
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), reference, rtol=0, atol=atol)


def test_steps_fuse_into_one_program_under_nnx_scan() -> None:
    model, optimizer = _sgd_setup()
    eager_model, eager_optimizer = _sgd_setup()
    step = GradientStep(_mse)
    base = jax.random.key(0)
    batches = [_batch(i) for i in range(3)]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *batches)

    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    def run(
        carry: tuple[_Regressor, nnx.Optimizer[Any]], batch: dict[str, jax.Array], index: jax.Array
    ) -> tuple[tuple[_Regressor, nnx.Optimizer[Any]], jax.Array]:
        model, optimizer = carry
        metrics = step(model, optimizer, batch, StepContext(step=index, base_key=base))
        return (model, optimizer), metrics["loss"]

    _, losses = run((model, optimizer), stacked, jnp.arange(3, dtype=jnp.int32))
    eager_losses = [
        step(eager_model, eager_optimizer, batch, StepContext.at(base, i))["loss"]
        for i, batch in enumerate(batches)
    ]

    assert losses.shape == (3,)
    _assert_float32_close(losses, jnp.stack(eager_losses))
    _assert_float32_close(model.linear.kernel[...], eager_model.linear.kernel[...])


def test_an_ensemble_steps_as_one_program_under_nnx_vmap() -> None:
    keys = jax.random.split(jax.random.key(0), 3)

    @nnx.vmap(in_axes=0, out_axes=0)
    def create(key: jax.Array) -> tuple[_Regressor, nnx.Optimizer[Any]]:
        model = _Regressor(rngs=nnx.Rngs(key))
        return model, nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)

    models, optimizers = create(keys)
    step = GradientStep(_mse)
    batch, ctx = _batch(), StepContext.at(jax.random.key(1), 0)

    @nnx.vmap(in_axes=(0, 0, None, None), out_axes=0)
    def ensemble_step(
        model: _Regressor,
        optimizer: nnx.Optimizer[Any],
        batch: dict[str, jax.Array],
        ctx: StepContext,
    ) -> dict[str, jax.Array]:
        return step(model, optimizer, batch, ctx)

    losses = ensemble_step(models, optimizers, batch, ctx)["loss"]
    member = _Regressor(rngs=nnx.Rngs(keys[1]))
    member_loss = step(member, nnx.Optimizer(member, optax.sgd(0.1), wrt=nnx.Param), batch, ctx)[
        "loss"
    ]

    assert losses.shape == (3,)
    _assert_float32_close(losses[1], member_loss)
    _assert_float32_close(models.linear.kernel[...][1], member.linear.kernel[...])


@pytest.mark.parametrize("step", [-1, 2**31])
def test_a_step_outside_the_int32_range_is_refused(step: int) -> None:
    with pytest.raises(ValueError, match=r"step"):
        StepContext.at(jax.random.key(0), step)
