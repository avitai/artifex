"""Gradient accumulation and loss scaling come from optax and flax.

artifex used to ship ``GradientAccumulator`` and ``DynamicLossScaler``. Both
duplicated upstream-owned tools: ``optax.MultiSteps`` accumulates microbatch
gradients inside the optimizer, and ``flax.training.dynamic_scale.DynamicScale``
scales the loss, skips non-finite steps and adapts the scale. These tests pin the
semantics the artifex classes promised, on the upstream tools, so the training
docs can point at them with a checked example.
"""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from flax.training.dynamic_scale import DynamicScale


ACCUMULATION_STEPS = 4


def _model() -> nnx.Linear:
    return nnx.Linear(3, 2, rngs=nnx.Rngs(0))


def _microbatches() -> list[jax.Array]:
    return [jax.random.normal(jax.random.key(seed), (5, 3)) for seed in range(ACCUMULATION_STEPS)]


def _loss(model: nnx.Linear, x: jax.Array) -> jax.Array:
    return jnp.mean(model(x) ** 2)


def test_multisteps_matches_one_update_on_the_mean_gradient() -> None:
    """k microbatches through MultiSteps equal one update on the averaged gradient."""
    accumulated = _model()
    optimizer = nnx.Optimizer(
        accumulated,
        optax.MultiSteps(optax.adam(1e-2), every_k_schedule=ACCUMULATION_STEPS),
        wrt=nnx.Param,
    )
    grads = []
    for x in _microbatches():
        grad = nnx.grad(_loss)(accumulated, x)
        grads.append(grad)
        optimizer.update(accumulated, grad)

    reference = _model()
    reference_optimizer = nnx.Optimizer(reference, optax.adam(1e-2), wrt=nnx.Param)
    mean_grad = jax.tree.map(lambda *leaves: sum(leaves) / ACCUMULATION_STEPS, *grads)
    reference_optimizer.update(reference, mean_grad)

    assert jnp.allclose(accumulated.kernel[...], reference.kernel[...], atol=1e-7)
    assert jnp.allclose(accumulated.bias[...], reference.bias[...], atol=1e-7)


def test_multisteps_leaves_parameters_untouched_before_the_kth_microbatch() -> None:
    """The first k-1 updates are no-ops; the k-th applies the accumulated step."""
    model = _model()
    initial = model.kernel[...].copy()
    optimizer = nnx.Optimizer(
        model,
        optax.MultiSteps(optax.adam(1e-2), every_k_schedule=ACCUMULATION_STEPS),
        wrt=nnx.Param,
    )
    changed = []
    for x in _microbatches():
        optimizer.update(model, nnx.grad(_loss)(model, x))
        changed.append(not bool(jnp.allclose(model.kernel[...], initial)))

    assert changed == [False] * (ACCUMULATION_STEPS - 1) + [True]


def test_dynamic_scale_skips_non_finite_steps_and_adapts_the_scale() -> None:
    """DynamicScale grows once growth_interval finite steps are counted, halves on overflow."""
    model = _model()
    graphdef, params = nnx.split(model, nnx.Param)
    x = _microbatches()[0]
    dynamic_scale = DynamicScale(growth_interval=2)

    def finite_loss(p: nnx.State) -> jax.Array:
        return _loss(nnx.merge(graphdef, p), x)

    def overflowing_loss(p: nnx.State) -> jax.Array:
        return finite_loss(p) * jnp.float32(3e38) * jnp.float32(3e38)

    initial_scale = float(dynamic_scale.scale)

    dynamic_scale, is_finite, _, grads = dynamic_scale.value_and_grad(finite_loss)(params)
    assert bool(is_finite)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(grads))
    assert float(dynamic_scale.scale) == initial_scale
    assert int(dynamic_scale.fin_steps) == 1

    dynamic_scale, is_finite, _, _ = dynamic_scale.value_and_grad(finite_loss)(params)
    assert bool(is_finite)
    assert float(dynamic_scale.scale) == initial_scale
    assert int(dynamic_scale.fin_steps) == 2

    # The scale grows on the finite step taken once growth_interval finite steps
    # have been counted, and the counter restarts.
    dynamic_scale, is_finite, _, _ = dynamic_scale.value_and_grad(finite_loss)(params)
    assert bool(is_finite)
    assert float(dynamic_scale.scale) == initial_scale * 2
    assert int(dynamic_scale.fin_steps) == 0

    dynamic_scale, is_finite, _, _ = dynamic_scale.value_and_grad(overflowing_loss)(params)
    assert not bool(is_finite)
    assert float(dynamic_scale.scale) == initial_scale


def test_dynamic_scale_gradients_are_unscaled_before_the_optimizer_sees_them() -> None:
    """The gradients DynamicScale returns equal the plain gradient of the unscaled loss."""
    model = _model()
    graphdef, params = nnx.split(model, nnx.Param)
    x = _microbatches()[0]

    def loss(p: nnx.State) -> jax.Array:
        return _loss(nnx.merge(graphdef, p), x)

    _, is_finite, _, scaled_grads = DynamicScale().value_and_grad(loss)(params)
    plain_grads = jax.grad(loss)(params)

    assert bool(is_finite)
    for scaled, plain in zip(
        jax.tree.leaves(scaled_grads), jax.tree.leaves(plain_grads), strict=True
    ):
        assert jnp.allclose(scaled, plain, atol=1e-6)


def test_artifex_no_longer_ships_its_own_accumulator_or_scaler() -> None:
    """The duplicated classes and their module are gone from the training package."""
    training = importlib.import_module("artifex.generative_models.training")

    for name in (
        "GradientAccumulator",
        "GradientAccumulatorConfig",
        "DynamicLossScaler",
        "DynamicLossScalerConfig",
    ):
        assert not hasattr(training, name), name
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("artifex.generative_models.training.gradient_accumulation")
