"""The training step: an objective, one gradient update, device metrics.

A training step is the compiled unit of work. :class:`StepContext` carries what a step needs
besides the model and the batch -- the global step, the random key for it and scalars the host
sets (loss weights, temperatures) -- all as device values, so every step runs one compiled
program. The key is derived from a base key and the step on the device, so a resumed run draws
the keys the uninterrupted run drew and nothing random is checkpointed but the base key.

:class:`GradientStep` is the step of one model and one optimizer: it differentiates an
:data:`Objective` with respect to the parameters the optimizer trains, updates them, keeps any
other state the objective changed (batch statistics, RNG counts), and returns the objective's
metrics with the loss, the global gradient norm and, for an optimizer built through substrax,
the learning rate the update applied. Steps are ``nnx`` modules, so a step or an objective that
carries state (a replay buffer, a loss balancer) is part of the compiled call and of a checkpoint.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from substrax.optim import current_learning_rate


_INT32_LIMIT = 2**31


@dataclasses.dataclass(frozen=True, slots=True)
class StepContext:
    """The step, its random key and host-set scalars, as device values.

    Attributes:
        step: The global step, an ``int32`` scalar.
        base_key: The run's base key; :attr:`key` derives this step's key from it.
        scalars: Values the host sets for the objective, by name (``float32`` scalars).
    """

    step: jax.Array
    base_key: jax.Array
    scalars: Mapping[str, jax.Array] = dataclasses.field(default_factory=dict)

    @property
    def key(self) -> jax.Array:
        """This step's key: the base key folded with the step, computed where it is used."""
        return jax.random.fold_in(self.base_key, self.step)

    @classmethod
    def at(
        cls, base_key: jax.Array, step: int, scalars: Mapping[str, float] | None = None
    ) -> StepContext:
        """Build the context of ``step`` on the host, for a driver or a test.

        The Trainer keeps the step on the device and builds the context inside the compiled
        step; this builds the same context from host values.

        Args:
            base_key: The run's base key.
            step: The global step, in ``[0, 2**31)``.
            scalars: Host values for the objective, stored as ``float32`` scalars.

        Returns:
            The context.

        Raises:
            ValueError: If ``step`` is outside the ``int32`` range the counter holds.
        """
        if not 0 <= step < _INT32_LIMIT:
            raise ValueError(f"step must be in [0, 2**31), got {step}")
        return cls(
            step=jnp.asarray(step, dtype=jnp.int32),
            base_key=base_key,
            scalars={
                name: jnp.asarray(value, dtype=jnp.float32)
                for name, value in (scalars or {}).items()
            },
        )


jax.tree_util.register_dataclass(
    StepContext, data_fields=["step", "base_key", "scalars"], meta_fields=[]
)

type Objective = Callable[
    [Any, Mapping[str, Any], StepContext], tuple[jax.Array, dict[str, jax.Array]]
]
"""``(model, batch, ctx) -> (loss, metrics)``: the scalar loss and named device metrics."""


@runtime_checkable
class Step(Protocol):
    """One compiled update: ``(model, batch, ctx) -> metrics``, owning its update state."""

    def __call__(
        self,
        model: nnx.Module,
        batch: Mapping[str, Any],
        ctx: StepContext,
        /,
    ) -> dict[str, jax.Array]:
        """Update ``model`` through the step's own state; return device metrics."""
        ...


class GradientStep(nnx.Module):
    """One model, one optimizer: differentiate the objective and apply the update."""

    def __init__(self, optimizer: nnx.Optimizer[Any], objective: Objective) -> None:
        """Bind ``objective`` to the optimizer this step owns.

        Args:
            optimizer: The optimizer over the model, retained as native NNX state.
            objective: The loss and metrics of the model on a batch.
        """
        self.optimizer = optimizer
        self.objective = objective

    def __call__(
        self,
        model: nnx.Module,
        batch: Mapping[str, Any],
        ctx: StepContext,
    ) -> dict[str, jax.Array]:
        """Apply one update of the parameters the owned optimizer trains.

        Args:
            model: The model; its trained parameters are updated and any other state the
                objective changes is kept.
            batch: The batch.
            ctx: The step's context.

        Returns:
            The objective's metrics with ``loss``, ``grad_norm`` (the global norm of the
            trained parameters' gradients) and, when the optimizer carries it,
            ``learning_rate`` (the rate the update applied).
        """

        def loss_fn(model: nnx.Module) -> tuple[jax.Array, dict[str, jax.Array]]:
            return self.objective(model, batch, ctx)

        (loss, metrics), grads = nnx.value_and_grad(
            loss_fn, argnums=nnx.DiffState(0, self.optimizer.wrt), has_aux=True
        )(model)
        self.optimizer.update(model, grads)
        out = {**metrics, "loss": loss, "grad_norm": optax.tree.norm(grads)}
        rate = _applied_learning_rate(self.optimizer)
        if rate is not None:
            out["learning_rate"] = rate
        return out


def _applied_learning_rate(optimizer: nnx.Optimizer[Any]) -> jax.Array | None:
    """The rate the last update applied, or ``None`` for a transformation that does not record it.

    substrax's builder records it (``optax.inject_hyperparams``); a transformation built by hand
    may not, and the step still runs. The check is on the state's structure, decided once at
    trace time.
    """
    try:
        return current_learning_rate(optimizer)
    except ValueError:
        return None
