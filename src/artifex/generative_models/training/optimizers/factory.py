"""The optimizer factory: from ``OptimizerConfig`` to the optimizer substrax builds.

``OptimizerConfig`` stays artifex's validated, user-facing configuration; the factory maps it
onto :class:`substrax.optim.OptimizerConfig` and returns the transformation
:func:`substrax.optim.create_transformation` builds for the model, with the schedule (or the
configured rate) as the optimizer's learning rate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from substrax.optim import create_transformation, MOMENTUM_TYPES, OptimizerConfig as OptimizerSpec

from artifex.generative_models.core.configuration import OptimizerConfig


if TYPE_CHECKING:
    import optax
    from flax import nnx


def create_optimizer(
    model: nnx.Module,
    config: OptimizerConfig,
    schedule: optax.Schedule | float | None = None,
) -> optax.GradientTransformation:
    """Create the optax transformation ``config`` describes for ``model``.

    Args:
        model: The model the optimizer will update; substrax reads its parameter tree.
        config: Optimizer configuration specifying type and hyperparameters.
        schedule: Optional learning rate, a float or an ``optax.Schedule``, that replaces
            ``config.learning_rate`` as the optimizer's learning rate.

    Returns:
        Optax gradient transformation (optimizer).

    Raises:
        ValueError: If substrax refuses the specification, for instance a weight decay on an
            optimizer without decoupled decay.
    """
    return create_transformation(model, optimizer_spec(config, schedule))


def optimizer_spec(
    config: OptimizerConfig, schedule: optax.Schedule | float | None = None
) -> OptimizerSpec:
    """Map ``config`` onto substrax's optimizer specification.

    ``momentum`` reaches only the optimizers that take it (``sgd``, ``rmsprop``), and a zero
    momentum means none.

    Args:
        config: Optimizer configuration specifying type and hyperparameters.
        schedule: Optional learning rate that replaces ``config.learning_rate``.

    Returns:
        The specification substrax builds.
    """
    momentum = (
        config.momentum
        if config.optimizer_type in MOMENTUM_TYPES and config.momentum > 0.0
        else None
    )
    return OptimizerSpec(
        optimizer_type=config.optimizer_type,  # type: ignore[arg-type]
        learning_rate=config.learning_rate if schedule is None else schedule,
        b1=config.beta1,
        b2=config.beta2,
        eps=config.eps,
        momentum=momentum,
        weight_decay=config.weight_decay,
        gradient_clip_norm=config.gradient_clip_norm,
        gradient_clip_value=config.gradient_clip_value,
    )
