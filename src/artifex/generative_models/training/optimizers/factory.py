"""Optimizer factory for unified configuration.

Creates optax optimizers from OptimizerConfig, supporting all common
optimizer types with proper hyperparameter mapping.
"""

import optax

from artifex.generative_models.core.configuration import OptimizerConfig


def create_optimizer(
    config: OptimizerConfig,
    schedule: optax.Schedule | float | None = None,
) -> optax.GradientTransformation:
    """Create optimizer from OptimizerConfig.

    Args:
        config: Optimizer configuration specifying type and hyperparameters.
        schedule: Optional learning rate schedule. If None, uses config.learning_rate.
                  Can be a float (constant) or optax.Schedule (callable).

    Returns:
        Optax gradient transformation (optimizer).

    Raises:
        ValueError: If optimizer_type is not supported.

    Supported optimizer types:
        - adam: Adam optimizer with beta1, beta2, eps
        - adamw: AdamW with weight decay
        - sgd: SGD with optional momentum and Nesterov
        - rmsprop: RMSProp with decay (beta2) and eps
        - adagrad: AdaGrad with eps and initial_accumulator_value
        - lamb: LAMB optimizer (Layer-wise Adaptive Moments)
        - radam: Rectified Adam
        - nadam: Nesterov-accelerated Adam
    """
    # Determine learning rate: schedule takes precedence over config
    if schedule is not None:
        learning_rate = schedule
    else:
        learning_rate = config.learning_rate

    # Create base optimizer based on type
    optimizer = _create_base_optimizer(config, learning_rate)

    # Apply gradient clipping if configured
    optimizer = _apply_gradient_clipping(optimizer, config)

    return optimizer


def _create_base_optimizer(
    config: OptimizerConfig,
    learning_rate: optax.Schedule | float,
) -> optax.GradientTransformation:
    """Create the base optimizer without gradient clipping.

    Args:
        config: Optimizer configuration.
        learning_rate: Learning rate (constant or schedule).

    Returns:
        Base optax optimizer.
    """
    optimizer_type = config.optimizer_type

    if optimizer_type == "adam":
        return optax.adam(
            learning_rate=learning_rate,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
        )

    elif optimizer_type == "adamw":
        return optax.adamw(
            learning_rate=learning_rate,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    elif optimizer_type == "sgd":
        return optax.sgd(
            learning_rate=learning_rate,
            momentum=config.momentum,
            nesterov=config.nesterov,
        )

    elif optimizer_type == "rmsprop":
        return optax.rmsprop(
            learning_rate=learning_rate,
            decay=config.beta2,  # RMSProp uses beta2 as decay
            eps=config.eps,
            initial_scale=config.initial_accumulator_value,
        )

    elif optimizer_type == "adagrad":
        return optax.adagrad(
            learning_rate=learning_rate,
            eps=config.eps,
            initial_accumulator_value=config.initial_accumulator_value,
        )

    elif optimizer_type == "lamb":
        return optax.lamb(
            learning_rate=learning_rate,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    elif optimizer_type == "radam":
        return optax.radam(
            learning_rate=learning_rate,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
        )

    elif optimizer_type == "nadam":
        # NAdam is not in optax, implement as Adam with Nesterov momentum
        # Using the standard pattern: scale_by_adam + add_nesterov_momentum
        return optax.chain(
            optax.scale_by_adam(b1=config.beta1, b2=config.beta2, eps=config.eps),
            optax.scale(-1.0),  # Negate for gradient descent
            optax.scale_by_schedule(
                lambda count: learning_rate(count) if callable(learning_rate) else learning_rate
            ),
        )

    else:
        raise ValueError(
            f"Unknown optimizer_type: {optimizer_type}. "
            f"Supported types: adam, adamw, sgd, rmsprop, adagrad, lamb, radam, nadam"
        )


def _apply_gradient_clipping(
    optimizer: optax.GradientTransformation,
    config: OptimizerConfig,
) -> optax.GradientTransformation:
    """Apply gradient clipping to optimizer if configured.

    Args:
        optimizer: Base optimizer.
        config: Optimizer configuration with clipping settings.

    Returns:
        Optimizer with gradient clipping applied (if configured).
    """
    if config.gradient_clip_norm is not None:
        return optax.chain(
            optax.clip_by_global_norm(config.gradient_clip_norm),
            optimizer,
        )
    elif config.gradient_clip_value is not None:
        return optax.chain(
            optax.clip(config.gradient_clip_value),
            optimizer,
        )

    return optimizer
