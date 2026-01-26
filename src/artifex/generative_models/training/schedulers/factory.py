"""Learning rate scheduler factory for unified configuration."""

import optax

from artifex.generative_models.core.configuration import SchedulerConfig


def create_scheduler(scheduler_config: SchedulerConfig, base_lr: float) -> optax.Schedule:
    """Create learning rate scheduler from SchedulerConfig.

    Args:
        scheduler_config: Scheduler configuration
        base_lr: Base learning rate

    Returns:
        Optax learning rate schedule
    """
    if scheduler_config.scheduler_type == "constant" or scheduler_config.scheduler_type == "none":
        return optax.constant_schedule(base_lr)

    elif scheduler_config.scheduler_type == "linear":
        if scheduler_config.total_steps is None:
            raise ValueError("total_steps must be specified for linear scheduler")

        if scheduler_config.warmup_steps > 0:
            return optax.join_schedules(
                schedules=[
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=base_lr,
                        transition_steps=scheduler_config.warmup_steps,
                    ),
                    optax.linear_schedule(
                        init_value=base_lr,
                        end_value=base_lr * scheduler_config.min_lr_ratio,
                        transition_steps=scheduler_config.total_steps
                        - scheduler_config.warmup_steps,
                    ),
                ],
                boundaries=[scheduler_config.warmup_steps],
            )
        else:
            return optax.linear_schedule(
                init_value=base_lr,
                end_value=base_lr * scheduler_config.min_lr_ratio,
                transition_steps=scheduler_config.total_steps,
            )

    elif scheduler_config.scheduler_type == "cosine":
        decay_steps = scheduler_config.cycle_length
        if decay_steps is None:
            raise ValueError("cycle_length must be specified for cosine scheduler")

        if scheduler_config.warmup_steps > 0:
            return optax.join_schedules(
                schedules=[
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=base_lr,
                        transition_steps=scheduler_config.warmup_steps,
                    ),
                    optax.cosine_decay_schedule(
                        init_value=base_lr,
                        decay_steps=decay_steps - scheduler_config.warmup_steps,
                        alpha=scheduler_config.min_lr_ratio,
                    ),
                ],
                boundaries=[scheduler_config.warmup_steps],
            )
        else:
            return optax.cosine_decay_schedule(
                init_value=base_lr,
                decay_steps=decay_steps,
                alpha=scheduler_config.min_lr_ratio,
            )

    elif scheduler_config.scheduler_type == "exponential":
        return optax.exponential_decay(
            init_value=base_lr,
            transition_steps=scheduler_config.decay_steps,
            decay_rate=scheduler_config.decay_rate,
        )

    elif scheduler_config.scheduler_type == "polynomial":
        if scheduler_config.total_steps is None:
            raise ValueError("total_steps must be specified for polynomial scheduler")

        return optax.polynomial_schedule(
            init_value=base_lr,
            end_value=base_lr * scheduler_config.min_lr_ratio,
            power=1.0,  # Linear by default
            transition_steps=scheduler_config.total_steps,
        )

    elif scheduler_config.scheduler_type == "step":
        # Create boundaries for step scheduler
        boundaries_and_scales = {}
        for i in range(1, 20):  # Support up to 20 steps
            step_boundary = scheduler_config.step_size * i
            boundaries_and_scales[step_boundary] = scheduler_config.gamma

        return optax.piecewise_constant_schedule(
            init_value=base_lr,
            boundaries_and_scales=boundaries_and_scales,
        )

    elif scheduler_config.scheduler_type == "multistep":
        if not scheduler_config.milestones:
            raise ValueError("milestones must be specified for multistep scheduler")

        boundaries_and_scales = {
            milestone: scheduler_config.gamma for milestone in sorted(scheduler_config.milestones)
        }

        return optax.piecewise_constant_schedule(
            init_value=base_lr,
            boundaries_and_scales=boundaries_and_scales,
        )

    elif scheduler_config.scheduler_type == "cyclic":
        # Simple cyclic schedule with triangular shape
        if scheduler_config.cycle_length is None:
            raise ValueError("cycle_length must be specified for cyclic scheduler")

        # Use a triangular wave pattern
        def cyclic_schedule(count):
            cycle_position = count % scheduler_config.cycle_length
            cycle_ratio = cycle_position / scheduler_config.cycle_length

            if cycle_ratio < 0.5:
                # Ascending phase
                return base_lr * scheduler_config.min_lr_ratio + (
                    base_lr - base_lr * scheduler_config.min_lr_ratio
                ) * (2 * cycle_ratio)
            else:
                # Descending phase
                return base_lr - (base_lr - base_lr * scheduler_config.min_lr_ratio) * (
                    2 * (cycle_ratio - 0.5)
                )

        return cyclic_schedule

    elif scheduler_config.scheduler_type == "one_cycle":
        # One cycle policy
        if scheduler_config.total_steps is None:
            raise ValueError("total_steps must be specified for one_cycle scheduler")

        # Split into 3 phases: warmup (30%), annealing (60%), cooldown (10%)
        warmup_steps = int(0.3 * scheduler_config.total_steps)
        annealing_steps = int(0.6 * scheduler_config.total_steps)
        cooldown_steps = scheduler_config.total_steps - warmup_steps - annealing_steps

        return optax.join_schedules(
            schedules=[
                # Warmup phase
                optax.linear_schedule(
                    init_value=base_lr * scheduler_config.min_lr_ratio,
                    end_value=base_lr,
                    transition_steps=warmup_steps,
                ),
                # Annealing phase
                optax.cosine_decay_schedule(
                    init_value=base_lr,
                    decay_steps=annealing_steps,
                    alpha=scheduler_config.min_lr_ratio,
                ),
                # Cooldown phase
                optax.linear_schedule(
                    init_value=base_lr * scheduler_config.min_lr_ratio,
                    end_value=base_lr * scheduler_config.min_lr_ratio * 0.1,
                    transition_steps=cooldown_steps,
                ),
            ],
            boundaries=[warmup_steps, warmup_steps + annealing_steps],
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_config.scheduler_type}")
