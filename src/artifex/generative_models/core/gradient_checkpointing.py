"""Gradient checkpointing (rematerialization) utilities.

Provides architecture-agnostic wrappers around ``nnx.remat`` with named
checkpoint policies.  Any model — MLP, Transformer, WaveNet, CNN, etc. —
can import ``apply_remat`` to trade compute for memory during backprop.

Example::

    from artifex.generative_models.core.gradient_checkpointing import apply_remat

    remated_fn = apply_remat(layer_fn, policy="dots_saveable")
    output = remated_fn(x)
"""

from collections.abc import Callable
from typing import Any

import jax
from flax import nnx


# ---------------------------------------------------------------------------
# Named checkpoint policies
# ---------------------------------------------------------------------------
CHECKPOINT_POLICIES: dict[str, Callable[..., bool]] = {
    "dots_saveable": jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    "everything_saveable": jax.checkpoint_policies.everything_saveable,
    "nothing_saveable": jax.checkpoint_policies.nothing_saveable,
    "checkpoint_dots": jax.checkpoint_policies.checkpoint_dots,
    "checkpoint_dots_no_batch": jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
}


def resolve_checkpoint_policy(
    policy: str | Callable[..., bool] | None,
) -> Callable[..., bool] | None:
    """Resolve a checkpoint policy from a string name or callable.

    Args:
        policy: A named policy string, a callable policy, or ``None``.

    Returns:
        The resolved callable policy, or ``None`` if *policy* is ``None``.

    Raises:
        ValueError: If *policy* is a string not found in
            :data:`CHECKPOINT_POLICIES`.
    """
    if policy is None:
        return None
    if callable(policy):
        return policy
    if policy not in CHECKPOINT_POLICIES:
        raise ValueError(
            f"Unknown checkpoint policy: {policy!r}. Available: {list(CHECKPOINT_POLICIES)}"
        )
    return CHECKPOINT_POLICIES[policy]


def apply_remat(
    fn: Callable[..., Any],
    *,
    policy: str | Callable[..., bool] | None = None,
    prevent_cse: bool = True,
    static_argnums: tuple[int, ...] = (),
) -> Callable[..., Any]:
    """Wrap *fn* with ``nnx.remat`` for gradient checkpointing.

    This is a thin, architecture-agnostic wrapper.  It resolves a named
    policy string (if given) and delegates to ``nnx.remat``, which correctly
    handles NNX module state during recomputation.

    Args:
        fn: The function to wrap (typically a module's ``__call__``).
        policy: Checkpoint policy — a string name, callable, or ``None``.
        prevent_cse: Whether to prevent common sub-expression elimination.
        static_argnums: Indices of static arguments (forwarded to
            ``nnx.remat``).

    Returns:
        A wrapped callable with the same signature as *fn*.
    """
    resolved_policy = resolve_checkpoint_policy(policy)

    kwargs: dict[str, Any] = {"prevent_cse": prevent_cse}
    if resolved_policy is not None:
        kwargs["policy"] = resolved_policy
    if static_argnums:
        kwargs["static_argnums"] = static_argnums

    return nnx.remat(fn, **kwargs)
