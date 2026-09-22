"""Chain diagnostics for the samplers, over BlackJAX's estimators.

BlackJAX is the only MCMC dependency, and every call into it lives in this package, so a
consumer reads a chain's diagnostics from artifex rather than importing BlackJAX itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from blackjax.diagnostics import effective_sample_size as _blackjax_effective_sample_size


# The estimator needs a few draws before its autocorrelations mean anything.
_MINIMUM_DRAWS = 4


def effective_sample_size(
    samples: jax.Array, *, chain_axis: int | None = None, sample_axis: int = 0
) -> jax.Array:
    """The number of independent draws the chains are worth, per parameter.

    ``blackjax.diagnostics.effective_sample_size``: the draw count over the integrated
    autocorrelation time, summed by Geyer's initial positive sequence over every lag the
    chains support (Vehtari et al. 2021, Bayesian Analysis 16(2), 667). A chain whose draws
    alternate around the mean is worth more than its length, so the result is not capped at
    the number of draws.

    Args:
        samples: Draws of shape ``(draws, *parameters)``, or of
            ``(chains, draws, *parameters)`` when ``chain_axis`` is given.
        chain_axis: The axis holding separate chains; ``None`` treats ``samples`` as one.
        sample_axis: The axis holding the draws of a chain.

    Returns:
        The effective sample size of each parameter, shaped like one draw.

    Raises:
        ValueError: If the chains hold fewer than four draws.
    """
    chains = samples if chain_axis is not None else samples[jnp.newaxis, ...]
    chain_axis, sample_axis = (0, 1) if chain_axis is None else (chain_axis, sample_axis)
    if chains.shape[sample_axis] < _MINIMUM_DRAWS:
        msg = (
            f"effective sample size needs at least {_MINIMUM_DRAWS} draws, "
            f"got {chains.shape[sample_axis]}"
        )
        raise ValueError(msg)
    parameters = tuple(
        size for axis, size in enumerate(chains.shape) if axis not in {chain_axis, sample_axis}
    )
    ess = _blackjax_effective_sample_size(chains, chain_axis=chain_axis, sample_axis=sample_axis)
    # BlackJAX squeezes a parameter axis of length one along with the chain and draw axes.
    return jnp.reshape(ess, parameters)
