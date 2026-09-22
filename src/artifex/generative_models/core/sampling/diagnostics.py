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
    chains support (Vehtari et al. 2021, Bayesian Analysis 16(2), 667).

    A chain whose draws alternate around the mean carries more information than its length,
    so the result is deliberately not capped at the number of draws. It is bounded by
    ``draws * log10(draws)``, the guard Vehtari et al. describe in section 3.2 and Stan and
    ArviZ apply as well: the estimator itself becomes erratic once single-lag correlations
    stay large while the paired ones vanish, not because more draws are impossible.

    A component whose draws never move is worth no draws and reads 0.0, and one holding a NaN
    is undefined and reads NaN.

    Args:
        samples: Draws of shape ``(draws, *parameters)``, or of
            ``(chains, draws, *parameters)`` when ``chain_axis`` is given.
        chain_axis: The axis holding separate chains; ``None`` treats ``samples`` as one.
        sample_axis: The axis holding the draws of a chain.

    Returns:
        The effective sample size of each parameter, shaped like one draw: 0.0 where the
        draws never move, NaN where they hold one.

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
    ess = jnp.reshape(ess, parameters)

    # Geyer's truncation gates on partial sums being positive, and every such comparison is
    # False for a component that never moves or holds a NaN. The truncation collapses, the
    # estimate falls back to the guard below and reports a large fictitious sample size:
    # blackjax 1.6.2 returns 6602 for 2000 constant draws. BlackJAX repairs this on main
    # (blackjax-devs/blackjax, `is_numerically_degenerate` and `has_nan` in diagnostics.py);
    # until that release, the two components are answered here.
    axes = (chain_axis, sample_axis)
    unmoving = ~(jnp.var(chains, axis=axes) > 0)
    undefined = jnp.any(jnp.isnan(chains), axis=axes)
    ess = jnp.where(jnp.reshape(unmoving, parameters), 0.0, ess)
    return jnp.where(jnp.reshape(undefined, parameters), jnp.nan, ess)
