# BlackJAX Samplers

**Module:** `artifex.generative_models.core.sampling.blackjax_samplers`

**Source:** `src/artifex/generative_models/core/sampling/blackjax_samplers.py`

## Overview

Artifex keeps BlackJAX as the canonical MCMC engine. The Artifex helper layer is
now intentionally thin: it prepares scalar joint log-density callables, manages
burn-in and thinning loops, and requires explicit RNG ownership from the caller.

Pass an explicit JAX key or `nnx.Rngs` to every public sampling helper. The
wrapper layer does not fabricate fallback keys, and the sampler constructors do
not own hidden RNG state.

## Public Helper Contract

- `hmc_sampling(...)` exposes live HMC controls: `step_size`,
  `num_integration_steps`, `inverse_mass_matrix`, and `thinning`.
- `nuts_sampling(...)` exposes live NUTS controls: `step_size`,
  `inverse_mass_matrix`, and `thinning`.
- `mala_sampling(...)` exposes live MALA controls: `step_size` and `thinning`.
- Passing a `Distribution` uses its `log_prob(...)` output as the log-density
  owner and collapses non-scalar outputs to one scalar joint log probability.

## Class Contract

`BlackJAXHMC`, `BlackJAXNUTS`, and `BlackJAXMALA` are thin stateful wrappers over
BlackJAX kernels. They prepare kernels and states, but they do not accept hidden
constructor-level RNG knobs. Sampling keys are supplied explicitly to `init(...)`
and advanced through `step(...)`.

## When To Use Direct BlackJAX

Use the direct BlackJAX API when you need engine-specific tuning or richer
per-step diagnostics than the Artifex helper layer publishes. Use the Artifex
helpers when the thinner public surface is sufficient and you want a fully owned
burn-in and thinning loop.
