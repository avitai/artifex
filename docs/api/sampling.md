# Sampling API Reference

This page documents the live top-level `artifex.generative_models.core.sampling`
package exports.

## Public Imports

```python
from artifex.generative_models.core.sampling import (
    BlackJAXHMC,
    BlackJAXMALA,
    BlackJAXNUTS,
    BlackJAXSamplerState,
    euler_maruyama_step,
    hmc_sampling,
    mala_sampling,
    mcmc_sampling,
    milstein_step,
    nuts_sampling,
    sde_sampling,
)
```

## Current Top-Level Surface

### BlackJAX wrappers

- `BlackJAXHMC`
- `BlackJAXMALA`
- `BlackJAXNUTS`
- `BlackJAXSamplerState`
- `hmc_sampling(...)`
- `mala_sampling(...)`
- `nuts_sampling(...)`

### Generic sampling helpers

- `mcmc_sampling(...)`
- `sde_sampling(...)`
- `euler_maruyama_step(...)`
- `milstein_step(...)`

## Submodule Note

Helpers that are not exported from the top-level package should be imported
from their owning submodules directly, for example `core.sampling.ancestral`
or `core.sampling.ode`.
