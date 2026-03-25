# Energy-Based Models API Reference

This page documents the live energy-model surface.

## Canonical Import

```python
from artifex.generative_models.models.energy import EnergyBasedModel
```

## Related Top-Level Exports

```python
from artifex.generative_models.models.energy import (
    CNNEnergyFunction,
    EBM,
    EnergyBasedModel,
    EnergyFunction,
    MLPEnergyFunction,
    create_cifar_ebm,
    create_mnist_ebm,
    create_simple_ebm,
    improved_langevin_dynamics,
    langevin_dynamics,
    persistent_contrastive_divergence,
)
```

## Base EBM Surface

`EnergyBasedModel` is the shared base for energy-model workflows.

- `energy(x)`
- `energy_outputs(x)`
- `unnormalized_log_prob(x)`
- `score(x)`
- `contrastive_divergence_loss(real_data, fake_data, alpha=0.01)`
- `loss_fn(batch, model_outputs, *, fake_samples, alpha=0.01, **kwargs)`

## Notes

- the live public package is `artifex.generative_models.models.energy`
- MCMC helpers such as `langevin_dynamics(...)` are exported from that same
  package alongside the base classes
- use family-owned energy functions or factory helpers rather than the dead
  historical `models.ebm.*` path
