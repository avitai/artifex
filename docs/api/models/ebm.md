# Energy-Based Models API Reference

Complete API documentation for energy-based models (EBMs) in Artifex.

!!! info "Coming Soon"
    Energy-based model implementations are planned for a future release. This documentation will be updated when the feature is available.

## Overview

Energy-based models will include:

- **Score Matching**: Noise-contrastive estimation
- **Contrastive Divergence**: CD-k training algorithms
- **Stein Variational Gradient Descent**: Particle-based inference
- **Langevin Dynamics**: MCMC-based sampling

## Planned API

### Base Class

```python
from artifex.generative_models.models.ebm.base import EnergyBasedModel

model = EnergyBasedModel(
    config: EBMConfig,
    *,
    rngs: nnx.Rngs,
)
```

### Configuration

```python
from artifex.generative_models.core.configuration import EBMConfig

config = EBMConfig(
    name="ebm_model",
    energy_net_hidden_dims=(256, 128),
    mcmc_steps=10,
    step_size=0.01,
)
```

### Energy Function

```python
# Define energy function
energy = model.energy(x)  # Lower = more likely

# Compute log probability
log_prob = -energy
```

### Sampling

```python
# Sample using Langevin dynamics
samples = model.sample(
    n_samples=100,
    n_steps=1000,
    step_size=0.01,
)
```

## Related Documentation

- [Energy-Based Models Guide](../../user-guide/models/ebm-guide.md) - Conceptual overview
- [EBM Concepts](../../user-guide/concepts/ebm-explained.md) - Understanding energy-based modeling
- [MCMC Sampling](../../api/sampling.md) - Sampling algorithms for EBMs

## References

- LeCun et al., "A Tutorial on Energy-Based Learning" (2006)
- Du & Mordatch, "Implicit Generation and Modeling with Energy-Based Models" (2019)
- Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution" (2019)
