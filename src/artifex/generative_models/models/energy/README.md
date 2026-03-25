# Energy-Based Models

This package provides typed-config and helper-function support for the Artifex
energy-model family.

## Main Runtime Surface

- `EBM`
- `DeepEBM`
- `EnergyFunction`
- `MLPEnergyFunction`
- `CNNEnergyFunction`
- `create_simple_ebm`
- `create_mnist_ebm`
- `create_cifar_ebm`

## Factory Usage

```python
from flax import nnx

from artifex.configs import (
    EBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)
from artifex.generative_models.factory import create_model

config = EBMConfig(
    name="mnist_ebm",
    input_dim=28 * 28,
    energy_network=EnergyNetworkConfig(
        name="mnist_energy_network",
        hidden_dims=(256, 128),
        network_type="mlp",
    ),
    mcmc=MCMCConfig(name="langevin", n_steps=60, step_size=0.01),
    sample_buffer=SampleBufferConfig(name="replay_buffer", capacity=8192),
    alpha=0.01,
)

model = create_model(config, rngs=nnx.Rngs(0))
```

## Helper Functions

Use the helper constructors when you want a ready-made EBM setup instead of
building the nested config explicitly:

```python
from flax import nnx

from artifex.generative_models.models.energy import create_mnist_ebm

model = create_mnist_ebm(rngs=nnx.Rngs(0))
```

## Sampling Notes

The EBM stack uses typed `MCMCConfig` and `SampleBufferConfig` objects to
control Langevin dynamics, replay-buffer behavior, and sampling temperature.
Those settings are part of the runtime config surface, not ad hoc kwargs hidden
behind a generic model config.
