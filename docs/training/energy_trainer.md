# Energy Trainer

**Status:** `Supported runtime training surface`

**Module:** `artifex.generative_models.training.trainers.energy_trainer`

**Source:** `src/artifex/generative_models/training/trainers/energy_trainer.py`

`EnergyTrainer` implements Contrastive Divergence, Persistent Contrastive
Divergence, and score-matching training with Langevin negative-sample updates.

## Quick Start

```python
from flax import nnx
import jax
import optax

from artifex.generative_models.training.trainers import (
    EnergyTrainer,
    EnergyTrainingConfig,
)

model = create_energy_model(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)
trainer = EnergyTrainer(
    EnergyTrainingConfig(
        training_method="pcd",
        mcmc_steps=20,
        step_size=0.01,
        noise_scale=0.005,
    )
)

key = jax.random.key(0)
loss, metrics = trainer.train_step(model, optimizer, batch, key)
```

## Configuration

::: artifex.generative_models.training.trainers.energy_trainer.EnergyTrainingConfig
    options:
      show_root_heading: true
      members_order: source

### Runtime-Active Fields

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training_method` | `"cd"` | One of `cd`, `pcd`, or `score_matching` |
| `mcmc_steps` | `20` | Number of Langevin updates for negative samples |
| `step_size` | `0.01` | Langevin step size |
| `noise_scale` | `0.005` | Noise multiplier in Langevin updates |
| `gradient_clipping` | `1.0` | Clip norm applied to Langevin input gradients |
| `replay_buffer_size` | `10000` | Replay-buffer capacity for `pcd` |
| `replay_buffer_init_prob` | `0.95` | Probability of drawing PCD starts from the replay buffer |
| `energy_regularization` | `0.0` | Optional energy-value penalty |
| `gradient_penalty_weight` | `0.0` | Optional gradient penalty weight |

## Langevin Sampling

The negative-sample path is always Langevin based:

```python
negatives = trainer.run_mcmc_chain(model, x_init, key, num_steps=20)
```

The trainer does not expose alternate sampler families in its public config.

## Training Methods

### Contrastive Divergence

```python
EnergyTrainingConfig(training_method="cd", mcmc_steps=20)
```

### Persistent Contrastive Divergence

```python
EnergyTrainingConfig(
    training_method="pcd",
    mcmc_steps=20,
    replay_buffer_size=10000,
)
```

### Score Matching

```python
EnergyTrainingConfig(training_method="score_matching")
```

## Related Documentation

- [Training Systems](index.md)
- [Energy Models](../user-guide/models/ebm-guide.md)
