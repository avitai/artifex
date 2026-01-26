# REINFORCE Trainer

**Module:** `artifex.generative_models.training.rl.reinforce`

The REINFORCE Trainer implements the basic policy gradient algorithm with variance reduction through return normalization and entropy bonus for exploration.

## Overview

REINFORCE is the simplest policy gradient algorithm, computing gradient updates based on discounted returns. This implementation includes:

- **Discounted Returns**: Efficient backward pass computation
- **Return Normalization**: Variance reduction for stable training
- **Entropy Bonus**: Encourages exploration and prevents premature convergence

## Quick Start

```python
from artifex.generative_models.training import (
    REINFORCEConfig,
    REINFORCETrainer,
)
from flax import nnx
import optax

# Create model and optimizer
model = PolicyModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

# Configure REINFORCE
config = REINFORCEConfig(
    gamma=0.99,
    normalize_returns=True,
    entropy_coeff=0.01,
)

trainer = REINFORCETrainer(model, optimizer, config)

# Training step
batch = {
    "observations": observations,
    "actions": actions,
    "rewards": rewards,
}
metrics = trainer.train_step(batch)
```

## Configuration

::: artifex.generative_models.training.rl.configs.REINFORCEConfig
    options:
      show_root_heading: true
      members_order: source

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma` | `float` | `0.99` | Discount factor for computing returns |
| `normalize_returns` | `bool` | `True` | Normalize returns for variance reduction |
| `entropy_coeff` | `float` | `0.01` | Coefficient for entropy bonus |

## Algorithm

REINFORCE computes the policy gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t \right]$$

Where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the discounted return from time step $t$.

### Variance Reduction

With `normalize_returns=True`, returns are normalized:

$$\hat{G}_t = \frac{G_t - \mu_G}{\sigma_G + \epsilon}$$

### Entropy Bonus

The entropy bonus encourages exploration:

$$\mathcal{L} = -\mathbb{E}[\log \pi(a|s) \cdot G] - \lambda_H H(\pi)$$

Where $H(\pi) = -\sum \pi(a|s) \log \pi(a|s)$ is the policy entropy.

## API Reference

::: artifex.generative_models.training.rl.reinforce.REINFORCETrainer
    options:
      show_root_heading: true
      members_order: source

## Training Metrics

| Metric | Description |
|--------|-------------|
| `policy_loss` | Policy gradient loss (negated for minimization) |

## Use Cases

REINFORCE is best suited for:

- **Simple baselines**: Quick experiments before more sophisticated methods
- **Low-dimensional action spaces**: Works well when action space is small
- **Research**: Understanding policy gradient fundamentals

For more stable training, consider [PPO](ppo.md) or [GRPO](grpo.md).

## Related Documentation

- [RL Training Guide](../user-guide/training/rl-training.md) - Comprehensive RL training guide
- [PPO Trainer](ppo.md) - More stable policy gradient training
- [GRPO Trainer](grpo.md) - Memory-efficient critic-free RL
