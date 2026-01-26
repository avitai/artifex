# PPO Trainer

**Module:** `artifex.generative_models.training.rl.ppo`

The PPO (Proximal Policy Optimization) Trainer provides stable policy gradient training through clipped surrogate objectives and Generalized Advantage Estimation (GAE).

## Overview

PPO is a state-of-the-art policy gradient method that maintains training stability through:

- **Clipped Surrogate Loss**: Prevents large policy updates
- **Generalized Advantage Estimation**: Balances bias-variance in advantage computation
- **Value Function Learning**: Learns state values for advantage estimation
- **Entropy Bonus**: Encourages exploration

## Quick Start

```python
from artifex.generative_models.training import PPOConfig, PPOTrainer
from flax import nnx
import optax

# Create actor-critic model
model = ActorCriticModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)

# Configure PPO
config = PPOConfig(
    gamma=0.99,
    gae_lambda=0.95,
    clip_param=0.2,
    vf_coeff=0.5,
    entropy_coeff=0.01,
    max_grad_norm=0.5,
)

trainer = PPOTrainer(model, optimizer, config)

# Training step with trajectory
trajectory = {
    "observations": observations,
    "actions": actions,
    "rewards": rewards,
    "values": values,
    "log_probs": old_log_probs,
    "dones": dones,
}
metrics = trainer.train_step(trajectory)
```

## Configuration

::: artifex.generative_models.training.rl.configs.PPOConfig
    options:
      show_root_heading: true
      members_order: source

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma` | `float` | `0.99` | Discount factor for GAE |
| `gae_lambda` | `float` | `0.95` | Lambda for GAE (bias-variance trade-off) |
| `clip_param` | `float` | `0.2` | Clipping parameter epsilon |
| `vf_coeff` | `float` | `0.5` | Value function loss coefficient |
| `entropy_coeff` | `float` | `0.01` | Entropy bonus coefficient |
| `max_grad_norm` | `float` | `0.5` | Maximum gradient norm for clipping |

## Algorithm

### Clipped Surrogate Objective

PPO uses a clipped surrogate objective to prevent large policy updates:

$$\mathcal{L}^{CLIP} = \mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio.

### Generalized Advantage Estimation

GAE computes advantages using TD residuals:

$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD residual.

**Lambda parameter:**

- $\lambda = 0$: TD(0) - low variance, high bias
- $\lambda = 1$: Monte Carlo - high variance, low bias
- $\lambda = 0.95$: Good balance (default)

### Value Function Loss

$$\mathcal{L}^{VF} = (V_\theta(s_t) - V_t^{target})^2$$

### Full Objective

$$\mathcal{L} = -\mathcal{L}^{CLIP} + c_1 \mathcal{L}^{VF} - c_2 H(\pi)$$

## API Reference

::: artifex.generative_models.training.rl.ppo.PPOTrainer
    options:
      show_root_heading: true
      members_order: source

## Training Metrics

| Metric | Description |
|--------|-------------|
| `policy_loss` | Clipped surrogate policy loss |
| `value_loss` | Value function MSE loss |
| `entropy` | Policy entropy (exploration measure) |

## Model Requirements

PPO requires an actor-critic model that outputs both action probabilities and value estimates:

```python
class ActorCriticModel(nnx.Module):
    def __call__(self, observations) -> tuple[jax.Array, jax.Array]:
        """Forward pass returning (log_probs, values).

        Args:
            observations: State observations.

        Returns:
            Tuple of:
                - log_probs: Action log probabilities, shape (batch, num_actions)
                - values: State value estimates, shape (batch,)
        """
        ...
```

## Hyperparameter Guidelines

### Clip Parameter (epsilon)

- **0.1-0.2**: Standard range, 0.2 is most common
- Lower values → more conservative updates
- Higher values → larger policy changes allowed

### GAE Lambda

- **0.95**: Good default for most tasks
- **0.99**: Lower bias, higher variance (longer-horizon tasks)
- **0.9**: Higher bias, lower variance (shorter-horizon tasks)

### Value Function Coefficient

- **0.5**: Standard choice
- Higher values → more emphasis on accurate value estimation

## Use Cases

PPO is recommended for:

- **Complex tasks**: When REINFORCE is too unstable
- **Continuous control**: Robotics, physics simulations
- **Games**: Atari, board games, video games
- **Large models**: When you can afford the value network memory

For memory-constrained settings, consider [GRPO](grpo.md).

## Related Documentation

- [RL Training Guide](../user-guide/training/rl-training.md) - Comprehensive RL training guide
- [REINFORCE Trainer](reinforce.md) - Simpler baseline algorithm
- [GRPO Trainer](grpo.md) - Memory-efficient alternative
- [DPO Trainer](dpo.md) - Preference-based learning
