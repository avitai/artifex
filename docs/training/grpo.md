# GRPO Trainer

**Module:** `artifex.generative_models.training.rl.grpo`

The GRPO (Group Relative Policy Optimization) Trainer provides memory-efficient RL training by eliminating the value network through group-normalized advantages.

## Overview

GRPO, pioneered by DeepSeek, achieves approximately **50% memory savings** compared to PPO:

- **No Value Network**: Eliminates critic, saving significant memory
- **Group Normalization**: Normalizes advantages within groups of generations
- **PPO-Style Clipping**: Maintains training stability
- **Optional KL Penalty**: Prevents policy drift from reference

## Quick Start

```python
from artifex.generative_models.training import GRPOConfig, GRPOTrainer
from flax import nnx
import optax

# Create policy model (no value head needed!)
model = PolicyModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

# Configure GRPO
config = GRPOConfig(
    num_generations=4,    # Generate 4 samples per prompt
    clip_param=0.2,
    beta=0.01,            # KL penalty coefficient
    entropy_coeff=0.01,
)

trainer = GRPOTrainer(model, optimizer, config)

# Training with grouped generations
# batch_size = num_prompts * num_generations
batch = {
    "observations": observations,  # (batch_size, ...)
    "actions": actions,            # (batch_size, ...)
    "rewards": rewards,            # (batch_size,)
    "log_probs": old_log_probs,    # (batch_size,)
}
metrics = trainer.train_step(batch)
```

## Configuration

::: artifex.generative_models.training.rl.configs.GRPOConfig
    options:
      show_root_heading: true
      members_order: source

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_generations` | `int` | `4` | Number of generations per prompt (group size G) |
| `clip_param` | `float` | `0.2` | PPO-style clipping parameter |
| `beta` | `float` | `0.01` | KL divergence penalty coefficient |
| `entropy_coeff` | `float` | `0.01` | Entropy bonus coefficient |
| `gamma` | `float` | `0.99` | Discount factor |

## Algorithm

### Group-Normalized Advantages

Instead of learning a value function, GRPO normalizes rewards within groups:

1. **Generate G samples per prompt**: For each prompt $x_i$, generate $G$ completions $\{y_{i,1}, ..., y_{i,G}\}$

2. **Compute rewards**: Evaluate each generation $r_{i,j} = R(x_i, y_{i,j})$

3. **Normalize within groups**:
   $$\hat{A}_{i,j} = \frac{r_{i,j} - \mu_i}{\sigma_i + \epsilon}$$
   where $\mu_i = \frac{1}{G}\sum_j r_{i,j}$ and $\sigma_i = \sqrt{\frac{1}{G}\sum_j (r_{i,j} - \mu_i)^2}$

4. **Apply PPO clipping**: Use normalized advantages with clipped surrogate loss

### Why It Works

- **Relative comparison**: By normalizing within groups, GRPO compares generations to each other rather than to an absolute baseline
- **Self-normalization**: Each prompt serves as its own baseline through group statistics
- **Memory efficient**: No value network parameters or forward passes needed

### KL Penalty (Optional)

To prevent the policy from drifting too far from the reference:

$$\mathcal{L}_{total} = \mathcal{L}_{policy} + \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})$$

Provide `ref_log_probs` in the batch to enable KL penalty:

```python
batch = {
    "observations": observations,
    "actions": actions,
    "rewards": rewards,
    "log_probs": old_log_probs,
    "ref_log_probs": ref_log_probs,  # From frozen reference model
}
```

## API Reference

::: artifex.generative_models.training.rl.grpo.GRPOTrainer
    options:
      show_root_heading: true
      members_order: source

## Training Metrics

| Metric | Description |
|--------|-------------|
| `policy_loss` | Clipped surrogate policy loss |
| `approx_kl` | Approximate KL divergence from old policy |
| `kl_penalty` | KL penalty term (if ref_log_probs provided) |

## Data Organization

GRPO expects data organized by groups. For `num_prompts=N` and `num_generations=G`:

```python
# Total batch size = N * G
batch_size = num_prompts * num_generations

# Data layout: [prompt1_gen1, prompt1_gen2, ..., prompt1_genG,
#               prompt2_gen1, prompt2_gen2, ..., prompt2_genG, ...]
observations = jnp.zeros((batch_size, obs_dim))
actions = jnp.zeros((batch_size, action_dim))
rewards = jnp.zeros((batch_size,))
log_probs = jnp.zeros((batch_size,))
```

### Custom Group Size

Override the default group size per batch:

```python
batch = {
    "observations": observations,
    "actions": actions,
    "rewards": rewards,
    "log_probs": log_probs,
    "group_size": 8,  # Override config.num_generations
}
```

## Complete Training Example

```python
from artifex.generative_models.training import GRPOConfig, GRPOTrainer
from flax import nnx
import optax
import jax.numpy as jnp

def train_with_grpo(
    model,
    reward_fn,
    prompts_loader,
    num_epochs: int = 10,
    num_generations: int = 4,
    learning_rate: float = 1e-4,
):
    """Train a generative model with GRPO."""
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    config = GRPOConfig(
        num_generations=num_generations,
        clip_param=0.2,
        beta=0.01,
    )
    trainer = GRPOTrainer(model, optimizer, config)

    for epoch in range(num_epochs):
        for prompts in prompts_loader:
            num_prompts = len(prompts)

            # Generate multiple samples per prompt
            all_samples = []
            all_log_probs = []
            for prompt in prompts:
                for _ in range(num_generations):
                    sample, log_prob = model.generate(prompt, return_log_prob=True)
                    all_samples.append(sample)
                    all_log_probs.append(log_prob)

            samples = jnp.stack(all_samples)
            log_probs = jnp.stack(all_log_probs)

            # Compute rewards
            rewards = reward_fn(samples, prompts.repeat(num_generations))

            # GRPO training step
            batch = {
                "observations": prompts.repeat(num_generations, axis=0),
                "actions": samples,
                "rewards": rewards,
                "log_probs": log_probs,
            }
            metrics = trainer.train_step(batch)

            print(f"Epoch {epoch}: loss={metrics['policy_loss']:.4f}")

    return model
```

## Memory Comparison

| Method | Policy Params | Value Params | Total Memory |
|--------|--------------|--------------|--------------|
| PPO | P | ~P | ~2P |
| GRPO | P | 0 | P |

For a 7B parameter model:

- **PPO**: ~14B parameters (policy + value head)
- **GRPO**: ~7B parameters (policy only)

This translates to approximately 50% memory savings.

## Hyperparameter Guidelines

### Number of Generations (G)

- **4**: Good default, balance of diversity and efficiency
- **2**: Minimum useful, less reliable normalization
- **8**: Better statistics, higher compute cost
- **16+**: Diminishing returns, very high compute

### Beta (KL Coefficient)

- **0.001-0.01**: More exploration, faster learning
- **0.01-0.1**: Standard range
- **0.1+**: Conservative updates, slower but safer

### Clip Parameter

- **0.2**: Standard PPO default
- **0.1-0.3**: Reasonable range

## Use Cases

GRPO is recommended for:

- **Large language models**: Where memory is constrained
- **Image generation**: Diffusion model fine-tuning with CLIP rewards
- **Resource-limited settings**: Single GPU training of large models
- **Fast iteration**: Simpler setup than PPO (no value network training)

## Comparison with PPO

| Aspect | GRPO | PPO |
|--------|------|-----|
| Memory | ~50% less | Higher |
| Value function | None | Required |
| Advantage estimation | Group normalization | GAE |
| Sample efficiency | Requires more generations | More efficient |
| Implementation complexity | Simpler | More complex |

## Related Documentation

- [RL Training Guide](../user-guide/training/rl-training.md) - Comprehensive RL training guide
- [PPO Trainer](ppo.md) - Traditional actor-critic RL
- [DPO Trainer](dpo.md) - Preference-based learning
- [REINFORCE Trainer](reinforce.md) - Basic policy gradient

## References

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [GRPO in Tunix](https://github.com/google/tunix) - Production JAX implementation
