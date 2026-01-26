# Reinforcement Learning Training

This guide covers reinforcement learning (RL) training in Artifex for fine-tuning generative models using reward signals. RL training enables optimization of models beyond standard likelihood-based objectives, allowing alignment with human preferences, aesthetic quality, or domain-specific metrics.

## Overview

Artifex provides a comprehensive RL training module with four main trainers:

| Trainer | Use Case | Memory | Best For |
|---------|----------|--------|----------|
| **REINFORCE** | Simple policy gradients | Low | Baselines, simple rewards |
| **PPO** | Proximal Policy Optimization | Medium | Stable training, complex tasks |
| **GRPO** | Group Relative Policy Optimization | Low | Large models, memory-constrained |
| **DPO** | Direct Preference Optimization | Low | Preference learning, no reward model |

### When to Use RL Training

RL training is particularly effective for:

- **Diffusion Models**: Fine-tuning for aesthetic quality, text-image alignment (CLIP scores), or domain-specific attributes
- **GANs**: Using discriminator feedback as rewards for generator improvement
- **VAEs**: Optimizing reconstruction quality or latent space properties
- **Flow Models**: Improving sample quality beyond maximum likelihood

## REINFORCE Trainer

REINFORCE implements the basic policy gradient algorithm with variance reduction through baseline subtraction.

### Configuration

```python
from artifex.generative_models.training import REINFORCEConfig, REINFORCETrainer

config = REINFORCEConfig(
    gamma=0.99,              # Discount factor for returns
    normalize_returns=True,  # Normalize returns for stability
    entropy_coeff=0.01,      # Entropy bonus for exploration
)
```

### REINFORCEConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma` | `float` | `0.99` | Discount factor for computing returns |
| `normalize_returns` | `bool` | `True` | Whether to normalize returns to zero mean and unit variance |
| `entropy_coeff` | `float` | `0.01` | Coefficient for entropy bonus (encourages exploration) |

### Basic Usage

```python
from flax import nnx
import optax
from artifex.generative_models.training import (
    REINFORCEConfig,
    REINFORCETrainer,
)

# Setup model and optimizer
model = YourPolicyModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

# Create trainer
config = REINFORCEConfig(gamma=0.99, entropy_coeff=0.01)
trainer = REINFORCETrainer(model, optimizer, config)

# Training step
batch = {
    "observations": observations,  # State observations
    "actions": actions,            # Actions taken
    "rewards": rewards,            # Rewards received
}
metrics = trainer.train_step(batch)
print(f"Policy loss: {metrics['policy_loss']:.4f}")
```

### How It Works

REINFORCE computes the policy gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t \right]$$

Where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the discounted return.

**Key features:**

- Returns are computed using `compute_discounted_returns()` from shared utilities
- Optional return normalization reduces variance
- Entropy bonus encourages exploration and prevents premature convergence

## PPO Trainer

Proximal Policy Optimization (PPO) provides stable training through clipped surrogate objectives and value function learning.

### Configuration

```python
from artifex.generative_models.training import PPOConfig, PPOTrainer

config = PPOConfig(
    gamma=0.99,           # Discount factor
    gae_lambda=0.95,      # GAE lambda for advantage estimation
    clip_param=0.2,       # PPO clipping parameter
    vf_coeff=0.5,         # Value function loss coefficient
    entropy_coeff=0.01,   # Entropy bonus coefficient
    max_grad_norm=0.5,    # Gradient clipping norm
)
```

### PPOConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma` | `float` | `0.99` | Discount factor for GAE computation |
| `gae_lambda` | `float` | `0.95` | Lambda for Generalized Advantage Estimation |
| `clip_param` | `float` | `0.2` | Clipping parameter epsilon for surrogate loss |
| `vf_coeff` | `float` | `0.5` | Coefficient for value function loss |
| `entropy_coeff` | `float` | `0.01` | Coefficient for entropy bonus |
| `max_grad_norm` | `float` | `0.5` | Maximum gradient norm for clipping |

### Training with PPO

```python
from flax import nnx
import optax
from artifex.generative_models.training import PPOConfig, PPOTrainer

# Actor-critic model with separate heads
model = ActorCriticModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)

config = PPOConfig(
    gamma=0.99,
    gae_lambda=0.95,
    clip_param=0.2,
    vf_coeff=0.5,
    entropy_coeff=0.01,
)
trainer = PPOTrainer(model, optimizer, config)

# Collect trajectory
trajectory = {
    "observations": observations,
    "actions": actions,
    "rewards": rewards,
    "values": values,           # Value estimates V(s)
    "log_probs": old_log_probs, # Log probs from old policy
    "dones": dones,             # Episode termination flags
}

# Train on trajectory
metrics = trainer.train_step(trajectory)
print(f"Policy loss: {metrics['policy_loss']:.4f}")
print(f"Value loss: {metrics['value_loss']:.4f}")
print(f"Entropy: {metrics['entropy']:.4f}")
```

### Generalized Advantage Estimation

PPO uses GAE for variance-reduced advantage estimation:

$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD residual.

**Benefits of GAE:**

- Balances bias-variance trade-off via $\lambda$ parameter
- $\lambda = 0$ gives TD(0) (low variance, high bias)
- $\lambda = 1$ gives Monte Carlo returns (high variance, low bias)
- $\lambda = 0.95$ is a good default for most applications

## GRPO Trainer

Group Relative Policy Optimization (GRPO) is a critic-free RL algorithm that normalizes advantages within groups of generations. This approach, pioneered by DeepSeek, provides approximately **50% memory savings** compared to PPO by eliminating the value network.

### Configuration

```python
from artifex.generative_models.training import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    num_generations=4,    # Generations per prompt (group size)
    clip_param=0.2,       # PPO-style clipping
    beta=0.01,            # KL penalty coefficient
    entropy_coeff=0.01,   # Entropy bonus
    gamma=0.99,           # Discount factor
)
```

### GRPOConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_generations` | `int` | `4` | Number of generations per prompt (group size G) |
| `clip_param` | `float` | `0.2` | PPO-style clipping parameter |
| `beta` | `float` | `0.01` | KL divergence penalty coefficient |
| `entropy_coeff` | `float` | `0.01` | Entropy bonus coefficient |
| `gamma` | `float` | `0.99` | Discount factor |

### How GRPO Works

GRPO eliminates the critic by normalizing rewards within groups:

1. **Generate G samples per prompt**: For each prompt, generate multiple completions
2. **Compute rewards**: Evaluate each generation with a reward function
3. **Normalize within groups**: Compute advantages as $(r - \mu_g) / \sigma_g$ within each group
4. **Apply PPO-style clipping**: Use the normalized advantages with clipped surrogate loss

```python
from flax import nnx
import optax
from artifex.generative_models.training import GRPOConfig, GRPOTrainer

# Policy model (no value head needed!)
model = PolicyModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

config = GRPOConfig(
    num_generations=4,  # Generate 4 samples per prompt
    clip_param=0.2,
    beta=0.01,
)
trainer = GRPOTrainer(model, optimizer, config)

# Batch with grouped generations
# If num_prompts=8 and num_generations=4, batch_size=32
batch = {
    "observations": observations,  # Shape: (batch_size, ...)
    "actions": actions,            # Shape: (batch_size, ...)
    "rewards": rewards,            # Shape: (batch_size,)
    "log_probs": old_log_probs,    # Shape: (batch_size,)
    "group_size": 4,               # Optional, defaults to config
}

metrics = trainer.train_step(batch)
print(f"Policy loss: {metrics['policy_loss']:.4f}")
print(f"Approx KL: {metrics['approx_kl']:.4f}")
```

### Advantages of GRPO

1. **Memory Efficient**: No value network means ~50% memory savings
2. **Simple Implementation**: No need to train a critic
3. **Effective for Generative Models**: Naturally fits the "generate multiple samples" paradigm
4. **Stable Training**: Group normalization provides consistent advantage scaling

### KL Divergence Penalty

GRPO can optionally include a KL penalty to prevent the policy from diverging too far from a reference:

```python
# With reference model for KL penalty
batch = {
    "observations": observations,
    "actions": actions,
    "rewards": rewards,
    "log_probs": old_log_probs,
    "ref_log_probs": ref_log_probs,  # From frozen reference model
}

# KL penalty is automatically applied when ref_log_probs is provided
metrics = trainer.train_step(batch)
print(f"KL penalty: {metrics.get('kl_penalty', 0):.4f}")
```

## DPO Trainer

Direct Preference Optimization (DPO) learns from preference pairs without requiring an explicit reward model or RL optimization loop.

### Configuration

```python
from artifex.generative_models.training import DPOConfig, DPOTrainer

config = DPOConfig(
    beta=0.1,              # Reward scaling temperature
    label_smoothing=0.0,   # Smoothing for preference labels
    reference_free=False,  # Enable SimPO mode
)
```

### DPOConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | `float` | `0.1` | Temperature parameter for reward scaling |
| `label_smoothing` | `float` | `0.0` | Label smoothing for robustness |
| `reference_free` | `bool` | `False` | Use SimPO (reference-free) mode |

### Standard DPO Training

```python
from flax import nnx
import optax
from artifex.generative_models.training import DPOConfig, DPOTrainer

model = PolicyModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-5), wrt=nnx.Param)

config = DPOConfig(beta=0.1)
trainer = DPOTrainer(model, optimizer, config)

# Preference pairs: chosen (preferred) vs rejected
batch = {
    "chosen_log_probs": chosen_log_probs,      # Log probs for preferred
    "rejected_log_probs": rejected_log_probs,  # Log probs for rejected
    "ref_chosen_log_probs": ref_chosen,        # Reference model log probs
    "ref_rejected_log_probs": ref_rejected,
}

metrics = trainer.train_step(batch)
print(f"DPO loss: {metrics['dpo_loss']:.4f}")
print(f"Reward accuracy: {metrics['reward_accuracy']:.2%}")
```

### SimPO: Reference-Free DPO

SimPO eliminates the need for a reference model by using length-normalized log probabilities:

```python
config = DPOConfig(
    beta=0.1,
    reference_free=True,  # Enable SimPO mode
)
trainer = DPOTrainer(model, optimizer, config)

# No reference log probs needed
batch = {
    "chosen_log_probs": chosen_log_probs,
    "rejected_log_probs": rejected_log_probs,
}

metrics = trainer.train_step(batch)
```

### How DPO Works

DPO directly optimizes the Bradley-Terry preference model:

$$\mathcal{L}_{DPO} = -\log \sigma\left(\beta \left[ \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right]\right)$$

Where:

- $y_w$ is the preferred (chosen) response
- $y_l$ is the rejected response
- $\pi_{ref}$ is the reference policy
- $\beta$ controls the implicit reward scaling

## Reward Functions

Artifex provides a flexible reward function interface for custom reward computation.

### Built-in Reward Functions

```python
from artifex.generative_models.training import (
    ConstantReward,
    CompositeReward,
    ThresholdReward,
    ScaledReward,
    ClippedReward,
)
```

### ConstantReward

Returns a fixed reward value (useful for testing):

```python
reward_fn = ConstantReward(value=1.0)
rewards = reward_fn(samples)  # Returns array of 1.0
```

### CompositeReward

Combines multiple reward functions with weights:

```python
reward_fn = CompositeReward(
    reward_fns=[aesthetic_reward, clip_reward],
    weights=[0.3, 0.7],  # 30% aesthetic, 70% CLIP
)
```

### ThresholdReward

Applies a threshold to convert scores to binary rewards:

```python
# Reward = 1.0 if base_reward > 0.5 else 0.0
reward_fn = ThresholdReward(
    reward_fn=base_reward,
    threshold=0.5,
    above_value=1.0,
    below_value=0.0,
)
```

### ScaledReward

Scales rewards by a constant factor:

```python
reward_fn = ScaledReward(reward_fn=base_reward, scale=10.0)
```

### ClippedReward

Clips rewards to a specified range:

```python
reward_fn = ClippedReward(
    reward_fn=base_reward,
    min_value=-1.0,
    max_value=1.0,
)
```

### Custom Reward Functions

Implement the `RewardFunction` protocol for custom rewards:

```python
from typing import Protocol
import jax
import jax.numpy as jnp

class RewardFunction(Protocol):
    """Protocol for reward function implementations."""

    def __call__(
        self,
        samples: jax.Array,
        conditions: jax.Array | None = None,
        **kwargs,
    ) -> jax.Array:
        """Compute rewards for generated samples.

        Args:
            samples: Generated samples to evaluate.
            conditions: Optional conditioning information.

        Returns:
            Reward values with shape (batch_size,).
        """
        ...

# Example: CLIP-based reward
class CLIPReward:
    def __init__(self, clip_model, target_text):
        self.clip_model = clip_model
        self.target_text = target_text
        self.target_embedding = clip_model.encode_text(target_text)

    def __call__(self, samples, conditions=None, **kwargs):
        image_embeddings = self.clip_model.encode_image(samples)
        # Cosine similarity as reward
        similarity = jnp.sum(
            image_embeddings * self.target_embedding,
            axis=-1,
        )
        return similarity


# Example: Multi-objective reward with learnable weights
class MultiObjectiveReward:
    """Complex reward combining multiple objectives with adaptive weighting."""

    def __init__(
        self,
        clip_model,
        aesthetic_model,
        safety_classifier,
        target_text: str,
        clip_weight: float = 0.4,
        aesthetic_weight: float = 0.3,
        safety_weight: float = 0.3,
    ):
        self.clip_model = clip_model
        self.aesthetic_model = aesthetic_model
        self.safety_classifier = safety_classifier
        self.target_embedding = clip_model.encode_text(target_text)

        # Weights can be tuned during training
        self.weights = {
            "clip": clip_weight,
            "aesthetic": aesthetic_weight,
            "safety": safety_weight,
        }

    def __call__(self, samples, conditions=None, **kwargs):
        batch_size = samples.shape[0]

        # CLIP alignment score (cosine similarity)
        image_emb = self.clip_model.encode_image(samples)
        clip_score = jnp.sum(image_emb * self.target_embedding, axis=-1)

        # Aesthetic quality score (normalized to [0, 1])
        aesthetic_score = self.aesthetic_model(samples)
        aesthetic_score = jax.nn.sigmoid(aesthetic_score)

        # Safety score (1.0 = safe, 0.0 = unsafe)
        safety_logits = self.safety_classifier(samples)
        safety_score = jax.nn.softmax(safety_logits, axis=-1)[:, 0]  # P(safe)

        # Combine with penalty for unsafe content
        combined_reward = (
            self.weights["clip"] * clip_score +
            self.weights["aesthetic"] * aesthetic_score +
            self.weights["safety"] * safety_score
        )

        # Apply safety penalty: zero reward for unsafe samples
        safety_mask = safety_score > 0.5
        combined_reward = jnp.where(safety_mask, combined_reward, -1.0)

        return combined_reward


# Example: Sequence-level reward for text generation
class SequenceReward:
    """Reward function for autoregressive text models."""

    def __init__(self, reward_model, tokenizer):
        self.reward_model = reward_model
        self.tokenizer = tokenizer

    def __call__(self, samples, conditions=None, **kwargs):
        # samples: token IDs of shape (batch, seq_len)
        # Compute reward at sequence level
        rewards = self.reward_model(samples)

        # Apply length penalty to avoid reward hacking
        lengths = jnp.sum(samples != self.tokenizer.pad_id, axis=-1)
        length_penalty = jnp.log(lengths + 1) / jnp.log(100)  # Normalize

        return rewards - 0.1 * length_penalty
```

## Utility Functions

The RL module provides shared utility functions following the DRY principle:

```python
from artifex.generative_models.training.rl.utils import (
    compute_discounted_returns,
    compute_gae_advantages,
    normalize_advantages,
    compute_policy_entropy,
    compute_kl_divergence,
    compute_clipped_surrogate_loss,
)
```

### compute_discounted_returns

```python
# Compute G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
returns = compute_discounted_returns(rewards, gamma=0.99)
```

### compute_gae_advantages

```python
# GAE with bias-variance trade-off
advantages = compute_gae_advantages(
    rewards=rewards,
    values=values,        # V(s) estimates including V(s_T+1)
    dones=dones,          # Episode termination flags
    gamma=0.99,
    gae_lambda=0.95,
)
```

### normalize_advantages

```python
# Zero mean, unit variance normalization
normalized = normalize_advantages(advantages, eps=1e-8)
```

### compute_policy_entropy

```python
# H = -sum(p * log(p))
entropy = compute_policy_entropy(log_probs)
```

### compute_kl_divergence

```python
# KL(policy || reference)
kl = compute_kl_divergence(policy_log_probs, ref_log_probs)
```

### compute_clipped_surrogate_loss

```python
# PPO clipped objective
loss = compute_clipped_surrogate_loss(
    log_probs=current_log_probs,
    old_log_probs=old_log_probs,
    advantages=advantages,
    clip_param=0.2,
)
```

## Integration with Model-Specific Trainers

RL trainers integrate with Artifex's model-specific trainers for fine-tuning:

### Diffusion Model Fine-Tuning with GRPO

```python
from artifex.generative_models.training import GRPOConfig, GRPOTrainer
from artifex.generative_models.training.trainers import DiffusionTrainer

# Standard diffusion trainer for pre-training
diffusion_trainer = DiffusionTrainer(model, optimizer, diffusion_config)

# Fine-tune with GRPO
grpo_config = GRPOConfig(num_generations=4, beta=0.01)
grpo_trainer = GRPOTrainer(model, optimizer, grpo_config)

# Generate samples and compute rewards
for batch in dataloader:
    # Generate multiple samples per condition
    samples = generate_samples(model, batch["conditions"], num_samples=4)
    rewards = reward_fn(samples, batch["conditions"])

    # GRPO training step
    rl_batch = {
        "observations": batch["conditions"],
        "actions": samples,
        "rewards": rewards,
        "log_probs": compute_log_probs(model, samples, batch["conditions"]),
    }
    metrics = grpo_trainer.train_step(rl_batch)
```

### VAE Latent Space Optimization with PPO

```python
from artifex.generative_models.training import PPOConfig, PPOTrainer

# Use encoder as policy, decoder as environment
ppo_config = PPOConfig(gamma=0.99, gae_lambda=0.95)
ppo_trainer = PPOTrainer(encoder, optimizer, ppo_config)

# Train encoder to produce latents that decode well
for batch in dataloader:
    # Encoder produces latent "actions"
    latents, log_probs = encoder(batch["images"], return_log_prob=True)

    # Decoder reconstructs, computing "rewards"
    reconstructions = decoder(latents)
    rewards = compute_reconstruction_reward(batch["images"], reconstructions)

    trajectory = {
        "observations": batch["images"],
        "actions": latents,
        "rewards": rewards,
        "values": value_estimates,
        "log_probs": log_probs,
        "dones": jnp.zeros(batch_size),
    }
    metrics = ppo_trainer.train_step(trajectory)
```

### GAN Discriminator as Reward with REINFORCE

```python
from artifex.generative_models.training import REINFORCEConfig, REINFORCETrainer

reinforce_config = REINFORCEConfig(entropy_coeff=0.01)
reinforce_trainer = REINFORCETrainer(generator, optimizer, reinforce_config)

# Use discriminator output as reward
for batch in dataloader:
    # Generate samples
    generated = generator(batch["noise"])

    # Discriminator provides reward signal
    rewards = discriminator(generated)  # Higher = more realistic

    rl_batch = {
        "observations": batch["noise"],
        "actions": generated,
        "rewards": rewards,
    }
    metrics = reinforce_trainer.train_step(rl_batch)
```

## Best Practices

### Choosing the Right Algorithm

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Memory-constrained | GRPO (no value network) |
| Stable training needed | PPO (clipped updates) |
| Preference data available | DPO (no reward model) |
| Simple baseline | REINFORCE |
| Large language models | GRPO or DPO |
| Image generation | GRPO with CLIP reward |

### Hyperparameter Guidelines

**REINFORCE:**

- Start with `gamma=0.99` for long-horizon tasks
- Use `normalize_returns=True` for stability
- `entropy_coeff=0.01` is a good default

**PPO:**

- `clip_param=0.2` is the standard choice
- `gae_lambda=0.95` balances bias-variance
- Multiple epochs per batch (3-10) improve sample efficiency

**GRPO:**

- `num_generations=4-8` typically works well
- Lower `beta` (0.001-0.01) for more exploration
- Higher `beta` (0.1) to stay close to reference

**DPO:**

- `beta=0.1` is a common starting point
- Lower `beta` for stronger preferences
- Use `label_smoothing=0.1` for noisy preferences

### Common Pitfalls

1. **Reward hacking**: Models may find unintended ways to maximize rewards
   - Use composite rewards with multiple objectives
   - Monitor sample quality qualitatively

2. **Training instability**: Large policy updates can destabilize training
   - Use PPO clipping or GRPO's group normalization
   - Apply gradient clipping

3. **Forgetting**: RL fine-tuning can degrade base model capabilities
   - Use KL penalty (GRPO's `beta` parameter)
   - Mix RL objectives with supervised loss

4. **Sparse rewards**: Infrequent rewards make learning difficult
   - Use reward shaping with intermediate signals
   - Consider dense reward proxies

## API Reference

For complete API documentation, see the [Trainer API Reference](../../api/training/trainer.md).

All RL training components are exported from the main training module:

```python
from artifex.generative_models.training import (
    # Configurations
    REINFORCEConfig,
    PPOConfig,
    GRPOConfig,
    DPOConfig,

    # Trainers
    REINFORCETrainer,
    PPOTrainer,
    GRPOTrainer,
    DPOTrainer,

    # Reward functions
    RewardFunction,
    ConstantReward,
    CompositeReward,
    ThresholdReward,
    ScaledReward,
    ClippedReward,
)
```

## Using Callbacks with RL Trainers

RL trainers integrate seamlessly with Artifex callbacks for logging, checkpointing, and profiling:

```python
from artifex.generative_models.training import GRPOTrainer, GRPOConfig
from artifex.generative_models.training.callbacks import (
    WandbLoggerCallback,
    WandbLoggerConfig,
    ModelCheckpoint,
    CheckpointConfig,
    JAXProfiler,
    ProfilingConfig,
    ProgressBarCallback,
)

# Configure RL trainer
grpo_config = GRPOConfig(
    num_generations=4,
    clip_param=0.2,
    beta=0.01,
)
trainer = GRPOTrainer(model, optimizer, grpo_config)

# Setup callbacks for RL training
callbacks = [
    # Log RL-specific metrics (rewards, advantages, KL divergence)
    WandbLoggerCallback(WandbLoggerConfig(
        project="rl-finetuning",
        name="grpo-experiment",
        config={
            "algorithm": "GRPO",
            "num_generations": 4,
            "beta": 0.01,
        },
        log_every_n_steps=10,
    )),

    # Save best model based on mean reward
    ModelCheckpoint(CheckpointConfig(
        dirpath="checkpoints/grpo",
        monitor="mean_reward",
        mode="max",  # Higher reward is better
        save_top_k=1,  # Keep only the best checkpoint
    )),

    # Profile RL training (useful for debugging generation bottlenecks)
    JAXProfiler(ProfilingConfig(
        log_dir="logs/rl_profiles",
        start_step=50,
        end_step=60,
    )),

    # Progress bar with RL metrics
    ProgressBarCallback(),
]

# Training loop with callbacks
for callback in callbacks:
    callback.on_train_begin(trainer)

for step, batch in enumerate(dataloader):
    # Generate samples and compute rewards
    metrics = trainer.train_step(batch, reward_fn, key)

    # Log metrics via callbacks
    for callback in callbacks:
        callback.on_train_batch_end(trainer, metrics, step)

for callback in callbacks:
    callback.on_train_end(trainer)
```

### RL-Specific Metrics Logged

Different RL trainers log different metrics:

| Trainer | Metrics |
|---------|---------|
| **REINFORCE** | `loss`, `mean_reward`, `reward_std`, `entropy` |
| **PPO** | `policy_loss`, `value_loss`, `mean_reward`, `advantage_mean`, `kl_divergence` |
| **GRPO** | `loss`, `mean_reward`, `group_advantage_std`, `kl_penalty` |
| **DPO** | `loss`, `chosen_reward`, `rejected_reward`, `reward_margin` |

## Trainer Class Hierarchy

Artifex uses a hierarchical trainer architecture for flexibility:

```
Trainer (base)
├── VAETrainer        → ELBO loss, KL annealing, free bits
├── GANTrainer        → Adversarial training, multiple loss types
├── DiffusionTrainer  → Denoising, noise scheduling, EMA
├── FlowTrainer       → Flow matching, OT-CFM, rectified flow
├── EnergyTrainer     → Contrastive divergence, MCMC sampling
├── AutoregressiveTrainer → Teacher forcing, scheduled sampling
└── RL Trainers
    ├── REINFORCETrainer → Policy gradient, variance reduction
    ├── PPOTrainer       → Actor-critic, GAE, clipping
    ├── GRPOTrainer      → Critic-free, group normalization
    └── DPOTrainer       → Preference learning, SimPO mode
```

Each model-specific trainer:

1. **Inherits core functionality** from the base `Trainer` class (optimizer management, callbacks, checkpointing)
2. **Implements model-specific loss computation** via `compute_loss()` method
3. **Provides specialized training utilities** (e.g., `generate()` for autoregressive, `sample_negatives()` for energy)
4. **Exposes model-specific configuration** via dataclass configs

## Related Documentation

- [Training Guide](training-guide.md) - Core training patterns and callbacks
- [Advanced Features](advanced-features.md) - Gradient accumulation and loss scaling
- [Logging & Experiment Tracking](logging.md) - W&B, TensorBoard, and progress bar integration
- [Performance Profiling](profiling.md) - JAX trace profiling and memory tracking
- [Distributed Training](../advanced/distributed.md) - Multi-device RL training
- [Configuration System](configuration.md) - Training configuration options
