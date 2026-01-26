# DPO Trainer

**Module:** `artifex.generative_models.training.rl.dpo`

The DPO (Direct Preference Optimization) Trainer enables learning from preference pairs without requiring an explicit reward model or RL optimization loop.

## Overview

DPO directly optimizes the policy to prefer chosen responses over rejected ones:

- **No Reward Model**: Learns directly from preferences
- **Stable Training**: Uses supervised-learning-style updates
- **SimPO Support**: Reference-free variant for simpler setup
- **Label Smoothing**: Robustness to noisy preferences

## Quick Start

```python
from artifex.generative_models.training import DPOConfig, DPOTrainer
from flax import nnx
import optax

# Create policy model
model = PolicyModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-5), wrt=nnx.Param)

# Configure DPO
config = DPOConfig(
    beta=0.1,
    label_smoothing=0.0,
    reference_free=False,
)

trainer = DPOTrainer(model, optimizer, config)

# Training with preference pairs
batch = {
    "chosen_log_probs": chosen_log_probs,
    "rejected_log_probs": rejected_log_probs,
    "ref_chosen_log_probs": ref_chosen,
    "ref_rejected_log_probs": ref_rejected,
}
metrics = trainer.train_step(batch)
```

## Configuration

::: artifex.generative_models.training.rl.configs.DPOConfig
    options:
      show_root_heading: true
      members_order: source

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | `float` | `0.1` | Temperature parameter for reward scaling |
| `label_smoothing` | `float` | `0.0` | Label smoothing for robustness |
| `reference_free` | `bool` | `False` | Use SimPO (reference-free) mode |

## Algorithm

### Standard DPO

DPO optimizes the Bradley-Terry preference model:

$$\mathcal{L}_{DPO} = -\log \sigma\left(\beta \left[ \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right]\right)$$

Where:

- $y_w$ is the preferred (chosen) response
- $y_l$ is the rejected response
- $\pi_{ref}$ is the reference policy (frozen)
- $\beta$ controls the implicit reward scaling

### SimPO (Reference-Free)

SimPO eliminates the reference model by using length-normalized log probabilities:

$$\mathcal{L}_{SimPO} = -\log \sigma\left(\beta \left[ \frac{\log \pi_\theta(y_w|x)}{|y_w|} - \frac{\log \pi_\theta(y_l|x)}{|y_l|} \right]\right)$$

Enable with `reference_free=True`:

```python
config = DPOConfig(
    beta=0.1,
    reference_free=True,  # SimPO mode
)
```

### Label Smoothing

For robustness to noisy preference labels:

```python
config = DPOConfig(
    beta=0.1,
    label_smoothing=0.1,  # 10% label smoothing
)
```

## API Reference

::: artifex.generative_models.training.rl.dpo.DPOTrainer
    options:
      show_root_heading: true
      members_order: source

## Training Metrics

| Metric | Description |
|--------|-------------|
| `dpo_loss` | DPO/SimPO loss value |
| `reward_accuracy` | Fraction where chosen > rejected reward |

## Data Format

### Standard DPO

Requires log probabilities from both policy and reference model:

```python
batch = {
    # Policy log probs
    "chosen_log_probs": policy_chosen,      # shape: (batch,)
    "rejected_log_probs": policy_rejected,  # shape: (batch,)
    # Reference model log probs (frozen)
    "ref_chosen_log_probs": ref_chosen,     # shape: (batch,)
    "ref_rejected_log_probs": ref_rejected, # shape: (batch,)
}
```

### SimPO (Reference-Free)

Only requires policy log probabilities:

```python
batch = {
    "chosen_log_probs": chosen_log_probs,
    "rejected_log_probs": rejected_log_probs,
    # No reference model log probs needed
}
```

## Beta Parameter

The `beta` parameter controls the sharpness of the preference:

- **Lower beta (0.01-0.05)**: Softer preferences, more exploration
- **Standard beta (0.1)**: Default, good balance
- **Higher beta (0.5-1.0)**: Sharper preferences, stronger alignment

## Preparing Preference Data

```python
def prepare_dpo_batch(
    model,
    ref_model,
    prompts,
    chosen_responses,
    rejected_responses,
):
    """Prepare batch for DPO training.

    Args:
        model: Policy model being trained
        ref_model: Frozen reference model
        prompts: Input prompts
        chosen_responses: Preferred completions
        rejected_responses: Non-preferred completions

    Returns:
        Batch dict for DPO trainer
    """
    # Compute log probs from policy
    chosen_log_probs = compute_log_probs(model, prompts, chosen_responses)
    rejected_log_probs = compute_log_probs(model, prompts, rejected_responses)

    # Compute log probs from reference (no gradients)
    ref_chosen = compute_log_probs(ref_model, prompts, chosen_responses)
    ref_rejected = compute_log_probs(ref_model, prompts, rejected_responses)

    return {
        "chosen_log_probs": chosen_log_probs,
        "rejected_log_probs": rejected_log_probs,
        "ref_chosen_log_probs": ref_chosen,
        "ref_rejected_log_probs": ref_rejected,
    }
```

## Use Cases

DPO is recommended for:

- **Alignment**: When you have human preference data
- **No reward model**: Simpler than RLHF pipeline
- **Fine-tuning LLMs**: Preference tuning for language models
- **Image generation**: Preference-based image quality tuning

## Comparison with RL Methods

| Aspect | DPO | PPO/GRPO |
|--------|-----|----------|
| Requires reward model | No | Yes |
| Training stability | High | Medium |
| Sample efficiency | High | Lower |
| Flexibility | Less | More |
| Online learning | No | Yes |

## Related Documentation

- [RL Training Guide](../user-guide/training/rl-training.md) - Comprehensive RL training guide
- [PPO Trainer](ppo.md) - Policy gradient with value function
- [GRPO Trainer](grpo.md) - Memory-efficient RL
- [REINFORCE Trainer](reinforce.md) - Basic policy gradient

## References

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
