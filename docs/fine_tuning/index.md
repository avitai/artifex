# Fine-Tuning

Comprehensive fine-tuning infrastructure for adapting pre-trained generative models, including parameter-efficient methods, knowledge distillation, and reinforcement learning from human feedback.

## Overview

<div class="grid cards" markdown>

- :material-tune:{ .lg .middle } **Parameter-Efficient Adapters**

    ---

    LoRA, Prefix Tuning, and Prompt Tuning for efficient adaptation

- :material-school:{ .lg .middle } **Knowledge Distillation**

    ---

    Transfer knowledge from large to small models

- :material-brain:{ .lg .middle } **Reinforcement Learning**

    ---

    RLHF, DPO, and PPO for alignment training

- :material-transfer:{ .lg .middle } **Transfer Learning**

    ---

    Few-shot and domain adaptation techniques

</div>

## Quick Start

### LoRA Fine-Tuning

```python
from artifex.fine_tuning.adapters import LoRAAdapter

# Create LoRA adapter
adapter = LoRAAdapter(
    rank=8,
    alpha=16,
    dropout=0.1,
    target_modules=["query", "value"],
)

# Apply to model
adapted_model = adapter.apply(pretrained_model)

# Train with frozen base weights
trainer.train(adapted_model, train_data)
```

### DPO Training

```python
from artifex.fine_tuning.rl import DPOTrainer

# Create DPO trainer
trainer = DPOTrainer(
    model=model,
    ref_model=reference_model,
    beta=0.1,
)

# Train on preference data
trainer.train(preference_dataset)
```

## Parameter-Efficient Adapters

### LoRA (Low-Rank Adaptation)

Efficient fine-tuning with low-rank matrix decomposition.

```python
from artifex.fine_tuning.adapters import LoRAAdapter, LoRAConfig

config = LoRAConfig(
    rank=8,           # Rank of low-rank matrices
    alpha=16,         # Scaling factor
    dropout=0.1,      # Dropout rate
    target_modules=[  # Modules to adapt
        "attention.query",
        "attention.value",
        "mlp.dense",
    ],
)

adapter = LoRAAdapter(config)
adapted_model = adapter.apply(model)

# Get trainable parameters only
trainable_params = adapter.get_trainable_params()
print(f"Trainable: {trainable_params:,} params")
```

[:octicons-arrow-right-24: LoRA Reference](lora.md)

### Prefix Tuning

Learn continuous task-specific prefixes.

```python
from artifex.fine_tuning.adapters import PrefixTuning, PrefixConfig

config = PrefixConfig(
    prefix_length=20,      # Number of prefix tokens
    num_layers=12,         # Layers to add prefixes
    hidden_dim=768,        # Prefix hidden dimension
    init_method="random",  # Initialization method
)

prefix_tuner = PrefixTuning(config)
adapted_model = prefix_tuner.apply(model)
```

[:octicons-arrow-right-24: Prefix Tuning Reference](prefix_tuning.md)

### Prompt Tuning

Learn soft prompts for task adaptation.

```python
from artifex.fine_tuning.adapters import PromptTuning, PromptConfig

config = PromptConfig(
    num_tokens=10,           # Number of learnable tokens
    init_from_vocab=True,    # Initialize from vocabulary
    init_text="Generate:",   # Text for initialization
)

prompt_tuner = PromptTuning(config)
adapted_model = prompt_tuner.apply(model)
```

[:octicons-arrow-right-24: Prompt Tuning Reference](prompt_tuning.md)

## Fine-Tuning Methods

### Knowledge Distillation

Transfer knowledge from teacher to student models.

```python
from artifex.fine_tuning import DistillationTrainer, DistillationConfig

config = DistillationConfig(
    temperature=4.0,       # Softmax temperature
    alpha=0.5,             # Balance between hard/soft labels
    loss_type="kl_div",    # Distillation loss type
)

trainer = DistillationTrainer(
    student=small_model,
    teacher=large_model,
    config=config,
)

trainer.train(train_data)
```

[:octicons-arrow-right-24: Distillation Reference](distillation.md)

### Few-Shot Learning

Adapt models with limited examples.

```python
from artifex.fine_tuning import FewShotTrainer, FewShotConfig

config = FewShotConfig(
    n_ways=5,              # Number of classes
    n_shots=5,             # Examples per class
    n_queries=15,          # Query examples per class
    meta_batch_size=4,     # Tasks per batch
)

trainer = FewShotTrainer(model, config)
trainer.train(support_set, query_set)
```

[:octicons-arrow-right-24: Few-Shot Reference](few_shot.md)

### Transfer Learning

Transfer pre-trained models to new domains.

```python
from artifex.fine_tuning import TransferTrainer, TransferConfig

config = TransferConfig(
    freeze_encoder=True,   # Freeze feature extractor
    new_head=True,         # Add new classification head
    layer_wise_lr={        # Layer-wise learning rates
        "encoder": 1e-5,
        "decoder": 1e-4,
        "head": 1e-3,
    },
)

trainer = TransferTrainer(pretrained_model, config)
trainer.train(target_dataset)
```

[:octicons-arrow-right-24: Transfer Learning Reference](transfer.md)

## Reinforcement Learning

### RLHF (Reinforcement Learning from Human Feedback)

Align models with human preferences.

```python
from artifex.fine_tuning.rl import RLHFTrainer, RLHFConfig

config = RLHFConfig(
    reward_model_path="reward_model.ckpt",
    kl_coef=0.1,           # KL divergence coefficient
    clip_range=0.2,        # PPO clip range
    value_loss_coef=0.5,   # Value function loss weight
)

trainer = RLHFTrainer(
    policy_model=model,
    reward_model=reward_model,
    config=config,
)

trainer.train(prompts_dataset)
```

[:octicons-arrow-right-24: RLHF Reference](rlhf.md)

### DPO (Direct Preference Optimization)

Direct optimization on preference pairs without reward model.

```python
from artifex.fine_tuning.rl import DPOTrainer, DPOConfig

config = DPOConfig(
    beta=0.1,              # Temperature parameter
    label_smoothing=0.0,   # Label smoothing
    loss_type="sigmoid",   # Loss function type
)

trainer = DPOTrainer(
    model=model,
    ref_model=reference_model,
    config=config,
)

# Preference data format: (prompt, chosen, rejected)
trainer.train(preference_dataset)
```

[:octicons-arrow-right-24: DPO Reference](dpo.md)

### PPO (Proximal Policy Optimization)

Policy gradient with clipped surrogate objective.

```python
from artifex.fine_tuning.rl import PPOTrainer, PPOConfig

config = PPOConfig(
    clip_range=0.2,        # Policy clip range
    clip_range_vf=0.2,     # Value clip range
    entropy_coef=0.01,     # Entropy bonus
    vf_coef=0.5,           # Value function coefficient
    max_grad_norm=0.5,     # Gradient clipping
    n_epochs=4,            # PPO epochs per batch
)

trainer = PPOTrainer(
    model=model,
    reward_model=reward_model,
    config=config,
)

trainer.train(prompts_dataset)
```

[:octicons-arrow-right-24: PPO Reference](ppo.md)

## Best Practices

### Choosing an Adapter

| Method | Parameters | Memory | Best For |
|--------|------------|--------|----------|
| LoRA | ~0.1-1% | Low | General fine-tuning |
| Prefix Tuning | ~0.1% | Low | Sequence tasks |
| Prompt Tuning | ~0.01% | Very Low | Few-shot adaptation |

### Training Tips

```python
# 1. Start with smaller rank for LoRA
lora_config = LoRAConfig(rank=4, alpha=8)

# 2. Use learning rate warmup
scheduler = create_scheduler(
    "cosine",
    warmup_steps=100,
    total_steps=10000,
)

# 3. Monitor KL divergence in RLHF
callbacks = [
    KLDivergenceCallback(threshold=0.1),
]

# 4. Save adapter weights separately
adapter.save_weights("adapter_weights.ckpt")
```

## Module Reference

| Category | Modules |
|----------|---------|
| **Adapters** | [lora](lora.md), [prefix_tuning](prefix_tuning.md), [prompt_tuning](prompt_tuning.md) |
| **Fine-Tuning** | [distillation](distillation.md), [few_shot](few_shot.md), [transfer](transfer.md) |
| **RL** | [dpo](dpo.md), [ppo](ppo.md), [rlhf](rlhf.md) |

## Related Documentation

- [Training Guide](../user-guide/training/training-guide.md) - Complete training guide
- [RL Training Guide](../user-guide/training/rl-training.md) - RL fine-tuning details
- [Training Systems](../training/index.md) - Training infrastructure
