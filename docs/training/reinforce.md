# REINFORCE Trainer

**Status:** `Supported runtime training surface`

**Module:** `artifex.generative_models.training.rl.reinforce`

**Source:** `src/artifex/generative_models/training/rl/reinforce.py`

`REINFORCETrainer` trains over typed autoregressive rollout batches rather than
ad-hoc dictionaries.

## Quick Start

```python
from flax import nnx
import jax.numpy as jnp
import optax

from artifex.generative_models.training import REINFORCEConfig, REINFORCETrainer
from artifex.generative_models.training.rl import (
    GeneratedBatch,
    GeneratedSequenceBatch,
    SequenceRolloutBatch,
)

model = PolicyModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)
trainer = REINFORCETrainer(model, optimizer, REINFORCEConfig(entropy_coeff=0.01))

sequence_batch = GeneratedSequenceBatch(
    generation=GeneratedBatch(
        outputs=token_sequences,
        rewards=sequence_rewards,
    ),
    response_mask=response_mask,
)
rollout = SequenceRolloutBatch(
    sequence_batch=sequence_batch,
    token_rewards=token_rewards,
)

loss, metrics = trainer.train_step(rollout)
```

## Typed Batch Contract

`REINFORCETrainer` consumes `SequenceRolloutBatch`:

- `GeneratedBatch` stores the generated outputs and optional sequence rewards
- `GeneratedSequenceBatch` adds token-aligned masks for prompt/response splits
- `SequenceRolloutBatch` carries rollout-specific tensors such as
  `token_rewards`, `returns`, `advantages`, and `old_log_probs`

If `returns` are absent, the trainer derives them from `token_rewards` or from
terminal `sequence_rewards`.

## Configuration

::: artifex.generative_models.training.rl.configs.REINFORCEConfig
    options:
      show_root_heading: true
      members_order: source

## Related Documentation

- [PPO Trainer](ppo.md)
- [GRPO Trainer](grpo.md)
- [RL Training Guide](../user-guide/training/rl-training.md)
