# PPO Trainer

**Status:** `Supported runtime training surface`

**Module:** `artifex.generative_models.training.rl.ppo`

**Source:** `src/artifex/generative_models/training/rl/ppo.py`

`PPOTrainer` operates on typed autoregressive rollout batches with explicit
old-policy log probabilities, returns, and advantages.

## Quick Start

```python
from flax import nnx
import optax

from artifex.generative_models.training import PPOConfig, PPOTrainer
from artifex.generative_models.training.rl import (
    GeneratedBatch,
    GeneratedSequenceBatch,
    SequenceRolloutBatch,
)

model = ActorCriticModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)
trainer = PPOTrainer(model, optimizer, PPOConfig())

sequence_batch = GeneratedSequenceBatch(
    generation=GeneratedBatch(outputs=token_sequences),
    response_mask=response_mask,
)
rollout = SequenceRolloutBatch(
    sequence_batch=sequence_batch,
    old_log_probs=old_log_probs,
    returns=returns,
    advantages=advantages,
    dones=dones,
)

loss, metrics = trainer.train_step(rollout)
```

## Typed Batch Contract

`PPOTrainer` requires a `SequenceRolloutBatch` with:

- `old_log_probs=` aligned to `sequences[:, 1:]`
- `returns=` aligned to the same action-token layout
- `advantages=` aligned to the same action-token layout

The sequence wrapper comes from `GeneratedSequenceBatch`, which itself wraps the
generic `GeneratedBatch`.

## Configuration

::: artifex.generative_models.training.rl.configs.PPOConfig
    options:
      show_root_heading: true
      members_order: source

## Related Documentation

- [REINFORCE Trainer](reinforce.md)
- [GRPO Trainer](grpo.md)
- [RL Training Guide](../user-guide/training/rl-training.md)
