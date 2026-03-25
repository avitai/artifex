# GRPO Trainer

**Status:** `Supported runtime training surface`

**Module:** `artifex.generative_models.training.rl.grpo`

**Source:** `src/artifex/generative_models/training/rl/grpo.py`

`GRPOTrainer` consumes grouped typed rollout batches. The runtime expects
prompt-group structure to be explicit through `GroupRolloutBatch`.

## Quick Start

```python
from flax import nnx
import optax

from artifex.generative_models.training import GRPOConfig, GRPOTrainer
from artifex.generative_models.training.rl import (
    GeneratedBatch,
    GeneratedSequenceBatch,
    GroupRolloutBatch,
    SequenceRolloutBatch,
)

model = PolicyModel(rngs=nnx.Rngs(0))
reference_model = PolicyModel(rngs=nnx.Rngs(1))
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

trainer = GRPOTrainer(
    model,
    optimizer,
    GRPOConfig(num_generations=4, clip_param=0.2, beta=0.01),
    reference_model=reference_model,
)

sequence_batch = GeneratedSequenceBatch(
    generation=GeneratedBatch(
        outputs=token_sequences,
        rewards=sequence_rewards,
    ),
    response_mask=response_mask,
)
rollout = SequenceRolloutBatch(
    sequence_batch=sequence_batch,
    old_log_probs=old_log_probs,
)
grouped_rollout = GroupRolloutBatch(
    rollout=rollout,
    group_size=4,
)

loss, metrics = trainer.train_step(grouped_rollout)
```

## Typed Batch Contract

`GRPOTrainer` expects:

- a `SequenceRolloutBatch` carrying `old_log_probs=`
- sequence-level rewards on the wrapped `GeneratedBatch` via `sequence_rewards=`
- a `GroupRolloutBatch` with explicit `group_size=`

When a `reference_model` is provided, KL regularization is computed from the
reference policy directly. Batch-level reference log probabilities are not part
of the public GRPO surface.

## Configuration

::: artifex.generative_models.training.rl.configs.GRPOConfig
    options:
      show_root_heading: true
      members_order: source

## Related Documentation

- [PPO Trainer](ppo.md)
- [DPO Trainer](dpo.md)
- [RL Training Guide](../user-guide/training/rl-training.md)
