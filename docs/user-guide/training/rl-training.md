# Reinforcement Learning Training

Artifex RL training is sequence-first and type-driven. The RL trainers are
standalone optimizer helpers, not subclasses of the shared epoch-oriented
`Trainer`.

## Trainer Shape

The RL trainers are standalone optimizer helpers. They own algorithm-specific
loss computation and a `train_step(...)`, but they do not own a callback-driven
epoch loop.

If you need surrounding orchestration, create it explicitly and use
`CallbackList` in that outer loop.

## Core Typed Contracts

The shared RL surface is built from four main batch types:

- `GeneratedBatch`
- `GeneratedSequenceBatch`
- `SequenceRolloutBatch`
- `GroupRolloutBatch`

Preference learning adds:

- `PreferenceBatch`

`GeneratedSequenceBatch` stores token sequences plus optional `response_mask`
and prompt masking. `SequenceRolloutBatch` adds rollout tensors such as
`old_log_probs`, token rewards, returns, and advantages. `GroupRolloutBatch`
adds prompt-group structure for grouped algorithms such as GRPO.

## REINFORCE

```python
from flax import nnx
import optax

from artifex.generative_models.training import REINFORCEConfig, REINFORCETrainer
from artifex.generative_models.training.rl import (
    GeneratedBatch,
    GeneratedSequenceBatch,
    SequenceRolloutBatch,
)

model = PolicyModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)
trainer = REINFORCETrainer(model, optimizer, REINFORCEConfig())

rollout = SequenceRolloutBatch(
    sequence_batch=GeneratedSequenceBatch(
        generation=GeneratedBatch(outputs=token_sequences, rewards=sequence_rewards),
        response_mask=response_mask,
    ),
    token_rewards=token_rewards,
)

loss, metrics = trainer.train_step(rollout)
```

## PPO

```python
from artifex.generative_models.training import PPOConfig, PPOTrainer
from artifex.generative_models.training.rl import (
    GeneratedBatch,
    GeneratedSequenceBatch,
    SequenceRolloutBatch,
)

trainer = PPOTrainer(model, optimizer, PPOConfig())

rollout = SequenceRolloutBatch(
    sequence_batch=GeneratedSequenceBatch(
        generation=GeneratedBatch(outputs=token_sequences),
        response_mask=response_mask,
    ),
    old_log_probs=old_log_probs,
    returns=returns,
    advantages=advantages,
    dones=dones,
)

loss, metrics = trainer.train_step(rollout)
```

`PPOTrainer` requires `old_log_probs=`, `returns=`, and `advantages=` on the
typed rollout batch.

## GRPO

```python
from artifex.generative_models.training import GRPOConfig, GRPOTrainer
from artifex.generative_models.training.rl import (
    GeneratedBatch,
    GeneratedSequenceBatch,
    GroupRolloutBatch,
    SequenceRolloutBatch,
)

trainer = GRPOTrainer(
    model,
    optimizer,
    GRPOConfig(num_generations=4),
    reference_model=reference_model,
)

rollout = SequenceRolloutBatch(
    sequence_batch=GeneratedSequenceBatch(
        generation=GeneratedBatch(outputs=token_sequences, rewards=sequence_rewards),
        response_mask=response_mask,
    ),
    old_log_probs=old_log_probs,
)
grouped_rollout = GroupRolloutBatch(rollout=rollout, group_size=4)

loss, metrics = trainer.train_step(grouped_rollout)
```

GRPO uses `GroupRolloutBatch` so prompt-level grouping is explicit. When
`reference_model` is provided, KL penalties are computed from that model rather
than from batch-carried reference log probabilities.

## DPO

```python
from artifex.generative_models.training import DPOConfig, DPOTrainer
from artifex.generative_models.training.rl import (
    GeneratedSequenceBatch,
    PreferenceBatch,
)

trainer = DPOTrainer(
    model=model,
    reference_model=reference_model,
    optimizer=optimizer,
    config=DPOConfig(beta=0.1),
)

preference_batch = PreferenceBatch(
    chosen=GeneratedSequenceBatch.from_sequences(
        chosen_sequences,
        response_mask=chosen_loss_mask,
    ),
    rejected=GeneratedSequenceBatch.from_sequences(
        rejected_sequences,
        response_mask=rejected_loss_mask,
    ),
)

loss, metrics = trainer.train_step(preference_batch)
```

For prompt-conditioned preference data, convert `chosen_loss_mask` and
`rejected_loss_mask` into the chosen and rejected `response_mask` values on the
typed `GeneratedSequenceBatch` instances.

## Orchestrating RL Runs

RL trainers do not inherit from the shared low-level `Trainer`, but you can
still use callback utilities around your own rollout loop:

```python
from artifex.generative_models.training.callbacks import CallbackList

callbacks = CallbackList([])

for step, rollout in enumerate(rollout_stream):
    callbacks.on_batch_begin(None, step)
    loss, metrics = trainer.train_step(rollout)
    callbacks.on_batch_end(None, step, metrics)
```

## Related Documentation

- [REINFORCE Reference](../../training/reinforce.md)
- [PPO Reference](../../training/ppo.md)
- [GRPO Reference](../../training/grpo.md)
- [DPO Reference](../../training/dpo.md)
