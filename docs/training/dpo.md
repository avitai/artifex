# DPO Trainer

**Status:** `Supported runtime training surface`

**Module:** `artifex.generative_models.training.rl.dpo`

**Source:** `src/artifex/generative_models/training/rl/dpo.py`

`DPOTrainer` scores typed preference batches built from sequence-native
generation contracts.

## Quick Start

```python
from flax import nnx
import optax

from artifex.generative_models.training import DPOConfig, DPOTrainer
from artifex.generative_models.training.rl import (
    GeneratedSequenceBatch,
    PreferenceBatch,
)

model = PolicyModel(rngs=nnx.Rngs(0))
reference_model = PolicyModel(rngs=nnx.Rngs(1))
optimizer = nnx.Optimizer(model, optax.adam(1e-5), wrt=nnx.Param)

trainer = DPOTrainer(
    model=model,
    reference_model=reference_model,
    optimizer=optimizer,
    config=DPOConfig(beta=0.1, label_smoothing=0.0),
)

batch = PreferenceBatch(
    chosen=GeneratedSequenceBatch.from_sequences(
        chosen_sequences,
        response_mask=chosen_loss_mask,
    ),
    rejected=GeneratedSequenceBatch.from_sequences(
        rejected_sequences,
        response_mask=rejected_loss_mask,
    ),
)

loss, metrics = trainer.train_step(batch)
```

## Typed Preference Contract

`DPOTrainer` consumes `PreferenceBatch[GeneratedSequenceBatch]`.

- `PreferenceBatch` keeps chosen and rejected samples aligned
- `GeneratedSequenceBatch` carries the token sequences and optional
  `response_mask`
- `chosen_loss_mask` and `rejected_loss_mask` should be converted into the
  corresponding `response_mask` values on each side

## SimPO / Reference-Free Mode

```python
trainer = DPOTrainer(
    model=model,
    reference_model=None,
    optimizer=optimizer,
    config=DPOConfig(beta=0.1, reference_free=True),
)
```

## Configuration

::: artifex.generative_models.training.rl.configs.DPOConfig
    options:
      show_root_heading: true
      members_order: source

## Related Documentation

- [GRPO Trainer](grpo.md)
- [RL Training Guide](../user-guide/training/rl-training.md)
