# Fine-Tuning

**Status:** `Current runtime fine-tuning surface`

There is no standalone `artifex.fine_tuning` package in the current runtime.
Artifex currently ships fine-tuning support only through the reinforcement-
learning trainers exported from `artifex.generative_models.training`.

## Available Today

The retained fine-tuning surface is the RL training layer under
`artifex.generative_models.training`:

- [`REINFORCETrainer`](../training/reinforce.md) for policy-gradient updates on typed sequence rollouts
- [`PPOTrainer`](../training/ppo.md) for clipped policy optimization on `SequenceRolloutBatch`
- [`GRPOTrainer`](../training/grpo.md) for grouped rollout optimization with explicit prompt grouping
- [`DPOTrainer`](../training/dpo.md) for preference optimization on `PreferenceBatch`

For shared rollout contracts and orchestration guidance, see the
[RL Training Guide](../user-guide/training/rl-training.md).

## Coming Soon

These topics remain relevant to the roadmap but are not shipped yet as runtime
modules or trainers:

- LoRA
- Prefix Tuning
- Prompt Tuning
- Distillation
- Few-Shot Learning
- Transfer Learning
- RLHF

See [Planned Modules](../roadmap/planned-modules.md) for the current roadmap
status of the broader fine-tuning surface.
