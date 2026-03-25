# CycleGAN

**Module:** `artifex.generative_models.models.gan.cyclegan`

**Source:** `src/artifex/generative_models/models/gan/cyclegan.py`

## Overview

CycleGAN implements unpaired image-to-image translation with two generators and
two discriminators:

- `generator_a_to_b`: translates samples from domain A to domain B
- `generator_b_to_a`: translates samples from domain B to domain A
- `discriminator_a`: scores samples in domain A
- `discriminator_b`: scores samples in domain B

The training objective is split, not unified:

- generator-side objective: adversarial + cycle-consistency + identity terms
- discriminator-side objective: separate adversarial objectives for domain A and domain B

CycleGAN training does not expose a combined `loss_fn(...)`. Use
`generator_objective(...)` and `discriminator_objective(...)` directly, or use
the trainer surface that manages separate optimization steps.

## Public Methods

### `__call__(batch)`

Runs the discriminators on real inputs and returns model outputs for inference
or inspection. It is not the training objective.

### `generate(inputs, direction="a_to_b")`

Translates inputs from one domain to the other.

Supported directions:

- `"a_to_b"`
- `"b_to_a"`

### `generator_objective(batch)`

Returns the generator-side optimization target and metrics. The result includes:

- `total_loss`
- `generator_loss`
- `adversarial_loss`
- `generator_a_to_b_loss`
- `generator_b_to_a_loss`
- `cycle_loss`
- `identity_loss`

### `discriminator_objective(batch)`

Returns the discriminator-side optimization target and metrics. The result includes:

- `total_loss`
- `discriminator_loss`
- `discriminator_a_loss`
- `discriminator_b_loss`

## Training Example

```python
generator_metrics = cyclegan.generator_objective(batch)
discriminator_metrics = cyclegan.discriminator_objective(batch)

generator_loss = generator_metrics["total_loss"]
discriminator_loss = discriminator_metrics["total_loss"]
```

## Design Notes

- CycleGAN is a multi-objective adversarial family.
- The model intentionally rejects a fake single-objective `loss_fn(...)` entrypoint.
- Trainer code should optimize generator and discriminator objectives in separate steps.
