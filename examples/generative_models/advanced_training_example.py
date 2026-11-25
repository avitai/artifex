# %% [markdown]
"""
# Training a Classifier from Artifex's Optimizer and Scheduler Configs

| Metadata | Value |
| --- | --- |
| **Tier** | Tutorial |
| **Devices** | CPU-compatible |
| **Prerequisites** | Flax NNX modules, optax optimizers |

## Overview

Artifex describes an optimizer and its learning-rate schedule as typed configurations,
`OptimizerConfig` and `SchedulerConfig`, and builds them with `create_scheduler` and
`create_optimizer`. This example builds AdamW with decoupled weight decay and global-norm gradient
clipping, under a schedule that warms up linearly and then follows a cosine, trains a small
classifier on synthetic data with a jitted step, and evaluates it in eval mode.

## Learning Goals

1. Describe an optimizer and a learning-rate schedule with artifex's typed configurations
2. Build the schedule and the optimizer from them, and check the schedule's values
3. Train with dropout keys passed at call time, and evaluate in eval mode
"""

# %% [markdown]
"""
## Setup
"""

# %%
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from artifex.generative_models.core.configuration import OptimizerConfig, SchedulerConfig
from artifex.generative_models.training import create_optimizer, create_scheduler


# %% [markdown]
"""
## Core Concepts

The `cosine` schedule with `warmup_steps > 0` rises linearly from 0 to the peak learning rate over
`warmup_steps`, then decays along a cosine to `min_lr_ratio` times the peak at step
`cycle_length`. `create_optimizer` hands the optimizer configuration to substrax, which applies
the weight decay decoupled from the gradient (AdamW) and clips each update's gradients to a global
norm of `gradient_clip_norm` before the step.

## Implementation

### Step 1: Describe the optimizer and the schedule
"""

# %%
STEPS = 300
PEAK_LEARNING_RATE = 3e-3

optimizer_config = OptimizerConfig(
    name="adamw",
    optimizer_type="adamw",
    learning_rate=PEAK_LEARNING_RATE,
    weight_decay=1e-4,
    gradient_clip_norm=1.0,
)
scheduler_config = SchedulerConfig(
    name="warmup_cosine",
    scheduler_type="cosine",
    warmup_steps=30,
    cycle_length=STEPS,
    min_lr_ratio=0.1,
)


# %% [markdown]
"""
### Step 2: Make a dataset

Four classes of 16 features, each class a Gaussian cloud around its own centre, drawn from a typed
key so every run sees the same data.
"""


# %%
def make_dataset(key: jax.Array, count: int) -> tuple[jax.Array, jax.Array]:
    """Draw labelled points from four Gaussian clouds.

    Args:
        key: The typed key the draw uses.
        count: The number of points.

    Returns:
        The features, shaped ``(count, 16)``, and the integer labels, shaped ``(count,)``.
    """
    centres_key, labels_key, noise_key = jax.random.split(key, 3)
    centres = 2.0 * jax.random.normal(centres_key, (4, 16))
    labels = jax.random.randint(labels_key, (count,), 0, 4)
    features = centres[labels] + jax.random.normal(noise_key, (count, 16))
    return features, labels


# %% [markdown]
"""
### Step 3: Define the classifier

The dropout layer holds no key of its own: a training call passes `rngs`, and `.eval()` turns
dropout off.
"""


# %%
class Classifier(nnx.Module):
    """A two-layer classifier with dropout between the layers."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        """Build the layers.

        Args:
            rngs: Keys for the parameters' initialisation.
        """
        self.hidden = nnx.Linear(16, 64, rngs=rngs)
        self.dropout = nnx.Dropout(0.2)
        self.out = nnx.Linear(64, 4, rngs=rngs)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Return the class logits.

        Args:
            x: The features, shaped ``(batch, 16)``.
            rngs: Dropout keys for this call; required to train with dropout.

        Returns:
            The logits, shaped ``(batch, 4)``.
        """
        return self.out(self.dropout(nnx.relu(self.hidden(x)), rngs=rngs))


# %% [markdown]
"""
### Step 4: Build the schedule and the optimizer, then train

Each step draws its dropout key by folding the step number into one base key, so the jitted step
sees a new key every call.
"""


# %%
@nnx.jit
def train_step(
    model: Classifier, optimizer: nnx.Optimizer, x: jax.Array, y: jax.Array, key: jax.Array
) -> jax.Array:
    """Take one optimizer step on a batch and return its loss.

    Args:
        model: The classifier, in train mode.
        optimizer: The optimizer that updates its parameters.
        x: The batch's features.
        y: The batch's labels.
        key: The dropout key for this step.

    Returns:
        The batch's mean cross-entropy.
    """

    def loss_fn(model: Classifier) -> jax.Array:
        logits = model(x, rngs=nnx.Rngs(dropout=key))
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def accuracy(model: Classifier, x: jax.Array, y: jax.Array) -> float:
    """Return the fraction of points the model labels correctly, in eval mode.

    Args:
        model: The classifier.
        x: The features.
        y: The labels.

    Returns:
        The accuracy.
    """
    model.eval()
    return float(jnp.mean(jnp.argmax(model(x), axis=-1) == y))


# %% [markdown]
"""
## Results

### Schedule, loss and accuracy

`main()` checks the schedule against its closed form at three steps (start, end of warmup, end of
the cycle), trains for `STEPS` steps in batches of 64, and reports the first and last batch losses
and the held-out accuracy in eval mode.
"""


# %%
def main() -> dict[str, float]:
    """Run the example and return its results.

    Returns:
        The learning rate at the start, the end of warmup and the end of the cycle, the first and
        last training losses, and the held-out accuracy.
    """
    schedule = create_scheduler(scheduler_config, base_lr=optimizer_config.learning_rate)
    rates = {
        "lr_start": float(jnp.asarray(schedule(0))),
        "lr_warmup_end": float(jnp.asarray(schedule(scheduler_config.warmup_steps))),
        "lr_cycle_end": float(jnp.asarray(schedule(scheduler_config.cycle_length))),
    }
    print(f"learning rate at steps 0, 30, 300: {rates}")
    print(f"closed form: 0, {PEAK_LEARNING_RATE}, {PEAK_LEARNING_RATE * 0.1:.6g}")

    data_key, model_key, dropout_key = jax.random.split(jax.random.key(0), 3)
    features, labels = make_dataset(data_key, 2048)
    train_x, test_x = features[:1536], features[1536:]
    train_y, test_y = labels[:1536], labels[1536:]

    model = Classifier(rngs=nnx.Rngs(params=model_key))
    optimizer = nnx.Optimizer(
        model, create_optimizer(model, optimizer_config, schedule), wrt=nnx.Param
    )

    model.train()
    losses = []
    for step in range(STEPS):
        start = (step * 64) % train_x.shape[0]
        batch = slice(start, start + 64)
        key = jax.random.fold_in(dropout_key, step)
        losses.append(float(train_step(model, optimizer, train_x[batch], train_y[batch], key)))

    test_accuracy = accuracy(model, test_x, test_y)
    print(f"loss at the first and last step: {losses[0]:.4f}, {losses[-1]:.4f}")
    print(f"held-out accuracy in eval mode: {test_accuracy:.4f}")
    return {**rates, "first_loss": losses[0], "last_loss": losses[-1], "accuracy": test_accuracy}


if __name__ == "__main__":
    main()


# %% [markdown]
"""
## Troubleshooting

### `ValueError: cycle_length must be specified for cosine scheduler`

Symptom: `create_scheduler` raises before training. Cause: the `cosine` schedule decays over
`cycle_length` steps and has no default. Solution: set `cycle_length` to the number of training
steps.

### `ValueError` naming a weight decay on `adam`

Symptom: `create_optimizer` refuses the configuration. Cause: plain Adam has no decoupled weight
decay. Solution: use `optimizer_type="adamw"`, or set `weight_decay=0.0`.

## Next Steps

- [The shared example guide](https://substrax.readthedocs.io/en/latest/guides/writing-examples/)
- [Training guide](https://artifex.readthedocs.io/en/latest/user-guide/training/training-guide/)

### Experiments to Try

- Set `warmup_steps=0` and compare the first losses.
- Raise `gradient_clip_norm` to 10 and watch the early loss curve.
- Change `scheduler_type` to `"linear"` with `total_steps=STEPS`.
"""
