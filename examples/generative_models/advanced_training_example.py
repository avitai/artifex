#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
r"""
# Advanced Training Example

**Author:** Artifex Team
**Last Updated:** 2025-10-22
**Difficulty:** Intermediate
**Runtime:** ~2 minutes
**Format:** Dual (.py script | .ipynb notebook)

## Overview

This comprehensive example demonstrates production-ready training patterns using the
Artifex framework. Learn how to implement robust training loops, manage optimizer
configurations, track metrics, and implement checkpointing strategies.

## Learning Objectives

After completing this example, you will understand:

- [ ] How to implement a complete training pipeline with proper validation
- [ ] Optimizer and learning rate scheduler configuration
- [ ] Metrics tracking and visualization during training
- [ ] Checkpoint management and model persistence
- [ ] Best practices for training loop organization

## Prerequisites

- Basic understanding of neural network training
- Familiarity with JAX and Flax NNX
- Understanding of gradient descent and backpropagation
- Knowledge of learning rate scheduling concepts

## Key Concepts

### Training Loop Components

A production training loop requires several key components:

1. **Data Management**: Efficient batching and shuffling
2. **Optimization**: Gradient computation and parameter updates
3. **Metrics Tracking**: Monitor training and validation performance
4. **Checkpointing**: Save model state for recovery and deployment
5. **Validation**: Monitor generalization to unseen data

### Learning Rate Scheduling

Learning rate schedules improve training by:

- **Warmup**: Gradual increase to avoid instability
- **Decay**: Reduce learning rate as training progresses
- **Cosine Annealing**: Smooth decrease with periodic restarts

The formula for cosine decay is:

$$
\\eta_t = \\eta_{\\text{min}} + \\frac{1}{2}(\\eta_{\\text{max}} - \\eta_{\\text{min}})
\\left(1 + \\cos\\left(\\frac{t}{T}\\pi\\right)\\right)
$$

where $\\eta_t$ is the learning rate at step $t$, and $T$ is the total number of steps.

## Installation

This example requires the Artifex library with standard dependencies:

```bash
pip install artifex[examples]
```

## Usage

Run the Python script directly:

```bash
python examples/generative_models/advanced_training_example.py
```

Or open the notebook:

```bash
jupyter notebook examples/generative_models/advanced_training_example.ipynb
```
"""

# %% [markdown]
"""
## Imports and Setup

Import required modules from JAX, Flax NNX, and Artifex.
"""

# %%
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

from artifex.generative_models.core.configuration import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)


# %% [markdown]
"""
## Data Utilities

Create synthetic datasets and data loaders for demonstration purposes.
"""


# %%
def create_synthetic_dataset(num_samples=1000, input_dim=784, num_classes=10):
    """Create synthetic dataset for demonstration.

    Args:
        num_samples: Number of samples
        input_dim: Input dimension
        num_classes: Number of classes for classification

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    key = jax.random.key(42)
    keys = jax.random.split(key, 6)

    # Create train/val/test splits
    train_x = jax.random.normal(keys[0], (num_samples, input_dim))
    train_y = jax.random.randint(keys[1], (num_samples,), 0, num_classes)

    val_x = jax.random.normal(keys[2], (num_samples // 5, input_dim))
    val_y = jax.random.randint(keys[3], (num_samples // 5,), 0, num_classes)

    test_x = jax.random.normal(keys[4], (num_samples // 5, input_dim))
    test_y = jax.random.randint(keys[5], (num_samples // 5,), 0, num_classes)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


# %%
def create_data_loader(data, batch_size=32, shuffle=True):
    """Create a simple data loader.

    Args:
        data: Tuple of (features, labels)
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Yields:
        Batches of data
    """
    x, y = data
    num_samples = len(x)
    indices = jnp.arange(num_samples)

    if shuffle:
        key = jax.random.key(np.random.randint(0, 10000))
        indices = jax.random.permutation(key, indices)

    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield x[batch_indices], y[batch_indices]


# %% [markdown]
"""
## Metrics Tracker

Track and visualize training and validation metrics over time.
"""


# %%
class TrainingMetrics:
    """Simple metrics tracker for training."""

    def __init__(self):
        """Initialize the metrics tracker."""
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def update(self, metrics: dict[str, float]):
        """Update metrics."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(float(value))

    def plot(self, save_path=None):
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        if self.history["train_loss"]:
            ax1.plot(self.history["train_loss"], label="Train")
        if self.history["val_loss"]:
            ax1.plot(self.history["val_loss"], label="Validation")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot accuracy
        if self.history["train_acc"]:
            ax2.plot(self.history["train_acc"], label="Train")
        if self.history["val_acc"]:
            ax2.plot(self.history["val_acc"], label="Validation")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig


# %% [markdown]
"""
## Model Definition

Define a simple classifier using Flax NNX for demonstration.
"""


# %%
class SimpleClassifier(nnx.Module):
    """Simple classifier for demonstration."""

    def __init__(self, input_dim, hidden_dims, num_classes, *, rngs: nnx.Rngs):
        """Initialize the classifier.

        Args:
            input_dim: Input dimension.
            hidden_dims: List of hidden layer dimensions.
            num_classes: Number of output classes.
            rngs: Random number generators.
        """
        super().__init__()

        layers: list[nnx.Module] = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            layers.append(nnx.relu)
            layers.append(nnx.Dropout(rate=0.1, rngs=rngs))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nnx.Linear(prev_dim, num_classes, rngs=rngs))

        self.net = nnx.Sequential(*layers)

    def __call__(self, x, *, training=False):
        """Forward pass through the classifier.

        Args:
            x: Input tensor.
            training: Whether in training mode.

        Returns:
            Logits for each class.
        """
        # Note: In real code, you'd properly handle dropout training mode
        return self.net(x)


# %% [markdown]
"""
## Optimizer and Scheduler Creation

Create optimizers and learning rate schedulers from configuration objects.
"""


# %%
def create_optimizer(config: OptimizerConfig):
    """Create optimizer from configuration.

    Args:
        config: Optimizer configuration

    Returns:
        Optax optimizer
    """
    if config.optimizer_type == "adam":
        return optax.adam(
            learning_rate=config.learning_rate,
            b1=config.beta1,
            b2=config.beta2,
        )
    elif config.optimizer_type == "sgd":
        return optax.sgd(
            learning_rate=config.learning_rate,
            momentum=config.momentum,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_type}")


# %%
def create_scheduler(base_optimizer, scheduler_config: SchedulerConfig):
    """Create learning rate scheduler.

    Args:
        base_optimizer: Base optimizer
        scheduler_config: Scheduler configuration

    Returns:
        Optimizer with scheduler
    """
    if scheduler_config is None:
        return base_optimizer

    if scheduler_config.scheduler_type == "cosine":
        schedule = optax.cosine_decay_schedule(
            init_value=1.0,
            decay_steps=scheduler_config.total_steps or 10000,
        )
    elif scheduler_config.scheduler_type == "exponential":
        schedule = optax.exponential_decay(
            init_value=1.0,
            transition_steps=scheduler_config.decay_steps,
            decay_rate=scheduler_config.decay_rate,
        )
    else:
        return base_optimizer

    # Combine with base optimizer
    return optax.chain(
        optax.scale_by_schedule(schedule),
        base_optimizer,
    )


# %% [markdown]
"""
## Training and Evaluation Functions

Implement the core training step and evaluation logic.
"""


# %%
def train_step(model, optimizer, batch_x, batch_y, loss_fn):
    """Single training step.

    Args:
        model: Model to train
        optimizer: Optimizer
        batch_x: Input batch
        batch_y: Target batch
        loss_fn: Loss function

    Returns:
        Tuple of (loss, accuracy)
    """

    def compute_loss(model):
        logits = model(batch_x, training=True)

        # Cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y)
        loss = jnp.mean(loss)

        # Accuracy
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == batch_y)

        return loss, accuracy

    (loss, accuracy), grads = nnx.value_and_grad(compute_loss, has_aux=True)(model)
    optimizer.update(model, grads)

    return loss, accuracy


# %%
def evaluate(model, data_loader):
    """Evaluate model on dataset.

    Args:
        model: Model to evaluate
        data_loader: Data loader

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    total_loss = 0
    total_acc = 0
    num_batches = 0

    for batch_x, batch_y in data_loader:
        logits = model(batch_x, training=False)

        # Loss
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y)
        loss = jnp.mean(loss)

        # Accuracy
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == batch_y)

        total_loss += loss
        total_acc += accuracy
        num_batches += 1

    return total_loss / num_batches, total_acc / num_batches


# %% [markdown]
"""
## Checkpointing

Save model checkpoints during training for recovery and deployment.
"""


# %%
def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        checkpoint_dir: Directory to save checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # In real implementation, you'd use orbax or similar
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_dir}")


# %% [markdown]
"""
## Main Training Loop

Execute the complete training pipeline with all components integrated.
"""


# %%
def main():
    """Run the advanced training example."""
    print("=" * 60)
    print("Advanced Training Example")
    print("=" * 60)

    # Configuration
    print("\n1. Setting up configuration...")

    # Model configuration
    model_config = ModelConfig(
        name="classifier",
        model_class="simple_classifier",
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10,
        dropout_rate=0.1,
        parameters={},
    )

    # Optimizer configuration
    optimizer_config = OptimizerConfig(
        name="training_optimizer",
        optimizer_type="adam",
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        weight_decay=1e-4,
    )

    # Scheduler configuration
    scheduler_config = SchedulerConfig(
        name="cosine_scheduler",
        scheduler_type="cosine",
        total_steps=1000,
        warmup_steps=100,
    )

    # Training configuration
    training_config = TrainingConfig(
        name="training",
        batch_size=32,
        num_epochs=10,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        checkpoint_dir="./checkpoints/advanced_example",
        save_frequency=5,
    )

    print(f"  Model: {model_config.name}")
    print(f"  Optimizer: {optimizer_config.optimizer_type}")
    print(f"  Scheduler: {scheduler_config.scheduler_type}")
    print(f"  Epochs: {training_config.num_epochs}")
    print(f"  Batch size: {training_config.batch_size}")

    # Create dataset
    print("\n2. Creating synthetic dataset...")
    train_data, val_data, test_data = create_synthetic_dataset(
        num_samples=1000, input_dim=784, num_classes=10
    )
    print(f"  Train samples: {len(train_data[0])}")
    print(f"  Validation samples: {len(val_data[0])}")
    print(f"  Test samples: {len(test_data[0])}")

    # Create model
    print("\n3. Creating model...")
    key = jax.random.key(42)
    rngs = nnx.Rngs(params=key, dropout=key)

    model = SimpleClassifier(input_dim=784, hidden_dims=[256, 128], num_classes=10, rngs=rngs)
    print(f"  Model created with {len(model_config.hidden_dims)} hidden layers")

    # Create optimizer
    print("\n4. Setting up optimizer and scheduler...")
    base_optimizer = create_optimizer(optimizer_config)
    optimizer_with_schedule = create_scheduler(base_optimizer, scheduler_config)
    optimizer = nnx.Optimizer(model, optimizer_with_schedule, wrt=nnx.All(nnx.Param))

    # Training metrics
    metrics = TrainingMetrics()

    # Training loop
    print("\n5. Starting training...")
    print("-" * 40)

    for epoch in range(training_config.num_epochs):
        # Training
        train_loss = 0
        train_acc = 0
        num_train_batches = 0

        train_loader = create_data_loader(
            train_data, batch_size=training_config.batch_size, shuffle=True
        )

        for batch_x, batch_y in train_loader:
            loss, acc = train_step(model, optimizer, batch_x, batch_y, None)
            train_loss += loss
            train_acc += acc
            num_train_batches += 1

        train_loss /= num_train_batches
        train_acc /= num_train_batches

        # Validation
        val_loader = create_data_loader(
            val_data, batch_size=training_config.batch_size, shuffle=False
        )
        val_loss, val_acc = evaluate(model, val_loader)

        # Update metrics
        metrics.update(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        # Print progress
        print(f"Epoch {epoch + 1}/{training_config.num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save checkpoint
        if (epoch + 1) % training_config.save_frequency == 0:
            save_checkpoint(model, optimizer, epoch + 1, training_config.checkpoint_dir)

    print("-" * 40)

    # Final evaluation on test set
    print("\n6. Evaluating on test set...")
    test_loader = create_data_loader(
        test_data, batch_size=training_config.batch_size, shuffle=False
    )
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    # Plot training curves
    print("\n7. Plotting training curves...")
    output_dir = "examples_output"
    os.makedirs(output_dir, exist_ok=True)

    metrics.plot(save_path=f"{output_dir}/training_curves.png")
    print(f"  Training curves saved to {output_dir}/training_curves.png")

    print()
    print("=" * 60)
    print("✅ Advanced training example completed successfully!")
    print("=" * 60)

    print("\nKey takeaways:")
    print("- Use configuration objects for all settings")
    print("- Implement proper training and validation loops")
    print("- Track metrics throughout training")
    print("- Save checkpoints regularly")
    print("- Evaluate on held-out test set")


# %% [markdown]
"""
## Summary and Next Steps

### Key Takeaways

This example demonstrated:

1. **Configuration Management**: Use Pydantic configuration objects for all settings
2. **Training Pipeline**: Implement proper training and validation loops
3. **Metrics Tracking**: Monitor performance throughout training
4. **Checkpointing**: Save model state regularly for recovery
5. **Learning Rate Scheduling**: Apply adaptive learning rate strategies

### Experiments to Try

1. **Different Optimizers**: Compare Adam, SGD, AdamW performance
2. **Scheduler Variations**: Test exponential decay vs cosine annealing
3. **Architecture Changes**: Experiment with different hidden layer sizes
4. **Regularization**: Add L2 regularization or different dropout rates
5. **Early Stopping**: Implement early stopping based on validation loss

### Next Steps

- Explore more advanced models (VAEs, GANs, Diffusion Models)
- Learn about distributed training strategies
- Study advanced optimization techniques (gradient clipping, mixed precision)
- Implement custom callbacks and monitoring tools

### Additional Resources

- [Artifex Documentation](https://docs.artifex.ai)
- [Flax NNX Guide](https://flax.readthedocs.io/en/latest/)
- [Optax Documentation](https://optax.readthedocs.io/)
- [JAX Training Best Practices](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)
"""

# %%
if __name__ == "__main__":
    main()
