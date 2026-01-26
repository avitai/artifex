# Training Strategies

Advanced training strategies and patterns for Artifex generative models.

## Overview

This guide covers advanced training strategies for optimizing your generative model training workflows.

<div class="grid cards" markdown>

- :material-layers-triple:{ .lg .middle } **Gradient Accumulation**

    ---

    Train with larger effective batch sizes on limited memory

    [:octicons-arrow-right-24: Gradient Accumulation](#gradient-accumulation)

- :material-speedometer:{ .lg .middle } **Mixed Precision Training**

    ---

    Speed up training with bfloat16 computation

    [:octicons-arrow-right-24: Mixed Precision](#mixed-precision-training)

- :material-scatter-plot:{ .lg .middle } **Distributed Training**

    ---

    Scale training across multiple devices

    [:octicons-arrow-right-24: Distributed Training](#distributed-training)

- :material-tune:{ .lg .middle } **Curriculum Learning**

    ---

    Progressive training for improved convergence

    [:octicons-arrow-right-24: Curriculum Learning](#curriculum-learning)

</div>

---

## Gradient Accumulation

Accumulate gradients over multiple steps to simulate larger batch sizes.

```python
from artifex.generative_models.training import Trainer, TrainingConfig

config = TrainingConfig(
    num_epochs=100,
    batch_size=16,
    gradient_accumulation_steps=4,  # Effective batch size: 64
)

trainer = Trainer(model=model, training_config=config)
trainer.train(train_data)
```

---

## Mixed Precision Training

Use bfloat16 for faster training while maintaining accuracy.

```python
import jax

# Enable mixed precision
config = TrainingConfig(
    dtype="bfloat16",
    num_epochs=100,
)

trainer = Trainer(model=model, training_config=config)
```

---

## Distributed Training

Scale training across multiple GPUs or TPUs.

```python
import jax

# Detect available devices
devices = jax.local_devices()
print(f"Available devices: {len(devices)}")

# Data parallel training
config = TrainingConfig(
    num_epochs=100,
    distributed=True,
)
```

---

## Curriculum Learning

Progressive training from simple to complex samples.

```python
def create_curriculum_loader(dataset, epoch, max_epochs):
    """Create data loader with curriculum difficulty."""
    difficulty = min(1.0, epoch / (max_epochs * 0.5))
    return filter_by_difficulty(dataset, max_difficulty=difficulty)
```

---

## Related Documentation

- [Training Guide](../../user-guide/training/training-guide.md) - Core training concepts
- [Configuration Guide](../../user-guide/training/configuration.md) - Training configuration options
- [Framework Features Demo](framework-features-demo.md) - Comprehensive framework example
