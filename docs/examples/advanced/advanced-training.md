# Advanced Training Pipeline

**Level:** Intermediate
**Runtime:** ~2 minutes
**Format:** Dual (.py script | .ipynb notebook)

Production-ready training patterns including optimizer configuration, learning rate scheduling, metrics tracking, and checkpointing strategies.

## Files

- **Python Script:** [`examples/generative_models/advanced_training_example.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/advanced_training_example.py)
- **Jupyter Notebook:** [`examples/generative_models/advanced_training_example.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/advanced_training_example.ipynb)

## Quick Start

```bash
# Run the Python script
python examples/generative_models/advanced_training_example.py

# Or open the Jupyter notebook
jupyter notebook examples/generative_models/advanced_training_example.ipynb
```

## Overview

This example demonstrates how to build a complete, production-ready training pipeline using the Artifex framework. You'll learn essential patterns for training deep learning models including configuration management, optimization strategies, metrics tracking, and model checkpointing.

### Learning Objectives

After completing this example, you will understand:

- [x] How to implement a complete training pipeline with proper validation
- [x] Optimizer and learning rate scheduler configuration
- [x] Metrics tracking and visualization during training
- [x] Checkpoint management and model persistence
- [x] Best practices for training loop organization

### Prerequisites

- Basic understanding of neural network training
- Familiarity with JAX and Flax NNX
- Understanding of gradient descent and backpropagation
- Knowledge of learning rate scheduling concepts

## Theory and Key Concepts

### Training Loop Components

A production training loop requires several key components working together:

1. **Data Management**: Efficient batching and shuffling strategies
2. **Optimization**: Gradient computation and parameter updates
3. **Metrics Tracking**: Monitor training and validation performance
4. **Checkpointing**: Save model state for recovery and deployment
5. **Validation**: Monitor generalization to unseen data

### Learning Rate Scheduling

Learning rate schedules improve training stability and convergence by adapting the learning rate during training:

**Warmup**: Gradually increase learning rate from zero to avoid early instability
**Decay**: Reduce learning rate as training progresses to enable fine-grained convergence
**Cosine Annealing**: Smooth decrease following a cosine curve

The formula for cosine decay is:

$$
\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)
$$

where $\eta_t$ is the learning rate at step $t$, and $T$ is the total number of steps.

### Optimization Algorithms

**Adam (Adaptive Moment Estimation)**: Combines momentum and adaptive learning rates

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

**SGD with Momentum**: Accelerates convergence by accumulating gradients

$$
v_t = \mu v_{t-1} + \eta g_t
$$

$$
\theta_{t+1} = \theta_t - v_t
$$

## Code Walkthrough

### 1. Configuration Setup

The example uses frozen dataclass configuration objects for all settings:

```python
from dataclasses import dataclass

# Model configuration (simple frozen dataclass)
@dataclass(frozen=True)
class ClassifierConfig:
    """Configuration for a simple classifier."""
    name: str = "classifier"
    input_dim: int = 784
    hidden_dims: tuple[int, ...] = (256, 128)  # Tuple for frozen dataclass
    output_dim: int = 10
    dropout_rate: float = 0.1

# Training configuration using Artifex's ModelConfig
from artifex.generative_models.core.configuration import ModelConfig

# ModelConfig is a general-purpose training configuration
config = ModelConfig(
    name="advanced_training",
    batch_size=32,
    num_epochs=10,
    learning_rate=1e-3,
    optimizer_type="adam",
    checkpoint_dir="./checkpoints/advanced_example",
    save_frequency=5,
    # Scheduler settings
    scheduler_type="cosine",
    warmup_steps=100,
    total_steps=1000,
    # Optimizer settings
    beta1=0.9,
    beta2=0.999,
    weight_decay=1e-4,
)

# Create model config
model_config = ClassifierConfig(
    input_dim=784,
    hidden_dims=(256, 128),  # Tuple for frozen dataclass
    output_dim=10,
    dropout_rate=0.1,
)
```

This approach centralizes all hyperparameters using frozen dataclasses, making experiments reproducible and configuration management type-safe.

### 2. Data Loading

The example implements a simple data loader with shuffling:

```python
def create_data_loader(data, batch_size=32, shuffle=True):
    """Create a simple data loader."""
    x, y = data
    num_samples = len(x)
    indices = jnp.arange(num_samples)

    if shuffle:
        key = jax.random.key(np.random.randint(0, 10000))
        indices = jax.random.permutation(key, indices)

    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield x[batch_indices], y[batch_indices]
```

In production, you would use more sophisticated data loading strategies like TensorFlow Datasets or PyTorch DataLoader equivalents.

### 3. Model Definition

A simple classifier using Flax NNX demonstrates proper module patterns:

```python
class SimpleClassifier(nnx.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, *, rngs: nnx.Rngs):
        super().__init__()  # Always call this

        layers = []
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
        return self.net(x)
```

### 4. Training Step

The core training step computes loss, gradients, and updates parameters:

```python
def train_step(model, optimizer, batch_x, batch_y, loss_fn):
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
```

This pattern uses NNX's `value_and_grad` for efficient gradient computation with auxiliary outputs (accuracy).

### 5. Main Training Loop

The main loop orchestrates all components:

```python
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
    metrics.update({
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    })

    # Save checkpoint
    if (epoch + 1) % training_config.save_frequency == 0:
        save_checkpoint(model, optimizer, epoch + 1, training_config.checkpoint_dir)
```

### 6. Metrics Tracking

The example includes a custom metrics tracker with visualization:

```python
class TrainingMetrics:
    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def update(self, metrics: dict[str, float]):
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(float(value))

    def plot(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        # Plot loss and accuracy curves
        # ...
```

This enables real-time monitoring and post-training analysis.

## Expected Output

When you run the example, you should see:

```
============================================================
Advanced Training Example
============================================================

1. Setting up configuration...
  Model: classifier
  Optimizer: adam
  Scheduler: cosine
  Epochs: 10
  Batch size: 32

2. Creating synthetic dataset...
  Train samples: 1000
  Validation samples: 200
  Test samples: 200

3. Creating model...
  Model created with 2 hidden layers

4. Setting up optimizer and scheduler...

5. Starting training...
----------------------------------------
Epoch 1/10
  Train - Loss: 2.3965, Acc: 0.1025
  Val   - Loss: 2.3957, Acc: 0.1071
...
Epoch 10/10
  Train - Loss: 0.0137, Acc: 1.0000
  Val   - Loss: 4.3435, Acc: 0.0670
----------------------------------------

6. Evaluating on test set...
  Test Loss: 4.1130
  Test Accuracy: 0.1205

7. Plotting training curves...
  Training curves saved to examples_output/training_curves.png

✅ Advanced training example completed successfully!
```

The example will also save a visualization of the training curves to `examples_output/training_curves.png`.

## Experiments to Try

1. **Different Optimizers**: Compare Adam, SGD with momentum, and AdamW

   ```python
   optimizer_config.optimizer_type = "sgd"
   optimizer_config.momentum = 0.9
   ```

2. **Scheduler Variations**: Test exponential decay vs cosine annealing

   ```python
   scheduler_config.scheduler_type = "exponential"
   scheduler_config.decay_steps = 100
   scheduler_config.decay_rate = 0.96
   ```

3. **Architecture Changes**: Experiment with different hidden layer configurations

   ```python
   model_config.hidden_dims = [512, 256, 128]  # Deeper network
   ```

4. **Regularization**: Adjust dropout and weight decay

   ```python
   model_config.dropout_rate = 0.2
   optimizer_config.weight_decay = 1e-3
   ```

5. **Early Stopping**: Implement early stopping based on validation loss

   ```python
   # Track best validation loss
   # Stop training if no improvement for N epochs
   ```

## Troubleshooting

### High Validation Loss

If validation loss is much higher than training loss:

- Reduce model complexity or add regularization
- Increase dropout rate
- Add weight decay to optimizer
- Use more training data

### Slow Convergence

If training is slow to converge:

- Increase learning rate (carefully)
- Use a learning rate warmup
- Try a different optimizer (e.g., Adam instead of SGD)
- Check gradient magnitudes

### Numerical Instability

If you encounter NaN or Inf values:

- Reduce learning rate
- Add gradient clipping
- Use mixed precision training
- Check for exploding/vanishing gradients

## Next Steps

<div class="grid cards" markdown>

- :material-brain: **VAE Training**

    ---

    Learn to train Variational Autoencoders with the ELBO loss

    [:octicons-arrow-right-24: VAE Examples](../basic/vae-mnist.md)

- :material-image-multiple: **GAN Training**

    ---

    Master adversarial training with generator and discriminator

    [:octicons-arrow-right-24: GAN Examples](../basic/simple-gan.md)

- :material-chart-line: **Advanced Optimization**

    ---

    Explore gradient clipping, mixed precision, and distributed training

    [:octicons-arrow-right-24: Advanced Techniques](../framework/training-strategies.md)

- :material-package-variant: **Model Deployment**

    ---

    Learn to export and deploy trained models

    [:octicons-arrow-right-24: Deployment Guide](../framework/model-deployment.md)

</div>

## Additional Resources

- [Flax NNX Training Guide](https://flax.readthedocs.io/en/latest/guides/training.html)
- [Optax Documentation](https://optax.readthedocs.io/)
- [JAX Training Best Practices](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)
- [Artifex Training Configuration](../../api/core/configuration.md)
