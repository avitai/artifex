# Autoregressive Models User Guide

Complete guide to building, training, and using Autoregressive Models with Artifex.

## Overview

This guide covers practical usage of autoregressive models in Artifex, from basic setup to advanced generation techniques. You'll learn how to:

<div class="grid cards" markdown>

- :material-cog: **Configure AR Models**

    ---

    Set up PixelCNN, WaveNet, and Transformer architectures

- :material-play: **Train Models**

    ---

    Train with teacher forcing and monitor perplexity

- :material-creation: **Generate Samples**

    ---

    Sequential generation with various sampling strategies

- :material-tune: **Optimize & Sample**

    ---

    Tune generation quality with temperature, top-k, and nucleus sampling

</div>

---

## Quick Start

### Basic Transformer Example

```python
import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.autoregressive_config import (
    TransformerConfig,
    TransformerNetworkConfig,
)
from artifex.generative_models.models.autoregressive import TransformerAutoregressiveModel

# Initialize RNGs
rngs = nnx.Rngs(params=0, dropout=1, sample=2)

# Configure the transformer network
network_config = TransformerNetworkConfig(
    name="transformer_network",
    hidden_dims=(512, 512),  # Hidden layer dimensions (required)
    activation="gelu",        # Activation function (required)
    embed_dim=512,
    num_heads=8,
    mlp_ratio=4.0,
    positional_encoding="sinusoidal",
    dropout_rate=0.1,
)

# Configure the transformer model
config = TransformerConfig(
    name="text_transformer",
    vocab_size=10000,
    sequence_length=512,
    network=network_config,
    num_layers=6,
    use_cache=True,
)

# Create model
model = TransformerAutoregressiveModel(config, rngs=rngs)

# Training: forward pass
sequences = jnp.array([[1, 2, 3, 4, 5]])  # [batch, seq_len]
outputs = model(sequences)
logits = outputs["logits"]  # [batch, seq_len, vocab_size]

print(f"Logits shape: {logits.shape}")

# Generation
model.eval()
samples = model.generate(n_samples=4, max_length=128, temperature=0.8)
print(f"Generated samples shape: {samples.shape}")
```

---

## Creating Autoregressive Models

### 1. PixelCNN (Image Generation)

For autoregressive image generation with masked convolutions:

```python
from artifex.generative_models.core.configuration.autoregressive_config import PixelCNNConfig
from artifex.generative_models.models.autoregressive import PixelCNN

# Configure PixelCNN for MNIST (28×28 grayscale)
config = PixelCNNConfig(
    name="pixelcnn_mnist",
    image_shape=(28, 28, 1),
    hidden_channels=128,
    num_layers=7,
    num_residual_blocks=5,
    kernel_size=3,
    dropout_rate=0.1,
)

# Create model
model = PixelCNN(config, rngs=rngs)

# Training
images = jnp.zeros((16, 28, 28, 1), dtype=jnp.int32)  # Values in [0, 255]
outputs = model(images, rngs=rngs, training=True)

# Loss
batch = {"images": images}
loss_dict = model.loss_fn(batch, outputs, rngs=rngs)
print(f"Loss: {loss_dict['loss']:.4f}")
print(f"Bits per dim: {loss_dict['bits_per_dim']:.4f}")

# Generation (pixel by pixel)
generated = model.generate(
    n_samples=16,
    temperature=1.0,
    rngs=rngs
)
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_shape` | - | (height, width, channels) |
| `num_layers` | 7 | Number of masked conv layers |
| `hidden_channels` | 128 | Hidden layer channels |
| `num_residual_blocks` | 5 | Residual block count |

**Use Cases:**

- Density estimation on images
- Lossless image compression
- Inpainting with spatial conditioning

### 2. WaveNet (Audio Generation)

For raw audio waveform modeling:

```python
from artifex.generative_models.core.configuration.autoregressive_config import WaveNetConfig
from artifex.generative_models.models.autoregressive import WaveNet

# Configure WaveNet
config = WaveNetConfig(
    name="wavenet_audio",
    vocab_size=256,            # 8-bit mu-law encoding
    sequence_length=16000,     # 1 second at 16kHz
    residual_channels=128,
    skip_channels=512,
    num_blocks=3,              # Number of dilation stacks
    layers_per_block=10,       # Layers per stack
    kernel_size=2,
    dilation_base=2,
    use_gated_activation=True,
)

# Create WaveNet
model = WaveNet(config, rngs=rngs)

# Training
waveform = jnp.zeros((4, 16000), dtype=jnp.int32)  # 1 second at 16kHz
outputs = model(waveform, rngs=rngs)

# Loss
batch = {"waveform": waveform}
loss_dict = model.loss_fn(batch, outputs, rngs=rngs)

# Generation
generated_audio = model.generate(
    n_samples=1,
    max_length=16000,  # 1 second
    temperature=0.9,
    rngs=rngs
)
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_layers` | 30 | Total dilated conv layers |
| `num_stacks` | 3 | Number of dilation stacks |
| `residual_channels` | 128 | Residual connection channels |
| `dilation_channels` | 256 | Dilated conv channels |

**Dilation Pattern:**

```python
# WaveNet uses exponentially increasing dilations
# Stack 1: dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# Stack 2: dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# Stack 3: dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# Receptive field = 1024 time steps
```

### 3. Transformer (Sequence Modeling)

For text, code, and general sequences:

```python
from artifex.generative_models.core.configuration.autoregressive_config import (
    TransformerConfig,
    TransformerNetworkConfig,
)
from artifex.generative_models.models.autoregressive import TransformerAutoregressiveModel

# Configure the transformer network
network_config = TransformerNetworkConfig(
    name="transformer_network",
    hidden_dims=(768, 768),
    activation="gelu",
    embed_dim=768,
    num_heads=12,
    mlp_ratio=4.0,
    positional_encoding="sinusoidal",
    dropout_rate=0.1,
)

# Configure the transformer model
config = TransformerConfig(
    name="text_transformer",
    vocab_size=50000,
    sequence_length=1024,
    network=network_config,
    num_layers=12,
    dropout_rate=0.1,
    use_cache=True,
)

# Create model
model = TransformerAutoregressiveModel(config, rngs=rngs)

# Training
sequences = jnp.zeros((8, 512), dtype=jnp.int32)  # [batch, seq_len]
outputs = model(sequences, rngs=rngs, training=True)
logits = outputs['logits']

# Compute loss
batch = {"sequences": sequences}
loss_dict = model.loss_fn(batch, outputs, rngs=rngs)
print(f"NLL Loss: {loss_dict['nll_loss']:.4f}")
print(f"Perplexity: {loss_dict['perplexity']:.2f}")
print(f"Accuracy: {loss_dict['accuracy']:.4f}")
```

**Architecture Scaling:**

```python
from artifex.generative_models.core.configuration.autoregressive_config import (
    TransformerConfig,
    TransformerNetworkConfig,
)

# Small (for experiments)
small_config = TransformerConfig(
    name="small_transformer",
    vocab_size=50000,
    sequence_length=512,
    network=TransformerNetworkConfig(
        name="small_net",
        hidden_dims=(256, 256),
        activation="gelu",
        embed_dim=256,
        num_heads=4,
        mlp_ratio=4.0,
    ),
    num_layers=4,
)

# Medium (GPT-2 small)
medium_config = TransformerConfig(
    name="medium_transformer",
    vocab_size=50000,
    sequence_length=1024,
    network=TransformerNetworkConfig(
        name="medium_net",
        hidden_dims=(768, 768),
        activation="gelu",
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
    ),
    num_layers=12,
)

# Large (GPT-2 medium)
large_config = TransformerConfig(
    name="large_transformer",
    vocab_size=50000,
    sequence_length=1024,
    network=TransformerNetworkConfig(
        name="large_net",
        hidden_dims=(1024, 1024),
        activation="gelu",
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
    ),
    num_layers=24,
)
```

---

## Training Autoregressive Models

### Teacher Forcing Training

Standard training uses ground truth previous tokens:

```python
def train_step(model, batch, optimizer_state):
    """Standard teacher forcing training step."""

    def loss_fn(model):
        # Forward pass with ground truth input
        outputs = model(batch['sequences'], training=True, rngs=rngs)

        # Compute loss
        loss_dict = model.loss_fn(batch, outputs, rngs=rngs)

        return loss_dict['loss'], loss_dict

    # Compute gradients
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Update parameters
    optimizer_state = optimizer.update(grads, optimizer_state)

    return loss, metrics, optimizer_state

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        loss, metrics, optimizer_state = train_step(
            model, batch, optimizer_state
        )
```

### Monitoring Training

Track key metrics:

```python
def train_with_monitoring(model, train_loader, val_loader, num_epochs):
    """Training with detailed monitoring."""

    for epoch in range(num_epochs):
        # Training
        train_losses = []
        train_perplexities = []

        for step, batch in enumerate(train_loader):
            outputs = model(batch['sequences'], training=True, rngs=rngs)
            loss_dict = model.loss_fn(batch, outputs, rngs=rngs)

            train_losses.append(loss_dict['loss'])
            train_perplexities.append(loss_dict['perplexity'])

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}:")
                print(f"  Loss: {loss_dict['loss']:.4f}")
                print(f"  Perplexity: {loss_dict['perplexity']:.2f}")
                print(f"  Accuracy: {loss_dict['accuracy']:.4f}")

        # Validation
        if val_loader is not None:
            val_loss, val_ppl = evaluate(model, val_loader)
            print(f"\nEpoch {epoch} Validation:")
            print(f"  Loss: {val_loss:.4f}")
            print(f"  Perplexity: {val_ppl:.2f}")

def evaluate(model, val_loader):
    """Evaluate on validation set."""
    total_loss = 0
    total_tokens = 0

    for batch in val_loader:
        outputs = model(batch['sequences'], training=False, rngs=rngs)
        loss_dict = model.loss_fn(batch, outputs, rngs=rngs)

        batch_size, seq_len = batch['sequences'].shape
        total_loss += loss_dict['loss'] * batch_size * seq_len
        total_tokens += batch_size * seq_len

    avg_loss = total_loss / total_tokens
    perplexity = jnp.exp(avg_loss)

    return avg_loss, perplexity
```

### Learning Rate Scheduling

Transformers benefit from learning rate warmup:

```python
def transformer_lr_schedule(step, warmup_steps=4000, d_model=512):
    """Transformer learning rate schedule with warmup."""
    step = jnp.maximum(step, 1)
    arg1 = step ** -0.5
    arg2 = step * (warmup_steps ** -1.5)
    return (d_model ** -0.5) * jnp.minimum(arg1, arg2)

# Apply schedule
lr = transformer_lr_schedule(current_step, warmup_steps=4000, d_model=768)
```

---

## Generation and Sampling

### 1. Greedy Decoding

Most likely token at each step:

```python
# Greedy generation
samples = model.generate(
    n_samples=4,
    max_length=128,
    temperature=1.0,  # No effect with greedy (argmax)
    rngs=rngs
)
```

### 2. Temperature Sampling

Control randomness:

```python
# Low temperature (more deterministic)
deterministic_samples = model.generate(
    n_samples=4,
    max_length=128,
    temperature=0.5,  # More peaked distribution
    rngs=rngs
)

# High temperature (more random)
random_samples = model.generate(
    n_samples=4,
    max_length=128,
    temperature=1.5,  # Flatter distribution
    rngs=rngs
)
```

**Temperature Guidelines:**

- `0.5`: Very deterministic, repetitive
- `0.7`: Slightly creative, coherent
- `1.0`: Sample from true model distribution
- `1.2`: More diverse, less coherent
- `1.5+`: Very random, often incoherent

### 3. Top-k Sampling

Sample from k most likely tokens:

```python
# Top-k sampling
samples = model.generate(
    n_samples=4,
    max_length=128,
    temperature=1.0,
    top_k=40,  # Only consider top 40 tokens
    rngs=rngs
)
```

**Top-k Values:**

- `k=1`: Greedy (deterministic)
- `k=10`: Very focused
- `k=40`: Balanced (common for text)
- `k=100`: More diverse

### 4. Top-p (Nucleus) Sampling

Sample from smallest set with cumulative probability ≥ p:

```python
# Top-p (nucleus) sampling
samples = model.generate(
    n_samples=4,
    max_length=128,
    temperature=1.0,
    top_p=0.9,  # Nucleus with 90% probability mass
    rngs=rngs
)
```

**Top-p Values:**

- `p=0.5`: Very focused
- `p=0.7`: Focused but creative
- `p=0.9`: Balanced (recommended)
- `p=0.95`: More diverse
- `p=1.0`: No filtering

### 5. Beam Search

Maintain multiple hypotheses:

```python
# Beam search (returns tuple of sequences and scores)
prompt = jnp.array([1, 45, 23, 89])  # Token IDs
sequences, scores = model.beam_search(
    prompt=prompt,
    beam_size=5,
    max_length=128,
    rngs=rngs
)
print(f"Top sequence shape: {sequences.shape}")  # [beam_size, max_length]
print(f"Scores shape: {scores.shape}")            # [beam_size]
```

**Beam Search Use Cases:**

- Machine translation
- Summarization
- When likelihood is more important than diversity

### 6. Combined Strategies

Combine multiple techniques:

```python
# Recommended: temperature + top-p
samples = model.generate(
    n_samples=4,
    max_length=128,
    temperature=0.8,   # Slight sharpening
    top_p=0.9,         # Nucleus sampling
    rngs=rngs
)

# Alternative: temperature + top-k
samples = model.generate(
    n_samples=4,
    max_length=128,
    temperature=0.7,
    top_k=50,
    rngs=rngs
)
```

---

## Conditional Generation

### 1. Prompt-Based Generation

Generate from a prefix:

```python
# Start with a prompt
prompt = jnp.array([[1, 45, 23, 89]])  # Token IDs

# Continue from prompt
continuation = model.sample_with_conditioning(
    conditioning=prompt,
    n_samples=4,  # 4 completions for same prompt
    temperature=0.8,
    rngs=rngs
)

print(f"Prompt length: {prompt.shape[1]}")
print(f"Continuation shape: {continuation.shape}")
```

### 2. Class-Conditional Generation (PixelCNN)

For class-conditional image generation:

```python
# Add class conditioning to PixelCNN
class ConditionalPixelCNN(PixelCNN):
    def __init__(self, *args, num_classes=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.class_embedding = nnx.Embed(
            num_classes, self.hidden_channels, rngs=kwargs['rngs']
        )

# Generate specific class
class_label = 7
conditional_images = model.generate_conditional(
    class_label=class_label,
    n_samples=16,
    rngs=rngs
)
```

### 3. Inpainting (PixelCNN)

Spatial conditioning for inpainting:

```python
# Conditioning image with mask
conditioning = jnp.zeros((28, 28, 1), dtype=jnp.int32)
mask = jnp.zeros((28, 28))  # 0 = generate, 1 = keep

# Set known pixels
conditioning = conditioning.at[0:10, 0:10, :].set(known_values)
mask = mask.at[0:10, 0:10].set(1)

# Inpaint
inpainted = model.inpaint(
    conditioning=conditioning,
    mask=mask,
    n_samples=4,
    temperature=1.0,
    rngs=rngs
)
```

---

## Advanced Techniques

### 1. Caching for Faster Generation

Cache key-value pairs for Transformers:

```python
# TransformerAutoregressiveModel has a built-in generate_with_cache() method
# that handles KV caching automatically.

# Basic cached generation
samples = model.generate_with_cache(
    n_samples=4,
    max_length=128,
    temperature=0.8,
    rngs=rngs,
)

# Cached generation with prompt and top-p sampling
prompt = jnp.array([1, 45, 23, 89])  # Token IDs
samples = model.generate_with_cache(
    n_samples=4,
    prompt=prompt,
    max_length=128,
    temperature=0.8,
    top_p=0.9,
    rngs=rngs,
)
```

### 2. Speculative Sampling

Speed up generation with a draft model:

```python
def speculative_sampling(target_model, draft_model, n_samples, max_length):
    """Faster sampling using a smaller draft model."""
    sequences = jnp.zeros((n_samples, max_length), dtype=jnp.int32)

    pos = 0
    while pos < max_length:
        # Draft model generates k tokens quickly
        k = 5
        draft_tokens = draft_model.generate(
            conditioning=sequences[:, :pos],
            n_samples=k,
            rngs=rngs
        )

        # Target model verifies
        target_outputs = target_model(draft_tokens, rngs=rngs)
        target_probs = nnx.softmax(target_outputs['logits'], axis=-1)

        # Accept or reject based on probability ratios
        # ... acceptance logic ...

        pos += accepted_tokens

    return sequences
```

### 3. Prefix Tuning for Adaptation

Adapt to new tasks with prefix tuning:

```python
class PrefixTunedTransformer(TransformerAutoregressiveModel):
    """Transformer with learnable prefix for task adaptation."""

    def __init__(self, *args, prefix_length=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_length = prefix_length

        # Learnable prefix embeddings
        self.prefix_embeddings = nnx.Param(
            jax.random.normal(
                kwargs['rngs'].params(),
                (prefix_length, self.embed_dim)
            )
        )

    def forward_with_prefix(self, x, rngs):
        """Forward pass with prefix prepended."""
        batch_size = x.shape[0]

        # Expand prefix for batch
        prefix = jnp.tile(self.prefix_embeddings[None], (batch_size, 1, 1))

        # Embed input
        x_embedded = self.embedding(x)

        # Concatenate prefix and input
        x_with_prefix = jnp.concatenate([prefix, x_embedded], axis=1)

        # Forward through Transformer
        outputs = self.transformer(x_with_prefix, rngs=rngs)

        return outputs
```

---

## Troubleshooting

### Common Issues and Solutions

<div class="grid cards" markdown>

- :material-alert: **High Perplexity**

    ---

    **Symptoms**: Perplexity stays high, poor generation

    **Solutions**:
  - Increase model capacity
  - More training epochs
  - Better data preprocessing
  - Check for label smoothing

    ```python
    # Increase model size by creating a larger config
    network_config = TransformerNetworkConfig(
        name="larger_net",
        hidden_dims=(768, 768),
        activation="gelu",
        embed_dim=768,  # from 512
        num_heads=12,
        mlp_ratio=4.0,
    )
    config = TransformerConfig(
        name="larger_transformer",
        vocab_size=50000,
        sequence_length=1024,
        network=network_config,
        num_layers=12,  # from 6
    )
    ```

- :material-alert: **Slow Generation**

    ---

    **Symptoms**: Sequential generation takes too long

    **Solutions**:
  - Use KV caching (Transformers)
  - Reduce sequence length
  - Use smaller model for drafting
  - JIT compile generation

    ```python
    @jax.jit
    def fast_generate(model, n_samples, max_length):
        return model.generate(n_samples, max_length, rngs=rngs)
    ```

- :material-alert: **Repetitive Output**

    ---

    **Symptoms**: Model generates same tokens repeatedly

    **Solutions**:
  - Increase temperature
  - Use nucleus (top-p) sampling
  - Add repetition penalty
  - More diverse training data

    ```python
    # More diverse sampling
    samples = model.generate(
        temperature=1.2,  # Higher than 1.0
        top_p=0.95,       # Wider nucleus
        rngs=rngs
    )
    ```

- :material-alert: **Training Instability**

    ---

    **Symptoms**: Loss spikes, NaN gradients

    **Solutions**:
  - Lower learning rate
  - Add gradient clipping
  - Use warmup schedule
  - Check data preprocessing

    ```python
    # Gradient clipping
    grads = jax.tree_map(
        lambda g: jnp.clip(g, -1.0, 1.0), grads
    )
    ```

</div>

---

## Best Practices

### 1. Data Preprocessing

```python
def preprocess_text(text, tokenizer):
    """Proper text preprocessing."""
    # Tokenize
    tokens = tokenizer.encode(text)

    # Add special tokens
    tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]

    # Pad/truncate to fixed length
    max_length = 512
    if len(tokens) < max_length:
        tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]

    return jnp.array(tokens)

def preprocess_image(image):
    """Proper image preprocessing for PixelCNN."""
    # Ensure uint8 values [0, 255]
    image = jnp.clip(image, 0, 255).astype(jnp.uint8)
    return image
```

### 2. Start with Small Models

```python
# Quick iteration with small model
small_network = TransformerNetworkConfig(
    name="small_net",
    hidden_dims=(256, 256),
    activation="gelu",
    embed_dim=256,
    num_heads=4,
    mlp_ratio=4.0,
)

small_config = TransformerConfig(
    name="small_transformer",
    vocab_size=10000,
    sequence_length=128,  # Short sequences
    network=small_network,
    num_layers=4,         # Few layers
)

small_model = TransformerAutoregressiveModel(small_config, rngs=rngs)

# Train quickly, verify everything works
# Then scale up
```

### 3. Monitor Generation Quality

```python
def monitor_generation_quality(model, val_prompts, epoch):
    """Regularly check generation quality."""
    print(f"\nEpoch {epoch} - Generation Samples:")

    for i, prompt in enumerate(val_prompts[:3]):
        # Generate
        completion = model.sample_with_conditioning(
            conditioning=prompt,
            temperature=0.8,
            rngs=rngs
        )

        # Decode and display
        text = tokenizer.decode(completion[0])
        print(f"\nPrompt {i+1}: {tokenizer.decode(prompt[0])}")
        print(f"Completion: {text}")
```

### 4. Use Appropriate Metrics

```python
# Track multiple metrics
metrics = {
    "nll_loss": [],     # Negative log-likelihood
    "perplexity": [],   # exp(nll_loss)
    "accuracy": [],     # Token-level accuracy
    "bpd": [],          # Bits per dimension (for images)
}

# For text generation
def evaluate_text_generation(model, generated_samples):
    """Evaluate generation quality."""
    return {
        "diversity": compute_diversity(generated_samples),
        "coherence": compute_coherence(generated_samples),
        "fluency": compute_fluency(generated_samples),
    }
```

---

## Example: Complete Text Generation

```python
from artifex.generative_models.core.configuration.autoregressive_config import (
    TransformerConfig,
    TransformerNetworkConfig,
)
from artifex.generative_models.models.autoregressive import TransformerAutoregressiveModel
import tensorflow_datasets as tfds

# Load dataset (e.g., WikiText)
train_ds = tfds.load('wiki40b/en', split='train')

# Configure and create model
network_config = TransformerNetworkConfig(
    name="transformer_network",
    hidden_dims=(768, 768),
    activation="gelu",
    embed_dim=768,
    num_heads=12,
    mlp_ratio=4.0,
    positional_encoding="sinusoidal",
    dropout_rate=0.1,
)

config = TransformerConfig(
    name="text_transformer",
    vocab_size=50000,
    sequence_length=512,
    network=network_config,
    num_layers=12,
    dropout_rate=0.1,
    use_cache=True,
)

model = TransformerAutoregressiveModel(config, rngs=rngs)

# Training configuration
learning_rate = 1e-4
num_epochs = 10
batch_size = 32

# Training loop
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for step, batch in enumerate(train_ds.batch(batch_size)):
        # Preprocess
        sequences = preprocess_batch(batch)

        # Forward pass
        outputs = model(sequences, training=True, rngs=rngs)

        # Compute loss
        loss_dict = model.loss_fn(
            {"sequences": sequences}, outputs, rngs=rngs
        )

        # Backward pass (via optimizer)
        # ... update parameters ...

        if step % 100 == 0:
            print(f"  Step {step}: Loss={loss_dict['nll_loss']:.4f}, "
                  f"PPL={loss_dict['perplexity']:.2f}")

    # Generate samples
    prompt = "The quick brown fox"
    prompt_tokens = tokenizer.encode(prompt)
    completion = model.sample_with_conditioning(
        conditioning=jnp.array([prompt_tokens]),
        temperature=0.8,
        rngs=rngs
    )
    print(f"\nGeneration: {tokenizer.decode(completion[0])}")

print("Training complete!")
```

---

## Performance Optimization

### GPU Utilization

```python
# Move to GPU
from artifex.generative_models.core.device_manager import DeviceManager

device_manager = DeviceManager()
device = device_manager.get_device()

# Move model and data to GPU
model = jax.device_put(model, device)
batch = jax.device_put(batch, device)
```

### Batch Size Tuning

```python
# Larger batches for better GPU utilization
# But: limited by memory

# PixelCNN (memory intensive)
pixelcnn_batch_size = 32

# Transformer (depends on sequence length)
transformer_batch_sizes = {
    128: 256,   # Short sequences
    512: 64,    # Medium sequences
    1024: 16,   # Long sequences
}

# WaveNet (very memory intensive)
wavenet_batch_size = 4
```

### Mixed Precision Training

```python
# Use bfloat16 for faster training
from jax import config
config.update("jax_enable_x64", False)

# Model automatically uses bfloat16 on TPU
```

---

## Further Reading

- [Autoregressive Explained](../concepts/autoregressive-explained.md) - Theoretical foundations
- [AR API Reference](../../api/models/autoregressive.md) - Complete API documentation
- [Training Guide](../training/training-guide.md) - General training workflows
- [Examples](../../examples/text/simple-text-generation.md) - More AR examples

---

## Summary

**Key Takeaways:**

- Autoregressive models factorize probability via chain rule: $p(x) = \prod_i p(x_i | x_{<i})$
- Training uses teacher forcing with cross-entropy loss
- Generation is sequential, one token at a time
- Sampling strategies (temperature, top-k, top-p) control diversity vs quality
- PixelCNN for images, WaveNet for audio, Transformers for text

**Recommended Workflow:**

1. Choose architecture based on data type (PixelCNN/WaveNet/Transformer)
2. Start with small model for quick iteration
3. Train with teacher forcing, monitor perplexity
4. Generate samples with temperature=0.8, top_p=0.9
5. Scale up model size for better quality
6. Use caching and JIT for faster inference

For theoretical understanding, see the [Autoregressive Explained guide](../concepts/autoregressive-explained.md).
