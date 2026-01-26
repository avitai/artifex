# Multi-Modal Guide

This guide covers working with multi-modal data in Artifex, including aligned datasets, modality fusion, cross-modal generation, and best practices for multi-modal generative models.

## Overview

Artifex's multi-modal system enables working with multiple data modalities (image, text, audio) simultaneously, supporting alignment, fusion, and cross-modal generation tasks.

<div class="grid cards" markdown>

- :material-layers-plus:{ .lg .middle } **Modality Alignment**

    ---

    Create aligned multi-modal datasets with shared latent representations

- :material-link-variant:{ .lg .middle } **Paired Datasets**

    ---

    Handle explicitly paired multi-modal data with alignment scores

- :material-merge:{ .lg .middle } **Modality Fusion**

    ---

    Combine information from multiple modalities for joint representations

- :material-swap-horizontal:{ .lg .middle } **Cross-Modal Generation**

    ---

    Generate one modality from another (e.g., image from text)

- :material-chart-scatter-plot:{ .lg .middle } **Shared Latent Space**

    ---

    Learn unified representations across modalities

- :material-speedometer:{ .lg .middle } **JAX-Native**

    ---

    Full JAX compatibility with efficient batch processing

</div>

## Multi-Modal Datasets

### Aligned Multi-Modal Dataset

Create datasets with aligned samples across modalities:

```python
import jax
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.modalities.multi_modal.datasets import (
    create_synthetic_multi_modal_dataset
)

# Initialize RNG
rngs = nnx.Rngs(0)

# Create aligned multi-modal dataset
dataset = create_synthetic_multi_modal_dataset(
    modalities=["image", "text", "audio"],
    num_samples=1000,
    alignment_strength=0.8,  # 0.0 = random, 1.0 = perfectly aligned
    image_shape=(32, 32, 3),
    text_vocab_size=1000,
    text_sequence_length=50,
    audio_sample_rate=16000,
    audio_duration=1.0,
    rngs=rngs
)

# Access multi-modal sample
sample = dataset[0]
print(sample.keys())
# dict_keys(['image', 'text', 'audio', 'alignment_score', 'latent'])

print(f"Image shape: {sample['image'].shape}")  # (32, 32, 3)
print(f"Text shape: {sample['text'].shape}")  # (50,)
print(f"Audio shape: {sample['audio'].shape}")  # (16000,)
print(f"Alignment: {sample['alignment_score']}")  # 0.8
print(f"Shared latent: {sample['latent'].shape}")  # (32,)
```

### Understanding Alignment Strength

The alignment strength controls how strongly modalities are correlated:

```python
# Weakly aligned (more random)
weak_dataset = create_synthetic_multi_modal_dataset(
    modalities=["image", "text"],
    num_samples=1000,
    alignment_strength=0.3,  # 30% alignment
    rngs=rngs
)

# Moderately aligned
moderate_dataset = create_synthetic_multi_modal_dataset(
    modalities=["image", "text"],
    num_samples=1000,
    alignment_strength=0.6,  # 60% alignment
    rngs=rngs
)

# Strongly aligned
strong_dataset = create_synthetic_multi_modal_dataset(
    modalities=["image", "text"],
    num_samples=1000,
    alignment_strength=0.9,  # 90% alignment
    rngs=rngs
)

# Perfect alignment
perfect_dataset = create_synthetic_multi_modal_dataset(
    modalities=["image", "text"],
    num_samples=1000,
    alignment_strength=1.0,  # 100% alignment
    rngs=rngs
)
```

### Paired Multi-Modal Dataset

For explicitly paired data:

```python
from artifex.generative_models.modalities.multi_modal.datasets import (
    MultiModalPairedDataset
)

# Prepare paired data
image_data = jnp.array([...])  # (N, H, W, C)
text_data = jnp.array([...])  # (N, max_length)
audio_data = jnp.array([...])  # (N, n_samples)

# Define modality pairs
pairs = [
    ("image", "text"),
    ("image", "audio"),
    ("text", "audio")
]

# Optional alignment scores for each pair
alignments = jnp.ones((len(image_data),))  # All perfectly aligned

# Create paired dataset
paired_dataset = MultiModalPairedDataset(
    pairs=pairs,
    data={
        "image": image_data,
        "text": text_data,
        "audio": audio_data
    },
    alignments=alignments
)

# Access paired sample
sample = paired_dataset[0]
print(sample["image"].shape)  # (H, W, C)
print(sample["text"].shape)  # (max_length,)
print(sample["audio"].shape)  # (n_samples,)
print(sample["alignment_scores"])  # 1.0
print(sample["pairs"])  # [('image', 'text'), ('image', 'audio'), ('text', 'audio')]
```

### Batching Multi-Modal Data

```python
# Get batch from aligned dataset
batch = dataset.get_batch(batch_size=32)

print(batch["image"].shape)  # (32, 32, 32, 3)
print(batch["text"].shape)  # (32, 50)
print(batch["audio"].shape)  # (32, 16000)
print(batch["latent"].shape)  # (32, 32)

# Iterate over paired dataset
for i, sample in enumerate(paired_dataset):
    if i >= 3:
        break
    print(f"Sample {i}:")
    for modality in ["image", "text", "audio"]:
        print(f"  {modality}: {sample[modality].shape}")
```

## How Alignment Works

The synthetic multi-modal dataset creates aligned data through a shared latent representation:

```python
# Simplified alignment process
def generate_aligned_sample(latent, alignment_strength):
    """Generate aligned multi-modal sample.

    Args:
        latent: Shared latent vector (32,)
        alignment_strength: Strength of alignment (0-1)

    Returns:
        Dictionary with aligned modalities
    """
    # Generate image from latent
    image = generate_image_from_latent(latent, alignment_strength)

    # Generate text from latent
    text = generate_text_from_latent(latent, alignment_strength)

    # Generate audio from latent
    audio = generate_audio_from_latent(latent, alignment_strength)

    return {
        "image": image,
        "text": text,
        "audio": audio,
        "latent": latent,
        "alignment_score": alignment_strength
    }
```

### Image Generation from Latent

```python
def generate_image_from_latent(latent, alignment_strength, image_shape=(32, 32, 3)):
    """Generate image from latent representation.

    The latent vector modulates spatial patterns:
    - Higher alignment → stronger influence from latent
    - Lower alignment → more random noise

    Args:
        latent: Shared latent (32,)
        alignment_strength: Alignment factor
        image_shape: Target image shape

    Returns:
        Generated image
    """
    h, w, c = image_shape

    # Create spatial coordinate grids
    x = jnp.linspace(-1, 1, w)
    y = jnp.linspace(-1, 1, h)
    xx, yy = jnp.meshgrid(x, y)

    # Create patterns from latent
    pattern = jnp.zeros((h, w))
    for i in range(min(len(latent), 8)):
        freq = 2 + i
        phase = latent[i] * jnp.pi
        amplitude = jnp.abs(latent[i])
        pattern += amplitude * jnp.sin(freq * xx + phase) * jnp.cos(freq * yy + phase)

    # Normalize pattern
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
    pattern = jnp.stack([pattern] * c, axis=-1)

    # Mix with random noise based on alignment
    key = jax.random.key(0)
    noise = jax.random.normal(key, (h, w, c))

    image = alignment_strength * pattern + (1 - alignment_strength) * noise

    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    return image
```

### Text Generation from Latent

```python
def generate_text_from_latent(
    latent,
    alignment_strength,
    vocab_size=1000,
    seq_length=50
):
    """Generate text from latent representation.

    The latent vector biases token selection:
    - Higher alignment → stronger bias from latent
    - Lower alignment → more random tokens

    Args:
        latent: Shared latent (32,)
        alignment_strength: Alignment factor
        vocab_size: Vocabulary size
        seq_length: Sequence length

    Returns:
        Generated token sequence
    """
    # Expand latent to vocab size
    latent_expanded = jnp.tile(latent, (vocab_size // len(latent) + 1))
    latent_expanded = latent_expanded[:vocab_size]

    # Create token probabilities from latent
    token_logits = latent_expanded * alignment_strength
    token_probs = jax.nn.softmax(token_logits)

    # Sample tokens
    key = jax.random.key(0)
    tokens = []
    for i in range(seq_length):
        token_key = jax.random.fold_in(key, i)
        token = jax.random.choice(token_key, vocab_size, p=token_probs)
        tokens.append(token)

    return jnp.array(tokens)
```

### Audio Generation from Latent

```python
def generate_audio_from_latent(
    latent,
    alignment_strength,
    sample_rate=16000,
    duration=1.0
):
    """Generate audio from latent representation.

    The latent vector controls frequency content:
    - Higher alignment → stronger latent influence
    - Lower alignment → more random noise

    Args:
        latent: Shared latent (32,)
        alignment_strength: Alignment factor
        sample_rate: Sample rate in Hz
        duration: Duration in seconds

    Returns:
        Generated audio waveform
    """
    num_samples = int(sample_rate * duration)
    t = jnp.linspace(0, duration, num_samples)

    # Create audio as sum of sinusoids from latent
    waveform = jnp.zeros(num_samples)

    for i in range(min(len(latent), 10)):
        # Frequency from latent (100-2000 Hz)
        freq = 100 + 1900 * (jnp.abs(latent[i]) % 1)
        phase = latent[i] * 2 * jnp.pi
        amplitude = jnp.abs(latent[i]) * 0.1

        waveform += amplitude * jnp.sin(2 * jnp.pi * freq * t + phase)

    # Add noise based on alignment
    key = jax.random.key(0)
    noise = jax.random.normal(key, (num_samples,)) * 0.1

    waveform = alignment_strength * waveform + (1 - alignment_strength) * noise

    # Normalize
    waveform = waveform / (jnp.max(jnp.abs(waveform)) + 1e-8)

    return waveform
```

## Modality Fusion

Combine information from multiple modalities:

### Early Fusion

Concatenate raw features:

```python
def early_fusion(image, text, audio):
    """Concatenate features from all modalities.

    Args:
        image: Image features (H, W, C)
        text: Text features (seq_length,)
        audio: Audio features (n_samples,)

    Returns:
        Fused features
    """
    # Flatten each modality
    image_flat = image.reshape(-1)
    text_flat = text.reshape(-1)
    audio_flat = audio.reshape(-1)

    # Concatenate
    fused = jnp.concatenate([image_flat, text_flat, audio_flat])

    return fused

# Usage
sample = dataset[0]
fused_features = early_fusion(
    sample["image"],
    sample["text"],
    sample["audio"]
)
print(f"Fused features shape: {fused_features.shape}")
```

### Late Fusion

Combine high-level representations:

```python
def late_fusion(image_embedding, text_embedding, audio_embedding):
    """Combine embeddings from separate encoders.

    Args:
        image_embedding: Image encoder output (d_model,)
        text_embedding: Text encoder output (d_model,)
        audio_embedding: Audio encoder output (d_model,)

    Returns:
        Fused embedding
    """
    # Option 1: Concatenation
    fused_concat = jnp.concatenate([
        image_embedding,
        text_embedding,
        audio_embedding
    ])

    # Option 2: Average pooling
    fused_avg = (image_embedding + text_embedding + audio_embedding) / 3

    # Option 3: Weighted sum
    weights = jnp.array([0.4, 0.4, 0.2])  # image, text, audio
    fused_weighted = (
        weights[0] * image_embedding +
        weights[1] * text_embedding +
        weights[2] * audio_embedding
    )

    return fused_weighted

# Usage
# Assuming we have encoders for each modality
# image_emb = image_encoder(sample["image"])
# text_emb = text_encoder(sample["text"])
# audio_emb = audio_encoder(sample["audio"])
# fused = late_fusion(image_emb, text_emb, audio_emb)
```

### Attention-Based Fusion

Use attention to weight modalities:

```python
import jax.numpy as jnp
from flax import nnx

class MultiModalAttentionFusion(nnx.Module):
    """Attention-based multi-modal fusion."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        *,
        rngs: nnx.Rngs
    ):
        """Initialize attention fusion.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            rngs: Random number generators
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Projection layers
        self.query_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.key_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.value_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)

    def __call__(
        self,
        image_emb: jax.Array,
        text_emb: jax.Array,
        audio_emb: jax.Array,
        *,
        deterministic: bool = False
    ) -> jax.Array:
        """Fuse modality embeddings with attention.

        Args:
            image_emb: Image embedding (d_model,)
            text_emb: Text embedding (d_model,)
            audio_emb: Audio embedding (d_model,)
            deterministic: Whether in eval mode

        Returns:
            Fused embedding (d_model,)
        """
        # Stack embeddings (3, d_model)
        embeddings = jnp.stack([image_emb, text_emb, audio_emb])

        # Project to Q, K, V
        queries = self.query_proj(embeddings)
        keys = self.key_proj(embeddings)
        values = self.value_proj(embeddings)

        # Compute attention scores
        scores = jnp.matmul(queries, keys.T) / jnp.sqrt(self.d_model)
        attention_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention
        attended = jnp.matmul(attention_weights, values)

        # Pool across modalities (average)
        fused = jnp.mean(attended, axis=0)

        # Final projection
        output = self.out_proj(fused)

        return output

# Usage
rngs = nnx.Rngs(0)
fusion = MultiModalAttentionFusion(d_model=256, num_heads=4, rngs=rngs)

# image_emb = image_encoder(sample["image"])  # (256,)
# text_emb = text_encoder(sample["text"])  # (256,)
# audio_emb = audio_encoder(sample["audio"])  # (256,)

# fused = fusion(image_emb, text_emb, audio_emb)
# print(f"Fused embedding: {fused.shape}")  # (256,)
```

## Cross-Modal Generation

Generate one modality from another:

### Image from Text (Text-to-Image)

```python
from flax import nnx
import jax.numpy as jnp

class TextToImageGenerator(nnx.Module):
    """Generate images from text."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        image_shape: tuple,
        *,
        rngs: nnx.Rngs
    ):
        """Initialize text-to-image generator.

        Args:
            vocab_size: Text vocabulary size
            embed_dim: Embedding dimension
            image_shape: Target image shape (H, W, C)
            rngs: Random number generators
        """
        super().__init__()
        self.image_shape = image_shape

        # Text encoder
        self.text_embed = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.encoder = nnx.Linear(embed_dim, 512, rngs=rngs)

        # Image decoder
        self.decoder = nnx.Sequential(
            nnx.Linear(512, 1024, rngs=rngs),
            nnx.relu,
            nnx.Linear(1024, int(jnp.prod(jnp.array(image_shape))), rngs=rngs),
            nnx.sigmoid
        )

    def __call__(self, text_tokens: jax.Array) -> jax.Array:
        """Generate image from text.

        Args:
            text_tokens: Text token sequence (seq_length,)

        Returns:
            Generated image (H, W, C)
        """
        # Encode text
        text_emb = self.text_embed(text_tokens)  # (seq_length, embed_dim)
        text_feat = jnp.mean(text_emb, axis=0)  # Pool: (embed_dim,)
        encoded = self.encoder(text_feat)  # (512,)

        # Decode to image
        image_flat = self.decoder(encoded)
        image = image_flat.reshape(self.image_shape)

        return image

# Usage
# generator = TextToImageGenerator(
#     vocab_size=10000,
#     embed_dim=256,
#     image_shape=(32, 32, 3),
#     rngs=rngs
# )
# generated_image = generator(sample["text"])
```

### Text from Image (Image-to-Text)

```python
class ImageToTextGenerator(nnx.Module):
    """Generate text from images."""

    def __init__(
        self,
        image_shape: tuple,
        vocab_size: int,
        max_length: int,
        hidden_dim: int = 512,
        *,
        rngs: nnx.Rngs
    ):
        """Initialize image-to-text generator.

        Args:
            image_shape: Input image shape (H, W, C)
            vocab_size: Text vocabulary size
            max_length: Maximum text length
            hidden_dim: Hidden dimension
            rngs: Random number generators
        """
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size

        # Image encoder
        image_size = int(jnp.prod(jnp.array(image_shape)))
        self.encoder = nnx.Sequential(
            nnx.Linear(image_size, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        )

        # Text decoder
        self.decoder = nnx.Linear(hidden_dim, vocab_size, rngs=rngs)

    def __call__(self, image: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Generate text from image.

        Args:
            image: Input image (H, W, C)
            rngs: Random number generators

        Returns:
            Generated text tokens (max_length,)
        """
        # Encode image
        image_flat = image.reshape(-1)
        encoded = self.encoder(image_flat)  # (hidden_dim,)

        # Decode to text (simplified - sample tokens)
        tokens = []
        for i in range(self.max_length):
            logits = self.decoder(encoded)  # (vocab_size,)

            # Sample token
            if rngs and "sample" in rngs:
                key = rngs.sample()
                token = jax.random.categorical(key, logits)
            else:
                token = jnp.argmax(logits)

            tokens.append(token)

        return jnp.array(tokens)

# Usage
# generator = ImageToTextGenerator(
#     image_shape=(32, 32, 3),
#     vocab_size=10000,
#     max_length=50,
#     rngs=rngs
# )
# generated_text = generator(sample["image"], rngs=rngs)
```

### Audio from Text (Text-to-Speech)

```python
class TextToAudioGenerator(nnx.Module):
    """Generate audio from text."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        audio_length: int,
        *,
        rngs: nnx.Rngs
    ):
        """Initialize text-to-audio generator.

        Args:
            vocab_size: Text vocabulary size
            embed_dim: Embedding dimension
            audio_length: Target audio length in samples
            rngs: Random number generators
        """
        super().__init__()
        self.audio_length = audio_length

        # Text encoder
        self.text_embed = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.encoder = nnx.Linear(embed_dim, 512, rngs=rngs)

        # Audio decoder
        self.decoder = nnx.Sequential(
            nnx.Linear(512, 1024, rngs=rngs),
            nnx.relu,
            nnx.Linear(1024, audio_length, rngs=rngs),
            nnx.tanh  # Audio in [-1, 1]
        )

    def __call__(self, text_tokens: jax.Array) -> jax.Array:
        """Generate audio from text.

        Args:
            text_tokens: Text token sequence (seq_length,)

        Returns:
            Generated audio waveform (audio_length,)
        """
        # Encode text
        text_emb = self.text_embed(text_tokens)  # (seq_length, embed_dim)
        text_feat = jnp.mean(text_emb, axis=0)  # Pool
        encoded = self.encoder(text_feat)  # (512,)

        # Decode to audio
        audio = self.decoder(encoded)  # (audio_length,)

        return audio

# Usage
# generator = TextToAudioGenerator(
#     vocab_size=10000,
#     embed_dim=256,
#     audio_length=16000,
#     rngs=rngs
# )
# generated_audio = generator(sample["text"])
```

## Complete Multi-Modal Training Example

```python
import jax
import jax.numpy as jnp
from flax import nnx

# Setup
rngs = nnx.Rngs(0)

# Create multi-modal dataset
dataset = create_synthetic_multi_modal_dataset(
    modalities=["image", "text", "audio"],
    num_samples=10000,
    alignment_strength=0.8,
    image_shape=(32, 32, 3),
    text_vocab_size=1000,
    text_sequence_length=50,
    audio_sample_rate=16000,
    audio_duration=1.0,
    rngs=rngs
)

# Create validation dataset
val_dataset = create_synthetic_multi_modal_dataset(
    modalities=["image", "text", "audio"],
    num_samples=1000,
    alignment_strength=0.8,
    image_shape=(32, 32, 3),
    text_vocab_size=1000,
    text_sequence_length=50,
    audio_sample_rate=16000,
    audio_duration=1.0,
    rngs=rngs
)

# Training loop
batch_size = 32
num_epochs = 10
key = jax.random.key(42)

for epoch in range(num_epochs):
    num_batches = len(dataset) // batch_size

    for i in range(num_batches):
        # Get batch
        batch = dataset.get_batch(batch_size)

        # Extract modalities
        images = batch["image"]
        texts = batch["text"]
        audios = batch["audio"]
        latents = batch["latent"]

        # Training step (placeholder)
        # 1. Encode each modality
        # image_emb = image_encoder(images)
        # text_emb = text_encoder(texts)
        # audio_emb = audio_encoder(audios)

        # 2. Compute alignment loss
        # alignment_loss = contrastive_loss(image_emb, text_emb, audio_emb)

        # 3. Compute reconstruction losses
        # recon_loss_img = reconstruction_loss(images, reconstructed_images)
        # recon_loss_text = reconstruction_loss(texts, reconstructed_texts)
        # recon_loss_audio = reconstruction_loss(audios, reconstructed_audios)

        # 4. Total loss
        # loss = alignment_loss + recon_loss_img + recon_loss_text + recon_loss_audio

        # 5. Update parameters
        # params = optimizer.update(grads, params)

    # Validation
    val_batches = len(val_dataset) // batch_size
    for i in range(val_batches):
        val_batch = val_dataset.get_batch(batch_size)
        # Validation step
        # val_loss = validate_step(model, val_batch)

    print(f"Epoch {epoch + 1}/{num_epochs} complete")
```

## Best Practices

### DO

!!! tip "Multi-Modal Design"
    - Use shared latent representations for alignment
    - Balance modality contributions in fusion
    - Normalize features before fusion
    - Use attention for dynamic modality weighting
    - Test alignment quality visually/qualitatively
    - Cache aligned datasets when possible

!!! tip "Cross-Modal Generation"
    - Use separate encoders for each modality
    - Implement residual connections in decoders
    - Use appropriate loss functions per modality
    - Test generation quality separately per modality
    - Consider cycle consistency for bidirectional generation

!!! tip "Training"
    - Use contrastive losses for alignment
    - Balance reconstruction and alignment losses
    - Apply modality-specific augmentation
    - Monitor per-modality metrics
    - Use curriculum learning for complex tasks

### DON'T

!!! danger "Common Mistakes"
    - Mix different alignment strengths in same batch
    - Ignore modality-specific preprocessing
    - Use same architecture for all modalities
    - Apply same augmentation to all modalities
    - Forget to normalize embeddings before fusion
    - Ignore computational cost of attention

!!! danger "Alignment Issues"
    - Use too low alignment strength for supervised tasks
    - Mix aligned and unaligned samples
    - Ignore alignment scores during training
    - Use mismatched modality sizes
    - Forget to validate alignment quality

!!! danger "Performance"
    - Concatenate raw features from all modalities
    - Use very deep fusion networks
    - Process all modalities even when not needed
    - Ignore modality-specific batch sizes
    - Use attention for simple fusion tasks

## Summary

This guide covered:

- **Multi-modal datasets** - Aligned and paired datasets
- **Alignment** - Shared latent representations and alignment strength
- **Modality fusion** - Early, late, and attention-based fusion
- **Cross-modal generation** - Image↔Text, Text↔Audio
- **Complete example** - Multi-modal training pipeline
- **Best practices** - DOs and DON'Ts for multi-modal learning

## Next Steps

<div class="grid cards" markdown>

- :material-image:{ .lg .middle } **[Image Modality Guide](image.md)**

    ---

    Deep dive into image datasets, preprocessing, and augmentation

- :material-text:{ .lg .middle } **[Text Modality Guide](text.md)**

    ---

    Learn about text tokenization, vocabulary management, and sequences

- :material-volume-high:{ .lg .middle } **[Audio Modality Guide](audio.md)**

    ---

    Audio waveform processing, spectrograms, and audio augmentation

- :material-api:{ .lg .middle } **[Data API Reference](../../api/data/loaders.md)**

    ---

    Complete API documentation for all dataset classes and functions

</div>
