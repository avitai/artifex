# %% [markdown]
r"""
# Simple Image-Text Multimodal Learning

This example demonstrates multimodal learning by combining image and text
modalities in a unified model. Learn how to build encoders for different
modalities, create shared embedding spaces, and perform cross-modal retrieval.

## Learning Objectives

- [ ] Understand multimodal model architectures
- [ ] Implement separate encoders for different modalities
- [ ] Create shared embedding spaces for multiple modalities
- [ ] Compute cross-modal similarities
- [ ] Perform cross-modal retrieval tasks

## Prerequisites

- Understanding of CNNs for image processing
- Familiarity with text embeddings and RNNs
- Knowledge of similarity metrics
- Basic understanding of representation learning

## Key Concepts

### Multimodal Learning

Multimodal models learn joint representations from multiple input modalities
(e.g., images and text). The goal is to create a shared embedding space where
semantically similar inputs from different modalities are close together.

### Contrastive Learning

The model learns by maximizing similarity between matching pairs (e.g., an image and its caption)
while minimizing similarity between non-matching pairs:

$$\\mathcal{L} = -\\log \\frac{\\exp(\\text{sim}(f_I(I), f_T(T)) / \\tau)}
{\\sum_{i=1}^N \\exp(\\text{sim}(f_I(I), f_T(T_i)) / \\tau)}$$

where $f_I$ and $f_T$ are image and text encoders, and $\\tau$ is the temperature parameter.

### Cross-Modal Retrieval

Given a query in one modality, retrieve relevant items from another modality
by computing similarities in the shared embedding space.
"""

# %%
#!/usr/bin/env python
"""Simple image-text multimodal example using the Artifex framework.

This example demonstrates how to use the Artifex framework's modality
system to create multimodal models.

Source Code Dependencies:
    - flax.nnx: Neural network modules (Conv, Linear, Embed, Sequential)
    - jax.numpy: Array operations
    - matplotlib.pyplot: Visualization
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx


# %% [markdown]
"""
## Model Architecture

The multimodal model consists of three main components:

1. **Image Encoder**: CNN-based encoder for images → embeddings
2. **Text Encoder**: Embedding + MLP for text → embeddings
3. **Fusion Layer**: Combines modalities for joint predictions

Each encoder maps its input to a shared embedding space where
cross-modal similarities can be computed.
"""


# %%
class SimpleImageEncoder(nnx.Module):
    """Simple image encoder."""

    def __init__(self, image_size=32, embed_dim=128, *, rngs: nnx.Rngs):
        """Initialize the image encoder.

        Args:
            image_size: Size of input images.
            embed_dim: Dimension of output embeddings.
            rngs: Random number generators.
        """
        super().__init__()
        # Simple CNN encoder
        self.encoder = nnx.Sequential(
            nnx.Conv(3, 32, kernel_size=(3, 3), rngs=rngs),
            nnx.relu,
            nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs),
            nnx.relu,
            lambda x: jnp.mean(x, axis=(1, 2)),  # Global average pooling
            nnx.Linear(64, embed_dim, rngs=rngs),
        )

    def __call__(self, images):
        """Encode images to embeddings.

        Args:
            images: Input images.

        Returns:
            Image embeddings.
        """
        return self.encoder(images)


# %%
class SimpleTextEncoder(nnx.Module):
    """Simple text encoder."""

    def __init__(self, vocab_size=128, embed_dim=128, *, rngs: nnx.Rngs):
        """Initialize the text encoder.

        Args:
            vocab_size: Size of vocabulary.
            embed_dim: Dimension of output embeddings.
            rngs: Random number generators.
        """
        super().__init__()
        self.embedding = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.encoder = nnx.Sequential(
            nnx.Linear(embed_dim, 64, rngs=rngs), nnx.relu, nnx.Linear(64, embed_dim, rngs=rngs)
        )

    def __call__(self, text_ids):
        """Encode text to embeddings.

        Args:
            text_ids: Input text token IDs.

        Returns:
            Text embeddings.
        """
        # Embed and average pool
        embedded = self.embedding(text_ids)
        pooled = jnp.mean(embedded, axis=1)  # Average over sequence
        return self.encoder(pooled)


# %%
class SimpleMultimodalModel(nnx.Module):
    """Simple multimodal model combining image and text."""

    def __init__(
        self, image_size=32, vocab_size=128, embed_dim=128, output_dim=10, *, rngs: nnx.Rngs
    ):
        """Initialize the multimodal model.

        Args:
            image_size: Size of input images
            vocab_size: Size of text vocabulary
            embed_dim: Dimension of shared embedding space
            output_dim: Dimension of output
            rngs: Random number generators
        """
        super().__init__()

        # Separate encoders for each modality
        self.image_encoder = SimpleImageEncoder(image_size, embed_dim, rngs=rngs)
        self.text_encoder = SimpleTextEncoder(vocab_size, embed_dim, rngs=rngs)

        # Fusion layer
        self.fusion = nnx.Sequential(
            nnx.Linear(embed_dim * 2, embed_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(embed_dim, output_dim, rngs=rngs),
        )

        self.embed_dim = embed_dim

    def encode_image(self, images):
        """Encode images to embeddings."""
        return self.image_encoder(images)

    def encode_text(self, text_ids):
        """Encode text to embeddings."""
        return self.text_encoder(text_ids)

    def __call__(self, images, text_ids):
        """Forward pass combining both modalities.

        Args:
            images: Input images [batch, h, w, c]
            text_ids: Input text token IDs [batch, seq_len]

        Returns:
            Output predictions [batch, output_dim]
        """
        # Encode each modality
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_ids)

        # Concatenate features
        combined = jnp.concatenate([image_features, text_features], axis=-1)

        # Apply fusion
        output = self.fusion(combined)

        return output

    def compute_similarity(self, images, text_ids):
        """Compute similarity between images and text.

        Args:
            images: Input images
            text_ids: Input text token IDs

        Returns:
            Similarity scores
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_ids)

        # Normalize features
        image_features = image_features / (
            jnp.linalg.norm(image_features, axis=-1, keepdims=True) + 1e-8
        )
        text_features = text_features / (
            jnp.linalg.norm(text_features, axis=-1, keepdims=True) + 1e-8
        )

        # Compute cosine similarity
        similarity = jnp.sum(image_features * text_features, axis=-1)

        return similarity


# %% [markdown]
"""
## Data Generation and Visualization

Create synthetic image-text pairs for demonstration and visualize
the learned embedding space.
"""


# %%
def create_synthetic_data(batch_size=8, image_size=32, seq_len=10):
    """Create synthetic image-text pairs for demonstration.

    Args:
        batch_size: Number of samples
        image_size: Size of images
        seq_len: Length of text sequences

    Returns:
        Tuple of (images, text_ids, labels)
    """
    key = jax.random.key(np.random.randint(0, 10000))

    # Create random images
    img_key, text_key, label_key = jax.random.split(key, 3)
    images = jax.random.uniform(img_key, (batch_size, image_size, image_size, 3))

    # Create random text (token IDs)
    text_ids = jax.random.randint(text_key, (batch_size, seq_len), 0, 128)

    # Create random labels for classification
    labels = jax.random.randint(label_key, (batch_size,), 0, 10)

    return images, text_ids, labels


# %%
def visualize_multimodal_embeddings(model, images, text_ids):
    """Visualize the embedding space of the multimodal model.

    Args:
        model: Multimodal model
        images: Input images
        text_ids: Input text token IDs
    """
    # Get embeddings
    image_embeddings = model.encode_image(images)
    text_embeddings = model.encode_text(text_ids)

    # Reduce to 2D for visualization using PCA-like projection
    key = jax.random.key(42)
    projection = jax.random.normal(key, (model.embed_dim, 2))

    image_2d = image_embeddings @ projection
    text_2d = text_embeddings @ projection

    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot image embeddings
    ax1.scatter(image_2d[:, 0], image_2d[:, 1], c="blue", label="Image", alpha=0.6)
    ax1.set_title("Image Embeddings")
    ax1.set_xlabel("Dim 1")
    ax1.set_ylabel("Dim 2")
    ax1.grid(True, alpha=0.3)

    # Plot text embeddings
    ax2.scatter(text_2d[:, 0], text_2d[:, 1], c="red", label="Text", alpha=0.6)
    ax2.set_title("Text Embeddings")
    ax2.set_xlabel("Dim 1")
    ax2.set_ylabel("Dim 2")
    ax2.grid(True, alpha=0.3)

    # Plot both together
    ax3.scatter(image_2d[:, 0], image_2d[:, 1], c="blue", label="Image", alpha=0.6)
    ax3.scatter(text_2d[:, 0], text_2d[:, 1], c="red", label="Text", alpha=0.6)

    # Draw lines connecting pairs
    for i in range(len(image_2d)):
        ax3.plot(
            [image_2d[i, 0], text_2d[i, 0]],
            [image_2d[i, 1], text_2d[i, 1]],
            "gray",
            alpha=0.3,
            linewidth=0.5,
        )

    ax3.set_title("Joint Embedding Space")
    ax3.set_xlabel("Dim 1")
    ax3.set_ylabel("Dim 2")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# %% [markdown]
"""
## Main Demonstration

Demonstrates:
- Model creation and initialization
- Forward pass with both modalities
- Individual encoder testing
- Cross-modal similarity computation
- Embedding space visualization
- Cross-modal retrieval (image-to-text and text-to-image)
"""


# %%
def main():
    """Run the multimodal example."""
    print("=" * 60)
    print("Simple Image-Text Multimodal Example")
    print("=" * 60)

    # Set random seed
    seed = 42
    key = jax.random.key(seed)
    rngs = nnx.Rngs(params=key)

    # Create multimodal model
    print("\nCreating multimodal model...")
    model = SimpleMultimodalModel(
        image_size=32, vocab_size=128, embed_dim=128, output_dim=10, rngs=rngs
    )

    print("Image encoder initialized")
    print("Text encoder initialized")
    print(f"Embedding dimension: {model.embed_dim}")

    # Create synthetic data
    print("\nCreating synthetic data...")
    batch_size = 16
    images, text_ids, labels = create_synthetic_data(
        batch_size=batch_size, image_size=32, seq_len=10
    )

    print(f"Images shape: {images.shape}")
    print(f"Text IDs shape: {text_ids.shape}")
    print(f"Labels shape: {labels.shape}")

    # Test forward pass
    print("\nTesting forward pass...")
    outputs = model(images, text_ids)
    print(f"Output shape: {outputs.shape}")

    # Test individual encoders
    print("\nTesting individual encoders...")
    image_features = model.encode_image(images)
    text_features = model.encode_text(text_ids)
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")

    # Compute similarities
    print("\nComputing image-text similarities...")
    similarities = model.compute_similarity(images, text_ids)
    print(f"Similarities shape: {similarities.shape}")
    print(f"Mean similarity: {jnp.mean(similarities):.3f}")
    print(f"Max similarity: {jnp.max(similarities):.3f}")
    print(f"Min similarity: {jnp.min(similarities):.3f}")

    # Visualize embeddings
    print("\nVisualizing embedding space...")
    fig = visualize_multimodal_embeddings(model, images, text_ids)

    # Save figure
    import os

    output_dir = "examples_output"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "multimodal_embeddings.png"))
    print(f"Embeddings visualization saved to {output_dir}/multimodal_embeddings.png")

    # Demonstrate cross-modal retrieval
    print()
    print("=" * 40)
    print("Cross-Modal Retrieval Example")
    print("=" * 40)

    # Create a batch of queries and gallery
    query_images, query_text, _ = create_synthetic_data(5, 32, 10)
    gallery_images, gallery_text, _ = create_synthetic_data(20, 32, 10)

    # Image-to-text retrieval
    print("\nImage-to-Text Retrieval:")
    query_img_features = model.encode_image(query_images[:1])  # Use first image as query
    gallery_text_features = model.encode_text(gallery_text)

    # Compute similarities
    query_img_norm = query_img_features / (
        jnp.linalg.norm(query_img_features, axis=-1, keepdims=True) + 1e-8
    )
    gallery_text_norm = gallery_text_features / (
        jnp.linalg.norm(gallery_text_features, axis=-1, keepdims=True) + 1e-8
    )

    sims = query_img_norm @ gallery_text_norm.T
    top_5 = jnp.argsort(sims[0])[-5:][::-1]

    print(f"Top 5 text matches for image query: {top_5}")
    print(f"Similarity scores: {sims[0][top_5]}")

    # Text-to-image retrieval
    print("\nText-to-Image Retrieval:")
    query_text_features = model.encode_text(query_text[:1])  # Use first text as query
    gallery_img_features = model.encode_image(gallery_images)

    # Compute similarities
    query_text_norm = query_text_features / (
        jnp.linalg.norm(query_text_features, axis=-1, keepdims=True) + 1e-8
    )
    gallery_img_norm = gallery_img_features / (
        jnp.linalg.norm(gallery_img_features, axis=-1, keepdims=True) + 1e-8
    )

    sims = query_text_norm @ gallery_img_norm.T
    top_5 = jnp.argsort(sims[0])[-5:][::-1]

    print(f"Top 5 image matches for text query: {top_5}")
    print(f"Similarity scores: {sims[0][top_5]}")

    print("\nSimple multimodal example completed successfully!")


# %% [markdown]
"""
## Summary and Key Takeaways

This example demonstrated multimodal learning fundamentals:

### What We Learned

1. **Separate Encoders**: Different architectures for different modalities
2. **Shared Embedding Space**: Unified representation for multiple modalities
3. **Cross-Modal Similarity**: Computing relationships between modalities
4. **Retrieval Tasks**: Finding relevant items across modalities
5. **Visualization**: Understanding learned embedding spaces

### Experiments to Try

1. **Architecture Improvements**:
   - Add attention mechanisms for better feature aggregation
   - Use pre-trained encoders (ResNet for images, BERT for text)
   - Implement transformer-based fusion

2. **Training Enhancements**:
   - Add contrastive loss (InfoNCE, SimCLR)
   - Implement hard negative mining
   - Use temperature-scaled training

3. **Advanced Features**:
   - Multi-head attention for fusion
   - Cross-modal attention mechanisms
   - Hierarchical embeddings

### Next Steps

Explore related examples:
- **CLIP-style Models**: Large-scale image-text models
- **Visual Question Answering**: Advanced multimodal reasoning
- **Image Captioning**: Generating text from images
"""

# %%
if __name__ == "__main__":
    main()
