# Simple Image-Text Multimodal Learning

<div class="example-badges">
<span class="badge badge-intermediate">Intermediate</span>
<span class="badge badge-runtime-10min">Runtime: ~10min</span>
<span class="badge badge-format-dual">📓 Dual Format</span>
</div>

**Status:** `Standalone pedagogy`
**Device:** CPU-compatible

This walkthrough is a standalone JAX/Flax NNX concept demo.
It does not instantiate shipped Artifex runtime owners.

## Files

- **Python Script**: [`simple_image_text.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/multimodal/simple_image_text.py)
- **Jupyter Notebook**: [`simple_image_text.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/multimodal/simple_image_text.ipynb)

## Quick Start

```bash
python examples/generative_models/multimodal/simple_image_text.py

# Or open the Jupyter notebook
jupyter lab examples/generative_models/multimodal/simple_image_text.ipynb
```

## Overview

This standalone walkthrough demonstrates multimodal learning by combining image and text encoders in a unified model. Learn how to build separate encoders for different modalities, create shared embedding spaces, and perform cross-modal retrieval tasks without claiming shipped Artifex multimodal owners.

### Learning Objectives

After completing this example, you will understand:

- [ ] Multimodal model architectures with separate encoders
- [ ] Creating shared embedding spaces for multiple modalities
- [ ] Computing cross-modal similarities
- [ ] Performing cross-modal retrieval (image-to-text, text-to-image)
- [ ] Visualizing multimodal embedding spaces

### Prerequisites

- Understanding of CNNs for image processing
- Familiarity with text embeddings and sequence models
- Knowledge of similarity metrics and representation learning
- Basic understanding of JAX/Flax NNX patterns

## Theory

### Multimodal Learning

Multimodal models learn joint representations from multiple input modalities. The goal is to create a shared embedding space where semantically similar inputs from different modalities are close together.

#### Contrastive Learning

The model learns by maximizing similarity between matching pairs while minimizing similarity between non-matching pairs:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(f_I(I), f_T(T)) / \tau)}{\sum_{i=1}^N \exp(\text{sim}(f_I(I), f_T(T_i)) / \tau)}$$

where:

- $f_I$ is the image encoder
- $f_T$ is the text encoder
- $\tau$ is the temperature parameter
- $\text{sim}$ is the similarity function (typically cosine similarity)

### Architecture Components

1. **Image Encoder**: CNN-based encoder mapping images to embeddings
2. **Text Encoder**: Embedding + MLP mapping text sequences to embeddings
3. **Fusion Layer**: Combines modalities for joint predictions

## Code Walkthrough

### 1. Image Encoder

```python
class SimpleImageEncoder(nnx.Module):
    def __init__(self, image_size=32, embed_dim=128, *, rngs: nnx.Rngs):
        super().__init__()
        # CNN encoder with global average pooling
        self.encoder = nnx.Sequential(
            nnx.Conv(3, 32, kernel_size=(3, 3), rngs=rngs),
            nnx.relu,
            nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs),
            nnx.relu,
            lambda x: jnp.mean(x, axis=(1, 2)),  # Global pooling
            nnx.Linear(64, embed_dim, rngs=rngs),
        )
```

### 2. Text Encoder

```python
class SimpleTextEncoder(nnx.Module):
    def __init__(self, vocab_size=128, embed_dim=128, *, rngs: nnx.Rngs):
        super().__init__()
        self.embedding = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.encoder = nnx.Sequential(
            nnx.Linear(embed_dim, 64, rngs=rngs),
            nnx.relu,
            nnx.Linear(64, embed_dim, rngs=rngs)
        )

    def __call__(self, text_ids):
        embedded = self.embedding(text_ids)
        pooled = jnp.mean(embedded, axis=1)  # Average pooling
        return self.encoder(pooled)
```

### 3. Multimodal Model

```python
class SimpleMultimodalModel(nnx.Module):
    def __init__(self, image_size=32, vocab_size=128,
                 embed_dim=128, output_dim=10, *, rngs: nnx.Rngs):
        super().__init__()
        # Separate encoders
        self.image_encoder = SimpleImageEncoder(image_size, embed_dim, rngs=rngs)
        self.text_encoder = SimpleTextEncoder(vocab_size, embed_dim, rngs=rngs)

        # Fusion layer
        self.fusion = nnx.Sequential(
            nnx.Linear(embed_dim * 2, embed_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(embed_dim, output_dim, rngs=rngs),
        )
```

### 4. Cross-Modal Similarity

```python
def compute_similarity(self, images, text_ids):
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
```

## Experiments to Try

### 1. Architecture Improvements

- Add attention mechanisms for better feature aggregation
- Use pre-trained encoders (ResNet for images, BERT for text)
- Implement transformer-based fusion layers
- Add residual connections

### 2. Training Enhancements

- Implement contrastive loss (InfoNCE, SimCLR)
- Add hard negative mining
- Use temperature-scaled training
- Implement data augmentation

### 3. Advanced Features

- Multi-head attention for fusion
- Cross-modal attention mechanisms
- Hierarchical embeddings
- Multi-task learning objectives

## Next Steps

<div class="grid cards" markdown>

- :material-view-grid: **Multimodal Guide**

    ---

    Compare this standalone walkthrough with the retained modality abstractions

    [:octicons-arrow-right-24: Multimodal guide](../../user-guide/modalities/multimodal.md)

- :material-image: **Image Modality**

    ---

    Understand the retained image-processing path used by Artifex models

    [:octicons-arrow-right-24: Image guide](../../user-guide/modalities/image.md)

- :material-format-text: **Text Modality**

    ---

    Review the retained token and text-preparation surfaces

    [:octicons-arrow-right-24: Text guide](../../user-guide/modalities/text.md)

- :material-map-outline: **Planned Topics**

    ---

    Track the still-unshipped CLIP, VQA, captioning, and retrieval examples

    [:octicons-arrow-right-24: Planned examples](../../roadmap/planned-examples.md#text-and-multimodal)

</div>

## Troubleshooting

### Common Issues

**Embedding Dimension Mismatch**:

- Ensure both encoders output same embedding dimension
- Check fusion layer input dimensions
- Verify concatenation axis

**Poor Similarity Scores**:

- Normalize features before computing similarity
- Check for numerical instability (add epsilon)
- Tune temperature parameter

**Memory Issues**:

- Reduce batch size or embedding dimensions
- Use gradient checkpointing
- Enable mixed precision training

## Additional Resources

### Documentation

- [Flax NNX Guide](https://flax.readthedocs.io/en/latest/nnx/index.html)
- [JAX Transformations](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [Multimodal Learning Survey](https://arxiv.org/abs/2209.03430)

### Research Papers

- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [ALIGN: Scaling Up Visual and Vision-Language Representation Learning](https://arxiv.org/abs/2102.05918)
- [Multimodal Learning with Transformers: A Survey](https://arxiv.org/abs/2206.06488)

---

**Author**: Artifex Team
**Last Updated**: 2025-10-22
**Difficulty**: Intermediate
**Time to Complete**: ~45 minutes
