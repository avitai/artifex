# Data Modalities

Modality-specific implementations for handling different data types in generative models, including adapters, evaluation metrics, and representation learning.

## Overview

<div class="grid cards" markdown>

- :material-image:{ .lg .middle } **Image**

    ---

    Convolutional architectures, FID/IS metrics, perceptual losses

- :material-text:{ .lg .middle } **Text**

    ---

    Tokenization, transformers, perplexity metrics

- :material-waveform:{ .lg .middle } **Audio**

    ---

    Spectrograms, MFCCs, audio quality metrics

- :material-molecule:{ .lg .middle } **Protein**

    ---

    Sequence encoding, structure prediction, SE(3) equivariance

</div>

## Quick Start

### Using Modalities

```python
from artifex.generative_models.modalities import get_modality

# Get image modality handler
image_modality = get_modality("image", rngs=rngs)

# Get evaluation metrics
evaluator = image_modality.get_evaluator()
metrics = evaluator.evaluate(generated_images, real_images)

# Get model adapter
adapter = image_modality.get_adapter("vae")
adapted_model = adapter.adapt(model, config)
```

## Image Modality

Full support for image generation with specialized components.

### Features

- Convolutional encoder/decoder architectures
- FID, Inception Score, LPIPS metrics
- Image-specific augmentations
- Perceptual losses with VGG features

### Usage

```python
from artifex.generative_models.modalities.image import (
    ImageModality,
    ImageEvaluator,
    ImageRepresentation,
)

# Create image modality
modality = ImageModality(image_size=(256, 256), channels=3)

# Evaluate generated images
evaluator = ImageEvaluator()
metrics = evaluator.evaluate(
    generated=fake_images,
    real=real_images,
    metrics=["fid", "inception_score", "lpips"],
)

# Extract image representations
repr_model = ImageRepresentation(pretrained="inception_v3")
features = repr_model.extract(images)
```

### Modules

| Module | Description |
|--------|-------------|
| [base](base.md) | Image modality base class |
| [adapters](adapters.md) | Model adapters for images |
| [datasets](datasets.md) | Image dataset utilities |
| [evaluation](evaluation.md) | Image quality metrics |
| [representations](representations.md) | Feature extraction |

[:octicons-arrow-right-24: Image Modality Guide](../user-guide/modalities/image.md)

## Text Modality

Support for text generation with language-specific components.

### Features

- Multiple tokenization strategies (BPE, SentencePiece)
- Transformer architectures
- Perplexity and BLEU metrics
- Language-specific preprocessing

### Usage

```python
from artifex.generative_models.modalities.text import (
    TextModality,
    TextEvaluator,
    TextRepresentation,
)

# Create text modality
modality = TextModality(vocab_size=32000, max_length=512)

# Evaluate generated text
evaluator = TextEvaluator()
metrics = evaluator.evaluate(
    generated=generated_texts,
    references=reference_texts,
    metrics=["perplexity", "bleu", "rouge"],
)

# Extract text representations
repr_model = TextRepresentation(model="bert-base")
embeddings = repr_model.extract(texts)
```

### Modules

| Module | Description |
|--------|-------------|
| [base](base.md) | Text modality base class |
| [datasets](datasets.md) | Text dataset utilities |
| [evaluation](evaluation.md) | Text quality metrics |
| [representations](representations.md) | Text embeddings |

[:octicons-arrow-right-24: Text Modality Guide](../user-guide/modalities/text.md)

## Audio Modality

Support for audio generation with signal processing components.

### Features

- Spectrogram representations (Mel, STFT)
- Audio quality metrics (FAD, MOS prediction)
- Time-frequency transformations
- Audio-specific augmentations

### Usage

```python
from artifex.generative_models.modalities.audio import (
    AudioModality,
    AudioEvaluator,
    AudioRepresentation,
)

# Create audio modality
modality = AudioModality(sample_rate=16000, n_mels=80)

# Evaluate generated audio
evaluator = AudioEvaluator()
metrics = evaluator.evaluate(
    generated=fake_audio,
    real=real_audio,
    metrics=["fad", "inception_score"],
)

# Extract audio representations
repr_model = AudioRepresentation(model="vggish")
features = repr_model.extract(audio)
```

### Modules

| Module | Description |
|--------|-------------|
| [base](base.md) | Audio modality base class |
| [datasets](datasets.md) | Audio dataset utilities |
| [evaluation](evaluation.md) | Audio quality metrics |
| [representations](representations.md) | Audio features |

[:octicons-arrow-right-24: Audio Modality Guide](../user-guide/modalities/audio.md)

## Protein Modality

Support for protein structure generation with biological constraints.

### Features

- Amino acid sequence encoding
- 3D structure representation
- SE(3) equivariant operations
- Structure quality metrics (RMSD, TM-score)

### Usage

```python
from artifex.generative_models.modalities.protein import (
    ProteinModality,
    ProteinEvaluator,
    ProteinRepresentation,
)

# Create protein modality
modality = ProteinModality(max_length=256)

# Evaluate generated structures
evaluator = ProteinEvaluator()
metrics = evaluator.evaluate(
    generated=predicted_structures,
    reference=native_structures,
    metrics=["rmsd", "tm_score", "gdt_ts"],
)

# Extract protein representations
repr_model = ProteinRepresentation(model="esm2")
embeddings = repr_model.extract(sequences)
```

### Modules

| Module | Description |
|--------|-------------|
| [adapters](adapters.md) | Model adapters for proteins |
| [config](config.md) | Protein configuration |
| [losses](losses.md) | Structure losses |
| [modality](modality.md) | Protein modality class |
| [utils](utils.md) | Protein utilities |

[:octicons-arrow-right-24: Protein Modeling Guide](../guides/protein-modeling.md)

## Multimodal

Support for combining multiple modalities.

### Features

- Cross-modal attention
- Joint embedding spaces
- Multi-task learning
- Modality alignment

### Usage

```python
from artifex.generative_models.modalities.multimodal import (
    MultiModalModality,
    MultiModalEvaluator,
)

# Create multimodal handler
modality = MultiModalModality(
    modalities=["image", "text"],
)

# Evaluate cross-modal generation
evaluator = MultiModalEvaluator()
metrics = evaluator.evaluate(
    generated={"image": images, "text": captions},
    real=real_data,
)
```

### Modules

| Module | Description |
|--------|-------------|
| [adapters](adapters.md) | Cross-modal adapters |
| [base](base.md) | Multimodal base class |
| [datasets](datasets.md) | Multimodal datasets |
| [evaluation](evaluation.md) | Cross-modal metrics |
| [representations](representations.md) | Joint embeddings |

[:octicons-arrow-right-24: Multimodal Guide](../user-guide/modalities/multimodal.md)

## Tabular Modality

Support for structured/tabular data generation.

### Usage

```python
from artifex.generative_models.modalities.tabular import (
    TabularModality,
    TabularEvaluator,
)

modality = TabularModality(
    categorical_columns=["category", "type"],
    continuous_columns=["value", "amount"],
)
```

## Time Series Modality

Support for temporal data generation.

### Usage

```python
from artifex.generative_models.modalities.timeseries import (
    TimeSeriesModality,
    TimeSeriesEvaluator,
)

modality = TimeSeriesModality(
    seq_length=100,
    num_features=5,
)
```

## Molecular Modality

Support for molecular generation.

### Usage

```python
from artifex.generative_models.modalities.molecular import (
    MolecularModality,
    MolecularAdapter,
)

modality = MolecularModality()
adapter = modality.get_adapter("flow")
```

## Modality Registry

Register and retrieve modalities:

```python
from artifex.generative_models.modalities import (
    get_modality,
    register_modality,
    list_modalities,
)

# List available modalities
available = list_modalities()
# ['image', 'text', 'audio', 'protein', 'multimodal', 'tabular', 'timeseries', 'molecular']

# Get modality by name
modality = get_modality("image", rngs=rngs)

# Register custom modality
register_modality("custom", CustomModality)
```

[:octicons-arrow-right-24: Registry](registry.md)

## Module Reference

| Modality | Modules |
|----------|---------|
| **Base** | [base](base.md), [registry](registry.md) |
| **Image** | [adapters](adapters.md), [base](base.md), [datasets](datasets.md), [evaluation](evaluation.md), [representations](representations.md) |
| **Text** | [base](base.md), [datasets](datasets.md), [evaluation](evaluation.md), [representations](representations.md) |
| **Audio** | [base](base.md), [datasets](datasets.md), [evaluation](evaluation.md), [representations](representations.md) |
| **Protein** | [adapters](adapters.md), [config](config.md), [losses](losses.md), [modality](modality.md), [utils](utils.md) |
| **Multimodal** | [adapters](adapters.md), [base](base.md), [datasets](datasets.md), [evaluation](evaluation.md), [representations](representations.md) |

## Related Documentation

- [Image Modality Guide](../user-guide/modalities/image.md)
- [Text Modality Guide](../user-guide/modalities/text.md)
- [Audio Modality Guide](../user-guide/modalities/audio.md)
- [Multimodal Guide](../user-guide/modalities/multimodal.md)
