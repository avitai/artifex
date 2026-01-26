# Data Processing

Comprehensive data loading, preprocessing, and augmentation pipelines for training generative models across multiple modalities.

## Overview

<div class="grid cards" markdown>

- :material-image-multiple:{ .lg .middle } **Multi-Modal Support**

    ---

    Image, text, audio, video, protein, and multimodal datasets

- :material-database:{ .lg .middle } **Standard Datasets**

    ---

    MNIST, CIFAR, ImageNet, FFHQ, LibriSpeech, and more

- :material-auto-fix:{ .lg .middle } **Augmentation**

    ---

    Modality-specific data augmentation pipelines

- :material-cloud-download:{ .lg .middle } **Streaming**

    ---

    WebDataset, TFRecord, and remote data loading

</div>

## Quick Start

### Loading Standard Datasets

```python
from artifex.data import load_dataset

# Load MNIST
train_data, test_data = load_dataset(
    "mnist",
    batch_size=128,
    split=("train", "test"),
)

# Load CIFAR-10
train_data = load_dataset(
    "cifar10",
    batch_size=64,
    augment=True,
)
```

### Custom Datasets

```python
from artifex.data import ImageDataset

dataset = ImageDataset(
    root="/path/to/images",
    image_size=(256, 256),
    normalize=True,
)
```

## Image Datasets

### Standard Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| [CIFAR-10/100](cifar.md) | 32x32 natural images | 60K/60K |
| [ImageNet](imagenet.md) | 1000-class natural images | 1.2M |
| [FFHQ](ffhq.md) | High-quality face images | 70K |
| [Custom Image](custom_image.md) | Load from directory | Variable |

### Loading Images

```python
from artifex.data.image import CIFARDataset, FFHQDataset

# CIFAR-10
cifar = CIFARDataset(
    root="./data",
    train=True,
    download=True,
)

# FFHQ
ffhq = FFHQDataset(
    root="/path/to/ffhq",
    resolution=256,
)
```

### Image Augmentation

```python
from artifex.data.augmentation import ImageAugmentation

augment = ImageAugmentation(
    random_flip=True,
    random_crop=True,
    color_jitter=0.1,
    random_rotation=15,
)

augmented = augment(images, key=prng_key)
```

[:octicons-arrow-right-24: Image Augmentation](image.md)

## Text Datasets

### Standard Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| [Wikipedia](wikipedia.md) | Wikipedia articles | 6M+ articles |
| [BookCorpus](bookcorpus.md) | Book text corpus | 11K books |
| [Custom Text](custom_text.md) | Load from files | Variable |

### Loading Text

```python
from artifex.data.text import WikipediaDataset

wiki = WikipediaDataset(
    language="en",
    max_length=512,
)
```

### Tokenizers

```python
from artifex.data.tokenizers import BPETokenizer, SentencePieceTokenizer

# BPE Tokenizer
tokenizer = BPETokenizer(vocab_size=32000)
tokenizer.fit(corpus)
tokens = tokenizer.encode(text)

# SentencePiece
tokenizer = SentencePieceTokenizer(model_path="model.spm")
```

[:octicons-arrow-right-24: BPE Tokenizer](bpe.md) | [:octicons-arrow-right-24: SentencePiece](sentencepiece.md)

## Audio Datasets

### Standard Datasets

| Dataset | Description | Hours |
|---------|-------------|-------|
| [LibriSpeech](librispeech.md) | Read English speech | 1000h |
| [VCTK](vctk.md) | Multi-speaker speech | 44h |
| [Custom Audio](custom_audio.md) | Load from files | Variable |

### Loading Audio

```python
from artifex.data.audio import LibriSpeechDataset

librispeech = LibriSpeechDataset(
    root="./data",
    subset="train-clean-100",
    sample_rate=16000,
)
```

### Audio Preprocessing

```python
from artifex.data.preprocessing import AudioPreprocessor

preprocessor = AudioPreprocessor(
    sample_rate=16000,
    n_mels=80,
    n_fft=1024,
    hop_length=256,
)

mel_spec = preprocessor.to_mel_spectrogram(audio)
```

[:octicons-arrow-right-24: Audio Preprocessing](audio.md)

## Video Datasets

### Standard Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| [UCF101](ucf101.md) | Action recognition | 13K clips |
| [Custom Video](custom_video.md) | Load from files | Variable |

### Loading Video

```python
from artifex.data.video import UCF101Dataset

ucf = UCF101Dataset(
    root="/path/to/ucf101",
    frames_per_clip=16,
    frame_rate=5,
)
```

[:octicons-arrow-right-24: Video Datasets](ucf101.md)

## Multimodal Datasets

### Standard Datasets

| Dataset | Description | Modalities |
|---------|-------------|------------|
| [COCO](coco.md) | Image + captions | Image, Text |
| [Custom Multimodal](custom_multimodal.md) | Custom pairs | Variable |

### Loading Multimodal Data

```python
from artifex.data.multimodal import COCODataset

coco = COCODataset(
    root="/path/to/coco",
    split="train2017",
    include_captions=True,
)

for batch in coco:
    images = batch["image"]
    captions = batch["caption"]
```

[:octicons-arrow-right-24: COCO Dataset](coco.md)

## Protein Datasets

```python
from artifex.data.protein import ProteinDataset

protein_data = ProteinDataset(
    pdb_dir="/path/to/structures",
    max_length=256,
    include_sequence=True,
    include_structure=True,
)
```

[:octicons-arrow-right-24: Protein Dataset](dataset.md)

## Data Pipeline

### Pipeline API

```python
from artifex.data import DataPipeline

pipeline = DataPipeline()
pipeline.add_step("load", loader)
pipeline.add_step("preprocess", preprocessor)
pipeline.add_step("augment", augmentation)
pipeline.add_step("batch", batcher)

for batch in pipeline(data):
    # Process batch
    pass
```

[:octicons-arrow-right-24: Pipeline Reference](pipeline.md)

### Collators

```python
from artifex.data import DynamicBatchCollator

collator = DynamicBatchCollator(
    pad_token_id=0,
    max_length=512,
)
```

[:octicons-arrow-right-24: Collators](collators.md)

### Samplers

```python
from artifex.data import DistributedSampler, WeightedSampler

# Distributed training
sampler = DistributedSampler(
    dataset=dataset,
    num_replicas=4,
    rank=0,
)

# Weighted sampling
sampler = WeightedSampler(
    weights=class_weights,
    num_samples=len(dataset),
)
```

[:octicons-arrow-right-24: Samplers](samplers.md)

## Streaming Data

### WebDataset

```python
from artifex.data.streaming import WebDatasetLoader

loader = WebDatasetLoader(
    urls="s3://bucket/data-{000..099}.tar",
    batch_size=64,
    shuffle=True,
)
```

[:octicons-arrow-right-24: WebDataset](webdataset.md)

### TFRecord

```python
from artifex.data.streaming import TFRecordLoader

loader = TFRecordLoader(
    pattern="/path/to/data-*.tfrecord",
    features={
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    },
)
```

[:octicons-arrow-right-24: TFRecord](tfrecord.md)

### Remote Loading

```python
from artifex.data.streaming import RemoteLoader

loader = RemoteLoader(
    endpoint="https://api.example.com/data",
    cache_dir="./cache",
)
```

[:octicons-arrow-right-24: Remote Loading](remote.md)

## Module Reference

| Category | Modules |
|----------|---------|
| **Image** | [cifar](cifar.md), [custom_image](custom_image.md), [ffhq](ffhq.md), [imagenet](imagenet.md) |
| **Text** | [bookcorpus](bookcorpus.md), [custom_text](custom_text.md), [wikipedia](wikipedia.md) |
| **Audio** | [custom_audio](custom_audio.md), [librispeech](librispeech.md), [vctk](vctk.md) |
| **Video** | [custom_video](custom_video.md), [ucf101](ucf101.md) |
| **Multimodal** | [coco](coco.md), [custom_multimodal](custom_multimodal.md) |
| **Protein** | [dataset](dataset.md) |
| **Tokenizers** | [bpe](bpe.md), [character](character.md), [sentencepiece](sentencepiece.md), [word](word.md) |
| **Augmentation** | [audio](audio.md), [image](image.md), [text](text.md), [video](video.md) |
| **Preprocessing** | [audio](audio.md), [base](base.md), [image](image.md), [text](text.md), [video](video.md) |
| **Loaders** | [audio](audio.md), [base](base.md), [collators](collators.md), [image](image.md), [pipeline](pipeline.md), [protein_dataset](protein_dataset.md), [registry](registry.md), [samplers](samplers.md), [structured](structured.md), [text](text.md), [video](video.md) |
| **Streaming** | [remote](remote.md), [tfrecord](tfrecord.md), [webdataset](webdataset.md) |

## Related Documentation

- [Data Loading Guide](../user-guide/data/data-guide.md) - Complete data loading guide
- [Image Modality](../user-guide/modalities/image.md) - Image-specific features
- [Text Modality](../user-guide/modalities/text.md) - Text-specific features
- [Audio Modality](../user-guide/modalities/audio.md) - Audio-specific features
