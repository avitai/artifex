# Autoregressive Models API Reference

Complete API documentation for autoregressive models in Artifex.

!!! info "Coming Soon"
    Autoregressive model implementations are planned for a future release. This documentation will be updated when the feature is available.

## Overview

Autoregressive models will include:

- **PixelCNN**: Pixel-by-pixel image generation
- **WaveNet**: Audio waveform generation
- **Transformer-based**: GPT-style text generation
- **MAR (Masked Autoregressive)**: Masked autoregressive models

## Planned API

### Base Class

```python
from artifex.generative_models.models.autoregressive.base import AutoregressiveModel

model = AutoregressiveModel(
    config: AutoregressiveConfig,
    *,
    rngs: nnx.Rngs,
)
```

### Configuration

```python
from artifex.generative_models.core.configuration import AutoregressiveConfig

config = AutoregressiveConfig(
    name="autoregressive_model",
    sequence_length=256,
    vocab_size=10000,
    hidden_dim=512,
    num_layers=6,
)
```

## Related Documentation

- [Autoregressive Models Guide](../../user-guide/models/autoregressive-guide.md) - Conceptual overview
- [Autoregressive Concepts](../../user-guide/concepts/autoregressive-explained.md) - Understanding autoregressive modeling

## References

- van den Oord et al., "Pixel Recurrent Neural Networks" (2016)
- van den Oord et al., "WaveNet: A Generative Model for Raw Audio" (2016)
- Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)
