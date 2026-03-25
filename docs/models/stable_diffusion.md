# Stable Diffusion

Supported owner: `artifex.generative_models.models.diffusion.stable_diffusion`

## Public Import

```python
from artifex.generative_models.models.diffusion.stable_diffusion import StableDiffusionModel
```

## Overview

`StableDiffusionModel` remains a live module-local owner for text-conditioned
latent diffusion.

It is not re-exported from `artifex.generative_models.models.diffusion`, so the
supported import path for this page stays module-local.

Text encoding, cross-attention UNet components, and spatial VAE pieces stay in
their own owner modules beneath the Stable Diffusion stack instead of being
published here as additional top-level symbols.

## Related Pages

- [Model Implementations](index.md)
- [Diffusion API Reference](../api/models/diffusion.md)
- [Diffusion Guide](../user-guide/models/diffusion-guide.md)
