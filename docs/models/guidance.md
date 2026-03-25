# Guidance

Supported owner: `artifex.generative_models.models.diffusion.guidance`

## Public Imports

```python
from artifex.generative_models.models.diffusion import (
    ClassifierFreeGuidance,
    ClassifierGuidance,
    ConditionalDiffusionMixin,
    GuidedDiffusionModel,
    apply_guidance,
    cosine_guidance_schedule,
    linear_guidance_schedule,
)
```

## Overview

The retained guidance surface is the top-level diffusion export set above.
These owners cover classifier-free guidance, classifier guidance, the
conditional diffusion mixin, the guided wrapper model, and the exported
schedule helpers.

Family-local closures and per-instance helper methods remain implementation
details rather than supported module-level API.

## Related Pages

- [Model Implementations](index.md)
- [Diffusion API Reference](../api/models/diffusion.md)
- [Diffusion Guide](../user-guide/models/diffusion-guide.md)
