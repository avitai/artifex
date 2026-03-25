# WGAN

Supported owner: `artifex.generative_models.models.gan.wgan`

## Public Imports

```python
from artifex.generative_models.models.gan import (
    WGAN,
    WGANDiscriminator,
    WGANGenerator,
    compute_gradient_penalty,
)
```

## Overview

The retained WGAN surface consists of the main `WGAN` model, the concrete
`WGANGenerator` and `WGANDiscriminator` owners, and the exported
`compute_gradient_penalty` helper.

Generation and objective behavior live on the model instances themselves rather
than as additional module-level public helpers.

## Related Pages

- [Model Implementations](index.md)
- [GAN API Reference](../api/models/gan.md)
- [GAN Guide](../user-guide/models/gan-guide.md)
