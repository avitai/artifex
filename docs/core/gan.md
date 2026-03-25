# GAN Configuration

Supported owner: `artifex.generative_models.core.configuration.gan_config`

Public imports for these config types are re-exported from
`artifex.generative_models.core.configuration`.

## Overview

The GAN configuration layer uses frozen dataclasses built on the shared typed
config foundation. The retained public types are:

- `GANConfig`
- `DCGANConfig`
- `WGANConfig`
- `LSGANConfig`
- `ConditionalGANConfig`
- `CycleGANConfig`

These configs compose the shared network config objects such as
`GeneratorConfig`, `DiscriminatorConfig`, `ConvGeneratorConfig`, and
`ConvDiscriminatorConfig` rather than relying on a deleted unified helper
module.

## Typical Imports

```python
from artifex.generative_models.core.configuration import (
    ConditionalGANConfig,
    CycleGANConfig,
    DCGANConfig,
    GANConfig,
    LSGANConfig,
    WGANConfig,
)
```

## Notes

- the canonical module path is
  `artifex.generative_models.core.configuration.gan_config`
- legacy configuration-class aliases are not part of the supported runtime
- helper facades from the removed unified-config story are not shipped in the
  current config surface

## Related Pages

- [Configuration Overview](configuration.md)
- [API Configuration Reference](../api/core/configuration.md)
- [Training Configuration Guide](../user-guide/training/configuration.md)
