# Model Implementations

This catalog tracks the live runtime surface shipped under
`artifex.generative_models.models`. It does not treat deleted stubs or
roadmap-only standalone pages as supported model API.

## Family Packages

| Family | Owner package | Representative exports |
| --- | --- | --- |
| VAE | `artifex.generative_models.models.vae` | `VAE`, `BetaVAE`, `ConditionalVAE`, `VQVAE` |
| GAN | `artifex.generative_models.models.gan` | `GAN`, `DCGAN`, `WGAN`, `LSGAN`, `ConditionalGAN`, `CycleGAN` |
| Diffusion | `artifex.generative_models.models.diffusion` | `DiffusionModel`, `DDPMModel`, `ScoreDiffusionModel`, `LDMModel`, `DiTModel`, guidance helpers |
| Flow | `artifex.generative_models.models.flow` | `NormalizingFlow`, `RealNVP`, `Glow`, `MAF`, `IAF`, `NeuralSplineFlow` |
| Energy | `artifex.generative_models.models.energy` | `EBM`, `DeepEBM`, `EnergyBasedModel`, Langevin helpers |
| Autoregressive | `artifex.generative_models.models.autoregressive` | `PixelCNN`, `WaveNet`, `TransformerAutoregressiveModel` |
| Geometric | `artifex.generative_models.models.geometric` | `GraphModel`, `MeshModel`, `PointCloudModel`, `VoxelModel`, protein geometric models |

## Diffusion Surface

Use the top-level diffusion package for the supported exported family surface:

```python
from artifex.generative_models.models.diffusion import (
    DDPMModel,
    DiTModel,
    DiffusionModel,
    LDMModel,
    ScoreDiffusionModel,
)
```

Stable Diffusion currently remains a module-local owner rather than a top-level
package export:

```python
from artifex.generative_models.models.diffusion.stable_diffusion import StableDiffusionModel
```

`StableDiffusionModel` is not re-exported from `artifex.generative_models.models.diffusion`.

## Backbone And Standalone-Page Boundaries

UNet is currently owned by the backbone package, not by a standalone
`models.common.unet` model module.

```python
from artifex.generative_models.core.configuration import UNetBackboneConfig
import artifex.generative_models.models.backbones.unet as diffusion_unet
```

Use `UNetBackboneConfig` and the live backbone owner
`artifex.generative_models.models.backbones.unet` for retained UNet building
blocks.

## Planned Standalone Pages

The following pages remain relevant to the roadmap, but they are not part of
the supported model-reference surface today:

- standalone UNet page
- shared conditioning module page
- standalone StyleGAN page

See [planned-modules.md](../roadmap/planned-modules.md) for the roadmap-only
entries behind those pages.
