# Data Modalities

The live Artifex modality registry currently exposes three runtime-backed
modalities:

- `image`
- `molecular`
- `protein`

This page documents that retained registry surface and then separates the
family-scoped owner pages that still live under the shared `docs/modalities`
folder.

## Registry-Backed Modalities

### Image

Use the registry-backed image modality for the default factory-ready example
path across VAE, GAN, diffusion, and flow-family models.

[:octicons-arrow-right-24: Image Modality Guide](../user-guide/modalities/image.md)

### Molecular

Use the registry-backed molecular modality when you need typed chemical
extension configuration and family-based adapter lookup.

[:octicons-arrow-right-24: Protein-Ligand Benchmark Example](../examples/protein/protein-ligand-benchmark-demo.md)

### Protein

Use the registry-backed protein modality when you need typed protein extension
bundles and retained geometric or diffusion adapter lookup.

[:octicons-arrow-right-24: Protein Modeling Guide](../guides/protein-modeling.md)

## Quick Start

Use the registry-backed modality entrypoint:

```python
from artifex.generative_models.modalities import get_modality, list_modalities

available = list_modalities()
# ['image', 'molecular', 'protein']

image_modality = get_modality('image', rngs=rngs)
image_adapter = image_modality.get_adapter('vae')
adapted_model = image_adapter.adapt(model, config)
```

## Family-Scoped Owner Pages

The shared filenames in this catalog are not modality-generic.

### Timeseries Helper Owners

These pages document the retained timeseries helper package, which is not part
of the shared registry-backed modality contract:

- [Timeseries Base](base.md)
- [Timeseries Adapters](adapters.md)
- [Timeseries Datasets](datasets.md)
- [Timeseries Evaluation](evaluation.md)
- [Timeseries Representations](representations.md)

### Protein Owner Pages

These pages are protein-specific owners rather than shared modality categories:

- [Protein Modality](modality.md)
- [Protein Config](config.md)
- [Protein Losses](losses.md)
- [Protein Utils](utils.md)

Image, text, audio, and multi-modal helper packages keep their own package-local docs.
The shared owner pages above do not apply to them.

## Modality Registry

The registry is the authoritative surface for the retained modality set:

```python
from artifex.generative_models.modalities import (
    get_modality,
    list_modalities,
    register_modality,
)

available = list_modalities()
# ['image', 'molecular', 'protein']

protein_modality = get_modality('protein', rngs=rngs)
register_modality('custom', CustomModality)
```

[:octicons-arrow-right-24: Registry Owner](registry.md)

## Related Documentation

- [Model Factory](../factory/index.md)
- [Image Modality Guide](../user-guide/modalities/image.md)
- [Protein Modeling Guide](../guides/protein-modeling.md)
- [Protein-Ligand Benchmark Example](../examples/protein/protein-ligand-benchmark-demo.md)
