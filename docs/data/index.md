# Data Package

The top-level `artifex.data` package is intentionally narrow.

Artifex does not ship a second generic data-processing framework with top-level
helpers such as `load_dataset`, `ImageDataset`, `DataPipeline`, or streaming
loader classes. The general data-loading story lives in datarax-backed guides
and in modality-local dataset helpers.

## Current Public Surface

Today the retained top-level data surface is:

- `artifex.data.protein`

That subpackage exports the protein-specific dataset helpers that are still part
of the checked-in runtime:

- `ProteinDataset`
- `ProteinDatasetConfig`
- `ProteinStructure`
- `protein_collate_fn`
- `create_synthetic_protein_dataset`
- `pdb_to_protein_example`

## Protein Dataset Example

```python
from artifex.data import protein
from artifex.data.protein import ProteinDataset, ProteinDatasetConfig
from datarax import Pipeline
from flax import nnx

config = ProteinDatasetConfig(max_seq_length=128)
dataset = ProteinDataset(config, data_dir="./protein-pickles")
batch = dataset.get_batch(4)
pipeline_batch = Pipeline(source=dataset, stages=[], batch_size=4, rngs=nnx.Rngs(0)).step()

assert protein.ProteinDataset is ProteinDataset
print(batch["atom_positions"].shape)
print(pipeline_batch["atom_positions"].shape)
```

`ProteinDataset` is backed by datarax's `DataSourceModule`, so it keeps the
standard Datarax indexing, iteration, batching, and `Pipeline(...)`
integration story. The local `get_batch(...)` helper can collate variable-length
protein examples, while the Datarax `get_batch_at(...)` and `Pipeline.step()`
path returns fixed-shape tensor fields padded to `ProteinDatasetConfig.max_seq_length`.

## Where The Broader Data Story Lives

For general data-loading guidance, use the datarax-backed docs that describe the
surviving runtime directly:

- [User Guide: Data Overview](../user-guide/data/overview.md)
- [User Guide: Data Guide](../user-guide/data/data-guide.md)
- [API Reference: Data Loaders](../api/data/loaders.md)

For modality-specific synthetic dataset helpers, use the owners under
`artifex.generative_models.modalities.*.datasets` rather than expecting extra
subpackages under `artifex.data`.
