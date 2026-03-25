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

config = ProteinDatasetConfig(max_seq_length=128)
dataset = ProteinDataset(config, data_dir="./protein-pickles")
batch = dataset.get_batch(4)

assert protein.ProteinDataset is ProteinDataset
print(batch["atom_positions"].shape)
```

`ProteinDataset` is backed by datarax's `DataSourceModule`, so it keeps the
standard Datarax indexing, iteration, batching, and `from_source(...)`
integration story.

## Where The Broader Data Story Lives

For general data-loading guidance, use the datarax-backed docs that describe the
surviving runtime directly:

- [User Guide: Data Overview](../user-guide/data/overview.md)
- [User Guide: Data Guide](../user-guide/data/data-guide.md)
- [API Reference: Data Loaders](../api/data/loaders.md)

For modality-specific synthetic dataset helpers, use the owners under
`artifex.generative_models.modalities.*.datasets` rather than expecting extra
subpackages under `artifex.data`.
