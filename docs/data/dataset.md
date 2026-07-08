# Protein Dataset Module

Canonical owner: `artifex.data.protein.dataset`

This page documents the retained protein dataset runtime. There is one current
module owner for this surface: `artifex.data.protein.dataset`.

## Current Runtime Surface

The canonical module exposes:

- `ProteinDataset`
- `ProteinDatasetConfig`
- `ProteinStructure`
- `protein_collate_fn`
- `create_synthetic_protein_dataset`
- `pdb_to_protein_example`

`ProteinDataset` is a Datarax-backed `DataSourceModule`. It accepts a
`ProteinDatasetConfig` plus either in-memory protein dictionaries or a
pickle-file / directory path. It exposes both local protein collation and the
Datarax indexed source contract:

- `get_batch(indices)` or `get_batch(batch_size)` uses `protein_collate_fn`.
- `get_batch_at(start, size, key)` returns fixed-shape tensor batches padded to
  `ProteinDatasetConfig.max_seq_length` for `Pipeline.step()` and Pipeline
  iteration.

## Example

```python
from datarax import Pipeline
from flax import nnx

from artifex.data.protein import ProteinDataset, ProteinDatasetConfig

config = ProteinDatasetConfig(max_seq_length=128)
dataset = ProteinDataset(config, data_dir="./protein-pickles")
batch = dataset.get_batch(4)
pipeline_batch = Pipeline(source=dataset, stages=[], batch_size=4, rngs=nnx.Rngs(0)).step()
```

For the broader Datarax data-loading story, see:

- [User Guide: Data Overview](../user-guide/data/overview.md)
- [API Reference: Data Loaders](../api/data/loaders.md)
