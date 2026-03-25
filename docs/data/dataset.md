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
pickle-file / directory path.

## Example

```python
from artifex.data.protein import ProteinDataset, ProteinDatasetConfig

config = ProteinDatasetConfig(max_seq_length=128)
dataset = ProteinDataset(config, data_dir="./protein-pickles")
batch = dataset.get_batch(4)
```

For the broader Datarax data-loading story, see:

- [User Guide: Data Overview](../user-guide/data/overview.md)
- [API Reference: Data Loaders](../api/data/loaders.md)
