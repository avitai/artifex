# Protein Dataset

This page is retained for historical links.

There is no `artifex.data.protein_dataset` module in the live runtime. The
canonical owner is `artifex.data.protein.dataset`, re-exported through
`artifex.data.protein`.

## Current Imports

```python
from artifex.data.protein import ProteinDataset, ProteinDatasetConfig

config = ProteinDatasetConfig(max_seq_length=128)
dataset = ProteinDataset(config, data_dir="./protein-pickles")
```

Use these current helpers from the canonical module:

- `ProteinDataset`
- `ProteinDatasetConfig`
- `ProteinStructure`
- `protein_collate_fn`
- `create_synthetic_protein_dataset`
- `pdb_to_protein_example`

See [Protein Dataset Module](dataset.md) for the maintained reference page.
