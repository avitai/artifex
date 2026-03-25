# Protein Diffusion Example

**Status:** `Exploratory workflow`
**Device:** `CPU-compatible`

This walkthrough uses real lower-level Artifex protein structure owners around
synthetic data and padded batches, but it does not demonstrate a shipped
high-level Artifex protein diffusion API. The historical filename is retained
for continuity with the existing example tree, while the published contract is
now explicit: this is an exploratory direct-owner workflow rather than a
canonical protein-diffusion tutorial.

## Files

- Python script: [`examples/generative_models/protein/protein_diffusion_example.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_diffusion_example.py)
- Jupyter notebook: [`examples/generative_models/protein/protein_diffusion_example.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_diffusion_example.ipynb)

## Run It

```bash
python examples/generative_models/protein/protein_diffusion_example.py
jupyter lab examples/generative_models/protein/protein_diffusion_example.ipynb
```

## What This Workflow Actually Uses

- `ProteinPointCloudModel` and `ProteinGraphModel` for direct model construction
- `ProteinDataset` plus `create_synthetic_protein_dataset` for retained
  synthetic protein data
- `protein_collate_fn` for padded batch creation
- `create_protein_structure_loss` for protein-specific loss evaluation
- optional visualization helpers from `artifex.visualization.protein_viz`

## Current Data And Extension Contract

The retained data and extension path is still explicit in the paired source:

```python
from artifex.data.protein import ProteinDataset, ProteinDatasetConfig
from artifex.generative_models.core.configuration import (
    ProteinDihedralConfig,
    ProteinExtensionConfig,
    ProteinExtensionsConfig,
)

dataset_config = ProteinDatasetConfig(max_seq_length=64)
dataset = ProteinDataset(dataset_config, data_dir="data/proteins")

extensions = ProteinExtensionsConfig(
    name="protein_extensions",
    backbone=ProteinExtensionConfig(name="backbone", weight=1.0),
    dihedral=ProteinDihedralConfig(name="dihedral", weight=0.3),
)

config = ProteinPointCloudConfig(
    name="protein_point_cloud_exploratory",
    extensions=extensions,
)
```

## Why It Is Exploratory

- it does not demonstrate a shipped high-level Artifex protein diffusion API
- it focuses on lower-level direct-owner surfaces and synthetic data instead of
  a polished public protein-diffusion facade
- it is useful for contributors inspecting the retained protein model and batch
  contracts, but it should not be read as the canonical end-to-end protein
  example for the library

## Use This When

Use this pair if you want to explore the retained protein point-cloud and graph
owners, inspect the current padded-batch contract, or prototype protein loss and
visualization work on top of synthetic inputs. For retained protein tutorials
with a narrower public claim, use the other protein pages in the main catalog.
