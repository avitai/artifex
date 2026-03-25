# Notebooks

This published page lives under `docs/notebooks/`, but the actual notebooks live in the repo-root `notebooks/` directory. The `docs/notebooks/` directory contains only this page; it is not the runnable local notebook folder.

## Checked-In Notebook Inventory

The current repo-root notebook inventory is:

- `notebooks/artifex_vae_celeba_optimized.ipynb`
  - modern VAE training workflow on CelebA
- `notebooks/protein_diffusion.ipynb`
  - retained protein diffusion research notebook

Only list notebooks here when the `.ipynb` file is checked in under the repo-root `notebooks/` directory. Example scripts under `examples/` remain part of the examples catalog, not the notebook inventory.

## Running A Notebook Locally

```bash
./setup.sh
source ./activate.sh
uv run jupyter lab
```

After Jupyter starts, open one of the checked-in notebook paths above from the repo root.

## Maintenance Rule

- add a notebook here only when the `.ipynb` file exists under `notebooks/`
- keep the documented path identical to the checked-in file path
- do not fold ordinary examples, setup notes, or non-notebook feature lists into this page
