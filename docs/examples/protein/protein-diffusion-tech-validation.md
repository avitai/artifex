# Protein Diffusion Technical Validation

**Status:** `Validation utility`
**Device:** `CPU-compatible`

This pair is a validation utility for JAX, Flax NNX, and optional BioPython
setup around protein experimentation. It does not instantiate shipped Artifex
protein model, modality, or data owners, so it is grouped as validation rather
than as a canonical modeling tutorial.

## Files

- Python script: [`examples/generative_models/protein/protein_diffusion_tech_validation.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_diffusion_tech_validation.py)
- Jupyter notebook: [`examples/generative_models/protein/protein_diffusion_tech_validation.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_diffusion_tech_validation.ipynb)

## Run It

```bash
python examples/generative_models/protein/protein_diffusion_tech_validation.py
jupyter lab examples/generative_models/protein/protein_diffusion_tech_validation.ipynb
```

## What It Checks

- JAX array creation and random-number generation
- Flax NNX module construction and forward passes
- optional BioPython availability for local PDB parsing experiments
- a minimal raw-NNX point-cloud transformation path

## What It Does Not Claim

- it does not instantiate shipped Artifex protein model, modality, or data owners
- it does not validate an end-to-end protein training or generation workflow
- it should not be read as the canonical entry point for protein modeling in Artifex

Use this pair when you want a fast environment sanity check before moving on to
the retained protein examples or lower-level exploratory work.
