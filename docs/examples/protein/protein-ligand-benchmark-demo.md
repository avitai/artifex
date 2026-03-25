# Protein-Ligand Benchmark Demo

**Status:** `Demo-only benchmark walkthrough`
**Device:** `CPU-compatible`

This walkthrough demonstrates the retained protein-ligand benchmark owners in explicit demo mode. The shipped script
uses mock CrossDocked-style complexes and retained heuristic drug-likeness scoring; it should not be read as a
benchmark-grade public CrossDocked runtime.

## Files

- Python script: [protein_ligand_benchmark_demo.py](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_ligand_benchmark_demo.py)
- Jupyter notebook: [protein_ligand_benchmark_demo.ipynb](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_ligand_benchmark_demo.ipynb)

## Run It

```bash
python examples/generative_models/protein/protein_ligand_benchmark_demo.py
jupyter lab examples/generative_models/protein/protein_ligand_benchmark_demo.ipynb
```

## What This Demo Actually Uses

- `CrossDockedDataset` with `metadata={"demo_mode": True, ...}`
- `ProteinLigandBenchmarkSuite(..., demo_mode=True)`
- `ProteinLigandCoDesignBenchmark(..., demo_mode=True)`
- `DrugLikenessMetric` in explicit demo mode plus retained validity and affinity helpers
- typed molecular constraints configured through `ChemicalConstraintConfig` under an explicit `extensions={...}` bundle

```python
from artifex.generative_models.core.configuration import ChemicalConstraintConfig

extensions={
    "chemical": ChemicalConstraintConfig(name="chemical_constraints")
}
```

## Why It Is Demo-Only

- the dataset owner still generates mock protein-ligand complexes instead of loading benchmark-grade CrossDocked assets
- drug-likeness scoring is retained heuristic logic, not a chemistry-backend benchmark path
- the page teaches explicit demo opt-in rather than a supported public benchmark command surface

## Use This When

Use this pair when you want to inspect the retained protein-ligand benchmark interfaces, prototype against the demo
batch shapes, or replace the demo data and heuristics with your own benchmark-grade backend from Python.
