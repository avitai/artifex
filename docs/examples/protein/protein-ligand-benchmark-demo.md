# Protein-Ligand Co-Design Benchmark

**Level:** Advanced | **Runtime:** ~3-5 minutes (CPU) / ~1-2 minutes (GPU) | **Format:** Python + Jupyter

**Prerequisites:** Understanding of protein-ligand interactions, drug discovery, and molecular modeling | **Target Audience:** Computational chemists and drug discovery researchers

## Overview

This example demonstrates a comprehensive benchmark suite for evaluating protein-ligand co-design models. Learn how to use the CrossDocked2020 dataset, compute binding affinity predictions, assess molecular validity, evaluate drug-likeness, and systematically compare model architectures for computational drug discovery.

## What You'll Learn

<div class="grid cards" markdown>

- :material-molecule: **Molecular Modality**

    ---

    Domain-specific framework for chemical structure representation

- :material-database-outline: **CrossDocked2020**

    ---

    Large-scale protein-ligand binding dataset with 22.5M complexes

- :material-chart-line: **Binding Affinity**

    ---

    Predict and evaluate protein-ligand binding energies (kcal/mol)

- :material-check-circle: **Molecular Validity**

    ---

    Assess chemical plausibility of generated structures

- :material-pill: **Drug-likeness (QED)**

    ---

    Quantify pharmaceutical potential using QED scores

- :material-compare: **Benchmark Suite**

    ---

    Systematically evaluate and compare model architectures

</div>

## Files

This example is available in two formats:

- **Python Script**: [`protein_ligand_benchmark_demo.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_ligand_benchmark_demo.py)
- **Jupyter Notebook**: [`protein_ligand_benchmark_demo.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_ligand_benchmark_demo.ipynb)

## Quick Start

### Run the Python Script

```bash
# Activate environment
source activate.sh

# Run the complete demo
python examples/generative_models/protein/protein_ligand_benchmark_demo.py
```

### Run the Jupyter Notebook

```bash
# Activate environment
source activate.sh

# Launch Jupyter
jupyter lab examples/generative_models/protein/protein_ligand_benchmark_demo.ipynb
```

## Key Concepts

### 1. Protein-Ligand Co-Design

Simultaneously optimizing both the protein binding site and ligand molecule for strong, specific binding:

```
Protein Pocket + Ligand → Protein-Ligand Complex
     ↓               ↓              ↓
  Flexibility    Chemistry    Binding Affinity
  Specificity    Drug-like    Stability
```

**Applications:**

- De novo drug design
- Lead optimization
- Binding site engineering
- Personalized medicine

### 2. CrossDocked2020 Dataset

Large-scale protein-ligand binding dataset:

```python
from artifex.benchmarks.datasets.crossdocked import CrossDockedDataset
from artifex.generative_models.core.configuration import DataConfig

# Create dataset configuration
dataset_config = DataConfig(
    name="crossdocked_demo",
    dataset_name="crossdocked",
    metadata={
        "num_samples": 50,
        "max_protein_atoms": 200,
        "max_ligand_atoms": 30,
        "pocket_radius": 8.0,
    },
)

dataset = CrossDockedDataset(
    data_path="./data/crossdocked",
    config=dataset_config,
    rngs=rngs,
)

# Get a sample
sample = dataset[0]
# sample = {
#     "protein_coords": (200, 3),      # Protein atom coordinates
#     "protein_types": (200,),         # Atom types (C, N, O, S, etc.)
#     "ligand_coords": (30, 3),        # Ligand atom coordinates
#     "ligand_types": (30,),           # Ligand atom types
#     "binding_affinity": -8.5,        # In kcal/mol (lower = stronger)
#     "pocket_indices": [12, 45, ...], # Binding pocket atom indices
# }
```

**Dataset Statistics:**

- Total complexes: 22.5 million docked pairs
- Protein size: ~50-500 atoms (binding pocket)
- Ligand size: ~10-50 atoms (drug-like)
- Binding affinity range: -15 to 0 kcal/mol

### 3. Molecular Modality Framework

Domain-specific functionality for chemical structures:

```python
from artifex.generative_models.modalities.molecular import MolecularModality

modality = MolecularModality(rngs=rngs)

# Chemical constraints
config = ModalityConfiguration(
    name="molecular_config",
    modality_name="molecular",
    metadata={
        "use_chemical_constraints": True,
        "bond_length_weight": 1.0,        # Enforce realistic bond lengths
        "bond_angle_weight": 0.5,         # Enforce bond angles
        "use_pharmacophore_features": True,
        "pharmacophore_types": [
            "donor",       # H-bond donors
            "acceptor",    # H-bond acceptors
            "hydrophobic"  # Hydrophobic regions
        ],
    }
)

extensions = modality.get_extensions(config, rngs=rngs)
# extensions = {
#     "chemical_constraints": <ConstraintModule>,
#     "pharmacophore_features": <PharmacophoreModule>,
# }
```

### 4. Binding Affinity Metric

Evaluates binding affinity prediction accuracy:

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (\Delta G_{\text{pred}}^i - \Delta G_{\text{true}}^i)^2}$$

```python
from artifex.benchmarks.metrics.protein_ligand import BindingAffinityMetric

metric = BindingAffinityMetric(rngs=rngs)

# True binding affinities (in kcal/mol)
true_affinities = jnp.array([-8.2, -6.5, -9.1, -7.8])

# Model predictions
predictions = jnp.array([-8.5, -6.2, -8.9, -7.5])

results = metric.compute(predictions, true_affinities)
# results = {
#     "rmse": 0.32,          # Root Mean Square Error (kcal/mol)
#     "pearson_r": 0.95,     # Correlation coefficient
#     "mae": 0.28,           # Mean Absolute Error
# }
```

**Performance Targets:**

- Excellent: RMSE < 0.5 kcal/mol
- Good: RMSE < 1.0 kcal/mol
- Acceptable: RMSE < 1.5 kcal/mol

### 5. Molecular Validity Metric

Checks chemical plausibility of generated molecules:

```python
from artifex.benchmarks.metrics.protein_ligand import MolecularValidityMetric

metric = MolecularValidityMetric(rngs=rngs)

results = metric.compute(
    coordinates=ligand_coords,  # (batch, num_atoms, 3)
    atom_types=atom_types,      # (batch, num_atoms)
    masks=atom_masks            # (batch, num_atoms)
)
# results = {
#     "validity_rate": 0.96,      # Overall validity (target: >0.95)
#     "bond_validity": 0.98,      # Valid bond lengths
#     "clash_free": 0.94,         # No atomic clashes
#     "connectivity": 0.97,       # Proper atom connectivity
# }
```

**Validity Checks:**

- **Bond lengths**: 1.2-2.0 Å for most bonds
- **No clashes**: Atoms >1.0 Å apart (except bonded)
- **Connectivity**: All atoms form connected graph
- **Valence**: Atoms respect valence rules

### 6. Drug-likeness Metric (QED)

Quantitative Estimate of Drug-likeness:

$$\text{QED} = \exp\left(\frac{1}{8}\sum_{i=1}^{8} \ln p_i\right)$$

where $p_i$ are desirability functions for 8 molecular properties.

```python
from artifex.benchmarks.metrics.protein_ligand import DrugLikenessMetric

metric = DrugLikenessMetric(rngs=rngs)

results = metric.compute(
    coordinates=ligand_coords,
    atom_types=atom_types,
    masks=atom_masks
)
# results = {
#     "qed_score": 0.75,             # Overall drug-likeness (target: >0.7)
#     "lipinski_compliance": 0.85,   # Lipinski's Rule of Five
#     "molecular_weight": 385.4,     # Daltons (target: 180-500)
#     "logp": 2.3,                   # Lipophilicity (target: 0-5)
#     "h_bond_donors": 2,            # Target: ≤5
#     "h_bond_acceptors": 4,         # Target: ≤10
# }
```

**Lipinski's Rule of Five:**

- Molecular weight ≤ 500 Da
- LogP ≤ 5
- H-bond donors ≤ 5
- H-bond acceptors ≤ 10

### 7. Benchmark Suite

Comprehensive evaluation across all metrics:

```python
from artifex.benchmarks.suites.protein_ligand_suite import ProteinLigandBenchmarkSuite

suite = ProteinLigandBenchmarkSuite(
    dataset_config={
        "num_samples": 50,
        "max_protein_atoms": 200,
        "max_ligand_atoms": 30,
    },
    benchmark_config={
        "num_samples": 20,
        "batch_size": 4,
    },
    rngs=rngs
)

# Run evaluation
results = suite.run_all(model)
# results = {
#     "binding_affinity": {
#         "rmse": 0.45,
#         "pearson_r": 0.92,
#     },
#     "molecular_validity": {
#         "validity_rate": 0.97,
#         "bond_validity": 0.98,
#     },
#     "drug_likeness": {
#         "qed_score": 0.78,
#         "lipinski_compliance": 0.89,
#     },
# }
```

## Code Structure

The example demonstrates six main components:

1. **Molecular Modality Framework** - Chemical constraints and pharmacophore features
2. **CrossDocked2020 Dataset** - Protein-ligand complex loading and statistics
3. **Binding Affinity Metric** - RMSE evaluation for binding predictions
4. **Molecular Validity Metric** - Chemical plausibility assessment
5. **Drug-likeness Metric** - QED and Lipinski compliance
6. **Benchmark Suite** - Comprehensive evaluation and model comparison

## Features Demonstrated

- ✅ Molecular modality with chemical constraints
- ✅ CrossDocked2020 dataset with pocket extraction
- ✅ Binding affinity prediction (RMSE, correlation)
- ✅ Molecular validity checks (bonds, clashes, connectivity)
- ✅ Drug-likeness evaluation (QED, Lipinski)
- ✅ Complete benchmark suite execution
- ✅ Model comparison across quality levels
- ✅ Performance target assessment

## Experiments to Try

1. **Adjust Model Quality**

   ```python
   model = ExampleProteinLigandModel(rngs)
   model.model_quality = "excellent"  # Try "poor", "good", or "excellent"
   results = suite.run_all(model)
   ```

2. **Increase Dataset Size**

   ```python
   dataset_config = {
       "num_samples": 100,    # More samples
       "max_protein_atoms": 300,
       "max_ligand_atoms": 40,
   }
   ```

3. **Custom Pocket Radius**

   ```python
   dataset_config = DataConfig(
       name="custom_pocket",
       dataset_name="crossdocked",
       metadata={
           "pocket_radius": 10.0,  # Larger binding pocket
           "num_samples": 50,
       },
   )
   dataset = CrossDockedDataset(
       data_path="./data/crossdocked",
       config=dataset_config,
       rngs=rngs,
   )
   ```

4. **Add Custom Metrics**

   ```python
   class CustomMetric(nnx.Module):
       def compute(self, predictions, targets):
           # Your custom evaluation logic
           return {"custom_score": score}
   ```

## Next Steps

<div class="grid cards" markdown>

- :material-arrow-right: **Molecular Generation**

    ---

    Generate novel drug-like molecules

    [:octicons-arrow-right-24: Molecule Generation](#)

- :material-arrow-right: **Protein Folding**

    ---

    Predict protein structures

    [:octicons-arrow-right-24: Protein Folding Demo](#)

- :material-arrow-right: **Advanced Docking**

    ---

    Learn molecular docking methods

    [:octicons-arrow-right-24: Docking Tutorial](#)

- :material-arrow-right: **Framework Features**

    ---

    Understand modality system

    [:octicons-arrow-right-24: Framework Demo](../framework/framework-features-demo.md)

</div>

## Troubleshooting

### ImportError for Molecular Modality

**Symptom:** Cannot import molecular modality classes

**Solution:** Install molecular extras

```bash
uv sync --extra molecular
```

### Dataset Loading Too Slow

**Symptom:** Long wait times for dataset initialization

**Solution:** Reduce number of samples

```python
dataset_config = {
    "num_samples": 20,  # Smaller dataset for faster loading
}
```

### CUDA Out of Memory

**Symptom:** GPU memory error during evaluation

**Solution:** Reduce batch size

```python
benchmark_config = {
    "batch_size": 2,  # Smaller batches
}
```

### Low Molecular Validity Rates

**Symptom:** Most generated molecules are invalid

**Cause:** Incorrect coordinate scaling or atom types

**Solution:** Check coordinate normalization

```python
# Ensure coordinates are in angstroms
coordinates = coordinates * coordinate_scale

# Use realistic atom types (1-6 for C, N, O, S, P, F)
atom_types = jax.random.randint(key, (batch, num_atoms), 1, 7)
```

## Additional Resources

### Documentation

- [Molecular Modality Guide](../../guides/modalities/molecular.md) - Chemical structure representation
- [Protein-Ligand Benchmarks](../../guides/benchmarks/protein-ligand.md) - Complete benchmarking guide
- [CrossDocked2020 Dataset](../../api/datasets/crossdocked.md) - Dataset API reference

### Related Examples

- [Geometric Benchmark Demo](../geometric/geometric-benchmark-demo.md) - 3D generation
- [Loss Examples](../losses/loss-examples.md) - Loss functions

### Papers and Resources

- **CrossDocked2020**: ["Protein-Ligand Docking and Scoring with Deep Learning" (Francoeur et al., 2020)](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00411)
- **QED**: ["Quantifying the chemical beauty of drugs" (Bickerton et al., 2012)](https://www.nature.com/articles/nchem.1243)
- **Lipinski's Rule**: ["Experimental and computational approaches to estimate solubility" (Lipinski et al., 2001)](https://www.sciencedirect.com/science/article/pii/S0169409X00001290)
- **Autodock Vina**: [Popular molecular docking software](http://vina.scripps.edu/)

### External Tools

- **RDKit**: [Open-source cheminformatics library](https://www.rdkit.org/)
- **Open Babel**: [Chemical toolbox for file conversion](http://openbabel.org/)
- **PyMOL**: [Molecular visualization](https://pymol.org/)
- **Protein Data Bank (PDB)**: [Protein structure database](https://www.rcsb.org/)
