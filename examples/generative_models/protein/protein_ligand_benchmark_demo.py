#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
"""
# Protein-Ligand Co-Design Benchmark Demo

**Level:** Advanced | **Runtime:** ~3-5 minutes (CPU), ~1-2 minutes (GPU)
**Format:** Python + Jupyter

## Overview

This example demonstrates a comprehensive protein-ligand co-design benchmark suite,
showcasing how to evaluate generative models for drug discovery applications.

## Source Code Dependencies

**Validated:** 2025-10-15

This example depends on the following Artifex source files:
- `src/artifex/benchmarks/datasets/crossdocked.py` - CrossDocked2020 dataset
- `src/artifex/benchmarks/metrics/protein_ligand.py` - Protein-ligand metrics
- `src/artifex/benchmarks/suites/protein_ligand_suite.py` - Benchmark suite
- `src/artifex/generative_models/modalities/molecular.py` - Molecular modality

**Validation Status:**
- ✅ All dependencies validated against `memory-bank/guides/flax-nnx-guide.md`
- ✅ No anti-patterns detected (RNG handling, module init, etc.)
- ✅ All tests passing for dependency files

**Note:** This example was validated as part of Week 0 source dependency validation.

## What You'll Learn

By running this example, you will understand:

1. **Molecular Modality Framework** - How to represent and manipulate molecular structures
2. **CrossDocked2020 Dataset** - Accessing protein-ligand binding data for benchmarks
3. **Protein-Ligand Metrics** - Evaluating binding affinity, molecular validity, and drug-likeness
4. **Benchmark Suites** - Running comprehensive evaluations across multiple metrics
5. **Model Comparison** - Systematically comparing different model architectures

## Key Features Demonstrated

- CrossDocked2020 dataset with realistic protein-ligand complexes
- Molecular modality framework for chemical structure representation
- Binding affinity prediction metrics (RMSE target: <1.0 kcal/mol)
- Molecular validity assessment (target: >95% valid structures)
- Drug-likeness evaluation using QED score (target: >0.7)
- Complete benchmark suite execution with multiple model qualities
- Systematic model comparison across performance metrics

## Prerequisites

- Artifex installed (`source activate.sh`)
- Understanding of protein-ligand interactions and drug discovery
- Familiarity with molecular representations and binding affinities
- Basic knowledge of generative models for molecules

## Usage

```bash
source activate.sh
python examples/generative_models/protein/protein_ligand_benchmark_demo.py

# Or run the Jupyter notebook for interactive exploration
jupyter lab examples/generative_models/protein/protein_ligand_benchmark_demo.ipynb
```

## Expected Output

The example will demonstrate:
1. Molecular modality initialization with extensions and adapters
2. CrossDocked2020 dataset loading and statistics
3. Three protein-ligand specific metrics in action
4. Full benchmark suite execution with poor/good/excellent models
5. Comparative analysis showing performance improvements

**Performance Targets:**
- Binding Affinity RMSE: <1.0 kcal/mol
- Molecular Validity Rate: >95%
- QED (Drug-likeness) Score: >0.7

## Estimated Runtime

- CPU: ~3-5 minutes
- GPU: ~1-2 minutes

## Key Concepts

### Protein-Ligand Co-Design

Protein-ligand co-design involves simultaneously optimizing both the protein binding
site and the ligand molecule to achieve strong, specific binding. This is a critical
challenge in computational drug discovery.

### CrossDocked2020 Dataset

A benchmark dataset containing 22.5 million docked protein-ligand pairs from the
CrossDock2020 database, with experimentally determined binding affinities and
3D structures.

### Binding Affinity

Binding affinity quantifies how strongly a ligand binds to a protein target, typically
measured in kcal/mol. Lower (more negative) values indicate stronger binding.

### Molecular Validity

Checks whether generated molecular structures satisfy chemical constraints:
- Valid bond lengths (1.2-2.0 Å for most bonds)
- No atomic clashes (atoms too close together)
- Chemically feasible atom connectivity

### Drug-likeness (QED)

Quantitative Estimate of Drug-likeness (QED) scores molecules based on properties
like molecular weight, lipophilicity, and structural features that correlate with
successful drugs.

## Implementation Details

This demo implements objectives from the generative models benchmark project:
1. Molecular modality framework for chemical representations
2. CrossDocked2020 dataset integration
3. Protein-ligand co-design benchmark with three key metrics

## Further Reading

- **CrossDocked2020 Paper**: "Improving Protein-Ligand Docking with Deep Learning"
- **Artifex Benchmarks**: `docs/user-guide/benchmarks/protein-ligand.md`
- **Molecular Modalities**: `docs/user-guide/modalities/molecular.md`
- **Related Examples**:
  - `protein_folding_demo.py` - Protein structure prediction
  - `geometric_benchmark_demo.py` - Geometric generative models

## Troubleshooting

**Issue:** ImportError for molecular modality
**Solution:** Ensure Artifex is installed with molecular extras: `uv sync --extra molecular`

**Issue:** Dataset loading too slow
**Solution:** Reduce `num_samples` parameter in dataset initialization

**Issue:** CUDA out of memory
**Solution:** Reduce `batch_size` in benchmark configuration

## Author

Artifex Team

## Last Updated

2025-10-15
"""

# %% [markdown]
"""
## Section 1: Imports and Setup

We import all necessary components for the protein-ligand benchmark:
- JAX for numerical operations and automatic differentiation
- Flax NNX for neural network components
- Artifex benchmark suites, datasets, and metrics
- Molecular modality for chemical structure representation
"""

# %%
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.benchmarks.datasets.crossdocked import CrossDockedDataset
from artifex.benchmarks.metrics.protein_ligand import (
    BindingAffinityMetric,
    DrugLikenessMetric,
    MolecularValidityMetric,
)
from artifex.benchmarks.suites.protein_ligand_suite import (
    ProteinLigandBenchmarkSuite,
    ProteinLigandCoDesignBenchmark,
)
from artifex.generative_models.core.configuration import DataConfig, ModalityConfig
from artifex.generative_models.modalities.molecular import MolecularModality


# Add the src directory to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# %% [markdown]
"""
## Section 2: Example Protein-Ligand Model

This mock model simulates a protein-ligand co-design system for demonstration purposes.
In practice, you would replace this with your actual generative model.

**Key Features:**
- Predicts binding affinities based on structural features
- Generates ligands conditioned on protein structure
- Supports different quality levels (poor/good/excellent)
"""


# %%
class ExampleProteinLigandModel:
    """Example protein-ligand co-design model for demonstration.

    This mock model simulates a protein-ligand co-design system with:
    - Binding affinity prediction
    - Ligand generation given protein structure
    - Realistic performance characteristics
    """

    def __init__(self, rngs: nnx.Rngs):
        """Initialize the example model."""
        self.rngs = rngs
        self.model_quality = "good"  # Can be "poor", "good", or "excellent"

    def predict_binding_affinity(
        self,
        protein_coords: jnp.ndarray,
        protein_types: jnp.ndarray,
        ligand_coords: jnp.ndarray,
        ligand_types: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        """Predict binding affinities for protein-ligand complexes.

        Args:
            protein_coords: Protein coordinates (batch_size, num_protein_atoms, 3)
            protein_types: Protein atom types (batch_size, num_protein_atoms)
            ligand_coords: Ligand coordinates (batch_size, num_ligand_atoms, 3)
            ligand_types: Ligand atom types (batch_size, num_ligand_atoms)

        Returns:
            Predicted binding affinities (batch_size,) in kcal/mol
        """
        batch_size = protein_coords.shape[0]

        # Simulate binding affinity prediction based on structural features
        key = jax.random.key(123)
        keys = jax.random.split(key, batch_size)

        predicted_affinities = []

        for i in range(batch_size):
            # Extract molecular features
            protein_center = jnp.mean(protein_coords[i], axis=0)
            ligand_center = jnp.mean(ligand_coords[i], axis=0)

            # Distance between centers (proxy for binding pocket fit)
            center_distance = jnp.linalg.norm(protein_center - ligand_center)

            # Number of atoms (proxy for size complementarity)
            num_protein_atoms = jnp.sum(protein_types[i] > 0)
            num_ligand_atoms = jnp.sum(ligand_types[i] > 0)

            # Size ratio (optimal around 0.05-0.1 for ligand/protein)
            size_ratio = num_ligand_atoms / (num_protein_atoms + 1e-6)

            # Base affinity prediction
            base_affinity = -8.0  # Moderate binding

            # Adjust based on features
            if center_distance < 2.0:  # Close contact is better
                base_affinity -= 1.5
            elif center_distance > 5.0:  # Too far apart
                base_affinity += 2.0

            if 0.05 <= size_ratio <= 0.15:  # Good size ratio
                base_affinity -= 1.0
            elif size_ratio > 0.3:  # Ligand too big
                base_affinity += 1.5

            # Add noise based on model quality
            if self.model_quality == "excellent":
                noise_scale = 0.3
            elif self.model_quality == "good":
                noise_scale = 0.8
            else:  # poor
                noise_scale = 2.0

            noise = jax.random.normal(keys[i]) * noise_scale
            predicted_affinity = base_affinity + noise

            predicted_affinities.append(predicted_affinity)

        return jnp.array(predicted_affinities)

    def generate_ligand(
        self,
        protein_coords: jnp.ndarray,
        protein_types: jnp.ndarray,
        num_ligand_atoms: int = 20,
        **kwargs,
    ) -> dict:
        """Generate ligands for given protein structures.

        Args:
            protein_coords: Protein coordinates (batch_size, num_protein_atoms, 3)
            protein_types: Protein atom types (batch_size, num_protein_atoms)
            num_ligand_atoms: Number of atoms in generated ligands

        Returns:
            Dictionary with generated ligand coordinates and atom types
        """
        batch_size = protein_coords.shape[0]
        key = jax.random.key(456)
        keys = jax.random.split(key, batch_size + 1)

        generated_coords = []
        generated_types = []

        for i in range(batch_size):
            # Find protein center for ligand placement
            protein_center = jnp.mean(protein_coords[i], axis=0)

            # Generate ligand near protein center (binding pocket)
            coord_keys = jax.random.split(keys[i], 2)

            # Place ligand center near protein center with some offset
            ligand_center_offset = jax.random.normal(coord_keys[0], (3,)) * 1.5
            ligand_center = protein_center + ligand_center_offset

            # Generate ligand coordinates around this center
            ligand_coords = (
                jax.random.normal(coord_keys[1], (num_ligand_atoms, 3)) * 1.2
                + ligand_center[None, :]
            )

            # Generate atom types (realistic drug-like distribution)
            type_key = keys[-1]
            type_probs = jnp.array([0.5, 0.2, 0.15, 0.1, 0.05])  # C, N, O, S, P
            ligand_atom_types = jax.random.choice(
                type_key, jnp.arange(1, 6), shape=(num_ligand_atoms,), p=type_probs
            )

            generated_coords.append(ligand_coords)
            generated_types.append(ligand_atom_types)

        return {
            "coordinates": jnp.stack(generated_coords),
            "atom_types": jnp.stack(generated_types),
        }


# %% [markdown]
"""
## Section 3: Molecular Modality Framework Demo

The molecular modality provides domain-specific functionality for working with
chemical structures, including:
- Chemical constraints (bond lengths, angles)
- Pharmacophore features (hydrogen bond donors/acceptors, hydrophobic regions)
- Adapters for different model types (diffusion, geometric, etc.)
"""


# %%
def demonstrate_molecular_modality():
    """Demonstrate the molecular modality framework."""
    print("\n" + "=" * 60)
    print("MOLECULAR MODALITY FRAMEWORK DEMO")
    print("=" * 60)

    rngs = nnx.Rngs(42)

    # Initialize molecular modality
    modality = MolecularModality(rngs=rngs)
    print(f"✅ Molecular modality initialized: {modality.name}")

    # Test extensions
    config = ModalityConfig(
        name="molecular_modality_config",
        modality_name="molecular",
        metadata={
            "use_chemical_constraints": True,
            "bond_length_weight": 1.0,
            "bond_angle_weight": 0.5,
            "use_pharmacophore_features": True,
            "pharmacophore_types": ["donor", "acceptor", "hydrophobic"],
        },
    )

    extensions = modality.get_extensions(config, rngs=rngs)
    print(f"✅ Extensions loaded: {list(extensions.keys())}")

    # Test adapters
    adapters = {
        "diffusion": modality.get_adapter("diffusion"),
        "geometric": modality.get_adapter("geometric"),
        "default": modality.get_adapter("default"),
    }
    print(f"✅ Adapters available: {list(adapters.keys())}")

    print("💡 Molecular modality framework ready for protein-ligand co-design!")


# %% [markdown]
"""
## Section 4: CrossDocked2020 Dataset Demo

The CrossDocked2020 dataset contains protein-ligand complexes with:
- 3D coordinates for protein and ligand atoms
- Atom type information
- Binding affinity measurements
- Pocket extraction capabilities
"""


# %%
def demonstrate_crossdocked_dataset():
    """Demonstrate the CrossDocked2020 dataset."""
    print("\n" + "=" * 60)
    print("CROSSDOCKED2020 DATASET DEMO")
    print("=" * 60)

    rngs = nnx.Rngs(123)

    # Initialize dataset with proper DataConfig
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

    print(f"✅ Dataset initialized with {len(dataset)} samples")

    # Test single sample
    sample = dataset[0]
    print("✅ Sample structure:")
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"  - {key}: {value.shape} {value.dtype}")
        else:
            print(f"  - {key}: {value}")

    # Test batch generation
    batch = dataset.get_batch(batch_size=4, start_idx=10)
    print("✅ Batch generation:")
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"  - {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  - {key}: list of {len(value)} items")

    # Test pocket extraction
    sample = dataset[5]
    pocket_coords, pocket_indices = dataset.extract_pocket(
        sample["protein_coords"], sample["ligand_coords"], radius=6.0
    )
    print(
        f"✅ Pocket extraction: {len(pocket_coords)}/{len(sample['protein_coords'])} "
        "atoms in pocket"
    )

    # Dataset statistics
    stats = dataset.get_statistics()
    print("✅ Dataset statistics:")
    print(
        f"  - Protein atoms: {stats['protein_atoms']['mean']:.1f} ± "
        f"{stats['protein_atoms']['std']:.1f} "
        f"(min: {stats['protein_atoms']['min']:.1f}, max: {stats['protein_atoms']['max']:.1f})"
    )
    print(
        f"  - Ligand atoms: {stats['ligand_atoms']['mean']:.1f} ± "
        f"{stats['ligand_atoms']['std']:.1f} "
        f"(min: {stats['ligand_atoms']['min']:.1f}, max: {stats['ligand_atoms']['max']:.1f})"
    )
    print(
        f"  - Binding affinity: {stats['binding_affinity']['mean']:.1f} ± "
        f"{stats['binding_affinity']['std']:.1f} kcal/mol "
        f"(min: {stats['binding_affinity']['min']:.1f}, "
        f"max: {stats['binding_affinity']['max']:.1f})"
    )


# %% [markdown]
"""
## Section 5: Protein-Ligand Metrics Demo

Three specialized metrics evaluate different aspects of protein-ligand modeling:

1. **Binding Affinity Metric**: RMSE between predicted and actual binding energies
   - Target: <1.0 kcal/mol RMSE
   - Also reports Pearson correlation coefficient

2. **Molecular Validity Metric**: Chemical plausibility of generated structures
   - Target: >95% valid molecules
   - Checks bond lengths, angles, and atomic clashes

3. **Drug-likeness Metric**: QED score and Lipinski's Rule of Five
   - Target: QED >0.7
   - Evaluates molecular weight, lipophilicity, etc.
"""


# %%
def demonstrate_protein_ligand_metrics():
    """Demonstrate protein-ligand specific metrics."""
    print("\n" + "=" * 60)
    print("PROTEIN-LIGAND METRICS DEMO")
    print("=" * 60)

    rngs = nnx.Rngs(456)

    # 1. Binding Affinity Metric
    print("\n1. Binding Affinity Metric (RMSE target: <1.0 kcal/mol)")
    affinity_metric = BindingAffinityMetric(rngs=rngs)

    # Simulate good predictions (should pass target)
    true_affinities = jnp.array([-8.2, -6.5, -9.1, -7.8, -5.9])
    good_predictions = true_affinities + jax.random.normal(jax.random.key(1), (5,)) * 0.4

    good_results = affinity_metric.compute(good_predictions, true_affinities)
    print(
        f"   Good model RMSE: {good_results['rmse']:.3f} kcal/mol "
        f"({'✅ PASS' if good_results['rmse'] < 1.0 else '❌ FAIL'})"
    )
    print(f"   Correlation: {good_results['pearson_r']:.3f}")

    # Simulate poor predictions (should fail target)
    poor_predictions = true_affinities + jax.random.normal(jax.random.key(2), (5,)) * 1.8
    poor_results = affinity_metric.compute(poor_predictions, true_affinities)
    print(
        f"   Poor model RMSE: {poor_results['rmse']:.3f} kcal/mol "
        f"({'✅ PASS' if poor_results['rmse'] < 1.0 else '❌ FAIL'})"
    )

    # 2. Molecular Validity Metric
    print("\n2. Molecular Validity Metric (target: >95%)")
    validity_metric = MolecularValidityMetric(rngs=rngs)

    # Generate reasonable molecular structures
    batch_size = 8
    num_atoms = 20
    coordinates = jax.random.normal(rngs.default(), (batch_size, num_atoms, 3)) * 1.5
    atom_types = jax.random.randint(rngs.default(), (batch_size, num_atoms), 1, 6)
    masks = jnp.ones((batch_size, num_atoms), dtype=jnp.bool_)

    validity_results = validity_metric.compute(coordinates, atom_types, masks)
    print(
        f"   Validity rate: {validity_results['validity_rate']:.3f} "
        f"({'✅ PASS' if validity_results['validity_rate'] > 0.95 else '❌ FAIL'})"
    )
    print(f"   Bond validity: {validity_results['bond_validity']:.3f}")
    print(f"   Clash-free: {validity_results['clash_free']:.3f}")

    # 3. Drug-likeness Metric
    print("\n3. Drug-likeness Metric (QED target: >0.7)")
    drug_metric = DrugLikenessMetric(rngs=rngs)

    # Generate drug-like molecules (moderate size)
    drug_coords = jax.random.normal(rngs.default(), (4, 25, 3)) * 2.0
    drug_types = jax.random.randint(rngs.default(), (4, 25), 1, 6)
    drug_masks = jnp.ones((4, 25), dtype=jnp.bool_)

    drug_results = drug_metric.compute(drug_coords, drug_types, drug_masks)
    print(
        f"   QED score: {drug_results['qed_score']:.3f} "
        f"({'✅ PASS' if drug_results['qed_score'] > 0.7 else '❌ FAIL'})"
    )
    print(f"   Lipinski compliance: {drug_results['lipinski_compliance']:.3f}")
    print(f"   Molecular weight: {drug_results['molecular_weight']:.1f} Da")


# %% [markdown]
"""
## Section 6: Complete Benchmark Suite Demo

The benchmark suite orchestrates comprehensive evaluation across all metrics.
This demo tests three model qualities (poor/good/excellent) to show how
performance varies across the target metrics.
"""


# %%
def demonstrate_benchmark_suite():
    """Demonstrate the complete protein-ligand benchmark suite."""
    print("\n" + "=" * 60)
    print("PROTEIN-LIGAND CO-DESIGN BENCHMARK SUITE")
    print("=" * 60)

    rngs = nnx.Rngs(789)

    # Initialize benchmark suite
    suite = ProteinLigandBenchmarkSuite(
        dataset_config={
            "num_samples": 30,  # Small for demo
            "max_protein_atoms": 150,
            "max_ligand_atoms": 25,
        },
        benchmark_config={
            "num_samples": 16,  # Even smaller for quick demo
            "batch_size": 4,
        },
        rngs=rngs,
    )

    print(f"✅ Benchmark suite initialized: {suite.name}")
    print(f"✅ Number of benchmarks: {len(suite.benchmarks)}")
    print(f"✅ Dataset size: {len(suite.dataset)} samples")

    # Test with different model qualities
    model_qualities = ["poor", "good", "excellent"]

    for quality in model_qualities:
        print(f"\n--- Testing {quality.upper()} Model ---")

        model = ExampleProteinLigandModel(rngs)
        model.model_quality = quality

        start_time = time.time()
        results = suite.run_all(model)
        end_time = time.time()

        print(f"⏱️  Benchmark completed in {end_time - start_time:.2f} seconds")

        # Extract key metrics
        for benchmark_name, result in results.items():
            metrics = result.metrics

            rmse = metrics.get("binding_affinity_rmse", 0.0)
            validity = metrics.get("molecular_validity_rate", 0.0)
            qed = metrics.get("qed_score", 0.0)

            print(f"📊 Results for {quality} model:")
            print(f"   Binding Affinity RMSE: {rmse:.3f} kcal/mol ({'✅' if rmse < 1.0 else '❌'})")
            print(f"   Molecular Validity: {validity:.3f} ({'✅' if validity > 0.95 else '❌'})")
            print(f"   QED Score: {qed:.3f} ({'✅' if qed > 0.7 else '❌'})")

            # Overall assessment
            targets_met = sum([rmse < 1.0, validity > 0.95, qed > 0.7])

            print(f"   Overall: {targets_met}/3 targets met")


# %% [markdown]
"""
## Section 7: Model Comparison Demo

This section demonstrates how to systematically compare multiple model architectures
or configurations using the benchmark suite. The comparison table shows clear
differences between baseline, improved, and state-of-the-art models.
"""


# %%
def demonstrate_model_comparison():
    """Demonstrate comparing multiple models."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON DEMO")
    print("=" * 60)

    rngs = nnx.Rngs(999)

    # Create a smaller benchmark for quick comparison
    dataset_config = DataConfig(
        name="crossdocked_comparison",
        dataset_name="crossdocked",
        metadata={
            "num_samples": 20,
            "max_protein_atoms": 100,
            "max_ligand_atoms": 20,
        },
    )
    dataset = CrossDockedDataset(
        data_path="./data/crossdocked",
        config=dataset_config,
        rngs=rngs,
    )

    benchmark = ProteinLigandCoDesignBenchmark(
        dataset=dataset, num_samples=8, batch_size=4, rngs=rngs
    )

    # Test multiple model configurations
    model_configs = [
        ("Baseline Model", "poor"),
        ("Improved Model", "good"),
        ("SOTA Model", "excellent"),
    ]

    comparison_results = {}

    for model_name, quality in model_configs:
        model = ExampleProteinLigandModel(rngs)
        model.model_quality = quality

        result = benchmark.run(model)
        comparison_results[model_name] = result.metrics

    # Print comparison table
    print("\n📊 Model Comparison Results:")
    print(f"{'Model':<15} {'RMSE':<8} {'Validity':<10} {'QED':<8} {'Status':<10}")
    print("-" * 55)

    for model_name, metrics in comparison_results.items():
        rmse = metrics.get("binding_affinity_rmse", 0.0)
        validity = metrics.get("molecular_validity_rate", 0.0)
        qed = metrics.get("qed_score", 0.0)

        # Check if all targets are met
        all_pass = rmse < 1.0 and validity > 0.95 and qed > 0.7
        status = "✅ PASS" if all_pass else "❌ FAIL"

        print(f"{model_name:<15} {rmse:<8.3f} {validity:<10.3f} {qed:<8.3f} {status:<10}")

    print("\nTargets: RMSE <1.0 kcal/mol, Validity >95%, QED >0.7")


# %% [markdown]
"""
## Section 8: Main Execution

This section orchestrates the complete demonstration, running all components
in sequence and providing a summary of the implementation.
"""


# %%
def main():
    """Run the complete demonstration."""
    print("🧬 PROTEIN-LIGAND CO-DESIGN BENCHMARK DEMO")
    print("=" * 80)
    print("This demonstration showcases the implementation of:")
    print("• Molecular Modality Framework")
    print("• CrossDocked2020 Dataset Implementation")
    print("• Protein-Ligand Co-Design Benchmark Suite")
    print("=" * 80)

    try:
        # Molecular Modality Framework
        demonstrate_molecular_modality()

        # CrossDocked2020 Dataset
        demonstrate_crossdocked_dataset()

        # Metrics demonstration
        demonstrate_protein_ligand_metrics()

        # Complete benchmark suite
        demonstrate_benchmark_suite()

        # Model comparison
        demonstrate_model_comparison()

        print("\n" + "=" * 80)
        print("🎉 IMPLEMENTATION COMPLETE!")
        print("=" * 80)
        print("✅ Molecular modality framework operational")
        print("✅ CrossDocked2020 dataset ready")
        print("✅ Protein-ligand metrics implemented")
        print("✅ Comprehensive benchmark suite functional")
        print("\n🎯 Ready for protein-ligand co-design benchmarks!")
        print("📊 Target metrics: RMSE <1.0 kcal/mol, >95% validity, QED >0.7")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

# %% [markdown]
"""
## Summary and Key Takeaways

### What You Learned

- ✅ **Molecular Modality Framework**: Representing chemical structures with
  domain-specific extensions
- ✅ **CrossDocked2020 Dataset**: Accessing realistic protein-ligand binding data
- ✅ **Binding Affinity Prediction**: Evaluating model accuracy with RMSE metrics
- ✅ **Molecular Validity**: Ensuring generated molecules satisfy chemical constraints
- ✅ **Drug-likeness**: Quantifying pharmaceutical potential with QED scores
- ✅ **Benchmark Suites**: Running comprehensive evaluations systematically
- ✅ **Model Comparison**: Identifying performance improvements across architectures

### Key Performance Targets

- **Binding Affinity RMSE**: <1.0 kcal/mol (excellent models achieve ~0.3-0.5)
- **Molecular Validity**: >95% (excellent models achieve >98%)
- **QED Score**: >0.7 (excellent models achieve >0.8)

### Experiments to Try

1. **Adjust Model Quality**: Modify the `model_quality` parameter to see how it affects metrics
2. **Dataset Size**: Increase `num_samples` to test scalability
3. **Batch Size**: Experiment with different `batch_size` values for performance tuning
4. **Custom Metrics**: Add your own protein-ligand specific metrics to the suite
5. **Real Models**: Replace the mock model with actual generative architectures

### Next Steps

- **Advanced Protein Modeling**: See `protein_folding_demo.py` for structure prediction
- **Geometric Generative Models**: Explore `geometric_benchmark_demo.py` for 3D generation
- **Custom Benchmarks**: Create domain-specific benchmark suites for your use case
- **Integration**: Combine protein-ligand benchmarks with full training pipelines

### Additional Resources

- **Papers**:
  - "CrossDocked2020: A Dataset for Protein-Ligand Structure Prediction"
  - "Quantifying the chemical beauty of drugs" (QED paper)
  - "Lipinski's Rule of Five"
- **Documentation**:
  - Artifex Benchmarks: `docs/user-guide/benchmarks/`
  - Molecular Modalities: `docs/user-guide/modalities/molecular.md`
- **Related Examples**:
  - `protein_sequence_generation.py`
  - `molecule_generation.py`
  - `geometric_models_demo.py`

### Troubleshooting Common Issues

**Problem:** Slow dataset loading
**Solution:** Reduce `num_samples` or `max_protein_atoms` parameters

**Problem:** CUDA out of memory
**Solution:** Decrease `batch_size` in benchmark configuration

**Problem:** Low molecular validity rates
**Solution:** Check coordinate scaling and atom type distributions

**Problem:** Poor binding affinity predictions
**Solution:** Ensure ligand placement near protein center (binding pocket)

---

**Congratulations!** You've completed the protein-ligand co-design benchmark demonstration.
You now understand how to evaluate generative models for computational drug discovery using
Artifex's comprehensive benchmarking framework.
"""
