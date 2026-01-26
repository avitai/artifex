# Protein Diffusion Technical Validation

<div class="example-badges">
  <span class="badge badge-beginner">Beginner</span>
  <span class="badge badge-runtime-fast">⚡ 5 seconds</span>
  <span class="badge badge-format-dual">📓 Dual Format</span>
</div>

A minimal validation script to verify your environment is correctly set up for protein diffusion modeling with JAX and Flax NNX.

## Files

- **Python Script**: [`protein_diffusion_tech_validation.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_diffusion_tech_validation.py)
- **Jupyter Notebook**: [`protein_diffusion_tech_validation.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_diffusion_tech_validation.ipynb)

## Quick Start

```bash
# Clone and setup
cd artifex
source activate.sh

# Run validation
python examples/generative_models/protein/protein_diffusion_tech_validation.py

# Expected output:
# JAX version: 0.7.2
# Biopython available: False
# Protein coordinates shape: (100, 3)
# Model output shape: (100, 3)
# Loss: 1.467...
# Technology validation successful!
```

## Overview

This script performs a quick technology stack validation for protein modeling. It's designed to be the first thing you run to ensure your environment is correctly configured.

### Learning Objectives

- [x] Validate JAX and Flax NNX installation
- [x] Understand protein point cloud representation
- [x] Implement a minimal protein structure model
- [x] Test forward pass and loss computation
- [x] Handle optional dependencies gracefully

### Prerequisites

- JAX installed
- Flax NNX installed
- Basic understanding of protein structure (helpful)
- Familiarity with point clouds (helpful)

## What Gets Validated

This script checks 5 critical components:

### 1. JAX Functionality

**Tests:**

- Random number generation (`jax.random.key`)
- Array operations (`jax.numpy`)
- Device placement (CPU/GPU)

**Expected:** JAX 0.7.2+ working correctly

### 2. Flax NNX

**Tests:**

- Module creation (`nnx.Module`)
- Linear layers (`nnx.Linear`)
- Activation functions (`nnx.relu`)
- Parameter initialization

**Expected:** Flax NNX modules instantiate and run

### 3. Protein Representation

**Tests:**

- Point cloud format (N × 3 arrays)
- C-alpha atom extraction (if BioPython available)
- Synthetic data generation (fallback)

**Expected:** Proteins represented as 3D point clouds

### 4. Forward Pass

**Tests:**

- Model inference
- Shape preservation
- Numerical stability

**Expected:** Output shape matches input shape

### 5. Loss Computation

**Tests:**

- MSE calculation
- Gradient flow (implicit)
- JAX autodiff compatibility

**Expected:** Loss value computed successfully

## Code Walkthrough

### Step 1: Import and Check Dependencies

```python
import jax
import jax.numpy as jnp
from flax import nnx

try:
    from Bio.PDB import PDBParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    print("Biopython not installed. Will use synthetic data.")
```

The script gracefully handles missing BioPython by falling back to synthetic data.

### Step 2: Define Simple Protein Model

```python
class SimpleProteinPointCloud(nnx.Module):
    features: int = 32
    hidden_dim: int = 64
    output_dim: int = 3  # 3D coordinates

    def __init__(self, rngs: nnx.Rngs):
        super().__init__()
        self.encoder = nnx.Linear(in_features=3, out_features=self.features, rngs=rngs)
        self.hidden = nnx.Linear(in_features=self.features, out_features=self.hidden_dim, rngs=rngs)
        self.decoder = nnx.Linear(in_features=self.hidden_dim, out_features=self.output_dim, rngs=rngs)

    def __call__(self, points):
        x = self.encoder(points)
        x = nnx.relu(x)
        x = self.hidden(x)
        x = nnx.relu(x)
        x = self.decoder(x)
        return x
```

**Architecture:**

- **Input**: (N, 3) point cloud
- **Encoder**: 3 → 32 dimensions
- **Hidden**: 32 → 64 dimensions
- **Decoder**: 64 → 3 dimensions
- **Output**: (N, 3) transformed point cloud

### Step 3: Generate or Load Protein Data

```python
def create_synthetic_protein_data(n_points=100):
    """Create synthetic protein point cloud data."""
    key = jax.random.key(42)
    points = jax.random.normal(key, (n_points, 3))
    return points
```

Synthetic data is a simple Gaussian distribution in 3D space, simulating protein atom positions.

**Real data** (with BioPython):

```python
def load_protein_from_pdb(pdb_file):
    """Load protein coordinates from a PDB file."""
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)

    # Extract C-alpha atoms
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_atom = residue["CA"]
                    coords.append(ca_atom.get_coord())

    return jnp.array(coords)
```

**C-alpha (CA) atoms** form the protein backbone and provide a coarse-grained representation.

### Step 4: Run Validation

```python
# Create model
rngs = nnx.Rngs(params=jax.random.key(0))
model = SimpleProteinPointCloud(rngs=rngs)

# Generate data
protein_coords = create_synthetic_protein_data(n_points=100)

# Forward pass
output = model(protein_coords)

# Compute loss
loss = jnp.mean((output - protein_coords) ** 2)
```

**Validation checklist:**

✅ Model instantiates without errors

✅ Forward pass completes

✅ Output shape is (100, 3)

✅ Loss is a valid number

## Expected Output

```
Biopython not installed. Will use synthetic data.
JAX version: 0.7.2
Biopython available: False
Protein coordinates shape: (100, 3)
Model output shape: (100, 3)
Loss: 1.4674725532531738
Technology validation successful!
```

**What each line means:**

| Output | Meaning |
|--------|---------|
| `JAX version: 0.7.2` | JAX is installed and version is ≥ 0.7 |
| `Biopython available: False` | Optional dependency status |
| `Protein coordinates shape: (100, 3)` | 100 atoms in 3D space |
| `Model output shape: (100, 3)` | Forward pass preserves shape |
| `Loss: 1.467...` | MSE between input and output |
| `Technology validation successful!` | All checks passed ✓ |

## Understanding Protein Point Clouds

### What are C-alpha (CA) Atoms?

Proteins are chains of amino acids. Each amino acid has:

- **N**: Nitrogen (backbone)
- **C-alpha (CA)**: Central carbon (backbone)
- **C**: Carbonyl carbon (backbone)
- **O**: Oxygen (backbone)
- **Sidechain**: Variable atoms (different for each amino acid)

**C-alpha atoms** trace the protein backbone and are commonly used for:

- Structural alignment
- Coarse-grained modeling
- Fast structure prediction
- Low-resolution analysis

### Point Cloud Representation

```
Protein sequence: M-E-T-H-I-O-N-I-N-E
                  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
CA atoms:        (x₁,y₁,z₁), (x₂,y₂,z₂), ...
```

**Point cloud format:**

```python
coordinates.shape  # (N, 3) where N = number of residues
coordinates[0]     # [x, y, z] for first CA atom
```

## Troubleshooting

### "No module named 'jax'"

**Cause**: JAX not installed

**Solution**:

```bash
uv sync --extra cuda-dev  # For GPU
# or
uv sync  # For CPU only
```

### "No module named 'flax'"

**Cause**: Flax not installed

**Solution**:

```bash
source activate.sh  # Activates environment with Flax
```

### "Biopython not installed" warning

**Cause**: BioPython is optional and not installed

**Impact**: None - synthetic data works fine for validation

**To install (optional)**:

```bash
uv add biopython
```

### Loss value is NaN or Inf

**Possible causes:**

1. **Model initialization issue**: Check RNG keys are valid
2. **Numerical instability**: Add layer normalization
3. **Data issue**: Verify protein_coords contains valid floats

**Debug:**

```python
print("Coords min/max:", protein_coords.min(), protein_coords.max())
print("Output min/max:", output.min(), output.max())
```

### Different loss value than documentation

**This is normal!** The exact loss depends on:

- Random initialization (from RNG seed)
- JAX version
- Hardware (CPU vs GPU)
- Floating point precision

**As long as** the loss is a reasonable number (not NaN/Inf), validation passed.

## Next Steps

<div class="grid cards" markdown>

- :material-atom: **Protein Point Cloud**

    ---

    Full protein structure modeling with constraints

    [:octicons-arrow-right-24: protein_point_cloud_example.py](protein-point-cloud-example.md)

- :material-puzzle: **Protein Extensions**

    ---

    Domain-specific extensions for proteins

    [:octicons-arrow-right-24: protein_extensions_example.py](protein-extensions-example.md)

- :material-cog: **Protein with Modality**

    ---

    Using the modality architecture for proteins

    [:octicons-arrow-right-24: protein_model_with_modality.py](protein-model-with-modality.md)

- :material-flask: **Protein-Ligand Benchmark**

    ---

    Advanced: SE(3) equivariant protein modeling

    [:octicons-arrow-right-24: protein_ligand_benchmark_demo.py](protein-ligand-benchmark-demo.md)

</div>

## Additional Resources

- [JAX Documentation](https://jax.readthedocs.io/) - JAX fundamentals
- [Flax NNX Guide](https://flax.readthedocs.io/en/latest/nnx/index.html) - NNX module system
- [BioPython Tutorial](https://biopython.org/wiki/Documentation) - PDB file parsing
- [Protein Data Bank (PDB)](https://www.rcsb.org/) - Download protein structures
- [AlphaFold](https://alphafold.ebi.ac.uk/) - Protein structure prediction
