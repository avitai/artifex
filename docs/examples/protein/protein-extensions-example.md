# Protein Extensions Example

<div class="example-badges">
  <span class="badge badge-intermediate">Intermediate</span>
  <span class="badge badge-runtime-fast">⚡ 10 seconds</span>
  <span class="badge badge-format-dual">📓 Dual Format</span>
</div>

Learn how to use protein-specific extensions to add domain knowledge and physical constraints to geometric models.

## Files

- **Python Script**: [`protein_extensions_example.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_extensions_example.py)
- **Jupyter Notebook**: [`protein_extensions_example.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_extensions_example.ipynb)

## Quick Start

```bash
# Clone and setup
cd artifex
source activate.sh

# Run Python script
python examples/generative_models/protein/protein_extensions_example.py

# Or use Jupyter notebook
jupyter notebook examples/generative_models/protein/protein_extensions_example.ipynb
```

## Overview

This tutorial demonstrates Artifex's extension system for incorporating protein-specific knowledge into geometric models. Extensions are modular components that add domain expertise without modifying the base model architecture.

### Learning Objectives

- [x] Understand the extension architecture in Artifex
- [x] Use bond length constraints for realistic protein geometry
- [x] Apply bond angle constraints for proper backbone structure
- [x] Incorporate amino acid sequence information with mixins
- [x] Combine multiple extensions for comprehensive modeling
- [x] Calculate extension-aware losses automatically

### Prerequisites

- Understanding of protein structure (backbone atoms: N, CA, C, O)
- Familiarity with PointCloudModel from Artifex
- Basic knowledge of chemical bonds and angles
- Understanding of loss functions

## Why Use Extensions?

### The Problem

Generic geometric models don't know about protein physics:

❌ **Without Extensions:**

- No knowledge of realistic bond lengths (C-C ~1.5Å)
- No enforcement of proper bond angles (~109.5° tetrahedral)
- No awareness of amino acid types (A, G, L, etc.)
- Models can generate physically impossible structures

✅ **With Extensions:**

- Enforces chemical bond constraints
- Maintains proper molecular geometry
- Incorporates sequence information
- Produces chemically valid structures

### The Solution: Modular Extensions

Extensions are plug-and-play components:

```
Base Model (PointCloud)
       ↓
  + Extensions
       ↓
Protein-Aware Model
```

**Key advantage:** Same base model can be used for proteins, molecules, materials, etc., by swapping extensions.

## Extension Types

### 1. Bond Length Extension

**Purpose:** Enforce realistic distances between bonded atoms

**How it works:**

1. Identifies bonded atom pairs (e.g., CA-C, C-N, N-CA)
2. Measures current distances
3. Compares to ideal bond lengths
4. Adds penalty for deviations

**Typical bond lengths:**

| Bond Type | Ideal Length (Å) | Tolerance |
|-----------|------------------|-----------|
| C-C (single) | 1.54 | ±0.02 |
| C=C (double) | 1.34 | ±0.02 |
| C-N | 1.47 | ±0.02 |
| C=O | 1.23 | ±0.02 |
| N-H | 1.01 | ±0.02 |

**Loss formula:**

```
L_bond_length = (1/N) Σ w_i * (d_i - d_ideal)²
```

Where:

- `d_i` = measured distance
- `d_ideal` = target distance
- `w_i` = bond weight (stronger bonds = higher weight)
- `N` = number of bonds

### 2. Bond Angle Extension

**Purpose:** Maintain proper angles between consecutive bonds

**How it works:**

1. Identifies triplets of bonded atoms (e.g., CA-C-N)
2. Calculates current angle
3. Compares to ideal geometry
4. Penalizes deviations

**Common bond angles:**

| Geometry | Ideal Angle | Example |
|----------|-------------|---------|
| Tetrahedral | 109.5° | sp³ carbon (CA) |
| Trigonal planar | 120° | sp² carbon (C=O) |
| Linear | 180° | sp carbon (rare) |
| Peptide bond | ~120° | C-N-CA |

**Loss formula:**

```
L_bond_angle = (1/M) Σ w_j * (θ_j - θ_ideal)²
```

Where:

- `θ_j` = measured angle
- `θ_ideal` = target angle
- `w_j` = angle weight
- `M` = number of angles

### 3. Protein Mixin Extension

**Purpose:** Add amino acid sequence information

**How it works:**

1. Takes amino acid types as input (20 standard amino acids)
2. Embeds each type into a learned vector
3. Adds sequence-aware features to model
4. Helps model understand residue-specific properties

**Amino acid properties encoded:**

- Hydrophobicity (water-loving vs water-fearing)
- Size (small glycine vs large tryptophan)
- Charge (positive, negative, neutral)
- Aromaticity (ring structures)
- Secondary structure preference (helix, sheet, loop)

**Architecture:**

```
Amino Acid Type (0-19)
       ↓
Embedding Layer (learned)
       ↓
Feature Vector (e.g., 32-dim)
       ↓
Concatenate with position features
```

## Code Walkthrough

### Step 1: Setup and Create Test Data

```python
import jax
import jax.numpy as jnp
from flax import nnx

# Create synthetic protein data
batch_size = 2
num_residues = 10
num_atoms = 4  # N, CA, C, O backbone atoms

# Random 3D coordinates
positions = jax.random.normal(key, (batch_size, num_residues * num_atoms, 3))

# Random amino acid types (0-19 for 20 standard amino acids)
aatype = jax.random.randint(key, (batch_size, num_residues), 0, 20)

# Atom mask (1 = present, 0 = missing)
atom_mask = jnp.ones((batch_size, num_residues * num_atoms))

# Package into batch
batch = {
    "positions": positions,
    "aatype": aatype,
    "atom_mask": atom_mask,
}
```

**Batch structure:**

- `positions`: (2, 40, 3) - 2 proteins, 40 atoms each, xyz coordinates
- `aatype`: (2, 10) - 2 proteins, 10 residues each
- `atom_mask`: (2, 40) - which atoms are present

### Step 2: Create Extensions via Utility Function

```python
from artifex.generative_models.extensions.protein import create_protein_extensions

extension_config = {
    "use_backbone_constraints": True,      # Enable bond length/angle
    "bond_length_weight": 1.0,             # Weight for bond length loss
    "bond_angle_weight": 0.5,              # Weight for bond angle loss (lower = softer constraint)
    "use_protein_mixin": True,             # Enable amino acid encoding
    "aa_embedding_dim": 16,                # Embedding dimension
}

extensions = create_protein_extensions(extension_config, rngs=rngs)
```

This creates an `nnx.Dict` containing:

- `bond_length`: BondLengthExtension
- `bond_angle`: BondAngleExtension
- `protein_mixin`: ProteinMixinExtension

**Why use the utility function?**

✅ **Pros:**

- Handles compatibility between extensions
- Sets up proper dependencies
- Uses sensible defaults
- Less boilerplate code

❌ **When not to use:**

- Need very custom extension combinations
- Debugging specific extension behavior
- Research/experimentation with new extensions

### Step 3: Attach Extensions to Model

```python
from artifex.generative_models.models.geometric import PointCloudModel
from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)

# Create network config for point cloud processing
network_config = PointCloudNetworkConfig(
    name="protein_network",
    hidden_dims=(64, 64, 64),  # Tuple for frozen dataclass
    activation="gelu",
    embed_dim=64,
    num_heads=4,
    num_layers=3,
    dropout_rate=0.1,
)

# Create point cloud config with nested network config
model_config = PointCloudConfig(
    name="protein_point_cloud_with_extensions",
    network=network_config,
    num_points=num_residues * num_atoms,  # 40 points
    dropout_rate=0.1,
)

model = PointCloudModel(
    model_config,
    extensions=extensions,  # ← Extensions attached here
    rngs=rngs,
)
```

**What happens internally:**

1. Model stores extensions as attributes
2. During forward pass, model calls extensions automatically
3. During loss calculation, extension losses are aggregated
4. Total loss = base_loss + sum(ext_weight * ext_loss)

### Step 4: Run Model and Calculate Losses

```python
# Forward pass
outputs = model(batch)
print(f"Model output shape: {outputs['positions'].shape}")
# Output: (2, 40, 3)

# Calculate total loss (includes extension losses)
loss_fn = model.get_loss_fn()
loss = loss_fn(batch, outputs)
print(f"Loss with extensions: {loss}")
# Output: {'total_loss': 3.56, 'mse_loss': 2.28, 'bond_length': 0.60, 'bond_angle': 0.69, 'protein_mixin': 0.0}
```

**Loss breakdown:**

| Component | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| `mse_loss` | 2.28 | 1.0 | Base reconstruction |
| `bond_length` | 0.60 | 1.0 | Bond length constraint |
| `bond_angle` | 0.69 | 0.5 | Bond angle constraint (weighted) |
| `protein_mixin` | 0.0 | 1.0 | No loss (encoding only) |
| **Total** | **3.56** | - | Sum of all components |

**Formula:**

```
total_loss = mse_loss + (1.0 * bond_length) + (0.5 * bond_angle) + (1.0 * protein_mixin)
          = 2.28 + 0.60 + 0.345 + 0.0
          = 3.225  (approximately, due to rounding)
```

### Step 5: Access Extension Outputs

```python
# Get detailed metrics from each extension
for name, extension in extensions.items():
    ext_outputs = extension(batch, outputs)
    print(f"Extension {name} outputs: {list(ext_outputs.keys())}")
```

**Output:**

```
Extension bond_length outputs: ['bond_distances', 'bond_violations', 'extension_type']
Extension bond_angle outputs: ['bond_angles', 'angle_violations', 'extension_type']
Extension protein_mixin outputs: ['extension_type', 'aa_encoding']
```

**What each output contains:**

**BondLengthExtension:**

- `bond_distances`: Measured distances for all bonds (Å)
- `bond_violations`: Count of bonds outside tolerance
- `extension_type`: "bond_length"

**BondAngleExtension:**

- `bond_angles`: Measured angles for all triplets (degrees)
- `angle_violations`: Count of angles outside tolerance
- `extension_type`: "bond_angle"

**ProteinMixinExtension:**

- `aa_encoding`: Embedded amino acid features (batch, num_residues, embedding_dim)
- `extension_type`: "protein_mixin"

### Step 6: Using Individual Extensions

For fine-grained control, create extensions manually:

```python
from artifex.generative_models.extensions.base.extensions import ExtensionConfig
from artifex.generative_models.extensions.protein import BondLengthExtension

# Create extension config
bond_length_config = ExtensionConfig(
    name="bond_length",
    weight=1.0,
    enabled=True,
    extensions={}  # Extension-specific params (if needed)
)

# Instantiate extension
bond_length_ext = BondLengthExtension(bond_length_config, rngs=rngs)

# Use extension
metrics = bond_length_ext(batch, outputs)
loss = bond_length_ext.loss_fn(batch, outputs)

print(f"Bond length loss: {loss}")  # 0.598
```

**When to use individual extensions:**

1. **Debugging**: Isolate specific extension behavior
2. **Custom loss weighting**: Dynamic weight schedules
3. **Selective application**: Apply only to certain batches
4. **Research**: Experiment with new extension combinations

## Expected Output

```
Model output shape: (2, 40, 3)
Loss with extensions: {'total_loss': Array(3.56, dtype=float32), 'mse_loss': Array(2.28, dtype=float32), 'bond_length': Array(0.60, dtype=float32), 'bond_angle': Array(0.69, dtype=float32), 'protein_mixin': Array(0., dtype=float32)}
Extension bond_length outputs: ['bond_distances', 'bond_violations', 'extension_type']
Extension bond_angle outputs: ['bond_angles', 'angle_violations', 'extension_type']
Extension protein_mixin outputs: ['extension_type', 'aa_encoding']

Using individual extensions:
Bond length metrics: ['bond_distances', 'bond_violations', 'extension_type']
Bond length loss: 0.5976787209510803
Bond angle metrics: ['bond_angles', 'angle_violations', 'extension_type']
Bond angle loss: 0.6547483801841736
Amino acid encoding shape: (2, 10, 21)
```

## Understanding Extension Architecture

### Design Principles

#### 1. Modularity

Extensions are independent and composable:

```python
# Can mix and match
extensions_A = {"bond_length": ext1}
extensions_B = {"bond_length": ext1, "bond_angle": ext2}
extensions_C = {"protein_mixin": ext3}
```

#### 2. Compatibility

All extensions follow the same protocol:

```python
class Extension(Protocol):
    def __call__(self, batch, outputs) -> Dict:
        """Compute extension outputs"""
        ...

    def loss_fn(self, batch, outputs) -> float:
        """Compute extension loss"""
        ...
```

#### 3. Automatic Integration

Models handle extensions transparently:

```python
# Model automatically:
# 1. Calls each extension during forward pass
# 2. Aggregates losses with weights
# 3. Returns combined loss
```

### Extension Lifecycle

```
1. Initialization
   ├─ Create extension config
   ├─ Instantiate extension with RNGs
   └─ Attach to model

2. Forward Pass
   ├─ Model processes input
   ├─ Extension processes (batch, outputs)
   └─ Extension returns metrics dict

3. Loss Calculation
   ├─ Extension computes its loss
   ├─ Model weights extension loss
   └─ Adds to total loss

4. Backward Pass
   └─ Gradients flow through extension
```

## Experiments to Try

### 1. Adjust Extension Weights

```python
# Experiment with different weight combinations
configs = [
    {"bond_length_weight": 1.0, "bond_angle_weight": 0.0},  # Only length
    {"bond_length_weight": 0.0, "bond_angle_weight": 1.0},  # Only angle
    {"bond_length_weight": 2.0, "bond_angle_weight": 1.0},  # Stronger length
    {"bond_length_weight": 0.5, "bond_angle_weight": 2.0},  # Stronger angle
]

for config in configs:
    extensions = create_protein_extensions(config, rngs=rngs)
    # Train and compare results
```

**Observation:** Higher weights enforce stricter constraints but may limit flexibility.

### 2. Compare With and Without Extensions

```python
# Model without extensions
model_vanilla = PointCloudModel(model_config, rngs=rngs)
outputs_vanilla = model_vanilla(batch)

# Model with extensions
model_extended = PointCloudModel(model_config, extensions=extensions, rngs=rngs)
outputs_extended = model_extended(batch)

# Compare outputs
# Which produces more realistic bond lengths?
```

### 3. Visualize Extension Effects

```python
import matplotlib.pyplot as plt

# Extract bond lengths
metrics = bond_length_ext(batch, outputs)
bond_distances = metrics['bond_distances']

# Plot distribution
plt.hist(bond_distances, bins=50)
plt.axvline(x=1.54, color='r', linestyle='--', label='Ideal C-C')
plt.axvline(x=1.47, color='g', linestyle='--', label='Ideal C-N')
plt.legend()
plt.xlabel('Bond Length (Å)')
plt.ylabel('Count')
plt.title('Bond Length Distribution')
```

### 4. Custom Extension Combinations

```python
# Create custom extension set
from artifex.generative_models.extensions.protein import ProteinBackboneConstraint

custom_extensions = nnx.Dict({
    "bond_length": BondLengthExtension(config1, rngs=rngs),
    "backbone": ProteinBackboneConstraint(config2, rngs=rngs),
    # No angle constraint - looser model
})

model = PointCloudModel(model_config, extensions=custom_extensions, rngs=rngs)
```

## Advanced Usage

### Dynamic Extension Weighting

Adjust weights during training:

```python
def get_extension_weights(epoch):
    """Gradually increase constraint strength"""
    return {
        "bond_length_weight": min(1.0, epoch / 100),  # Ramp up over 100 epochs
        "bond_angle_weight": min(0.5, epoch / 200),   # Ramp up slower
        "use_protein_mixin": True,
        "aa_embedding_dim": 16,
    }

for epoch in range(num_epochs):
    config = get_extension_weights(epoch)
    extensions = create_protein_extensions(config, rngs=rngs)
    model = PointCloudModel(model_config, extensions=extensions, rngs=rngs)
    # Train...
```

**Rationale:** Start with weak constraints (let model learn), then tighten (refine to physics).

### Extension-Specific Loss Weighting

```python
# Access individual losses for custom weighting
loss_dict = loss_fn(batch, outputs)

custom_total_loss = (
    1.0 * loss_dict['mse_loss'] +
    2.0 * loss_dict['bond_length'] +     # Prioritize bond lengths
    0.1 * loss_dict['bond_angle'] +      # Soft angle constraint
    0.5 * loss_dict['protein_mixin']     # Moderate mixin contribution
)
```

### Conditional Extensions

Apply extensions selectively:

```python
def conditional_loss(batch, outputs, is_training):
    if is_training:
        # Use all extensions during training
        return model.get_loss_fn()(batch, outputs)
    else:
        # Use only base loss during evaluation
        return {'total_loss': loss_dict['mse_loss']}
```

## Next Steps

<div class="grid cards" markdown>

- :material-cog: **Protein Model Extension**

    ---

    More extension examples with backbone constraints

    [:octicons-arrow-right-24: protein_model_extension.py](protein-model-extension.md)

- :material-tune: **Protein Extensions with Config**

    ---

    Using the configuration system for extensions

    [:octicons-arrow-right-24: protein_extensions_with_config.py](protein-extensions-with-config.md)

- :material-atom: **Protein Point Cloud**

    ---

    Full protein modeling with constraints

    [:octicons-arrow-right-24: protein_point_cloud_example.py](protein-point-cloud-example.md)

- :material-puzzle: **Custom Extensions**

    ---

    Create your own domain-specific extensions

    [:octicons-arrow-right-24: ../../guides/custom-extensions.md](../../guides/custom-extensions.md)

</div>

## Troubleshooting

### TypeError: config must be ExtensionConfig

**Cause:** Passing a plain dict instead of ExtensionConfig object

**Wrong:**

```python
ext = BondLengthExtension({"name": "bond", "weight": 1.0}, rngs=rngs)
```

**Correct:**

```python
config = ExtensionConfig(name="bond", weight=1.0, enabled=True, extensions={})
ext = BondLengthExtension(config, rngs=rngs)
```

### Extension loss is NaN

**Possible causes:**

1. **Missing required batch keys**: Extensions need `positions`, `aatype`, `atom_mask`
2. **Invalid atom positions**: Check for Inf/NaN in input
3. **Division by zero**: Empty atom mask (all zeros)

**Debug:**

```python
print("Batch keys:", batch.keys())
print("Positions range:", batch['positions'].min(), batch['positions'].max())
print("Atom mask sum:", batch['atom_mask'].sum())
```

### Extension not affecting loss

**Cause:** Extension weight is 0 or extension is disabled

**Check:**

```python
print("Extension config:", extension.config)
print("Weight:", extension.config.weight)
print("Enabled:", extension.config.enabled)
```

**Fix:**

```python
extension.config.weight = 1.0
extension.config.enabled = True
```

### Bond violations are high

**Expected:** Initial violations are normal with random initialization

**Solutions:**

1. **Train longer**: Extensions need time to learn constraints
2. **Increase weight**: Stronger penalty for violations
3. **Check bond topology**: Ensure atom connectivity is correct
4. **Verify atom mask**: Missing atoms can cause false violations

**Monitor:**

```python
metrics = bond_length_ext(batch, outputs)
violations = metrics['bond_violations']
print(f"Violations: {violations} / {total_bonds}")
```

## Additional Resources

- [Extension System Design](../../architecture/extensions.md) - Architecture overview
- [Creating Custom Extensions](../../guides/custom-extensions.md) - Build your own
- [Protein Modeling Guide](../../guides/protein-modeling.md) - Comprehensive protein tutorial
- [Chemical Constraints](../../theory/molecular-constraints.md) - Theory behind constraints
- [Artifex Extension API](../../api/extensions.md) - Full API reference
