# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   language_info:
#     name: python
# ---

# %% [markdown]
"""
# Protein Extensions Example

This example demonstrates how to use protein-specific extensions with geometric models
to incorporate domain knowledge and physical constraints.

## Learning Objectives

- Understand protein-specific extensions in Artifex
- Attach extensions to geometric models
- Use bond length and bond angle constraints
- Apply amino acid encoding mixins
- Calculate extension-aware losses

## Prerequisites

- Understanding of protein structure (backbone atoms)
- Familiarity with PointCloudModel
- Basic knowledge of chemical bonds

## What Are Protein Extensions?

Extensions are modular components that add domain-specific functionality:

- **Bond Length Extension**: Enforces realistic bond distances
- **Bond Angle Extension**: Maintains proper bond angles
- **Protein Mixin**: Adds amino acid type information

These extensions can be attached to any geometric model to inject protein physics.
"""

# %%
import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)
from artifex.generative_models.extensions.base.extensions import ExtensionConfig
from artifex.generative_models.extensions.protein import (
    BondAngleExtension,
    BondLengthExtension,
    create_protein_extensions,
    ProteinMixinExtension,
)
from artifex.generative_models.models.geometric import PointCloudModel


# %% [markdown]
"""
## 1. Setup and Create Test Data

We create synthetic protein data with:

- **Positions**: 3D coordinates of backbone atoms (N, CA, C, O)
- **Amino acid types** (aatype): Sequence information
- **Atom mask**: Which atoms are present
"""


# %%
# Set up random keys
key = jax.random.key(42)
key, subkey = jax.random.split(key)
rngs = nnx.Rngs(params=subkey)

# Create a simple test batch
batch_size = 2
num_residues = 10
num_atoms = 4  # N, CA, C, O

# Create random 3D coordinates for protein backbone atoms
positions = jax.random.normal(key, (batch_size, num_residues * num_atoms, 3))

# Create random amino acid types
aatype = jax.random.randint(key, (batch_size, num_residues), 0, 20)

# Create atom mask (all atoms present)
atom_mask = jnp.ones((batch_size, num_residues * num_atoms))

# Create batch dictionary
batch = {
    "positions": positions,
    "aatype": aatype,
    "atom_mask": atom_mask,
}


# %% [markdown]
"""
## 2. Create Extensions with Utility Function

The `create_protein_extensions()` function creates a collection of extensions
based on configuration. This is the recommended approach for most use cases.
"""

# %%
# Create extensions using the utility function
extension_config = {
    "use_backbone_constraints": True,
    "bond_length_weight": 1.0,
    "bond_angle_weight": 0.5,
    "use_protein_mixin": True,
    "aa_embedding_dim": 16,
}

# Create protein extensions (now returns nnx.Dict for Flax NNX 0.12.0+ compatibility)
extensions = create_protein_extensions(extension_config, rngs=rngs)


# %% [markdown]
"""
## 3. Attach Extensions to Model

Extensions are passed to the model during initialization. The model will
automatically use them during forward passes and loss calculations.
"""

# %%
# Create a point cloud model with extensions
# Create network config for point cloud
network_config = PointCloudNetworkConfig(
    name="protein_point_cloud_network",
    hidden_dims=(64, 64, 64),  # Tuple for frozen dataclass
    activation="gelu",
    embed_dim=64,
    num_heads=4,
    num_layers=3,
    dropout_rate=0.1,
)

# Create PointCloudConfig with nested network config
model_config = PointCloudConfig(
    name="protein_point_cloud_with_extensions",
    network=network_config,
    num_points=num_residues * num_atoms,  # 40 points (10 residues * 4 atoms)
    dropout_rate=0.1,
)

model = PointCloudModel(
    model_config,
    extensions=extensions,
    rngs=rngs,
)


# %% [markdown]
"""
## 4. Run Model and Calculate Losses

The model's loss function automatically incorporates extension losses:

**Total Loss = Base Loss + sum(extension_weight * extension_loss)**
"""

# %%
# Run the model
outputs = model(batch)
print(f"Model output shape: {outputs['positions'].shape}")

# Calculate loss with extensions
loss_fn = model.get_loss_fn()
loss = loss_fn(batch, outputs)
print(f"Loss with extensions: {loss}")

# Access extension outputs
for name, extension in extensions.items():
    ext_outputs = extension(batch, outputs)
    print(f"Extension {name} outputs: {list(ext_outputs.keys())}")


# %% [markdown]
"""
## 5. Using Individual Extensions

For fine-grained control, you can create and use extensions individually.
This is useful for debugging or custom loss combinations.
"""

# %%
# Alternative: Create and use extensions individually
print("\nUsing individual extensions:")

# Create bond length extension
bond_length_config = ExtensionConfig(name="bond_length", weight=1.0, enabled=True, extensions={})
bond_length_ext = BondLengthExtension(
    bond_length_config,
    rngs=rngs,
)

# Calculate bond length metrics
bond_metrics = bond_length_ext(batch, outputs)
print(f"Bond length metrics: {list(bond_metrics.keys())}")

# Calculate bond length loss
bond_loss = bond_length_ext.loss_fn(batch, outputs)
print(f"Bond length loss: {bond_loss}")

# Create and use bond angle extension
bond_angle_config = ExtensionConfig(name="bond_angle", weight=0.5, enabled=True, extensions={})
bond_angle_ext = BondAngleExtension(
    bond_angle_config,
    rngs=rngs,
)

# Calculate bond angle metrics
angle_metrics = bond_angle_ext(batch, outputs)
print(f"Bond angle metrics: {list(angle_metrics.keys())}")

# Calculate bond angle loss
angle_loss = bond_angle_ext.loss_fn(batch, outputs)
print(f"Bond angle loss: {angle_loss}")

# Create amino acid encoding extension
aa_config = ExtensionConfig(
    name="protein_mixin", weight=1.0, enabled=True, extensions={"embedding_dim": 32}
)
aa_ext = ProteinMixinExtension(
    aa_config,
    rngs=rngs,
)

# Get amino acid encodings
aa_outputs = aa_ext(batch, outputs)
if "aa_encoding" in aa_outputs:
    print(f"Amino acid encoding shape: {aa_outputs['aa_encoding'].shape}")


# %% [markdown]
"""
## Summary and Key Takeaways

This example demonstrated:

1. **Creating extensions**: Using `create_protein_extensions()` utility
2. **Attaching to models**: Pass extensions during model initialization
3. **Automatic loss integration**: Extensions contribute to total loss
4. **Individual usage**: Create and use extensions separately for control

### Extension Types

**Bond Length Extension:**

- Measures distances between bonded atoms
- Penalizes deviations from ideal bond lengths
- Typical bond lengths: C-C ~1.5A, C-N ~1.3A, C-O ~1.2A

**Bond Angle Extension:**

- Measures angles between three consecutive atoms
- Enforces realistic bond angles
- Typical angles: tetrahedral ~109.5 degrees, planar ~120 degrees

**Protein Mixin:**

- Encodes amino acid type information
- Adds sequence-aware features
- Uses learned embeddings (not one-hot)

### Next Steps

- See `protein_model_extension.py` for more extension examples
- Try `protein_point_cloud_example.py` for full protein modeling
- Explore `protein_extensions_with_config.py` for configuration system
"""


# %%
print("\nProtein extensions example completed!")
