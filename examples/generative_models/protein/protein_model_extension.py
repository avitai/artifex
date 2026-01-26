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
# # Protein Model Extensions Example
#
# This example demonstrates how to use protein-specific extensions with Artifex's
# geometric model framework, showing how to combine domain knowledge with
# general-purpose geometric models.
#
# ## Learning Objectives
#
# - Understand protein-specific extensions in Artifex
# - Learn how to combine multiple extensions (mixin, constraints, bond length/angle)
# - See how extensions enhance model outputs with domain knowledge
# - Understand the role of extensions in loss calculation
#
# ## Prerequisites
#
# - Basic understanding of protein structure (residues, backbone atoms)
# - Familiarity with point cloud models
# - Knowledge of Flax NNX modules

# %%
"""Protein model extension example."""

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
    ProteinMixinExtension,
)
from artifex.generative_models.extensions.protein.constraints import (
    ProteinBackboneConstraint,
)
from artifex.generative_models.models.geometric.point_cloud import (
    PointCloudModel,
)


# %% [markdown]
# ## 1. Initialize Random Keys
#
# We'll use JAX's random number generation throughout this example.
# The key will be split as needed to ensure independent randomness.

# %%
# Initialize random keys
key = jax.random.key(42)
key, dropout_key = jax.random.split(key)


# %% [markdown]
# ## 2. Configure Protein Structure
#
# We define a small protein with 10 residues. Each residue has 4 backbone
# atoms (N, CA, C, O), giving us 40 total points.

# %%
# Protein model configuration
num_residues = 10
atoms_per_residue = 4  # N, CA, C, O
num_points = num_residues * atoms_per_residue
embedding_dim = 64

# Point cloud model configuration using frozen dataclass configs
network_config = PointCloudNetworkConfig(
    name="protein_network",
    hidden_dims=(embedding_dim,) * 2,
    activation="gelu",
    embed_dim=embedding_dim,
    num_heads=4,
    num_layers=2,
    dropout_rate=0.1,
)

model_config = PointCloudConfig(
    name="protein_point_cloud",
    network=network_config,
    num_points=num_points,
    dropout_rate=0.1,
)


# %% [markdown]
# ## 3. Create Protein-Specific Extensions
#
# Extensions add domain knowledge to the base model. We'll add four types:
#
# 1. **Protein Mixin**: Integrates amino acid type information
# 2. **Protein Constraints**: Enforces backbone geometry constraints
# 3. **Bond Length Extension**: Monitors bond length violations
# 4. **Bond Angle Extension**: Monitors bond angle violations

# %%
# Create protein-specific extensions
key, mixin_key, constraint_key, backbone_key = jax.random.split(key, 4)
extensions_dict = {}

# Add protein mixin - provides amino acid type integration
mixin_config = ExtensionConfig(
    name="protein_mixin",
    weight=1.0,
    enabled=True,
    extensions={
        "embedding_dim": embedding_dim,
        "num_aa_types": 20,
    },
)
extensions_dict["protein_mixin"] = ProteinMixinExtension(
    config=mixin_config,
    rngs=nnx.Rngs(params=mixin_key),
)

# %%
# Add constraint extension - enforces chemical constraints
constraint_config = ExtensionConfig(
    name="protein_constraints",
    weight=1.0,
    enabled=True,
    extensions={
        "num_residues": num_residues,
        "backbone_indices": [0, 1, 2, 3],  # N, CA, C, O
    },
)
extensions_dict["protein_constraints"] = ProteinBackboneConstraint(
    config=constraint_config,
    rngs=nnx.Rngs(params=constraint_key),
)

# %%
# Add bond length extension
bond_length_config = ExtensionConfig(
    name="bond_length",
    weight=1.0,
    enabled=True,
    extensions={
        "num_residues": num_residues,
        "backbone_indices": [0, 1, 2, 3],
    },
)
extensions_dict["bond_length"] = BondLengthExtension(
    config=bond_length_config,
    rngs=nnx.Rngs(params=constraint_key),
)

# %%
# Add bond angle extension - helps model understand backbone geometry
bond_angle_config = ExtensionConfig(
    name="bond_angle",
    weight=0.5,
    enabled=True,
    extensions={
        "num_residues": num_residues,
        "backbone_indices": [0, 1, 2, 3],
    },
)
extensions_dict["bond_angle"] = BondAngleExtension(
    config=bond_angle_config,
    rngs=nnx.Rngs(params=backbone_key),
)

# Wrap extensions in nnx.Dict for Flax NNX 0.12.0+ compatibility
extensions = nnx.Dict(extensions_dict)
print(f"Created extensions: {', '.join(extensions.keys())}")


# %% [markdown]
# ## 4. Create Point Cloud Model with Extensions
#
# Now we create the point cloud model with all four extensions attached.
# The model will use these extensions during forward passes and loss calculations.

# %%
# Create the point cloud model with protein extensions
key, model_key = jax.random.split(key)
model = PointCloudModel(model_config, extensions=extensions, rngs=nnx.Rngs(params=model_key))
print(f"Created model: {model.__class__.__name__}")


# %% [markdown]
# ## 5. Create Test Batch
#
# We'll create a small test batch with:
# - Amino acid types (20 possible values)
# - 3D coordinates for all atoms
# - Mask indicating valid atoms (all 1s in this case)

# %%
# Create test batch
batch_size = 2

# Create amino acid type inputs
key, aa_key = jax.random.split(key)
aatype = jax.random.randint(aa_key, (batch_size, num_residues), 0, 20)

# Create random 3D coordinates
key, coord_key = jax.random.split(key)
coords = jax.random.normal(coord_key, (batch_size, num_points, 3)) * 10.0

# Create mask (all atoms are valid)
mask = jnp.ones((batch_size, num_points))

# Create test batch
batch = {
    "aatype": aatype,
    "positions": coords,
    "mask": mask,
}


# %% [markdown]
# ## 6. Forward Pass with Extensions
#
# During the forward pass, the model:
# 1. Processes the input through the transformer layers
# 2. Runs each extension to extract domain-specific information
# 3. Returns both the main output and extension outputs

# %%
# Forward pass with extensions
outputs = model(batch)

# Check model outputs
print("\nModel outputs:")
main_output_shape = outputs["positions"].shape
print(f"- Main output shape: {main_output_shape}")

# Check extension outputs
if "extension_outputs" in outputs:
    ext_outputs = outputs["extension_outputs"]
    print("- Extension outputs:")
    for ext_name in ext_outputs.keys():
        print(f"  - {ext_name}")


# %% [markdown]
# ## 7. Calculate Loss with Extensions
#
# The loss calculation combines:
# - Main reconstruction loss (MSE between input and output positions)
# - Extension losses (bond length violations, angle violations, etc.)
#
# Each extension contributes to the total loss according to its weight.

# %%
# Calculate loss with extension losses
loss_fn = model.get_loss_fn()
loss_outputs = loss_fn(batch, outputs)

print("\nLoss calculation:")
print(f"- Available loss keys: {list(loss_outputs.keys())}")

# Print main loss
if "total_loss" in loss_outputs:
    print(f"- Total loss: {loss_outputs['total_loss']:.2f}")
elif "mse_loss" in loss_outputs:
    print(f"- MSE loss: {loss_outputs['mse_loss']:.2f}")
elif "loss" in loss_outputs:
    print(f"- Main loss: {loss_outputs['loss']:.2f}")

# Print all loss components
for loss_key, value in loss_outputs.items():
    if "loss" in loss_key.lower():
        print(f"- {loss_key}: {value:.2f}")

# Check extension losses
if "extension_losses" in loss_outputs:
    print("- Extension losses:")
    for ext_name, ext_loss in loss_outputs["extension_losses"].items():
        print(f"  - {ext_name}: {ext_loss:.2f}")


# %% [markdown]
# ## Summary
#
# This example demonstrated:
#
# - How to create and configure protein-specific extensions
# - How to attach multiple extensions to a point cloud model
# - How extensions contribute to model outputs and losses
# - The role of domain knowledge in geometric modeling
#
# ## Key Takeaways
#
# 1. **Extensions are modular**: Each extension handles one aspect of domain knowledge
# 2. **Weights control influence**: Extension weights balance different constraints
# 3. **Extensions enhance outputs**: Domain-specific information is available in outputs
# 4. **Extensions guide learning**: Extension losses help learn realistic structures


# %%
print("\nProtein model extension demo completed successfully!")
