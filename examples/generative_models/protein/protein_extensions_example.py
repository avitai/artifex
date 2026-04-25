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
"""# Protein Extensions Example.

This example demonstrates the canonical Artifex extension contract for protein-aware
geometric models. Protein extensions are composed as a typed
`ProteinExtensionsConfig` bundle, then materialized through
`create_protein_extensions()`. Coordinate-aware extensions consume explicit
protein-shaped `atom_positions` tensors rather than flattened point clouds.

## Learning Objectives

- Compose protein extensions with the typed bundle surface
- Attach protein-aware extensions to a geometric model fed with explicit `atom_positions` tensors
- Inspect extension outputs and loss contributions
- Instantiate individual extensions directly when finer control is needed
"""

# %%
import logging

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.configs import (
    ProteinExtensionConfig,
    ProteinExtensionsConfig,
    ProteinMixinConfig,
)
from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)
from artifex.generative_models.extensions.protein import (
    BondAngleExtension,
    BondLengthExtension,
    create_protein_extensions,
    ProteinMixinExtension,
)
from artifex.generative_models.models.geometric import PointCloudModel


logger = logging.getLogger(__name__)


def show(message: str) -> None:
    """Emit example output through logging instead of raw print()."""
    logger.info(message)


# %% [markdown]
"""## 1. Build a Typed Protein Extension Bundle.

This bundle is the single supported composition surface for protein extensions.
Each optional field corresponds to one extension in the resulting collection.
"""

# %%
key = jax.random.key(42)
key, params_key, dropout_key = jax.random.split(key, 3)
rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
logging.basicConfig(level=logging.INFO, format="%(message)s")

extension_bundle = ProteinExtensionsConfig(
    name="protein_extensions_example",
    bond_length=ProteinExtensionConfig(
        name="bond_length",
        weight=1.0,
        bond_length_weight=1.0,
        ideal_bond_lengths={
            "N-CA": 1.45,
            "CA-C": 1.52,
            "C-N": 1.33,
        },
    ),
    bond_angle=ProteinExtensionConfig(
        name="bond_angle",
        weight=0.5,
        bond_angle_weight=0.5,
        ideal_bond_angles={
            "CA-C-N": 2.025,
            "C-N-CA": 2.11,
            "N-CA-C": 1.94,
        },
    ),
    mixin=ProteinMixinConfig(
        name="protein_mixin",
        embedding_dim=16,
        num_aa_types=21,
        use_one_hot=True,
    ),
)

extensions = create_protein_extensions(extension_bundle, rngs=rngs)
show(f"Created protein extensions: {', '.join(extensions.keys())}")


# %% [markdown]
"""## 2. Create a Protein-Aware Point Cloud Model.
"""

# %%
batch_size = 2
num_residues = 10
num_atoms = 4
num_points = num_residues * num_atoms

network_config = PointCloudNetworkConfig(
    name="protein_network",
    hidden_dims=(64, 64, 64),
    activation="gelu",
    embed_dim=64,
    num_heads=4,
    num_layers=3,
    dropout_rate=0.1,
)

model_config = PointCloudConfig(
    name="protein_point_cloud_with_extensions",
    network=network_config,
    num_points=num_points,
    dropout_rate=0.1,
)

model = PointCloudModel(model_config, extensions=extensions, rngs=rngs)


# %% [markdown]
"""## 3. Create a Synthetic Protein Batch.
"""

# %%
atom_positions = jax.random.normal(key, (batch_size, num_residues, num_atoms, 3))
aatype = jax.random.randint(key, (batch_size, num_residues), 0, 20)
atom_mask = jnp.ones((batch_size, num_residues, num_atoms))

batch = {
    "atom_positions": atom_positions,
    "aatype": aatype,
    "atom_mask": atom_mask,
}


# %% [markdown]
"""## 4. Run the Model and Inspect Extension Outputs.
"""

# %%
outputs = model(batch)
show(f"Model output shape: {outputs['atom_positions'].shape}")

loss_fn = model.get_loss_fn()
loss = loss_fn(batch, outputs)
show(f"Loss with extensions: {loss}")

for name, extension in extensions.items():
    extension_outputs = extension(batch, outputs)
    show(f"Extension {name} outputs: {list(extension_outputs.keys())}")


# %% [markdown]
"""## 5. Instantiate Extensions Directly.

The typed bundle is the recommended composition path. When you need finer
control, you can still instantiate the underlying extension modules directly
using the same typed configs.
"""

# %%
bond_length_extension = BondLengthExtension(
    ProteinExtensionConfig(
        name="bond_length",
        weight=1.0,
        bond_length_weight=1.0,
        ideal_bond_lengths={
            "N-CA": 1.45,
            "CA-C": 1.52,
            "C-N": 1.33,
        },
    ),
    rngs=rngs,
)
bond_angle_extension = BondAngleExtension(
    ProteinExtensionConfig(
        name="bond_angle",
        weight=0.5,
        bond_angle_weight=0.5,
        ideal_bond_angles={
            "CA-C-N": 2.025,
            "C-N-CA": 2.11,
            "N-CA-C": 1.94,
        },
    ),
    rngs=rngs,
)
protein_mixin_extension = ProteinMixinExtension(
    ProteinMixinConfig(
        name="protein_mixin",
        embedding_dim=32,
        num_aa_types=21,
        use_one_hot=False,
    ),
    rngs=rngs,
)

show("Individual extension loss samples:")
show(f"  bond_length: {bond_length_extension.loss_fn(batch, outputs)}")
show(f"  bond_angle: {bond_angle_extension.loss_fn(batch, outputs)}")
show(f"  protein_mixin keys: {list(protein_mixin_extension(batch, outputs).keys())}")


# %% [markdown]
"""## Summary.

- The canonical composition surface is `ProteinExtensionsConfig`
- `create_protein_extensions()` materializes that bundle into live extension modules
- The same typed configs power direct extension instantiation when needed
- Coordinate-aware helpers consume explicit `atom_positions` payloads end to end
- Protein-aware models stay modular because the base model never needs to know
  about protein-specific implementation details
"""
