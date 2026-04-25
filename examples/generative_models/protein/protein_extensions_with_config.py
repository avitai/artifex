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
"""# Protein Extensions with Configuration System.

This example demonstrates the canonical configuration-driven workflow for
protein extensions in Artifex. The shipped protein extension YAML is loaded as a
typed `ProteinExtensionsConfig`, then used directly to materialize the runtime
extension bundle. Coordinate-aware helpers consume explicit protein-shaped
`atom_positions` tensors rather than flattened point clouds.

## Learning Objectives

- Load the shipped protein extension bundle from YAML
- Inspect and modify the typed config object
- Create extensions directly from the typed bundle
- Attach the resulting extensions to a geometric model fed with explicit `atom_positions` tensors
"""

# %%
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.configs import (
    get_protein_extensions_config,
    ProteinExtensionConfig,
    ProteinExtensionsConfig,
    ProteinMixinConfig,
)
from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)
from artifex.generative_models.extensions.protein import create_protein_extensions
from artifex.generative_models.models.geometric import PointCloudModel


logger = logging.getLogger(__name__)


def show(message: str) -> None:
    """Emit example output through logging instead of raw print()."""
    logger.info(message)


# %% [markdown]
"""## 1. Load the Canonical Protein Extension Bundle.

The shipped YAML lives under `src/artifex/configs/defaults/extensions/protein.yaml`
and is loaded as a real `ProteinExtensionsConfig`.
"""

# %%
logging.basicConfig(level=logging.INFO, format="%(message)s")
bundle = get_protein_extensions_config("protein")
show(f"Loaded bundle: {bundle.name}")
show(f"Bundle fields: {list(bundle.to_dict().keys())}")


# %% [markdown]
"""## 2. Customize the Bundle Programmatically.

The YAML bundle is just a typed config object, so local overrides are explicit
and type-checked.
"""

# %%
custom_bundle = ProteinExtensionsConfig(
    name="protein_extensions_with_config",
    description="Example bundle derived from the shipped protein extension default",
    bond_length=ProteinExtensionConfig(
        **{
            **bundle.bond_length.to_dict(),
            "weight": 1.25,
            "bond_length_weight": 1.25,
        }
    )
    if bundle.bond_length is not None
    else None,
    bond_angle=bundle.bond_angle,
    backbone=bundle.backbone,
    mixin=ProteinMixinConfig(
        **{
            **bundle.mixin.to_dict(),
            "embedding_dim": 24,
            "use_one_hot": False,
        }
    )
    if bundle.mixin is not None
    else None,
)

show("Customized bundle:")
show(
    "  bond_length weight: "
    f"{custom_bundle.bond_length.weight if custom_bundle.bond_length else 'n/a'}"
)
show(
    f"  mixin embedding_dim: {custom_bundle.mixin.embedding_dim if custom_bundle.mixin else 'n/a'}"
)


# %% [markdown]
"""## 3. Materialize the Runtime Extensions.
"""

# %%
key = jax.random.key(123)
key, params_key, dropout_key = jax.random.split(key, 3)
rngs = nnx.Rngs(params=params_key, dropout=dropout_key)

extensions = create_protein_extensions(custom_bundle, rngs=rngs)
show(f"Created extensions: {', '.join(extensions.keys())}")


# %% [markdown]
"""## 4. Attach Extensions to a Point Cloud Model.
"""

# %%
num_residues = 12
num_atoms_per_residue = 4
num_points = num_residues * num_atoms_per_residue

network_config = PointCloudNetworkConfig(
    name="protein_network",
    hidden_dims=(64, 64),
    activation="gelu",
    embed_dim=64,
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

model = PointCloudModel(model_config, extensions=extensions, rngs=rngs)


# %% [markdown]
"""## 5. Run a Forward Pass.
"""

# %%
batch_size = 2
atom_positions = jax.random.normal(key, (batch_size, num_residues, num_atoms_per_residue, 3))
aatype = jax.random.randint(key, (batch_size, num_residues), 0, 20)
atom_mask = jnp.ones((batch_size, num_residues, num_atoms_per_residue))

batch = {
    "atom_positions": atom_positions,
    "aatype": aatype,
    "atom_mask": atom_mask,
}

outputs = model(batch)
show(f"Model output shape: {outputs['atom_positions'].shape}")

loss_fn = model.get_loss_fn()
loss = loss_fn(batch, outputs)
show(f"Total loss with extensions: {loss}")


# %% [markdown]
"""## 6. Save the Bundle Back to YAML.

Because the bundle is a real config object, it round-trips cleanly through the
same config machinery.
"""

# %%
output_path = Path("temp/protein_extensions_example.yaml")
custom_bundle.to_yaml(output_path)
show(f"Saved customized bundle to: {output_path}")


# %% [markdown]
"""## Summary.

- `get_protein_extensions_config()` loads the shipped bundle as a typed config
- `ProteinExtensionsConfig` is the one supported composition surface
- Coordinate-aware helpers consume explicit `atom_positions` payloads end to end
- YAML, code, and runtime extension creation all flow through the same object
"""
