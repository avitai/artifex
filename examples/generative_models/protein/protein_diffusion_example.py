#!/usr/bin/env python
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
r"""# Protein Diffusion Example.

**Status:** Exploratory workflow

This walkthrough uses retained lower-level Artifex protein structure owners,
synthetic data, and padded batches to inspect the current protein-model
boundary. It does not demonstrate a shipped high-level Artifex protein
diffusion API.

## What This Workflow Actually Uses

- `ProteinPointCloudModel` and `ProteinGraphModel` for direct owner construction
- `ProteinDataset`, `create_synthetic_protein_dataset`, and `protein_collate_fn`
  for retained protein data and batching
- `create_protein_structure_loss` for protein-specific loss evaluation
- optional helpers from `artifex.visualization.protein_viz`

## Usage

```bash
source ./activate.sh
uv run python examples/generative_models/protein/protein_diffusion_example.py
```
"""

# %%
import logging

import jax
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from artifex.data.protein import (
    create_synthetic_protein_dataset,
    protein_collate_fn,
    ProteinDataset,
    ProteinDatasetConfig,
)
from artifex.generative_models.modalities.protein.losses import (
    create_protein_structure_loss,
)
from artifex.generative_models.models.geometric.protein_graph import ProteinGraphModel
from artifex.generative_models.models.geometric.protein_point_cloud import (
    ProteinPointCloudModel,
)
from artifex.visualization.protein_viz import ProteinVisualizer


logger = logging.getLogger(__name__)
EXAMPLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    NotImplementedError,
    RuntimeError,
    TypeError,
    ValueError,
)


def show(message: str) -> None:
    """Emit example output through logging instead of raw print()."""
    logger.info(message)


# %% [markdown]
"""## Build Direct Protein Owners.

The current retained path in this file is deliberately direct: create
`ProteinPointCloudModel` or `ProteinGraphModel` from their live configuration
owners and keep the batching path honest with `ProteinDataset` plus
`protein_collate_fn`.
"""


# %%
def create_protein_model(
    model_type: str = "point_cloud",
    num_residues: int = 64,
    num_atoms_per_residue: int = 4,
    hidden_dim: int = 128,
    num_layers: int = 4,
    rngs_seed: int = 42,
) -> ProteinPointCloudModel | ProteinGraphModel:
    """Create a retained protein model owner for exploratory experiments."""
    key = jax.random.PRNGKey(rngs_seed)
    key, dropout_key = jax.random.split(key)
    rngs = nnx.Rngs(params=key, dropout=dropout_key, sample=key)

    from artifex.generative_models.core.configuration import (
        GraphNetworkConfig,
        PointCloudNetworkConfig,
        ProteinDihedralConfig,
        ProteinExtensionConfig,
        ProteinExtensionsConfig,
        ProteinGraphConfig,
        ProteinPointCloudConfig,
    )

    extensions = ProteinExtensionsConfig(
        name="protein_extensions",
        backbone=ProteinExtensionConfig(
            name="backbone",
            weight=1.0,
            bond_length_weight=1.0,
            bond_angle_weight=0.5,
        ),
        dihedral=ProteinDihedralConfig(
            name="dihedral",
            weight=0.3,
        ),
    )

    if model_type == "point_cloud":
        network_config = PointCloudNetworkConfig(
            name="protein_point_network",
            hidden_dims=(hidden_dim,) * num_layers,
            activation="gelu",
            embed_dim=hidden_dim,
            num_heads=8,
            num_layers=num_layers,
            dropout_rate=0.1,
        )

        config = ProteinPointCloudConfig(
            name="protein_point_cloud_exploratory",
            network=network_config,
            num_points=num_residues * num_atoms_per_residue,
            num_residues=num_residues,
            num_atoms_per_residue=num_atoms_per_residue,
            backbone_indices=(0, 1, 2, 3),
            extensions=extensions,
            dropout_rate=0.1,
        )
        return ProteinPointCloudModel(config, rngs=rngs)

    if model_type == "graph":
        network_config = GraphNetworkConfig(
            name="protein_graph_network",
            hidden_dims=(hidden_dim,) * num_layers,
            activation="gelu",
            node_features_dim=hidden_dim,
            edge_features_dim=hidden_dim,
            num_layers=num_layers,
        )

        config = ProteinGraphConfig(
            name="protein_graph_exploratory",
            network=network_config,
            num_residues=num_residues,
            num_atoms_per_residue=num_atoms_per_residue,
            backbone_indices=(0, 1, 2, 3),
            extensions=extensions,
        )
        return ProteinGraphModel(config, rngs=rngs)

    raise ValueError(f"Unknown model type: {model_type}")


# %% [markdown]
"""## Load Retained Protein Data.

This example keeps the data contract on retained Artifex owners. It uses the
current `ProteinDataset` surface and batches examples with `protein_collate_fn`
instead of a local or placeholder batch helper.
"""


# %%
def load_protein_dataset(
    data_dir: str | None = None,
    num_proteins: int = 32,
    max_seq_length: int = 64,
    use_synthetic: bool = True,
    random_seed: int = 42,
) -> ProteinDataset:
    """Load retained synthetic or file-backed protein data."""
    if use_synthetic or data_dir is None:
        return create_synthetic_protein_dataset(
            num_proteins=num_proteins,
            min_seq_length=max_seq_length // 2,
            max_seq_length=max_seq_length,
            random_seed=random_seed,
        )

    dataset_config = ProteinDatasetConfig(max_seq_length=max_seq_length)
    return ProteinDataset(dataset_config, data_dir=data_dir)


# %%
def prepare_batch(
    dataset: ProteinDataset,
    batch_size: int = 8,
    random_seed: int = 42,
) -> dict[str, jax.Array]:
    """Prepare a padded batch through the retained protein collate function."""
    rng = np.random.RandomState(random_seed)
    indices = rng.choice(len(dataset), size=batch_size, replace=False)
    examples = [dataset[int(idx)] for idx in indices]
    return protein_collate_fn(examples, max_seq_length=dataset.config.max_seq_length)


# %%
def add_noise_to_batch(
    batch: dict[str, jax.Array],
    noise_level: float = 0.1,
    random_seed: int = 42,
) -> dict[str, jax.Array]:
    """Add Gaussian noise to valid atom positions only."""
    key = jax.random.PRNGKey(random_seed)
    atom_positions = batch["atom_positions"]
    atom_mask = batch["atom_mask"]
    noise = jax.random.normal(key, shape=atom_positions.shape) * noise_level

    noisy_batch = dict(batch)
    noisy_batch["atom_positions"] = atom_positions + noise * atom_mask[:, :, :, None]
    return noisy_batch


# %% [markdown]
"""## Summarize Outputs.

The example keeps visualization optional. The core retained contract is that the
point-cloud owner runs on the padded batch, returns protein-shaped outputs, and
pairs cleanly with `create_protein_structure_loss`.
"""


# %%
def summarize_outputs(
    batch: dict[str, jax.Array],
    outputs: dict[str, jax.Array],
    losses: dict[str, jax.Array],
) -> None:
    """Log the current output and loss surface and show lightweight plots when possible."""
    show("Losses:")
    for key, value in losses.items():
        scalar = float(value) if np.ndim(value) == 0 else value
        show(f"- {key}: {scalar}")

    show(f"Output keys: {sorted(outputs.keys())}")
    positions = outputs["positions"]
    show(f"Predicted positions shape: {positions.shape}")

    target_pos = batch["atom_positions"][0]
    target_mask = batch["atom_mask"][0]
    pred_pos = positions[0]

    if pred_pos.ndim == 2:
        pred_pos = pred_pos.reshape(target_pos.shape)

    try:
        target_phi, target_psi = ProteinVisualizer.calculate_dihedral_angles(
            target_pos, target_mask
        )
        pred_phi, pred_psi = ProteinVisualizer.calculate_dihedral_angles(pred_pos, target_mask)

        ProteinVisualizer.plot_ramachandran(target_phi, target_psi, title="Target")
        plt.tight_layout()
        plt.show()

        ProteinVisualizer.plot_ramachandran(pred_phi, pred_psi, title="Predicted")
        plt.tight_layout()
        plt.show()
    except EXAMPLE_ERRORS as error:
        show(f"Skipping optional visualization: {error}")


# %% [markdown]
"""## Run The Exploratory Workflow.
"""


# %%
def main() -> None:
    """Run the exploratory direct-owner protein workflow."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    show("=== Protein Diffusion Example ===")
    show("Exploratory workflow: retained lower-level protein owners only")
    show("This file does not demonstrate a shipped high-level Artifex protein diffusion API.")

    random_seed = 42
    np.random.seed(random_seed)

    show("Creating point-cloud owner...")
    point_cloud_model = create_protein_model(
        model_type="point_cloud",
        num_residues=64,
        hidden_dim=128,
        num_layers=4,
        rngs_seed=random_seed,
    )
    show(f"- Point-cloud owner: {type(point_cloud_model).__name__}")

    show("Creating graph owner preview...")
    graph_model = create_protein_model(
        model_type="graph",
        num_residues=64,
        hidden_dim=128,
        num_layers=4,
        rngs_seed=random_seed,
    )
    show(f"- Graph owner: {type(graph_model).__name__}")

    show("Loading retained synthetic ProteinDataset...")
    dataset = load_protein_dataset(
        num_proteins=32,
        max_seq_length=64,
        use_synthetic=True,
        random_seed=random_seed,
    )
    show(f"- Dataset owner: {type(dataset).__name__}")
    show(f"- Number of examples: {len(dataset)}")

    show("Preparing padded batch through protein_collate_fn...")
    batch = prepare_batch(dataset, batch_size=8, random_seed=random_seed)
    noisy_batch = add_noise_to_batch(batch, noise_level=0.1, random_seed=random_seed)

    show("Creating protein loss surface...")
    loss_fn = create_protein_structure_loss(
        rmsd_weight=1.0,
        backbone_weight=0.5,
        dihedral_weight=0.3,
    )

    show("Running point-cloud owner...")
    outputs = point_cloud_model(noisy_batch)
    losses = loss_fn(batch, outputs)
    summarize_outputs(batch, outputs, losses)


if __name__ == "__main__":
    main()
