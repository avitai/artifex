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
# Protein Point Cloud Model Example

This example demonstrates the ProteinPointCloudModel, a specialized geometric model
designed for protein structure generation and refinement. The model combines point
cloud processing with protein-specific constraints (bond lengths, angles) to generate
physically plausible protein structures.

## Learning Objectives

- Understand the protein point cloud representation (atoms as 3D points)
- Learn to create and configure ProteinPointCloudModel
- Work with backbone-only protein representations
- Apply geometric constraints during generation
- Evaluate model outputs with protein-specific metrics

## Prerequisites

- Basic understanding of protein structure (residues, atoms, backbone)
- Familiarity with point cloud representations
- Knowledge of Flax NNX and JAX basics
- Understanding of generative models

## Key Concepts

**Point Cloud Representation**: Proteins are represented as sets of 3D points,
where each point corresponds to an atom. This representation is invariant to
rotation and translation, making it ideal for geometric modeling.

**Backbone Atoms**: The protein backbone consists of N (nitrogen), CA (alpha carbon),
C (carbon), and O (oxygen) atoms. These four atoms are present in every amino acid
and determine the overall protein structure.

**Geometric Constraints**: Physical constraints like bond lengths and angles are
enforced during generation to ensure the output structures are chemically valid.
"""

# %%
import os
import pickle
from pathlib import Path

import jax
import numpy as np
from flax import nnx

from artifex.data.protein.dataset import (
    ATOM_TYPES,
    BACKBONE_ATOM_INDICES,
    ProteinDataset,
)
from artifex.generative_models.core.configuration import (
    PointCloudNetworkConfig,
    ProteinConstraintConfig,
    ProteinPointCloudConfig,
)
from artifex.generative_models.models.geometric.protein_point_cloud import (
    ProteinPointCloudModel,
)


# %% [markdown]
"""
## Setup and Initialization

First, we'll set up our environment and initialize the random number generator.
Artifex uses Flax NNX's `Rngs` class for managing random state.
"""


# %%
# Create RNG keys
# We need separate keys for parameters and dropout
key = jax.random.key(42)
key, params_key, dropout_key = jax.random.split(key, 3)
rngs = nnx.Rngs(params=params_key, dropout=dropout_key)


def main():
    """Run the protein point cloud model example."""
    print("Creating protein point cloud model example...")

    # %% [markdown]
    #     ## Model Configuration
    #
    #     The ProteinPointCloudModel requires configuration that specifies:
    #     - Model architecture (embedding dimension, layers, attention heads)
    #     - Protein structure parameters (number of residues, atoms per residue)
    #     - Constraint parameters (bond and angle weights)
    #
    #     **Important**: The model uses attention mechanisms to capture long-range
    #     interactions between amino acids, making it suitable for modeling protein folding.
    #

    # %%
    # Model configuration with protein-specific parameters
    # Create network config for point cloud processing
    network_config = PointCloudNetworkConfig(
        name="protein_network",
        hidden_dims=(128, 128, 128, 128),  # 4 layers for hierarchical processing
        activation="gelu",
        embed_dim=128,  # Embedding dimension for each atom
        num_heads=4,  # Number of attention heads
        num_layers=4,  # Number of transformer layers
        dropout_rate=0.1,  # Dropout rate for regularization
    )

    # Create constraint config for structural constraints
    constraint_config = ProteinConstraintConfig(
        bond_weight=1.0,  # Weight for bond length constraints
        angle_weight=0.5,  # Weight for bond angle constraints
    )

    # Create protein point cloud config with nested configs
    # Note: backbone_indices is (0, 1, 2, 3) for the backbone-only view (N, CA, C, O)
    # BACKBONE_ATOM_INDICES from dataset.py is [0, 1, 2, 4] - indices in the FULL atom list
    config = ProteinPointCloudConfig(
        name="protein_example",
        network=network_config,
        num_points=128 * 4,  # num_residues * num_atoms (flattened)
        dropout_rate=0.1,
        # Protein-specific parameters
        num_residues=128,  # Maximum number of residues
        num_atoms_per_residue=4,  # Only backbone atoms (N, CA, C, O)
        backbone_indices=(0, 1, 2, 3),  # Sequential indices for backbone-only view
        use_constraints=True,  # Enable geometric constraints
        constraint_config=constraint_config,
    )

    print("\nModel configuration:")
    print(f"  Embedding dimension: {config.network.embed_dim}")
    print(f"  Number of layers: {config.network.num_layers}")
    print(f"  Attention heads: {config.network.num_heads}")
    print(f"  Constraints enabled: {config.use_constraints}")

    # %% [markdown]
    #     ## Create Model
    #
    #     Now we instantiate the ProteinPointCloudModel with our configuration.
    #     The model will initialize all parameters using the provided RNG keys.
    #

    # %%
    # Create model instance
    model = ProteinPointCloudModel(config, rngs=rngs)
    print(f"\nCreated model: {model.__class__.__name__}")

    # %% [markdown]
    #     ## Create Synthetic Dataset
    #
    #     For this example, we'll create synthetic protein structures with helical geometry.
    #     Real protein data would come from the Protein Data Bank (PDB), but synthetic data
    #     allows us to run the example without downloading large datasets.
    #
    #     The synthetic proteins have:
    #     - Alpha-helix backbone geometry (common secondary structure)
    #     - Only backbone atoms (N, CA, C, O)
    #     - Small amounts of Gaussian noise to simulate structural variation
    #

    # %%
    # Create synthetic dataset
    data_dir = Path("examples_output/protein")
    os.makedirs(data_dir, exist_ok=True)
    data_path = data_dir / "synthetic_proteins.pkl"

    # Create dataset with synthetic data if file doesn't exist
    if not data_path.exists():
        print(f"\nCreating synthetic protein dataset at {data_path}...")

        # Generate synthetic protein data
        num_examples = 10  # Number of protein structures
        max_seq_length = 128  # Maximum number of residues
        noise_level = 0.1  # Gaussian noise standard deviation
        rng = np.random.RandomState(42)

        data = []
        for i in range(num_examples):
            # Create random protein structure with varying length
            seq_length = rng.randint(32, max_seq_length + 1)

            # Initialize arrays for full atom set (from ATOM_TYPES)
            num_atoms = len(ATOM_TYPES)  # Total atom types
            atom_positions = np.zeros((seq_length, num_atoms, 3))
            atom_mask = np.zeros((seq_length, num_atoms))

            # Create alpha-helix backbone geometry
            for j in range(seq_length):
                # CA positions along a helix (helical rise and rotation)
                t = j * 0.5  # Parameter along helix
                atom_positions[j, 1, 0] = 3.0 * np.sin(t)  # CA = index 1
                atom_positions[j, 1, 1] = 3.0 * np.cos(t)
                atom_positions[j, 1, 2] = 1.5 * t  # Rise along z-axis

                # N positions relative to CA (N-CA bond length ~ 1.45 Å)
                atom_positions[j, 0, :] = atom_positions[j, 1, :] + np.array([-1.45, 0, 0])

                # C positions relative to CA (CA-C bond length ~ 1.52 Å)
                atom_positions[j, 2, :] = atom_positions[j, 1, :] + np.array([1.52, 0, 0])

                # O positions relative to C (C-O bond length ~ 1.23 Å)
                atom_positions[j, 4, :] = atom_positions[j, 2, :] + np.array([0, 1.23, 0])

                # Set mask to 1 only for backbone atoms
                for backbone_idx in BACKBONE_ATOM_INDICES:  # [0, 1, 2, 4]
                    atom_mask[j, backbone_idx] = 1.0

            # Add random noise to backbone atom positions
            for backbone_idx in BACKBONE_ATOM_INDICES:
                atom_positions[:, backbone_idx, :] += rng.normal(0, noise_level, (seq_length, 3))

            # Create residue indices (sequential numbering)
            residue_index = np.arange(seq_length)

            # Create example dictionary
            example = {
                "atom_positions": atom_positions,
                "atom_mask": atom_mask,
                "residue_index": residue_index,
            }

            data.append(example)

        # Save synthetic data
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Saved {num_examples} synthetic proteins to {data_path}")

    # %% [markdown]
    #     ## Load Dataset
    #
    #     Now we load the dataset using Artifex's ProteinDataset class.
    #     The `backbone_only=True` flag tells the dataset to only return backbone atoms,
    #     which matches our model configuration.
    #

    # %%
    # Load the dataset from the saved file
    print(f"\nLoading protein dataset from {data_path}...")
    dataset = ProteinDataset(data_path, backbone_only=True)
    print(f"  Dataset size: {len(dataset)} proteins")

    # Get a sample batch
    batch_indices = list(range(min(4, len(dataset))))
    batch = dataset.get_batch(batch_indices)

    print("\nInput batch shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")

    # %% [markdown]
    #     ## Model Forward Pass
    #
    #     Let's run the model on our batch to generate protein structures.
    #     The model outputs:
    #     - `coordinates`: Predicted 3D positions for each atom
    #     - `constraints`: Dictionary with bond length and angle violations (if enabled)
    #

    # %%
    # Run forward pass
    print("\nRunning model forward pass...")
    outputs = model(batch)

    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value.shape}")

    # %% [markdown]
    #     ## Calculate Loss
    #
    #     The model's loss function combines:
    #     - **Reconstruction loss**: MSE between predicted and target coordinates
    #     - **Constraint losses**: Penalties for violating bond lengths and angles
    #
    #     These losses ensure that the model generates physically plausible structures.
    #

    # %%
    # Calculate loss
    print("\nCalculating loss...")
    loss_fn = model.get_loss_fn()
    loss_dict = loss_fn(batch, outputs)

    print("\nLoss values:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    print("\nExample completed successfully!")


# %% [markdown]
"""
## Summary and Key Takeaways

In this example, you learned:

1. **Point Cloud Representation**: How to represent proteins as sets of 3D points
   for geometric modeling

2. **Model Configuration**: How to configure ProteinPointCloudModel with architecture
   and constraint parameters

3. **Synthetic Data Generation**: How to create synthetic protein structures with
   alpha-helix geometry for testing

4. **Geometric Constraints**: How to apply bond length and angle constraints during
   generation to ensure chemical validity

5. **Model Evaluation**: How to compute reconstruction and constraint losses to
   evaluate model performance

## Experiments to Try

1. **Modify Constraint Weights**: Increase `bond_weight` and `angle_weight` to see
   stricter geometric constraints

   ```python
   config.parameters["constraint_config"] = {
       "bond_weight": 2.0,  # Increased from 1.0
       "angle_weight": 1.0,  # Increased from 0.5
   }
   ```

2. **Larger Proteins**: Increase `max_seq_length` to generate longer protein
   structures

   ```python
   max_seq_length = 256  # Double the size
   ```

3. **Different Secondary Structures**: Modify the synthetic data generation to create
   beta-sheets instead of alpha-helices

4. **Add More Atoms**: Modify `num_atoms` to include side-chain atoms beyond the
   backbone

## Next Steps

- Explore `protein_model_with_modality.py` to learn about the modality architecture
- See `protein_extensions_with_config.py` for advanced protein constraint usage
- Check the Artifex documentation for more protein modeling examples
"""

# %%
if __name__ == "__main__":
    main()
