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
# Protein Models with Modality Architecture

This example demonstrates how to use Artifex's modality architecture to create
protein-specific generative models. The modality system provides a unified interface
for working with different data types while maintaining domain-specific capabilities.

## Learning Objectives

- Understand Artifex's modality architecture and its benefits
- Learn to create protein models using the factory system with modalities
- Explore different model types (PointCloudModel, GeometricModel) for proteins
- See how to use full module paths when working with the factory

## Prerequisites

- Basic understanding of protein structure (residues, atoms)
- Familiarity with generative models
- Knowledge of Flax NNX and JAX basics

## Key Concepts

**Modality Architecture**: Artifex uses a modality-based design where each data type
(image, text, protein, etc.) has its own modality class that handles domain-specific
preprocessing, evaluation metrics, and model adaptations.

**Factory System**: The `create_model()` factory allows you to instantiate models
using configuration objects and automatically applies modality-specific enhancements
when a modality is specified.

**Full Module Paths**: When using the factory system, model classes must be specified
with their full module path to avoid ambiguity.
"""

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import (
    GeometricConfig,
    PointCloudConfig,
    PointCloudNetworkConfig,
)
from artifex.generative_models.factory import create_model
from artifex.generative_models.modalities import list_modalities


# %% [markdown]
"""
## Setup and Initialization

First, we'll set up our environment and initialize the random number generator.
Artifex uses Flax NNX's `Rngs` class for managing random state across the model.
"""


# %%
# Create examples_output directory
output_dir = Path("examples_output")
output_dir.mkdir(exist_ok=True)

# Initialize random number generator
print("Initializing random number generator...")
rng_key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(params=rng_key)


def main():
    """Demonstrate using protein models through modality architecture."""
    print("Demonstration of the modality architecture with protein models")

    # %% [markdown]
    #     ## Available Modalities
    #
    #     Artifex provides several built-in modalities for different data types.
    #     Let's explore what's available in the system.
    #

    # %%
    # List available modalities
    print("\nListing available modalities:")
    modalities = list_modalities()
    for name, cls in modalities.items():
        print(f"  - {name}: {cls.__name__}")

    # %% [markdown]
    #     ## Model Configuration
    #
    #     We'll create a configuration for a protein point cloud model. Notice that
    #     we use the **full module path** for the model_class. This is required when
    #     using the factory system to avoid ambiguity.
    #
    #     **Important**: The model_class must be the complete path:
    #     - ✅ `"artifex.generative_models.models.geometric.point_cloud.PointCloudModel"`
    #     - ❌ `"PointCloudModel"` (will cause ValueError)
    #

    # %%
    # Create model configuration
    print("\nCreating model configuration...")

    # Create network config for point cloud
    network_config = PointCloudNetworkConfig(
        name="protein_point_cloud_network",
        hidden_dims=(64, 64),  # Tuple for frozen dataclass
        activation="gelu",
        embed_dim=64,
        num_heads=4,
        num_layers=3,
        dropout_rate=0.1,
    )

    # Create PointCloudConfig with nested network config
    model_config = PointCloudConfig(
        name="protein_point_cloud",
        network=network_config,
        num_points=128,
        dropout_rate=0.1,
    )

    # %% [markdown]
    #     ## Approach 1: Factory with Modality Parameter
    #
    #     The most common way to create protein models is using the generic factory
    #     with the `modality` parameter. This automatically applies protein-specific
    #     enhancements and preprocessing.
    #

    # %%
    # Approach 1: Using the generic factory with modality
    print("\nApproach 1: Using the generic factory with modality...")
    model1 = create_model(
        config=model_config,
        modality="protein",
        rngs=rngs,
    )
    print(f"  Created model: {model1.__class__.__name__}")
    has_ext = hasattr(model1, "extensions")
    ext_count = len(model1.extensions) if has_ext else 0
    print(f"  Extensions: {ext_count}")

    # %% [markdown]
    #     ## Approach 2: Different Model Types
    #
    #     The modality system works with any compatible model type. Here we create
    #     a GeometricModel instead of a PointCloudModel, demonstrating the flexibility
    #     of the architecture.
    #

    # %%
    # Approach 2: Using different config for different model types
    print("\nApproach 2: Using GeometricConfig for base GeometricModel...")
    # GeometricConfig is the base config for all geometric models
    model_config2 = GeometricConfig(
        name="protein_geometric",
        dropout_rate=0.1,
    )
    model2 = create_model(
        config=model_config2,
        modality="protein",
        rngs=rngs,
    )
    print(f"  Created model: {model2.__class__.__name__}")
    has_ext = hasattr(model2, "extensions")
    ext_count = len(model2.extensions) if has_ext else 0
    print(f"  Extensions: {ext_count}")

    # %% [markdown]
    #     ## Using the Model
    #
    #     Now let's use the model for inference. We'll create dummy protein data
    #     with the expected structure:
    #     - `aatype`: Amino acid types for each residue
    #     - `atom_positions`: 3D coordinates for each atom
    #     - `atom_mask`: Binary mask indicating which atoms are present
    #

    # %%
    # Demonstrating model usage
    print("\nDemonstrating model usage with PointCloudModel...")
    # Create dummy input data (batch of 2 proteins with 10 residues, 4 atoms each)
    # Total 40 points (10 residues × 4 atoms)
    num_residues = 10
    num_atoms = 4
    batch_size = 2

    dummy_input = {
        # [batch, num_residues] - amino acid types (using glycine = 7 for simplicity)
        "aatype": jnp.full((batch_size, num_residues), 7),  # All glycine residues
        # [batch, num_residues, num_atoms, 3] - atom positions
        "atom_positions": jnp.ones((batch_size, num_residues, num_atoms, 3)),
        # [batch, num_residues, num_atoms] - atom mask
        "atom_mask": jnp.ones((batch_size, num_residues, num_atoms)),
    }

    # Generate output using model1 (PointCloudModel)
    print("  Generating output...")
    model_outputs = model1(dummy_input, deterministic=True)

    print("  Output shapes:")
    for key, value in model_outputs.items():
        if hasattr(value, "shape"):
            print(f"    {key}: {value.shape}")

    print("\nExample completed successfully!")


# %% [markdown]
"""
## Summary and Key Takeaways

In this example, you learned:

1. **Modality Architecture**: How Artifex uses modalities to provide domain-specific
   functionality while maintaining a unified interface

2. **Factory System**: How to use `create_model()` with modality parameters to
   create protein-specific models

3. **Full Module Paths**: The importance of using complete module paths when
   specifying model classes in the factory system

4. **Model Flexibility**: How the same modality can work with different model
   types (PointCloudModel, GeometricModel)

## Experiments to Try

1. **Different Modalities**: Try using other modalities (image, text) to see
   how the interface remains consistent

2. **Custom Extensions**: Modify the metadata to enable/disable different
   protein-specific extensions

3. **Larger Proteins**: Increase `num_residues` and `num_atoms` to work with
   larger protein structures

4. **Training Loop**: Extend this example to include a simple training loop
   using the model's loss function

## Next Steps

- Explore `protein_point_cloud_example.py` for more details on protein-specific models
- See `protein_extensions_example.py` to learn about protein constraint extensions
- Check the documentation for other available modalities and model types
"""

# %%
if __name__ == "__main__":
    main()
