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

This example demonstrates how to use Artifex's modality architecture for
protein-oriented geometric workflows. The shared factory still returns the
generic model family selected by the typed config, while the typed protein
extension bundle remains the retained protein-specific runtime boundary.

## Learning Objectives

- Understand Artifex's modality architecture and its benefits
- Learn to create protein-oriented models using the factory system with modalities
- Explore different model types (PointCloudModel, GeometricModel) for protein data
- See how typed config families determine the model family used by the factory
- Identify where the typed protein extension bundle fits in the runtime story

## Prerequisites

- Basic understanding of protein structure (residues, atoms)
- Familiarity with generative models
- Knowledge of Flax NNX and JAX basics

## Key Concepts

**Modality Architecture**: Artifex uses a modality-based design where each data type
(image, text, protein, etc.) has its own modality class that owns adapter lookup,
input conventions, and typed extension bundles.

**Factory System**: The `create_model()` factory allows you to instantiate models
using configuration objects. With `modality="protein"`, the shared factory keeps
the generic model family selected by the typed config; the typed protein extension
bundle is where retained protein-specific runtime behavior lives.

**Typed Config Families**: When using the factory system, the config class selects
the base model family. `PointCloudConfig` creates a point-cloud path, while
`GeometricConfig` keeps you on the generic geometric base path.
"""

# %%
import logging
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


logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def echo(message: object = "") -> None:
    """Emit example output through the standard example logger."""
    LOGGER.info("%s", message)


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
echo("Initializing random number generator...")
rng_key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(params=rng_key)


def main():
    """Demonstrate using protein models through modality architecture."""
    echo("Demonstration of the modality architecture with protein models")

    # %% [markdown]
    # ## Available Modalities
    #
    # Artifex provides several built-in modalities for different data types.
    # Let's explore what's available in the system.

    # %%
    # List available modalities
    echo()
    echo("Listing available modalities:")
    modalities = list_modalities()
    for name, cls in modalities.items():
        echo(f"  - {name}: {cls.__name__}")

    # %% [markdown]
    # ## Model Configuration
    #
    # We'll create a configuration for a protein point cloud model.
    # The config type determines the model family that the factory builds.

    # %%
    # Create model configuration
    echo()
    echo("Creating model configuration...")

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
    #     #     #     #     #     ## Approach 1: Factory with Modality Parameter
    #     #     #     #     #
    #     #     #     #     #     The most common way to create protein-oriented
    #     #     #     #     #     models is using the generic factory with the
    #     #     #     #     #     `modality` parameter. The typed config still picks
    #     #     #     #     #     the generic model family, and the typed protein
    #     #     #     #     #     extension bundle owns retained protein-specific
    #     #     #     #     #     runtime behavior.
    #     #     #     #     #

    # %%
    # Approach 1: Using the generic factory with modality
    echo()
    echo("Approach 1: Using the generic factory with modality...")
    model1 = create_model(
        config=model_config,
        modality="protein",
        rngs=rngs,
    )
    echo(f"  Created model: {model1.__class__.__name__}")
    has_ext = hasattr(model1, "extensions")
    ext_count = len(model1.extensions) if has_ext else 0
    echo(f"  Extensions: {ext_count}")

    # %% [markdown]
    #     #     #     #     #     ## Approach 2: Different Model Types
    #     #     #     #     #
    #     #     #     #     #     The modality system works with any compatible
    #     #     #     #     #     model type. Here we create
    #     #     #     #     #     a GeometricModel instead of a PointCloudModel,
    #     #     #     #     #     demonstrating the flexibility
    #     #     #     #     #     of the architecture.
    #     #     #     #     #

    # %%
    # Approach 2: Using different config for different model types
    echo()
    echo("Approach 2: Using GeometricConfig for base GeometricModel...")
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
    echo(f"  Created model: {model2.__class__.__name__}")
    has_ext = hasattr(model2, "extensions")
    ext_count = len(model2.extensions) if has_ext else 0
    echo(f"  Extensions: {ext_count}")

    # %% [markdown]
    # ## Using the Model
    #
    # Now let's use the model for inference. We'll create dummy protein data with
    # the expected structure:
    # - `aatype`: Amino acid types for each residue
    # - `atom_positions`: 3D coordinates for each atom
    # - `atom_mask`: Binary mask indicating which atoms are present

    # %%
    # Demonstrating model usage
    echo()
    echo("Demonstrating model usage with PointCloudModel...")
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
    echo("  Generating output...")
    model_outputs = model1(dummy_input, deterministic=True)

    echo("  Output shapes:")
    for key, value in model_outputs.items():
        if hasattr(value, "shape"):
            echo(f"    {key}: {value.shape}")

    echo()
    echo("Example completed successfully!")


# %% [markdown]
"""
## Summary and Key Takeaways

In this example, you learned:

1. **Modality Architecture**: How Artifex uses modalities to provide domain-specific
   functionality while maintaining a unified interface

2. **Factory System**: How to use `create_model()` with `modality="protein"`
   while keeping the generic model family selected by the typed config

3. **Typed Protein Extensions**: How the typed protein extension bundle carries
   retained protein-specific runtime behavior

4. **Model Flexibility**: How the same modality boundary can work with different
   generic model types (PointCloudModel, GeometricModel)

## Experiments to Try

1. **Different Modalities**: Try using other modalities (image, text) to see
   how the interface remains consistent

2. **Custom Extensions**: Build a typed protein extension bundle with
   `ProteinExtensionsConfig` and `ProteinExtensionConfig`

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
