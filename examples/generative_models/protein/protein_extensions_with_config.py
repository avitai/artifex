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
"""
# Protein Extensions with Configuration System

This example demonstrates how to use protein extensions with Artifex's configuration
system. Protein extensions add domain-specific capabilities like backbone constraints
and amino acid embeddings to geometric models. You'll learn how to load configurations
from YAML files and integrate them with the extension mechanism.

## Learning Objectives

- Understand protein extensions and their purpose
- Load extension configurations from YAML files
- Create protein extensions programmatically
- Integrate extensions with geometric models
- Use configuration validation and serialization
- Calculate extension-specific losses

## Prerequisites

- Understanding of protein structure (residues, backbone)
- Familiarity with Artifex's configuration system
- Knowledge of geometric models (PointCloudModel)
- Basic understanding of dataclass configs

## Key Concepts

**Protein Extensions**: Modular components that add protein-specific functionality
to generic geometric models:
- **Backbone Constraints**: Enforce bond lengths and angles in the protein backbone
- **Protein Mixin**: Add amino acid type embeddings and processing
- **Extensible**: Easy to add new protein-specific features

**Configuration System**: Artifex uses frozen dataclass configurations for:
- **Type Safety**: Automatic validation of configuration parameters
- **Serialization**: Save/load configs from YAML/JSON files
- **Documentation**: Self-documenting through schema
- **Defaults**: Sensible default values for all parameters
"""

# %%
# !/usr/bin/env python3
import jax
import jax.numpy as jnp
from flax import nnx

from artifex.configs.schema.extensions import ProteinExtensionConfig
from artifex.configs.utils import create_config_from_yaml
from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)
from artifex.generative_models.extensions.protein import create_protein_extensions
from artifex.generative_models.models.geometric import PointCloudModel


# %% [markdown]
"""
## Setup and Initialization

First, we'll set up our environment and initialize the random number generator.
Artifex uses separate RNG keys for parameters and dropout operations.
"""


# %%
# Initialize random number generator with separate keys
rng_key = jax.random.PRNGKey(42)
rng_key, params_key, dropout_key = jax.random.split(rng_key, 3)
rngs = nnx.Rngs(params=params_key, dropout=dropout_key)


def main():
    """Run the protein extension example with configuration."""
    print("Protein Extensions with Configuration System")
    print("=" * 60)

    # %% [markdown]
    #     #     ## Load Extension Configuration
    #     #
    #     #     Artifex supports loading configurations from YAML files. This allows you to:
    #     #     - Version control your model configurations
    #     #     - Share configurations across experiments
    #     #     - Validate configurations at load time
    #     #     - Override defaults easily
    #     #
    #     #     If the YAML file is not found, we'll create a configuration programmatically
    #     #     to demonstrate the schema.
    #     #

    # %%
    # Attempt to load protein extension configuration from YAML
    config_dir = "../src/artifex/generative_models/configs/defaults"
    config_path = f"{config_dir}/extensions/protein.yaml"

    try:
        extension_config = create_config_from_yaml(config_path, ProteinExtensionConfig)
        print(f"\n✓ Loaded extension config from YAML: {extension_config.name}")
        config_source = "YAML file"
    except (FileNotFoundError, ImportError, Exception) as e:
        print(f"\nℹ Could not load YAML config ({e})")
        print("  Creating config programmatically to demonstrate the schema")

        # Fallback: create config programmatically
        extension_config = ProteinExtensionConfig(
            name="programmatic_protein_extensions",
            description="Programmatically created protein extension config",
            use_backbone_constraints=True,
            use_protein_mixin=True,
        )
        config_source = "programmatic creation"

    print(f"\nConfig source: {config_source}")
    print(f"Config description: {extension_config.description}")

    # %% [markdown]
    #     #     ## Convert Configuration to Extension Format
    #     #
    #     #     The ProteinExtensionConfig uses a high-level schema, while the extension
    #     #     creation function expects specific parameters. We convert the config to the
    #     #     expected format here.
    #     #

    # %%
    # Convert Pydantic config to dictionary
    extension_dict = extension_config.model_dump()

    # Map config fields to extension parameters
    protein_config = {
        # Constraint settings
        "use_backbone_constraints": extension_dict.get("use_backbone_constraints", True),
        "bond_length_weight": 1.0,  # Weight for bond length violations
        "bond_angle_weight": 0.5,  # Weight for bond angle violations
        # Protein mixin settings
        "use_protein_mixin": extension_dict.get("use_protein_mixin", True),
        "aa_embedding_dim": 16,  # Amino acid embedding dimension
        "num_aa_types": 20,  # Number of amino acid types (standard amino acids)
    }

    print("\nExtension configuration:")
    for key, value in protein_config.items():
        print(f"  {key}: {value}")

    # %% [markdown]
    #     #     ## Create Protein Extensions
    #     #
    #     #     Now we create the protein extensions using the configuration. The factory
    #     #     function `create_protein_extensions` returns a dictionary of extension objects.
    #     #

    # %%
    # Create protein extensions from config
    extensions = create_protein_extensions(protein_config, rngs=rngs)
    print(f"\n✓ Created protein extensions: {', '.join(extensions.keys())}")

    # %% [markdown]
    #     #     ## Configure the Model
    #     #
    #     #     We'll create a small PointCloudModel to demonstrate how extensions integrate
    #     #     with geometric models. The model processes 3D point clouds representing
    #     #     protein structures.
    #     #

    # %%
    # Define protein structure dimensions
    num_residues = 10  # Number of amino acids
    num_atoms_per_residue = 4  # Backbone atoms (N, CA, C, O)
    num_points = num_residues * num_atoms_per_residue  # Total points

    # Create model configuration using frozen dataclass configs
    network_config = PointCloudNetworkConfig(
        name="protein_network",
        hidden_dims=(64, 64),  # Tuple for frozen dataclass
        activation="gelu",
        embed_dim=64,  # Point embedding dimension
        num_heads=4,  # Attention heads
        num_layers=2,  # Number of processing layers
        dropout_rate=0.1,  # Dropout rate
    )

    model_config = PointCloudConfig(
        name="protein_point_cloud",
        network=network_config,
        num_points=num_points,  # Total number of points
        dropout_rate=0.1,
    )

    print("\nModel configuration:")
    print(
        f"  Number of points: {num_points} "
        f"({num_residues} residues × {num_atoms_per_residue} atoms)"
    )
    print(f"  Embedding dimension: {model_config.network.embed_dim}")
    print(f"  Layers: {model_config.network.num_layers}")

    # %% [markdown]
    #     #     ## Create Model with Extensions
    #     #
    #     #     The PointCloudModel accepts an `extensions` parameter that integrates the
    #     #     protein-specific capabilities. Extensions are applied during forward passes
    #     #     and contribute to the loss function.
    #     #

    # %%
    # Create model with extensions
    model = PointCloudModel(model_config, extensions=extensions, rngs=rngs)
    print(f"\n✓ Created model: {model.__class__.__name__}")
    print(f"  Extensions attached: {len(model.extensions) if hasattr(model, 'extensions') else 0}")

    # %% [markdown]
    #     #     ## Prepare Test Data
    #     #
    #     #     We'll create synthetic protein data to test the model and extensions.
    #     #     The data includes:
    #     #     - **aatype**: Amino acid types (0-19 for 20 standard amino acids)
    #     #     - **positions**: 3D coordinates for each atom
    #     #     - **mask**: Binary mask indicating which positions are valid
    #     #

    # %%
    batch_size = 2

    # Create amino acid type inputs (random types 0-19)
    aatype = jax.random.randint(rng_key, (batch_size, num_residues), 0, 20)

    # Create random 3D coordinates for atoms
    rng_key, coords_key = jax.random.split(rng_key)
    coords = jax.random.normal(coords_key, (batch_size, num_points, 3)) * 5.0

    # Create mask (all positions valid in this example)
    mask = jnp.ones((batch_size, num_points))

    # Create input batch dictionary
    batch = {
        "aatype": aatype,  # Amino acid types
        "positions": coords,  # 3D coordinates
        "mask": mask,  # Validity mask
    }

    print("\nTest batch:")
    print(f"  Amino acid types shape: {aatype.shape}")
    print(f"  Positions shape: {coords.shape}")
    print(f"  Mask shape: {mask.shape}")

    # %% [markdown]
    #     #     ## Model Forward Pass
    #     #
    #     #     Run the model on our test data. The model processes the 3D coordinates
    #     #     and outputs refined positions.
    #     #

    # %%
    # Forward pass (pass positions directly to the model)
    rng_key, forward_key = jax.random.split(rng_key)
    outputs = model(coords, deterministic=True)

    print("\nModel outputs:")
    print(f"  Positions shape: {outputs['positions'].shape}")

    # %% [markdown]
    #     #     ## Test Extension Functionality
    #     #
    #     #     Extensions can process the model outputs and batch data to compute
    #     #     protein-specific features and metrics.
    #     #

    # %%
    # Test extension functionality
    if hasattr(model, "extensions") and model.extensions:
        print("\nExtension outputs:")
        for ext_name, extension in model.extensions.items():
            try:
                ext_output = extension(batch, outputs)
                ext_type = ext_output.get("extension_type", "unknown")
                print(f"  ✓ {ext_name}: {ext_type}")

                # Show some output keys
                output_keys = [k for k in ext_output.keys() if k != "extension_type"]
                if output_keys:
                    print(f"    Outputs: {', '.join(output_keys[:3])}")
            except Exception as e:
                print(f"  ✗ {ext_name}: Error - {e}")

    # %% [markdown]
    #     #     ## Calculate Losses
    #     #
    #     #     The total loss combines:
    #     #     - **Position reconstruction loss**: MSE between input and output coordinates
    #     #     - **Extension losses**: Domain-specific losses from each extension
    #     #       - Bond length violations
    #     #       - Bond angle violations
    #     #       - Amino acid embedding regularization
    #     #

    # %%
    # Calculate position reconstruction loss
    position_loss = jnp.mean((coords - outputs["positions"]) ** 2)
    total_loss = position_loss

    print("\nLoss breakdown:")
    print(f"  Position reconstruction: {position_loss:.6f}")

    # Calculate extension losses if available
    if hasattr(model, "extensions") and model.extensions:
        for ext_name, extension in model.extensions.items():
            if hasattr(extension, "loss_fn"):
                try:
                    ext_loss = extension.loss_fn(batch, outputs)
                    if isinstance(ext_loss, (float, jnp.ndarray)):
                        total_loss += ext_loss
                        print(f"  {ext_name}: {ext_loss:.6f}")
                except Exception as e:
                    print(f"  {ext_name}: Could not calculate loss ({e})")

    print(f"\n  Total loss: {total_loss:.6f}")

    # %% [markdown]
    #     #     ## Configuration System Features
    #     #
    #     #     Artifex's configuration system provides powerful features for managing
    #     #     experiment settings.
    #     #

    # %%
    # Demonstrate configuration system features
    print()
    print("=" * 60)
    print("Configuration System Features")
    print("=" * 60)

    # Configuration serialization
    print("\nConfiguration as dictionary:")
    config_dict = extension_config.model_dump()
    for key, value in config_dict.items():
        print(f"  {key}: {value}")

    # Configuration validation
    print("\nConfiguration validation:")
    print(f"  ✓ Name: {extension_config.name}")
    print(f"  ✓ Backbone constraints enabled: {extension_config.use_backbone_constraints}")
    print(f"  ✓ Protein mixin enabled: {extension_config.use_protein_mixin}")

    print("\n✓ Successfully demonstrated protein extensions with configuration system!")


# %% [markdown]
"""
## Summary and Key Takeaways

In this example, you learned:

1. **Protein Extensions**: How to add domain-specific capabilities to geometric models
   through a modular extension system

2. **Configuration System**: How to use Pydantic-based configs for type-safe,
   validated, and serializable experiment settings

3. **YAML Integration**: How to load configurations from YAML files for easy
   version control and sharing

4. **Extension Integration**: How extensions integrate with models and contribute
   to the loss function

5. **Type Safety**: How the configuration system provides automatic validation
   and documentation

## Experiments to Try

1. **Modify Extension Weights**: Adjust constraint weights to see their effect

   ```python
   protein_config["bond_length_weight"] = 2.0  # Stricter bond constraints
   protein_config["bond_angle_weight"] = 1.0   # Stricter angle constraints
   ```

2. **Disable Extensions**: Try running without certain extensions

   ```python
   protein_config["use_backbone_constraints"] = False
   # See how this affects the loss
   ```

3. **Change Embedding Dimension**: Modify amino acid embedding size

   ```python
   protein_config["aa_embedding_dim"] = 32  # Increased from 16
   # Larger embeddings may capture more amino acid properties
   ```

4. **Add More Residues**: Scale to larger proteins

   ```python
   num_residues = 50  # Increased from 10
   num_points = num_residues * num_atoms_per_residue
   # Update model config accordingly
   ```

5. **Create Custom YAML Config**: Save your configuration to a file

   ```python
   import yaml

   config_dict = extension_config.model_dump()
   with open("my_protein_config.yaml", "w") as f:
       yaml.dump(config_dict, f)
   ```

## Next Steps

- Explore `protein_point_cloud_example.py` for detailed protein point cloud modeling
- See `protein_model_with_modality.py` to learn about the modality architecture
- Check Artifex docs for creating custom protein extensions
- Read about Pydantic models for advanced configuration features
"""

# %%
if __name__ == "__main__":
    main()
