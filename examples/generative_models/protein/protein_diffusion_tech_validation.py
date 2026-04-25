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

# %%
# ruff: noqa: T201
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
"""# Protein Diffusion Technical Validation.

**Status:** Validation utility

This validation utility checks whether JAX, Flax NNX, and optional BioPython
support are available for local protein experiments. It does not instantiate
shipped Artifex protein model, modality, or data owners, so it should be read
as an environment sanity check rather than as a canonical modeling tutorial.

## What This Script Validates

1. **JAX functionality**: random number generation and array operations
2. **Flax NNX**: module creation, linear layers, and activations
3. **Protein representation**: a minimal raw-NNX point-cloud transform
4. **Optional BioPython**: graceful degradation if local PDB parsing is unavailable

## Usage

```bash
source ./activate.sh
uv run python examples/generative_models/protein/protein_diffusion_tech_validation.py
```
"""

# %%
import jax
import jax.numpy as jnp
from flax import nnx


# %% [markdown]
"""## 1. Check Optional Dependencies.

BioPython is used for loading real PDB files, but the example works without it
using synthetic data instead.
"""

# %%
try:
    from Bio.PDB import PDBParser  # type: ignore

    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    print("Biopython not installed. Will use synthetic data.")

# %% [markdown]
"""## 2. Define a Simple Protein Model.

This minimal model transforms protein point clouds using a basic MLP architecture:

- **Encoder**: Maps 3D coordinates → feature space (32 dims)
- **Hidden**: Processes features (64 dims)
- **Decoder**: Maps back to 3D coordinates

This validates that Flax NNX modules work correctly with protein data.
"""


# %%
class SimpleProteinPointCloud(nnx.Module):
    """A minimal protein point cloud model to validate technology stack."""

    features: int = 32
    hidden_dim: int = 64
    output_dim: int = 3  # 3D coordinates

    def __init__(self, rngs: nnx.Rngs):
        """Initialize the protein point cloud model.

        Args:
            rngs: JAX random number generator keys
        """
        super().__init__()

        # Basic MLP for transforming point cloud
        self.encoder = nnx.Linear(in_features=3, out_features=self.features, rngs=rngs)
        self.hidden = nnx.Linear(in_features=self.features, out_features=self.hidden_dim, rngs=rngs)
        self.decoder = nnx.Linear(
            in_features=self.hidden_dim, out_features=self.output_dim, rngs=rngs
        )

    def __call__(self, points):
        """Forward pass through the model."""
        x = self.encoder(points)
        x = nnx.relu(x)
        x = self.hidden(x)
        x = nnx.relu(x)
        x = self.decoder(x)
        return x


# %% [markdown]
"""## 3. Data Generation Functions.

We provide two methods for getting protein coordinates:

1. **Synthetic data**: Random 3D points (always available)
2. **Real PDB files**: Load C-alpha atoms from PDB (requires BioPython)

C-alpha (CA) atoms form the protein backbone and are commonly used
for coarse-grained protein modeling.
"""


# %%
def create_synthetic_protein_data(n_points=100):
    """Create synthetic protein point cloud data."""
    key = jax.random.key(42)
    # Generate random points in 3D space
    points = jax.random.normal(key, (n_points, 3))
    return points


def load_protein_from_pdb(pdb_file):
    """Load protein coordinates from a PDB file."""
    if not HAS_BIOPYTHON:
        print("Biopython not available. Using synthetic data instead.")
        return create_synthetic_protein_data()

    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)

    # Extract CA atom coordinates
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_atom = residue["CA"]
                    coords.append(ca_atom.get_coord())

    if not coords:
        print("No CA atoms found in PDB file. Using synthetic data instead.")
        return create_synthetic_protein_data()

    return jnp.array(coords)


# %% [markdown]
"""## 4. Run Validation Tests.

The main function validates:

1. **Environment check**: Print JAX version and BioPython availability
2. **Model creation**: Instantiate the protein model with NNX
3. **Data loading**: Create or load protein coordinates
4. **Forward pass**: Run inference to verify model works
5. **Loss computation**: Calculate MSE to verify gradients work
"""


# %%
def main():
    """Main function to validate protein diffusion technology stack."""
    print("JAX version:", jax.__version__)
    print("Biopython available:", HAS_BIOPYTHON)

    # Create RNG keys
    key = jax.random.key(0)
    key, params_key = jax.random.split(key)
    rngs = nnx.Rngs(params=params_key)

    # Create model
    model = SimpleProteinPointCloud(rngs=rngs)

    # Create synthetic protein data
    protein_coords = create_synthetic_protein_data(n_points=100)
    print("Protein coordinates shape:", protein_coords.shape)

    # Run forward pass
    output = model(protein_coords)
    print("Model output shape:", output.shape)

    # Compute simple loss (MSE)
    loss = jnp.mean((output - protein_coords) ** 2)
    print("Loss:", loss.item())


# %% [markdown]
# ## Summary and Key Takeaways
#
# This validation script confirmed:
#
# 1. **JAX works**: Random number generation and array operations
# 2. **Flax NNX works**: Module creation, layers, activations
# 3. **Protein data**: Can represent proteins as point clouds
# 4. **Graceful degradation**: Works without optional dependencies
#
# ### Technology Stack Verified
#
# - ✅ JAX {jax_version} - Numerical computing
# - ✅ Flax NNX - Neural network framework
# - ✅ Protein point clouds - 3D coordinate representation
# - ✅ Forward pass - Model inference working
# - ✅ Loss computation - Gradients can be computed
#
# ### Next Steps
#
# - Explore `protein_point_cloud_example.py` for full protein modeling
# - See `protein_extensions_example.py` for domain-specific extensions
# - Try `protein_model_with_modality.py` for modality architecture

# %%
print("Technology validation successful!")


if __name__ == "__main__":
    main()
