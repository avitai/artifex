"""Mesh generative model."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.geometric_config import (
    MeshConfig,
)
from artifex.generative_models.core.losses.geometric import get_mesh_loss
from artifex.generative_models.models.geometric.base import GeometricModel


class SiLU(nnx.Module):
    """SiLU activation function module."""

    def __call__(self, x):
        """Apply the SiLU activation function.

        Args:
            x: Input tensor

        Returns:
            Tensor with SiLU activation applied
        """
        return jax.nn.silu(x)


class MeshModel(GeometricModel):
    """Model for generating 3D meshes.

    Meshes consist of vertices (3D points) and faces (triangles) that
    connect the vertices.
    """

    def __init__(self, config: MeshConfig, *, rngs: nnx.Rngs):
        """Initialize the mesh model.

        Args:
            config: MeshConfig dataclass with model parameters.
            rngs: Random number generator keys.

        Raises:
            TypeError: If config is not a MeshConfig
        """
        super().__init__(config, rngs=rngs)

        # Model parameters from dataclass config
        self.embed_dim = config.network.embed_dim
        self.num_vertices = config.num_vertices
        hidden_dims = list(config.network.hidden_dims)

        # Create latent embedding - use embed_dim as latent_dim
        self.latent_dim = config.network.embed_dim

        # MLP for vertex deformation
        layers: list[nnx.Module] = []
        in_dim = self.latent_dim
        for dim in hidden_dims:
            layers.append(nnx.Linear(in_features=in_dim, out_features=dim, rngs=rngs))
            # Use custom activation module
            layers.append(SiLU())
            in_dim = dim

        # Output layer for vertices
        layers.append(
            nnx.Linear(
                in_features=in_dim,
                out_features=self.num_vertices * 3,  # x,y,z for each vertex
                rngs=rngs,
            )
        )

        self.vertex_mlp = nnx.Sequential(*layers)

        # Default to a simple sphere template
        self.template_vertices = self._create_sphere_template(self.num_vertices)
        self.faces = self._create_sphere_faces(self.num_vertices)

    def _create_sphere_template(self, num_vertices: int) -> jax.Array:
        """Create a sphere template with the given number of vertices.

        Args:
            num_vertices: Number of vertices

        Returns:
            Array of vertex coordinates with shape [num_vertices, 3]
        """
        # Placeholder for a simple sphere generation
        # In a real implementation, this would use a proper algorithm
        # to distribute points evenly on a sphere
        phi = jnp.linspace(0, jnp.pi, int(jnp.sqrt(num_vertices)))
        theta = jnp.linspace(0, 2 * jnp.pi, int(jnp.sqrt(num_vertices)))
        phi, theta = jnp.meshgrid(phi, theta)

        x = jnp.sin(phi) * jnp.cos(theta)
        y = jnp.sin(phi) * jnp.sin(theta)
        z = jnp.cos(phi)

        vertices = jnp.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
        # Pad or truncate to the right number of vertices
        if vertices.shape[0] < num_vertices:
            pad = num_vertices - vertices.shape[0]
            vertices = jnp.pad(vertices, ((0, pad), (0, 0)))
        else:
            vertices = vertices[:num_vertices]

        return vertices

    def _create_sphere_faces(self, num_vertices: int) -> jax.Array:
        """Create faces for a sphere template.

        Args:
            num_vertices: Number of vertices

        Returns:
            Array of face indices with shape [num_faces, 3]
        """
        # Very simplified face generation
        # This is just a placeholder; real implementation would be more complex
        n = int(jnp.sqrt(num_vertices))
        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                idx = i * n + j
                faces.append([idx, idx + 1, idx + n])
                faces.append([idx + 1, idx + n + 1, idx + n])

        return jnp.array(faces)

    def __call__(
        self,
        x: jax.Array | dict[str, Any] | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        deterministic: bool = False,
    ) -> dict[str, jax.Array]:
        """Process input and generate a mesh.

        Args:
            x: Input data, either latent vector or dictionary with 'z' key
            rngs: Optional RNG keys for stochastic operations
            deterministic: Whether to run in deterministic mode

        Returns:
            dictionary containing 'vertices' and 'faces' for the generated mesh
        """
        # Generate random latent if not provided
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

        # Handle dictionary input format
        batch_size = 1
        faces = self.faces

        if isinstance(x, dict):
            # Extract vertices from dictionary
            vertices_input = x.get("vertices")
            # Store faces for later if provided
            if "faces" in x:
                faces = x["faces"]

            # Set batch size based on faces if vertices are None
            if vertices_input is None and faces is not None:
                batch_size = faces.shape[0]
            elif vertices_input is not None:
                batch_size = vertices_input.shape[0]
        else:
            # Direct array input
            vertices_input = x
            if vertices_input is not None:
                batch_size = vertices_input.shape[0]

        # If input is None, we need to generate latent vectors
        if vertices_input is None:
            # Use the provided rngs for random number generation
            key = rngs.params()
            latent = jax.random.normal(key, shape=(batch_size, self.latent_dim))
        else:
            # Use input vertices as latent directly
            if vertices_input.shape[-1] != self.latent_dim:
                # Reshape if needed - just flatten and take the first latent_dim values
                latent = vertices_input.reshape(batch_size, -1)
                latent = latent[:, : self.latent_dim]
            else:
                latent = vertices_input

        # Generate vertex deformations
        deformations = self.vertex_mlp(latent)
        deformations = deformations.reshape(batch_size, self.num_vertices, 3)

        # Apply deformations to template
        # Expand template to batch size
        template = jnp.tile(self.template_vertices[None], (batch_size, 1, 1))
        vertices = template + deformations

        return {"vertices": vertices, "latent": latent, "faces": faces}

    def sample(self, n_samples: int, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Generate mesh samples.

        Args:
            n_samples: Number of samples to generate.
            rngs: Optional random number generator keys.

        Returns:
            Generated vertex positions with shape [n_samples, num_vertices, 3]
        """
        # Use the provided rngs or create a default one
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

        # Get the params key for random number generation
        key = rngs.params()

        # Generate latent vectors
        latents = jax.random.normal(key, shape=(n_samples, self.latent_dim))

        # Forward pass for all samples at once (batched)
        outputs = self(latents, deterministic=True)
        return outputs["vertices"]

    def generate(self, n_samples: int = 1, *, rngs: nnx.Rngs | None = None, **kwargs) -> jax.Array:
        """Generate a batch of meshes. Alias for sample.

        Args:
            n_samples: Number of samples to generate
            rngs: Optional random number generator keys
            **kwargs: Additional keyword arguments passed to sample

        Returns:
            Array of vertex coordinates
        """
        # Call sample with the appropriate parameters
        return self.sample(n_samples=n_samples, rngs=rngs)

    def get_loss_fn(self, auxiliary: dict[str, Any] | None = None) -> Any:
        """Get loss function for mesh generation based on configuration.

        Args:
            auxiliary: Optional auxiliary outputs to use in the loss

        Returns:
            Mesh loss function
        """
        # Get loss weights from config or use defaults
        vertex_weight = self.config.get("vertex_loss_weight", 1.0)
        normal_weight = self.config.get("normal_loss_weight", 0.1)
        edge_weight = self.config.get("edge_loss_weight", 0.1)

        # Get the mesh loss function with configured weights
        return get_mesh_loss(
            vertex_weight=vertex_weight,
            normal_weight=normal_weight,
            edge_weight=edge_weight,
        )
