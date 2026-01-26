"""Model adapters for protein models in the artifex benchmarking system.

This module provides adapters for protein models to work with the benchmark
metrics, particularly the precision-recall benchmarks. All adapters follow
the NNX requirements outlined in the critical technical guidelines.
"""

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from artifex.benchmarks.model_adapters import (
    BenchmarkModelAdapter,
    register_adapter,
)


class ProteinPointCloudAdapter(BenchmarkModelAdapter):
    """Adapter for protein point cloud models to work with benchmark metrics.

    This adapter handles the conversion between the protein-specific data
    structures and the formats expected by benchmark metrics like
    precision-recall. It ensures all NNX requirements are met.
    """

    def __init__(self, model: Any, *, point_dim: int = 3) -> None:
        """Initialize the protein point cloud adapter.

        Args:
            model: The protein model to adapt.
            point_dim: The dimension of the point cloud coordinates.
                Default is 3 for XYZ coordinates.
        """
        super().__init__(model)
        self.point_dim = point_dim
        self._model_name = "protein_point_cloud_model"  # Override the model name

    @classmethod
    def can_adapt(cls, model: Any) -> bool:
        """Check if this adapter can adapt the given model.

        Args:
            model: The model to check.

        Returns:
            True if this adapter can adapt the model, False otherwise.
        """
        # Check if the model has protein generation methods
        has_protein_methods = (
            hasattr(model, "generate_protein")
            or hasattr(model, "sample_protein")
            or hasattr(model, "generate_structure")
        )
        # Model must be NNX-based and have protein methods
        return isinstance(model, nnx.Module) and has_protein_methods

    def predict(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        """Make predictions using the protein model.

        Args:
            x: Input protein data.
            rngs: NNX Rngs objects for stochastic operations.

        Returns:
            Model predictions in a format compatible with benchmarks.
        """

        # Try different prediction methods based on the model interface
        if hasattr(self.model, "predict"):
            result = self.model.predict(x, rngs=rngs)
        elif hasattr(self.model, "__call__"):
            result = self.model(x, rngs=rngs)
        elif hasattr(self.model, "predict_structure"):
            result = self.model.predict_structure(x, rngs=rngs)
        else:
            raise ValueError(f"Model {self.model_name} has no predict method")

        # Convert protein-specific output to benchmark-compatible format
        return self._convert_protein_output_to_benchmark_format(result)

    def sample(
        self,
        *,
        batch_size: int = 1,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Generate protein samples from the model.

        Args:
            batch_size: Number of samples to generate.
            rngs: NNX Rngs objects for stochastic operations.

        Returns:
            Generated protein samples in benchmark-compatible format.
        """

        # Try different sampling methods based on the model interface
        if hasattr(self.model, "sample_protein"):
            result = self.model.sample_protein(batch_size=batch_size, rngs=rngs)
        elif hasattr(self.model, "generate_protein"):
            result = self.model.generate_protein(batch_size=batch_size, rngs=rngs)
        elif hasattr(self.model, "generate_structure"):
            result = self.model.generate_structure(batch_size=batch_size, rngs=rngs)
        elif hasattr(self.model, "sample"):
            # General sampling method
            result = self.model.sample(batch_size=batch_size, rngs=rngs)
        else:
            raise ValueError(f"Model {self.model_name} has no sample method")

        # Convert protein-specific output to benchmark-compatible format
        return self._convert_protein_output_to_benchmark_format(result)

    def _convert_protein_output_to_benchmark_format(self, protein_output: Any) -> jax.Array:
        """Convert protein-specific output to a benchmark-compatible format.

        Args:
            protein_output: The protein output from the model.

        Returns:
            The protein data in a format compatible with benchmarks.
        """
        # Extract protein coordinates from various formats
        if isinstance(protein_output, dict):
            # Case 1: Dictionary with coordinates
            if "coordinates" in protein_output:
                coords = protein_output["coordinates"]
            elif "atom_positions" in protein_output:
                coords = protein_output["atom_positions"]
            elif "point_cloud" in protein_output:
                coords = protein_output["point_cloud"]
            else:
                raise ValueError("Cannot find coordinates in protein output")
        # Case 2: Object with coordinates attribute
        elif hasattr(protein_output, "coordinates"):
            coords = protein_output.coordinates
        elif hasattr(protein_output, "atom_positions"):
            coords = protein_output.atom_positions
        elif hasattr(protein_output, "point_cloud"):
            coords = protein_output.point_cloud
        # Case 3: Already in array format
        elif isinstance(protein_output, (np.ndarray, jax.Array)):
            coords = protein_output
        else:
            raise ValueError(f"Unsupported protein output format: {type(protein_output)}")

        # Convert to JAX array if needed
        if isinstance(coords, np.ndarray):
            coords = jnp.asarray(coords)

        # Ensure we have a 3D array: [batch_size, num_points, feature_dim]
        if coords.ndim == 2:
            # Single protein point cloud: [num_points, feature_dim]
            coords = coords.reshape(1, coords.shape[0], -1)
        elif coords.ndim == 3:
            # Already in correct shape: [batch_size, num_points, feature_dim]
            pass
        else:
            raise ValueError(f"Unexpected coordinate shape: {coords.shape}")

        # Ensure the feature dimension is appropriate (e.g., 2D for benchmarks)
        feature_dim = coords.shape[-1]
        if feature_dim != self.point_dim:
            # If we have 3D points but need 2D for the benchmark, just use x,y
            if feature_dim > self.point_dim:
                coords = coords[..., : self.point_dim]
            # If we have fewer dims than needed, pad with zeros
            elif feature_dim < self.point_dim:
                padding = [(0, 0), (0, 0), (0, self.point_dim - feature_dim)]
                coords = jnp.pad(coords, padding)

        return coords


# Register the protein model adapter
register_adapter(ProteinPointCloudAdapter)
