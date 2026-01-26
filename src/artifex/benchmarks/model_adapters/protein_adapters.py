"""Model adapters for protein generative models.

This module provides adapters for protein-specific model types to be used with
the benchmark system. These adapters implement the ModelProtocol interface and
leverage the base NNXModelAdapter functionality.
"""

from typing import Any

import flax.nnx as nnx
import jax
import numpy as np

from artifex.benchmarks.model_adapters import NNXModelAdapter, register_adapter
from artifex.generative_models.models.geometric.protein_point_cloud import ProteinPointCloudModel


class ProteinPointCloudAdapter(NNXModelAdapter):
    """Adapter for Protein Point Cloud models.

    This adapter is designed for protein point cloud models that generate
    3D protein structures.
    """

    @classmethod
    def can_adapt(cls, model: Any) -> bool:
        """Check if this adapter can adapt the given model."""
        # Check if the model is a ProteinPointCloudModel
        return isinstance(model, ProteinPointCloudModel)

    def sample(self, *, batch_size: int = 1, rngs: nnx.Rngs) -> jax.Array:
        """Generate protein structure samples from the model.

        Args:
            batch_size: Number of samples to generate.
            rngs: RNGs for stochastic operations.

        Returns:
            Generated protein structures in 3D coordinates.
        """
        # Use the provided rngs directly

        # Use the model's sample method with the proper NNX approach
        samples = self.model.sample(batch_size, rngs=rngs)

        # Ensure samples have proper shape for benchmark processing
        # If the output is in protein format [batch, residues, atoms, 3]
        # Reshape to [batch, residues*atoms, 3] for easier benchmark processing
        if len(samples.shape) == 4:  # [batch, residues, atoms, 3]
            batch_size, num_residues, num_atoms, coords = samples.shape
            samples = samples.reshape(batch_size, num_residues * num_atoms, coords)

        return samples

    def predict(self, x: np.ndarray | jax.Array, *, rngs: nnx.Rngs) -> np.ndarray | jax.Array:
        """Make predictions using the protein model.

        Args:
            x: Input data.
            rngs: RNGs for stochastic operations.

        Returns:
            Protein structure predictions.
        """
        # Call the model with proper shape management
        outputs = self.model(x, rngs=rngs)

        # Extract the predicted coordinates or positions
        if "coordinates" in outputs:
            pred = outputs["coordinates"]
        elif "positions" in outputs:
            pred = outputs["positions"]
        else:
            # No coordinates found, return the outputs directly
            # (probably not a position-generating model)
            if isinstance(outputs, dict):
                # Try to find any position/coordinate-like output
                for key in ["atom_positions", "coords", "xyz", "output"]:
                    if key in outputs:
                        return outputs[key]
                # Fall back to the first value in the dictionary
                return next(iter(outputs.values()))
            return outputs

        # If predictions are in protein format [batch, residues, atoms, 3]
        # reshape to [batch, residues*atoms, 3] for consistent benchmark processing
        if len(pred.shape) == 4:  # [batch, residues, atoms, 3]
            batch_size, num_residues, num_atoms, coords = pred.shape
            pred = pred.reshape(batch_size, num_residues * num_atoms, coords)

        return pred


# Register the protein adapters
register_adapter(ProteinPointCloudAdapter)
