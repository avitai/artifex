"""Retained protein model adapter for the benchmark layer.

This module owns the single protein benchmark adapter implementation. The
legacy `artifex.benchmarks.protein_model_adapters` module is now only a thin
compatibility re-export of this class.
"""

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from artifex.benchmarks.model_adapters.generative import (
    NNXGenerativeModelAdapter,
    register_adapter,
)
from artifex.generative_models.models.geometric.protein_point_cloud import ProteinPointCloudModel


class ProteinPointCloudAdapter(NNXGenerativeModelAdapter):
    """Adapter for protein point-cloud style benchmark inputs and outputs."""

    def __init__(self, model: Any, *, point_dim: int = 3) -> None:
        super().__init__(model)
        self.point_dim = point_dim
        if self._model_name_str == "unknown":
            self._model_name_str = "protein_point_cloud_model"

    @classmethod
    def can_adapt(cls, model: Any) -> bool:
        """Check whether the adapter can handle the given protein model."""
        protein_method_names = (
            "generate_protein",
            "sample_protein",
            "generate_structure",
            "predict_structure",
        )
        has_protein_methods = any(
            callable(getattr(model, name, None)) for name in protein_method_names
        )
        return isinstance(model, ProteinPointCloudModel) or (
            isinstance(model, nnx.Module) and has_protein_methods
        )

    def _get_model_callable(self, *names: str) -> Any:
        """Return the first callable attribute exposed by the wrapped model."""
        for name in names:
            attr = getattr(self.model, name, None)
            if callable(attr):
                return attr
        return None

    def predict(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        """Make predictions using the protein model and normalize the output shape."""
        predict_fn = self._get_model_callable("predict", "__call__", "predict_structure")
        if predict_fn is None:
            raise ValueError(f"Model {self.model_name} has no predict method")

        result = predict_fn(x, rngs=rngs)
        return self._convert_protein_output_to_benchmark_format(result)

    def sample(
        self,
        *,
        batch_size: int = 1,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Generate protein samples and normalize them for benchmark use."""
        sample_fn = self._get_model_callable(
            "sample_protein",
            "generate_protein",
            "generate_structure",
            "sample",
        )
        if sample_fn is None:
            raise ValueError(f"Model {self.model_name} has no sample method")

        result = sample_fn(batch_size=batch_size, rngs=rngs)
        return self._convert_protein_output_to_benchmark_format(result)

    def _convert_protein_output_to_benchmark_format(self, protein_output: Any) -> jax.Array:
        """Convert protein-specific outputs to the common benchmark tensor shape."""
        if isinstance(protein_output, dict):
            if "coordinates" in protein_output:
                coords = protein_output["coordinates"]
            elif "atom_positions" in protein_output:
                coords = protein_output["atom_positions"]
            elif "point_cloud" in protein_output:
                coords = protein_output["point_cloud"]
            elif "positions" in protein_output:
                coords = protein_output["positions"]
            else:
                raise ValueError("Cannot find coordinates in protein output")
        elif hasattr(protein_output, "coordinates"):
            coords = protein_output.coordinates
        elif hasattr(protein_output, "atom_positions"):
            coords = protein_output.atom_positions
        elif hasattr(protein_output, "point_cloud"):
            coords = protein_output.point_cloud
        elif hasattr(protein_output, "positions"):
            coords = protein_output.positions
        elif isinstance(protein_output, (np.ndarray, jax.Array)):
            coords = protein_output
        else:
            raise ValueError(f"Unsupported protein output format: {type(protein_output)}")

        if isinstance(coords, np.ndarray):
            coords = jnp.asarray(coords)
        elif not isinstance(coords, jax.Array):
            coords = jnp.asarray(coords)

        if coords.ndim == 2:
            coords = coords.reshape(1, coords.shape[0], -1)
        elif coords.ndim == 3:
            pass
        elif coords.ndim == 4:
            batch_size, num_residues, num_atoms, feature_dim = coords.shape
            coords = coords.reshape(batch_size, num_residues * num_atoms, feature_dim)
        else:
            raise ValueError(f"Unexpected coordinate shape: {coords.shape}")

        feature_dim = coords.shape[-1]
        if feature_dim != self.point_dim:
            if feature_dim > self.point_dim:
                coords = coords[..., : self.point_dim]
            else:
                padding = [(0, 0), (0, 0), (0, self.point_dim - feature_dim)]
                coords = jnp.pad(coords, padding)

        return coords


register_adapter(ProteinPointCloudAdapter)
