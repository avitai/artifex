"""Cases for the geometric family, ``artifex.generative_models.models.geometric``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from substrax.testing.nnx_conformance import Case
from substrax.typing import PyTree

from tests.artifex.nnx_conformance.cases.inputs import BATCH, normal


_GEOMETRIC = "artifex.generative_models.models.geometric"
_NODES, _NODE_FEATURES, _HIDDEN = 6, 4, 8
_RESIDUES, _ATOMS = 3, 4
_POINTS, _EMBED = 6, 8


def _graph(key: jax.Array) -> tuple[jax.Array, ...]:
    """Node features, edge features and coordinates of a fully connected graph."""
    k1, k2, k3 = jax.random.split(key, 3)
    return (
        jax.random.normal(k1, (BATCH, _NODES, _NODE_FEATURES)),
        jax.random.normal(k2, (BATCH, _NODES, _NODES, _HIDDEN)),
        jax.random.normal(k3, (BATCH, _NODES, 3)),
    )


def _call_graph(
    module: nnx.Module,
    nodes: jax.Array,
    edges: jax.Array,
    coordinates: jax.Array,
    *,
    rngs: nnx.Rngs | None = None,
) -> PyTree:
    return module(
        {
            "node_features": nodes,
            "edge_features": edges,
            "coordinates": coordinates,
            "adjacency": jnp.ones((nodes.shape[0], _NODES, _NODES)),
            "mask": jnp.ones((nodes.shape[0], _NODES)),
        },
        rngs=rngs,
    )


def _protein(key: jax.Array) -> tuple[jax.Array, ...]:
    """Amino-acid types and atom positions of a short chain."""
    k1, k2 = jax.random.split(key)
    return (
        jax.random.randint(k1, (BATCH, _RESIDUES), 0, 20),
        jax.random.normal(k2, (BATCH, _RESIDUES, _ATOMS, 3)),
    )


def _call_protein(
    module: nnx.Module, aatype: jax.Array, positions: jax.Array, *, rngs: nnx.Rngs | None = None
) -> PyTree:
    return module({"aatype": aatype, "atom_positions": positions}, rngs=rngs)


def _point_cloud(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2 = jax.random.split(key)
    return (
        jax.random.normal(k1, (BATCH, _POINTS, 3)),
        jax.random.normal(k2, (BATCH, _POINTS, _EMBED)),
    )


def _graph_network() -> object:
    from artifex.generative_models.core.configuration.geometric_config import (  # noqa: PLC0415
        GraphNetworkConfig,
    )

    # R272/R281: an edge width other than the hidden width fails a dot_general.
    return GraphNetworkConfig(
        name="graph",
        hidden_dims=(_HIDDEN,),
        activation="silu",
        node_features_dim=_NODE_FEATURES,
        edge_features_dim=_HIDDEN,
        num_layers=1,
    )


def _point_cloud_network() -> object:
    from artifex.generative_models.core.configuration.geometric_config import (  # noqa: PLC0415
        PointCloudNetworkConfig,
    )

    return PointCloudNetworkConfig(
        name="point_cloud",
        hidden_dims=(_EMBED,),
        activation="silu",
        embed_dim=_EMBED,
        num_heads=2,
        num_layers=1,
    )


def cases() -> dict[str, Case]:
    """Return the geometric cases, by the qualified name of the class each builds."""

    from artifex.generative_models.core.configuration.geometric_config import (  # noqa: PLC0415
        GraphConfig,
        MeshConfig,
        MeshNetworkConfig,
        PointCloudConfig,
        ProteinGraphConfig,
        ProteinPointCloudConfig,
        VoxelConfig,
        VoxelNetworkConfig,
    )
    from artifex.generative_models.models.geometric import (  # noqa: PLC0415
        graph,
        mesh,
        point_cloud,
        protein_graph,
        protein_point_cloud,
        voxel,
    )

    graph_config = GraphConfig(
        name="graph", network=_graph_network(), max_nodes=_NODES, dropout_rate=0.5
    )
    protein_graph_config = ProteinGraphConfig(
        name="protein_graph",
        network=_graph_network(),
        max_nodes=_RESIDUES * _ATOMS,
        num_residues=_RESIDUES,
        num_atoms_per_residue=_ATOMS,
        dropout_rate=0.5,
    )
    point_cloud_config = PointCloudConfig(
        name="point_cloud", network=_point_cloud_network(), num_points=_POINTS, dropout_rate=0.5
    )
    protein_point_cloud_config = ProteinPointCloudConfig(
        name="protein_point_cloud",
        network=_point_cloud_network(),
        num_points=_RESIDUES * _ATOMS,
        num_residues=_RESIDUES,
        num_atoms_per_residue=_ATOMS,
        dropout_rate=0.5,
    )
    latent = 8
    mesh_config = MeshConfig(
        name="mesh",
        network=MeshNetworkConfig(
            name="mesh", hidden_dims=(8,), activation="silu", embed_dim=latent
        ),
        num_vertices=16,
    )
    voxel_config = VoxelConfig(
        name="voxel",
        network=VoxelNetworkConfig(
            name="voxel", hidden_dims=(16,), activation="relu", base_channels=16
        ),
        voxel_size=4,
    )
    return {
        f"{_GEOMETRIC}.graph.GraphModel": Case(
            build=lambda r: graph.GraphModel(graph_config, rngs=r),
            inputs=_graph,
            call=_call_graph,
            has_dropout=True,
            rng_streams=("dropout",),
        ),
        f"{_GEOMETRIC}.protein_graph.ProteinGraphModel": Case(
            build=lambda r: protein_graph.ProteinGraphModel(protein_graph_config, rngs=r),
            inputs=_protein,
            call=_call_protein,
            has_dropout=True,
            rng_streams=("dropout",),
        ),
        f"{_GEOMETRIC}.point_cloud.PointCloudModel": Case(
            build=lambda r: point_cloud.PointCloudModel(point_cloud_config, rngs=r),
            inputs=_point_cloud,
            call=lambda module, positions, features, *, rngs: module(
                {"positions": positions, "features": features}, rngs=rngs
            ),
            has_dropout=True,
            rng_streams=("dropout",),
        ),
        f"{_GEOMETRIC}.protein_point_cloud.ProteinPointCloudModel": Case(
            build=lambda r: protein_point_cloud.ProteinPointCloudModel(
                protein_point_cloud_config, rngs=r
            ),
            inputs=_protein,
            call=_call_protein,
            has_dropout=True,
            rng_streams=("dropout",),
        ),
        f"{_GEOMETRIC}.mesh.SiLU": Case(
            build=lambda r: mesh.SiLU(),
            inputs=normal(5),
            is_differentiable=False,
        ),
        f"{_GEOMETRIC}.mesh.MeshModel": Case(
            build=lambda r: mesh.MeshModel(mesh_config, rngs=r),
            inputs=normal(latent),
        ),
        f"{_GEOMETRIC}.voxel.VoxelModel": Case(
            build=lambda r: voxel.VoxelModel(voxel_config, rngs=r),
            inputs=normal(16),
            has_batch_stats=True,
        ),
    }


EXEMPT: dict[str, str] = {
    f"{_GEOMETRIC}.base.GeometricModel": (
        "base class whose __call__ raises NotImplementedError; each subclass's case runs it"
    ),
    f"{_GEOMETRIC}.point_cloud.TransformerBlock": (
        "R519/R424: constructed nowhere in src, and every call raises (its MultiHeadAttention is"
        " called with no decode argument); the geometric fix branch wires it in or deletes it"
    ),
}
