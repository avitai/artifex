"""Geometric model configuration classes.

This module provides dataclass-based configuration for geometric models
(PointCloud, Mesh, Voxel, Graph), following the established pattern from
GAN, VAE, Diffusion, Flow, and Autoregressive configs.

Geometric models generate 3D structures like point clouds, meshes, voxels,
and graphs.
"""

import dataclasses
from typing import Any

from .base_dataclass import BaseConfig
from .base_network import BaseNetworkConfig
from .validation import (
    validate_dropout_rate,
    validate_positive_int,
)


@dataclasses.dataclass(frozen=True)
class PointCloudNetworkConfig(BaseNetworkConfig):
    """Configuration for point cloud network architecture.

    This configures the transformer-based network used in point cloud models.
    It extends BaseNetworkConfig to inherit hidden_dims, activation, etc.

    Attributes:
        embed_dim: Embedding dimension for point representations
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        use_positional_encoding: Whether to use learned positional embeddings
    """

    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    use_positional_encoding: bool = True

    def __post_init__(self) -> None:
        """Validate point cloud network configuration."""
        super().__post_init__()

        validate_positive_int(self.embed_dim, "embed_dim")
        validate_positive_int(self.num_heads, "num_heads")
        validate_positive_int(self.num_layers, "num_layers")

        # Validate embed_dim is divisible by num_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PointCloudNetworkConfig":
        """Create config from dictionary."""
        data = data.copy()

        if "hidden_dims" in data and isinstance(data["hidden_dims"], list):
            data["hidden_dims"] = tuple(data["hidden_dims"])

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class GeometricConfig(BaseConfig):
    """Base configuration for geometric models.

    This is the base class for all geometric model configurations.
    Geometric models generate 3D structures like point clouds, meshes,
    voxels, and graphs.

    Attributes:
        dropout_rate: Global dropout rate for regularization
    """

    dropout_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate geometric configuration."""
        super().__post_init__()
        validate_dropout_rate(self.dropout_rate)


@dataclasses.dataclass(frozen=True)
class PointCloudConfig(GeometricConfig):
    """Configuration for point cloud models.

    Point cloud models generate sets of 3D points, typically using
    transformer-based architectures with attention mechanisms.

    Attributes:
        network: Configuration for the point cloud network architecture
        num_points: Number of points in each point cloud
        point_dim: Dimensionality of each point (typically 3 for XYZ)
        use_normals: Whether to include point normals
        global_features_dim: Dimension of global feature vector
    """

    # Nested network configuration (required)
    network: PointCloudNetworkConfig | None = None

    # Point cloud-specific fields
    num_points: int = 1024
    point_dim: int = 3
    use_normals: bool = False
    global_features_dim: int = 1024

    def __post_init__(self) -> None:
        """Validate PointCloud configuration."""
        super().__post_init__()

        # Validate network is provided
        if self.network is None:
            raise ValueError("network is required and cannot be None")

        # Validate network type
        if not isinstance(self.network, PointCloudNetworkConfig):
            raise TypeError(
                f"network must be PointCloudNetworkConfig, got {type(self.network).__name__}"
            )

        validate_positive_int(self.num_points, "num_points")
        validate_positive_int(self.point_dim, "point_dim")
        validate_positive_int(self.global_features_dim, "global_features_dim")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with nested config handling."""
        data = super().to_dict()

        if self.network is not None:
            data["network"] = self.network.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PointCloudConfig":
        """Create config from dictionary with nested config handling."""
        data = data.copy()

        if "network" in data and isinstance(data["network"], dict):
            data["network"] = PointCloudNetworkConfig.from_dict(data["network"])

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class MeshNetworkConfig(BaseNetworkConfig):
    """Configuration for mesh network architecture.

    Attributes:
        embed_dim: Embedding dimension for vertex/face representations
        num_heads: Number of attention heads
        num_layers: Number of mesh processing layers
        edge_features_dim: Dimension of edge features
    """

    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    edge_features_dim: int = 64

    def __post_init__(self) -> None:
        """Validate mesh network configuration."""
        super().__post_init__()

        validate_positive_int(self.embed_dim, "embed_dim")
        validate_positive_int(self.num_heads, "num_heads")
        validate_positive_int(self.num_layers, "num_layers")
        validate_positive_int(self.edge_features_dim, "edge_features_dim")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeshNetworkConfig":
        """Create config from dictionary."""
        data = data.copy()

        if "hidden_dims" in data and isinstance(data["hidden_dims"], list):
            data["hidden_dims"] = tuple(data["hidden_dims"])

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class MeshConfig(GeometricConfig):
    """Configuration for mesh models.

    Mesh models generate 3D meshes consisting of vertices and faces.

    Attributes:
        network: Configuration for the mesh network architecture
        num_vertices: Maximum number of vertices in the mesh
        num_faces: Maximum number of faces in the mesh
        vertex_dim: Dimensionality of each vertex (typically 3 for XYZ)
    """

    network: MeshNetworkConfig | None = None

    num_vertices: int = 2048
    num_faces: int = 4096
    vertex_dim: int = 3

    def __post_init__(self) -> None:
        """Validate Mesh configuration."""
        super().__post_init__()

        if self.network is None:
            raise ValueError("network is required and cannot be None")

        if not isinstance(self.network, MeshNetworkConfig):
            raise TypeError(f"network must be MeshNetworkConfig, got {type(self.network).__name__}")

        validate_positive_int(self.num_vertices, "num_vertices")
        validate_positive_int(self.num_faces, "num_faces")
        validate_positive_int(self.vertex_dim, "vertex_dim")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with nested config handling."""
        data = super().to_dict()

        if self.network is not None:
            data["network"] = self.network.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeshConfig":
        """Create config from dictionary."""
        data = data.copy()

        if "network" in data and isinstance(data["network"], dict):
            data["network"] = MeshNetworkConfig.from_dict(data["network"])

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class VoxelNetworkConfig(BaseNetworkConfig):
    """Configuration for voxel network architecture.

    Attributes:
        base_channels: Base number of channels in the 3D CNN
        num_layers: Number of 3D convolutional layers
        kernel_size: Size of 3D convolution kernels
        use_residual: Whether to use residual connections
    """

    base_channels: int = 64
    num_layers: int = 4
    kernel_size: int = 3
    use_residual: bool = True

    def __post_init__(self) -> None:
        """Validate voxel network configuration."""
        super().__post_init__()

        validate_positive_int(self.base_channels, "base_channels")
        validate_positive_int(self.num_layers, "num_layers")
        validate_positive_int(self.kernel_size, "kernel_size")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoxelNetworkConfig":
        """Create config from dictionary."""
        data = data.copy()

        if "hidden_dims" in data and isinstance(data["hidden_dims"], list):
            data["hidden_dims"] = tuple(data["hidden_dims"])

        return cls(**data)


# Valid loss types for voxel models
VALID_VOXEL_LOSS_TYPES = ("bce", "dice", "focal", "mse")


@dataclasses.dataclass(frozen=True)
class VoxelConfig(GeometricConfig):
    """Configuration for voxel models.

    Voxel models generate 3D volumetric representations using 3D CNNs.

    Attributes:
        network: Configuration for the voxel network architecture
        voxel_size: Resolution of the voxel grid (voxel_size^3)
        voxel_dim: Number of channels per voxel (e.g., 1 for occupancy)
        use_sparse: Whether to use sparse voxel representation
        loss_type: Type of loss function ("bce", "mse", "focal")
        focal_gamma: Gamma parameter for focal loss (only used if loss_type="focal")
    """

    network: VoxelNetworkConfig | None = None

    voxel_size: int = 32
    voxel_dim: int = 1
    use_sparse: bool = False
    loss_type: str = "bce"
    focal_gamma: float = 2.0

    def __post_init__(self) -> None:
        """Validate Voxel configuration."""
        super().__post_init__()

        if self.network is None:
            raise ValueError("network is required and cannot be None")

        if not isinstance(self.network, VoxelNetworkConfig):
            raise TypeError(
                f"network must be VoxelNetworkConfig, got {type(self.network).__name__}"
            )

        validate_positive_int(self.voxel_size, "voxel_size")
        validate_positive_int(self.voxel_dim, "voxel_dim")

        if self.loss_type not in VALID_VOXEL_LOSS_TYPES:
            raise ValueError(
                f"loss_type must be one of {VALID_VOXEL_LOSS_TYPES}, got '{self.loss_type}'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with nested config handling."""
        data = super().to_dict()

        if self.network is not None:
            data["network"] = self.network.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoxelConfig":
        """Create config from dictionary."""
        data = data.copy()

        if "network" in data and isinstance(data["network"], dict):
            data["network"] = VoxelNetworkConfig.from_dict(data["network"])

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class GraphNetworkConfig(BaseNetworkConfig):
    """Configuration for graph network architecture.

    Attributes:
        node_features_dim: Dimension of node features
        edge_features_dim: Dimension of edge features
        num_layers: Number of message passing layers
        num_mlp_layers: Number of MLP layers in each EGNN block
        aggregation: Type of aggregation ("mean", "sum", "max")
        use_attention: Whether to use attention for message passing
        norm_coordinates: Whether to normalize coordinates
        residual: Whether to use residual connections
    """

    node_features_dim: int = 64
    edge_features_dim: int = 32
    num_layers: int = 4
    num_mlp_layers: int = 2
    aggregation: str = "mean"
    use_attention: bool = True
    norm_coordinates: bool = True
    residual: bool = True

    def __post_init__(self) -> None:
        """Validate graph network configuration."""
        super().__post_init__()

        validate_positive_int(self.node_features_dim, "node_features_dim")
        validate_positive_int(self.edge_features_dim, "edge_features_dim")
        validate_positive_int(self.num_layers, "num_layers")
        validate_positive_int(self.num_mlp_layers, "num_mlp_layers")

        valid_aggregations = {"mean", "sum", "max"}
        if self.aggregation not in valid_aggregations:
            raise ValueError(
                f"aggregation must be one of {valid_aggregations}, got '{self.aggregation}'"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphNetworkConfig":
        """Create config from dictionary."""
        data = data.copy()

        if "hidden_dims" in data and isinstance(data["hidden_dims"], list):
            data["hidden_dims"] = tuple(data["hidden_dims"])

        return cls(**data)


@dataclasses.dataclass(frozen=True)
class GraphConfig(GeometricConfig):
    """Configuration for graph models.

    Graph models generate graph structures with nodes and edges,
    using message passing neural networks.

    Attributes:
        network: Configuration for the graph network architecture
        max_nodes: Maximum number of nodes in the graph
        max_edges: Maximum number of edges in the graph
        directed: Whether the graph is directed
    """

    network: GraphNetworkConfig | None = None

    max_nodes: int = 1024
    max_edges: int = 4096
    directed: bool = False

    def __post_init__(self) -> None:
        """Validate Graph configuration."""
        super().__post_init__()

        if self.network is None:
            raise ValueError("network is required and cannot be None")

        if not isinstance(self.network, GraphNetworkConfig):
            raise TypeError(
                f"network must be GraphNetworkConfig, got {type(self.network).__name__}"
            )

        validate_positive_int(self.max_nodes, "max_nodes")
        validate_positive_int(self.max_edges, "max_edges")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with nested config handling."""
        data = super().to_dict()

        if self.network is not None:
            data["network"] = self.network.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphConfig":
        """Create config from dictionary."""
        data = data.copy()

        if "network" in data and isinstance(data["network"], dict):
            data["network"] = GraphNetworkConfig.from_dict(data["network"])

        return cls(**data)


# ============================================================================
# Protein-specific geometric configurations
# ============================================================================


@dataclasses.dataclass(frozen=True)
class ProteinConstraintConfig:
    """Configuration for protein structural constraints.

    Attributes:
        backbone_weight: Weight for backbone bond length constraint
        bond_weight: Weight for general bond constraints
        angle_weight: Weight for bond angle constraints
        dihedral_weight: Weight for dihedral angle constraints
        phi_weight: Weight for phi angles (backbone dihedral)
        psi_weight: Weight for psi angles (backbone dihedral)
    """

    backbone_weight: float = 1.0
    bond_weight: float = 1.0
    angle_weight: float = 0.5
    dihedral_weight: float = 0.3
    phi_weight: float = 0.5
    psi_weight: float = 0.5


@dataclasses.dataclass(frozen=True)
class ProteinPointCloudConfig(PointCloudConfig):
    """Configuration for protein point cloud models.

    Extends PointCloudConfig with protein-specific parameters for
    modeling protein structures as point clouds of atoms.

    Attributes:
        num_residues: Number of amino acid residues
        num_atoms_per_residue: Atoms per residue (default 4 for backbone: N, CA, C, O)
        backbone_indices: Indices of backbone atoms in each residue
        use_constraints: Whether to apply protein structural constraints
        constraint_config: Configuration for structural constraints
    """

    num_residues: int = 10
    num_atoms_per_residue: int = 4
    backbone_indices: tuple[int, ...] = (0, 1, 2, 3)
    use_constraints: bool = True
    constraint_config: ProteinConstraintConfig | None = None

    def __post_init__(self) -> None:
        """Validate protein point cloud configuration."""
        super().__post_init__()

        validate_positive_int(self.num_residues, "num_residues")
        validate_positive_int(self.num_atoms_per_residue, "num_atoms_per_residue")

        # Validate backbone_indices
        if not self.backbone_indices:
            raise ValueError("backbone_indices cannot be empty")
        if max(self.backbone_indices) >= self.num_atoms_per_residue:
            raise ValueError(
                f"backbone_indices ({self.backbone_indices}) contains index >= "
                f"num_atoms_per_residue ({self.num_atoms_per_residue})"
            )

    @property
    def total_atoms(self) -> int:
        """Total number of atoms in the protein."""
        return self.num_residues * self.num_atoms_per_residue


@dataclasses.dataclass(frozen=True)
class ProteinGraphConfig(GraphConfig):
    """Configuration for protein graph models.

    Extends GraphConfig with protein-specific parameters for
    modeling protein structures as graphs with nodes (atoms/residues)
    and edges (bonds/contacts).

    Attributes:
        num_residues: Number of amino acid residues
        num_atoms_per_residue: Atoms per residue (default 4 for backbone)
        backbone_indices: Indices of backbone atoms in each residue
        use_constraints: Whether to apply protein structural constraints
        constraint_config: Configuration for structural constraints
        node_dim: Dimension of node features (defaults to network.node_features_dim)
        edge_dim: Dimension of edge features (defaults to network.edge_features_dim)
    """

    num_residues: int = 10
    num_atoms_per_residue: int = 4
    backbone_indices: tuple[int, ...] = (0, 1, 2, 3)
    use_constraints: bool = True
    constraint_config: ProteinConstraintConfig | None = None

    def __post_init__(self) -> None:
        """Validate protein graph configuration."""
        super().__post_init__()

        validate_positive_int(self.num_residues, "num_residues")
        validate_positive_int(self.num_atoms_per_residue, "num_atoms_per_residue")

        # Validate backbone_indices
        if not self.backbone_indices:
            raise ValueError("backbone_indices cannot be empty")
        if max(self.backbone_indices) >= self.num_atoms_per_residue:
            raise ValueError(
                f"backbone_indices ({self.backbone_indices}) contains index >= "
                f"num_atoms_per_residue ({self.num_atoms_per_residue})"
            )

    @property
    def total_atoms(self) -> int:
        """Total number of atoms in the protein."""
        return self.num_residues * self.num_atoms_per_residue

    @property
    def node_dim(self) -> int:
        """Node feature dimension from network config."""
        if self.network is not None:
            return self.network.node_features_dim
        return 64

    @property
    def edge_dim(self) -> int:
        """Edge feature dimension from network config."""
        if self.network is not None:
            return self.network.edge_features_dim
        return 32
