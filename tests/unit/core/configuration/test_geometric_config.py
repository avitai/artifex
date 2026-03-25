"""Tests for geometric configuration validation contracts."""

import dataclasses
import inspect

import pytest

from artifex.generative_models.core.configuration import ProteinExtensionsConfig
from artifex.generative_models.core.configuration.geometric_config import (
    MeshConfig,
    PointCloudNetworkConfig,
    ProteinPointCloudConfig,
    VoxelConfig,
)


class TestProteinPointCloudConfigExtensions:
    """Validation tests for the protein geometric extension bundle seam."""

    def test_accepts_canonical_protein_extension_bundle(self) -> None:
        """Protein geometric configs should accept the shared protein bundle."""
        network = PointCloudNetworkConfig(
            name="network",
            hidden_dims=(16,),
            activation="relu",
            embed_dim=16,
            num_heads=4,
            num_layers=1,
        )
        bundle = ProteinExtensionsConfig(name="protein_extensions")
        config = ProteinPointCloudConfig(
            name="protein_point_cloud",
            network=network,
            num_points=16,
            point_dim=3,
            num_residues=4,
            num_atoms_per_residue=4,
            backbone_indices=(0, 1, 2, 3),
            extensions=bundle,
        )

        assert config.extensions is bundle

    @pytest.mark.parametrize(
        "invalid_extensions",
        ["invalid", 1.0, {"backbone": "not_typed"}],
    )
    def test_rejects_non_bundle_extensions(self, invalid_extensions) -> None:
        """Protein geometric configs should reject ad hoc extension payloads."""
        network = PointCloudNetworkConfig(
            name="network",
            hidden_dims=(16,),
            activation="relu",
            embed_dim=16,
            num_heads=4,
            num_layers=1,
        )

        with pytest.raises(TypeError, match="ProteinExtensionsConfig"):
            ProteinPointCloudConfig(
                name="protein_point_cloud",
                network=network,
                num_points=16,
                point_dim=3,
                num_residues=4,
                num_atoms_per_residue=4,
                backbone_indices=(0, 1, 2, 3),
                extensions=invalid_extensions,
            )

    def test_is_frozen_slotted_keyword_only_dataclass(self) -> None:
        """Protein point-cloud config should follow the shared config contract."""
        params = ProteinPointCloudConfig.__dataclass_params__
        signature = inspect.signature(ProteinPointCloudConfig)

        assert dataclasses.is_dataclass(ProteinPointCloudConfig)
        assert params.frozen is True
        assert hasattr(ProteinPointCloudConfig, "__slots__")
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in signature.parameters.values()
        )


class TestGeometricConfigTruthfulness:
    """Validation tests for the retained mesh and voxel config contracts."""

    def test_mesh_config_rejects_removed_num_faces_alias(self) -> None:
        """MeshConfig should reject the removed decorative num_faces alias."""
        with pytest.raises(TypeError):
            MeshConfig.from_dict(
                {
                    "name": "mesh",
                    "network": {
                        "name": "mesh_network",
                        "hidden_dims": [16],
                        "activation": "relu",
                        "embed_dim": 16,
                        "num_heads": 4,
                        "num_layers": 1,
                        "edge_features_dim": 8,
                    },
                    "num_vertices": 32,
                    "num_faces": 64,
                }
            )

    @pytest.mark.parametrize(
        ("stale_key", "stale_value"),
        [
            ("resolution", 16),
            ("use_conditioning", True),
            ("conditioning_dim", 10),
            ("model_type", "cnn"),
        ],
    )
    def test_voxel_config_rejects_stale_public_aliases(self, stale_key, stale_value) -> None:
        """VoxelConfig should reject the removed top-level alias fields."""
        data = {
            "name": "voxel",
            "network": {
                "name": "voxel_network",
                "hidden_dims": [16],
                "activation": "relu",
                "base_channels": 16,
                "num_layers": 2,
                "kernel_size": 3,
                "use_residual": True,
            },
            "voxel_size": 8,
            "voxel_dim": 1,
        }
        data[stale_key] = stale_value

        with pytest.raises(TypeError):
            VoxelConfig.from_dict(data)
