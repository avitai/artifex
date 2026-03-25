"""Utility functions for protein extensions."""

from flax import nnx

from artifex.generative_models.core.configuration import (
    ProteinDihedralConfig,
    ProteinExtensionConfig,
    ProteinExtensionsConfig,
    ProteinMixinConfig,
)
from artifex.generative_models.extensions.base import ExtensionDict, ModelExtension
from artifex.generative_models.extensions.protein.backbone import (
    BondAngleExtension,
    BondLengthExtension,
)
from artifex.generative_models.extensions.protein.constraints import (
    ProteinBackboneConstraint,
    ProteinDihedralConstraint,
)
from artifex.generative_models.extensions.protein.mixin import (
    ProteinMixinExtension,
)


def create_protein_extensions(
    config: ProteinExtensionsConfig,
    *,
    rngs: nnx.Rngs,
) -> ExtensionDict:
    """Create protein extensions from the canonical typed protein bundle.

    Args:
        config: Typed protein extension bundle.
        rngs: Random number generator keys.

    Returns:
        ExtensionDict mapping extension names to extension instances.

    Raises:
        TypeError: If config is not a ProteinExtensionsConfig.
    """
    if not isinstance(config, ProteinExtensionsConfig):
        raise TypeError(f"config must be ProteinExtensionsConfig, got {type(config).__name__}")

    extensions_dict: dict[str, ModelExtension] = {}

    def _add_if_enabled(
        name: str,
        extension_config: (
            ProteinExtensionConfig | ProteinDihedralConfig | ProteinMixinConfig | None
        ),
        extension_class: type[ModelExtension],
    ) -> None:
        if extension_config is None or not extension_config.enabled:
            return
        extensions_dict[name] = extension_class(extension_config, rngs=rngs)

    _add_if_enabled("bond_length", config.bond_length, BondLengthExtension)
    _add_if_enabled("bond_angle", config.bond_angle, BondAngleExtension)
    _add_if_enabled("backbone", config.backbone, ProteinBackboneConstraint)
    _add_if_enabled("dihedral", config.dihedral, ProteinDihedralConstraint)
    _add_if_enabled("protein_mixin", config.mixin, ProteinMixinExtension)

    return ExtensionDict(extensions_dict)
