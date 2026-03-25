"""Legacy model-zoo tombstone for the removed preset runtime surface."""

from __future__ import annotations

from typing import Any, NoReturn


_REMOVED_MODEL_ZOO_MESSAGE = (
    "The legacy Artifex model zoo has been removed. "
    "Create a family-specific typed config (for example VAEConfig, DDPMConfig, "
    "FlowConfig, EBMConfig, or PointCloudConfig) and pass it to "
    "artifex.generative_models.factory.create_model(...)."
)


def _raise_removed_model_zoo() -> NoReturn:
    raise RuntimeError(_REMOVED_MODEL_ZOO_MESSAGE)


class ModelZoo:
    """Removed legacy preset surface kept only as an explicit migration boundary."""

    def __init__(self) -> None:
        """Fail fast with the supported model-creation migration path."""
        _raise_removed_model_zoo()


class _RemovedModelZooProxy:
    """Attribute proxy that raises the same migration error on first use."""

    def __getattr__(self, name: str) -> Any:
        _raise_removed_model_zoo()

    def __repr__(self) -> str:
        return f"<RemovedModelZooProxy message={_REMOVED_MODEL_ZOO_MESSAGE!r}>"


zoo = _RemovedModelZooProxy()

__all__ = ["ModelZoo", "zoo"]
