"""Contracts for the checkpoint surface artifex delegates to substrax."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER = REPO_ROOT / "src" / "artifex" / "generative_models" / "training" / "trainer.py"
DOCS_NAMING_THE_STORE = (
    "docs/user-guide/training/training-guide.md",
    "docs/user-guide/advanced/checkpointing.md",
    "docs/user-guide/integrations/deployment.md",
    "docs/user-guide/inference/overview.md",
    "docs/training/checkpoint.md",
)


def test_the_core_checkpointing_module_is_gone() -> None:
    """Orbax checkpointing has one home: substrax's OrbaxCheckpointStore."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("artifex.generative_models.core.checkpointing")


def test_checkpoint_callback_saves_through_the_substrax_store() -> None:
    """The callback module binds substrax's store, not a manager of its own."""
    from substrax.checkpoint import OrbaxCheckpointStore

    callback_module = importlib.import_module(
        "artifex.generative_models.training.callbacks.checkpoint"
    )

    assert callback_module.OrbaxCheckpointStore is OrbaxCheckpointStore


def test_trainer_checkpoints_are_not_pickles() -> None:
    """The trainer saves through the store; nothing in it imports pickle."""
    source = TRAINER.read_text(encoding="utf-8")

    assert "pickle" not in source
    assert "OrbaxCheckpointStore" in source


@pytest.mark.parametrize("relative_path", DOCS_NAMING_THE_STORE)
def test_checkpoint_docs_teach_the_substrax_store(relative_path: str) -> None:
    """Each guide that persists state names substrax's store and no deleted helper."""
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert "OrbaxCheckpointStore" in text
    for banned in (
        "core.checkpointing",
        "setup_checkpoint_manager",
        "save_checkpoint_with_optimizer",
        "validate_checkpoint(",
        "recover_from_corruption(",
    ):
        assert banned not in text, (relative_path, banned)
