"""Contracts for the callback surface artifex re-exports from substrax."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_GUIDE = REPO_ROOT / "docs" / "user-guide" / "training" / "training-guide.md"
SHARED_NAMES = (
    "BaseCallback",
    "CallbackList",
    "EarlyStoppingCallback",
    "EarlyStoppingConfig",
    "TrainerLike",
    "TrainingCallback",
)


@pytest.mark.parametrize("name", SHARED_NAMES)
def test_callback_package_re_exports_substrax(name: str) -> None:
    """The artifex callbacks package hands out substrax's objects, not copies."""
    artifex_callbacks = importlib.import_module("artifex.generative_models.training.callbacks")
    substrax_callbacks = importlib.import_module("substrax.callbacks")

    assert getattr(artifex_callbacks, name) is getattr(substrax_callbacks, name)


@pytest.mark.parametrize(
    "module",
    [
        "artifex.generative_models.training.callbacks.base",
        "artifex.generative_models.training.callbacks.early_stopping",
    ],
)
def test_the_duplicated_callback_modules_are_gone(module: str) -> None:
    """The base classes and the early-stopping callback have one home."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_callback_package_no_longer_names_the_old_early_stopping_class() -> None:
    """``EarlyStopping`` was the callback's old name; substrax's ``EarlyStopping`` is a tracker."""
    artifex_callbacks = importlib.import_module("artifex.generative_models.training.callbacks")

    assert "EarlyStopping" not in artifex_callbacks.__all__
    assert not hasattr(artifex_callbacks, "EarlyStopping")


def test_training_guide_teaches_the_shared_early_stopping_callback() -> None:
    """The guide constructs ``EarlyStoppingCallback`` from the artifex callbacks package."""
    text = TRAINING_GUIDE.read_text(encoding="utf-8")

    assert "EarlyStoppingCallback(" in text
    assert "EarlyStopping(" not in text
    assert "substrax" in text
