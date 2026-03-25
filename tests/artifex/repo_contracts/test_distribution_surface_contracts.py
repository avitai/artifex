import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_distribution_base_lives_under_core_distributions() -> None:
    """The concrete Distribution base should live under core.distributions."""
    module = importlib.import_module("artifex.generative_models.core.distributions.base")

    assert hasattr(module, "Distribution")
    assert module.Distribution.__module__ == "artifex.generative_models.core.distributions.base"


def test_core_interfaces_module_is_not_shipped() -> None:
    """The old concrete core.interfaces surface should stay removed."""
    assert not (REPO_ROOT / "src/artifex/generative_models/core/interfaces.py").exists()


def test_distributions_package_exports_distribution() -> None:
    """The distributions package should export the shared Distribution base."""
    module = importlib.import_module("artifex.generative_models.core.distributions")

    assert hasattr(module, "Distribution")
    assert "Distribution" in module.__all__
