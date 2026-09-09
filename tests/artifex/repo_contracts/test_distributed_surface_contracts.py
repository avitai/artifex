"""Contracts for the distributed surface artifex delegates to substrax."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
DISTRIBUTED_GUIDE = REPO_ROOT / "docs" / "user-guide" / "advanced" / "distributed.md"
REMOVED_PACKAGES = (
    "artifex.generative_models.scaling",
    "artifex.generative_models.training.distributed",
)


@pytest.mark.parametrize("package", REMOVED_PACKAGES)
def test_the_duplicated_distributed_packages_are_gone(package: str) -> None:
    """The mesh, sharding, placement and collective code lives in substrax alone."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(package)


def test_production_optimizer_takes_substrax_parallelism_config() -> None:
    """The one consumer of ParallelismConfig reads it from substrax."""
    from substrax.mesh import ParallelismConfig

    production = importlib.import_module(
        "artifex.generative_models.inference.optimization.production"
    )

    assert production.ParallelismConfig is ParallelismConfig


def test_distributed_guide_teaches_the_substrax_surface() -> None:
    """The guide names the substrax modules and no deleted artifex module."""
    text = DISTRIBUTED_GUIDE.read_text(encoding="utf-8")

    for required in ("substrax.mesh", "substrax.spmd", "substrax.devices", "DeviceMeshManager"):
        assert required in text
    for banned in ("generative_models.scaling", "training.distributed", "DistributedTrainer"):
        assert banned not in text


def test_no_tracked_file_names_the_removed_packages() -> None:
    """Docs, source, tests and examples reference no deleted module.

    The changelog records the removal and the repository contracts pin it, so
    both are outside the sweep; the changelog doubles as the positive control.
    """
    result = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "-E",
            r"generative_models\.scaling\b|training\.distributed\b|generative_models/scaling/|training/distributed/",
            "--",
            "src",
            "tests",
            ":(exclude)tests/artifex/repo_contracts",
            "docs",
            "examples",
            "scripts",
            "mkdocs.yml",
            "README.md",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    control = subprocess.run(
        ["git", "grep", "-l", "generative_models.scaling", "--", "CHANGELOG.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert control.stdout.strip() == "CHANGELOG.md"
    assert result.stdout.strip() == "", result.stdout
