"""Contracts for the optional subsystems that ship behind extras."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from substrax.testing import ChildResult
from tests.utils.fresh_interpreter import run_repo_python, WITHOUT_SEARCH_PATHS


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_SURFACE = ("artifex", "artifex.generative_models")
OPTIONAL_SURFACES = {
    "typer": ("cli", "artifex.cli.__main__"),
    "trimesh": ("benchmarks", "artifex.benchmarks"),
    "graphviz": ("analysis", "artifex.generative_models.utils.code_analysis"),
}


def _project() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def _names(requirements: list[str]) -> set[str]:
    return {
        requirement.split("[")[0].split(">")[0].split("=")[0].strip()
        for requirement in requirements
    }


def _import_without(package: str, modules: tuple[str, ...]) -> ChildResult:
    script = (
        "import importlib, sys\n"
        f"sys.modules[{package!r}] = None\n"
        f"for module in {modules!r}:\n"
        "    importlib.import_module(module)\n"
    )
    return run_repo_python(script, env=WITHOUT_SEARCH_PATHS)


def test_optional_packages_are_declared_as_extras_not_runtime_dependencies() -> None:
    """typer, trimesh and graphviz live behind extras and reach the test environment."""
    project = _project()
    runtime = _names(project["dependencies"])
    extras = project["optional-dependencies"]

    assert not {"typer", "trimesh", "graphviz"} & runtime
    assert _names(extras["cli"]) == {"typer"}
    assert _names(extras["geometric"]) == {"trimesh"}
    assert _names(extras["analysis"]) == {"graphviz"}
    assert "avitai-artifex[cli,geometric]" in extras["benchmarks"]
    assert "avitai-artifex[analysis,cli,geometric]" in extras["test"]
    assert "avitai-artifex[analysis]" in extras["dev"]


@pytest.mark.parametrize("package", sorted(OPTIONAL_SURFACES))
def test_base_surface_imports_without_the_optional_package(package: str) -> None:
    """The package users get from a bare install never needs the optional extras."""
    result = _import_without(package, BASE_SURFACE)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("package", sorted(OPTIONAL_SURFACES))
def test_optional_surface_names_its_extra_when_the_package_is_missing(package: str) -> None:
    """Importing an optional subsystem without its package fails loudly and names the extra."""
    extra, module = OPTIONAL_SURFACES[package]
    result = _import_without(package, (module,))

    assert result.returncode == 1
    assert "ImportError" in result.stderr
    assert f"avitai-artifex[{extra}]" in result.stderr


@pytest.mark.parametrize("package", sorted(OPTIONAL_SURFACES))
def test_optional_surface_imports_when_the_package_is_present(package: str) -> None:
    """Positive control: with the package installed the same subsystem imports."""
    _, module = OPTIONAL_SURFACES[package]
    result = run_repo_python(f"import {package}, {module}", env=WITHOUT_SEARCH_PATHS)

    assert result.returncode == 0, result.stderr
