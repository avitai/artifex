from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def _load_yaml(relative_path: str) -> dict[str, object]:
    with (REPO_ROOT / relative_path).open() as handle:
        return yaml.load(handle, Loader=yaml.BaseLoader)


def _repo_standards() -> dict[str, object]:
    pyproject = _load_pyproject()
    tool = pyproject["tool"]
    artifex = tool["artifex"]
    return artifex["repo_standards"]


def _yaml_bool(value: object) -> bool:
    return str(value).lower() == "true"


def test_pyproject_declares_canonical_repo_standards_contract() -> None:
    """The retained root policy should live in one explicit pyproject section."""
    standards = _repo_standards()

    assert standards["line_length"] == 100
    assert standards["interpreter"]["minimum_supported"] == "3.11"
    assert standards["interpreter"]["tooling"] == "3.11"
    assert standards["interpreter"]["docs"] == "3.11"
    assert standards["coverage"]["project_fail_under"] == 70
    assert standards["coverage"]["new_code_target"] == 80
    assert standards["typing"]["pyright_mode"] == "basic"
    assert standards["docs"]["main_strict"] is True
    assert standards["docs"]["development_strict"] is False
    assert standards["docs"]["rtd_fail_on_warning"] is True


def test_pyright_and_coverage_settings_follow_repo_standards_contract() -> None:
    """Tool settings should trace back to the canonical retained standards contract."""
    pyproject = _load_pyproject()
    standards = _repo_standards()

    requires_python = pyproject["project"]["requires-python"]
    pyright = pyproject["tool"]["pyright"]
    pytest_options = pyproject["tool"]["pytest"]["ini_options"]
    coverage_report = pyproject["tool"]["coverage"]["report"]
    ruff = pyproject["tool"]["ruff"]
    ruff_format = ruff["format"]

    assert requires_python == f">={standards['interpreter']['minimum_supported']}"
    assert pyright["pythonVersion"] == standards["interpreter"]["tooling"]
    assert pyright["typeCheckingMode"] == standards["typing"]["pyright_mode"]
    assert coverage_report["fail_under"] == standards["coverage"]["project_fail_under"]
    assert ruff["line-length"] == standards["line_length"]
    assert "*.ipynb" in ruff_format["exclude"]

    addopts = shlex.split(pytest_options["addopts"])
    cov_fail_under = next(
        argument for argument in addopts if argument.startswith("--cov-fail-under=")
    )
    assert cov_fail_under == (f"--cov-fail-under={standards['coverage']['project_fail_under']}")


def test_docs_configs_follow_repo_standards_contract() -> None:
    """MkDocs and Read the Docs should share one strict main-docs policy."""
    standards = _repo_standards()
    interpreter = standards["interpreter"]
    docs_policy = standards["docs"]

    mkdocs = _load_yaml("mkdocs.yml")
    mkdocs_dev = _load_yaml("mkdocs-dev.yml")
    readthedocs = _load_yaml(".readthedocs.yaml")

    assert _yaml_bool(mkdocs["strict"]) is docs_policy["main_strict"]
    assert _yaml_bool(mkdocs_dev["strict"]) is docs_policy["development_strict"]
    assert mkdocs_dev["INHERIT"] == "mkdocs.yml"

    assert readthedocs["mkdocs"]["configuration"] == "mkdocs.yml"
    assert (
        _yaml_bool(readthedocs["mkdocs"]["fail_on_warning"]) is docs_policy["rtd_fail_on_warning"]
    )
    assert readthedocs["build"]["tools"]["python"] == interpreter["docs"]
    assert "jobs" not in readthedocs["build"]

    installs = readthedocs["python"]["install"]
    assert len(installs) == 1
    assert installs[0]["method"] == "pip"
    assert installs[0]["path"] == "."
    assert installs[0]["extra_requirements"] == ["docs"]


def test_pre_commit_uses_shared_excludes_and_repo_pyright_environment() -> None:
    """Pre-commit should centralize exclusions and reuse the project Pyright environment."""
    contents = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
    pre_commit = _load_yaml(".pre-commit-config.yaml")

    pyright_repo = None
    pyright_hook = None
    for repo in pre_commit["repos"]:
        for hook in repo["hooks"]:
            if hook["id"] == "pyright":
                pyright_repo = repo["repo"]
                pyright_hook = hook
                break
        if pyright_hook is not None:
            break

    assert pyright_repo == "local"
    assert pyright_hook is not None
    assert pyright_hook["entry"] == "uv run pyright --warnings"
    assert pyright_hook["language"] == "system"
    assert pyright_hook["files"] == "^src/.*\\.py$"
    assert "additional_dependencies" not in pyright_hook

    assert "https://github.com/RobertCraigie/pyright-python" not in contents
    assert "git+https://github.com/avitai/datarax" not in contents
    assert re.search(r"exclude:\s*&repo_non_source_exclude\b", contents)
    assert contents.count("exclude: *repo_non_source_exclude") >= 6
    assert re.search(r"exclude:\s*&repo_non_source_and_tests_exclude\b", contents)
    assert contents.count("exclude: *repo_non_source_and_tests_exclude") >= 2


def test_ruff_per_file_policy_keeps_blocking_lint_focused_on_maintained_runtime_code() -> None:
    """Ruff should stay strict on runtime code while allowing interactive docs, examples, and tests."""
    pyproject = _load_pyproject()
    per_file_ignores = pyproject["tool"]["ruff"]["lint"]["per-file-ignores"]

    assert set(per_file_ignores["*.ipynb"]) >= {"F821", "F541", "T201", "UP006", "UP035", "BLE001"}
    assert set(per_file_ignores["docs/**/*.py"]) >= {"T201"}
    assert set(per_file_ignores["examples/**/*.py"]) >= {"T201", "BLE001"}
    assert set(per_file_ignores["tests/*.py"]) >= {"E501", "T201", "BLE001"}
    assert set(per_file_ignores["tests/**/*.py"]) >= {"E501", "T201", "BLE001"}

    assert "src/**/*.py" not in per_file_ignores
    assert "T201" not in pyproject["tool"]["ruff"]["lint"]["ignore"]
    assert "BLE001" not in pyproject["tool"]["ruff"]["lint"]["ignore"]
