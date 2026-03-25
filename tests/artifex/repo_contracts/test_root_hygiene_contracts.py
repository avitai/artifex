from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def _repo_standards() -> dict[str, object]:
    pyproject = _load_pyproject()
    tool = pyproject["tool"]
    artifex = tool["artifex"]
    return artifex["repo_standards"]


def test_internal_tool_ignores_are_explicit_and_narrow() -> None:
    """Private tooling ignores should name the retained repo-local paths exactly."""
    ignore_lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {"memory-bank/", ".claude/", ".cursor/", "CLAUDE.md"} <= ignore_lines

    for wildcard_pattern in {"m*ry-bank/", ".c*de/", ".c*sor/", "C*DE.md"}:
        assert wildcard_pattern not in ignore_lines


def test_python_pin_and_lockfile_follow_repo_python_policy() -> None:
    """Local tooling pins and the derived lockfile should trace back to canonical policy."""
    pyproject = _load_pyproject()
    standards = _repo_standards()
    ci_policy = pyproject["tool"]["artifex"]["ci"]
    uv_lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    python_pin = (REPO_ROOT / ".python-version").read_text().strip()

    assert python_pin == standards["interpreter"]["tooling"]
    assert python_pin == ci_policy["tooling_python"]
    assert python_pin in ci_policy["compatibility_python"]
    assert uv_lock["requires-python"] == pyproject["project"]["requires-python"]
    assert uv_lock["requires-python"] == f">={standards['interpreter']['minimum_supported']}"


def test_markdownlint_exceptions_are_reviewed_and_narrowed() -> None:
    """The root markdownlint profile should keep only the still-reviewed exceptions."""
    config = json.loads((REPO_ROOT / ".markdownlint.json").read_text())
    disabled_rules = sorted(
        rule for rule, value in config.items() if rule != "default" and value is False
    )

    assert config["default"] is True
    assert disabled_rules == [
        "MD013",
        "MD024",
        "MD029",
        "MD033",
        "MD040",
        "MD046",
        "MD056",
    ]


def test_gitattributes_keeps_only_explicit_binary_rules() -> None:
    """Binary tracking rules should stay minimal and explicit."""
    active_rules = [
        line.strip()
        for line in (REPO_ROOT / ".gitattributes").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert active_rules == ["*.svg binary"]


def test_jupytext_pairing_stays_deterministic() -> None:
    """Notebook pairing should remain concise and deterministic."""
    config = tomllib.loads((REPO_ROOT / ".jupytext.toml").read_text())

    assert config["default_jupytext_formats"] == "ipynb,py:percent"
    assert (
        config["notebook_metadata_filter"]
        == "kernelspec,language_info,-jupytext.text_representation.jupytext_version,"
        "-jupytext.notebook_metadata_filter,-jupytext.cell_metadata_filter"
    )
    assert config["cell_metadata_filter"] == "-all"


def test_license_remains_standard_mit_text() -> None:
    """The retained root license should stay the standard MIT grant and disclaimer."""
    license_text = (REPO_ROOT / "LICENSE").read_text()

    assert license_text.startswith("MIT License\n")
    assert "Permission is hereby granted, free of charge" in license_text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in license_text


TRACKED_TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".pyi",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".txt",
    ".sh",
    ".ini",
    ".cfg",
    ".env",
    ".properties",
    ".rst",
}
LOCAL_WORKSPACE_PATTERNS = (
    re.compile("/" + "media" + r"/[^/\n]+/[^/\n]+/"),
    re.compile("/" + "Users" + r"/[^/\n]+/"),
    re.compile("/" + "home" + r"/[^/\n]+/"),
    re.compile(r"[A-Za-z]:\\" + "Users" + r"\\[^\n]+\\"),
)


def _tracked_text_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=False,
        check=True,
    )
    tracked_paths = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(raw_path.decode("utf-8"))
        if relative.suffix in TRACKED_TEXT_SUFFIXES:
            tracked_paths.append(REPO_ROOT / relative)
    return tracked_paths


def test_tracked_text_files_do_not_embed_local_workspace_paths() -> None:
    """Tracked text files should not leak developer-specific absolute workspace paths."""
    offenders: list[str] = []

    for path in _tracked_text_files():
        contents = path.read_text(encoding="utf-8")
        if any(pattern.search(contents) for pattern in LOCAL_WORKSPACE_PATTERNS):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []
