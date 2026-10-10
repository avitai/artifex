"""``scripts/require_python.py`` fails a CI leg whose interpreter is not its matrix version."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from substrax.testing import ChildResult
from tests.utils.fresh_interpreter import run_repo_python, WITHOUT_SEARCH_PATHS


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "require_python.py"
RUNNING = f"{sys.version_info.major}.{sys.version_info.minor}"
OTHER = "3.12" if RUNNING != "3.12" else "3.13"


def _run(*args: str) -> ChildResult:
    return run_repo_python(SCRIPT, *args, env=WITHOUT_SEARCH_PATHS)


def test_the_running_version_is_accepted() -> None:
    """A leg whose interpreter is its matrix version passes and names the interpreter."""
    result = _run(RUNNING)

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"wanted Python {RUNNING}" in result.stdout
    assert sys.version.split()[0] in result.stdout


def test_another_version_is_refused_naming_both() -> None:
    """A leg running another interpreter fails, saying which it runs and which it wanted."""
    result = _run(OTHER)

    assert result.returncode != 0
    assert f"this environment runs Python {RUNNING}, not {OTHER}" in result.stderr


@pytest.mark.parametrize(
    "arguments", [(), ("3",), ("3.13.1",), ("three.thirteen",), ("3.12", "3.13")]
)
def test_anything_but_one_major_minor_version_is_refused(arguments: tuple[str, ...]) -> None:
    """The check takes exactly one ``major.minor`` version, so a typo cannot pass a leg."""
    result = _run(*arguments)

    assert result.returncode != 0
    assert "usage: require_python.py <major.minor>" in result.stderr
