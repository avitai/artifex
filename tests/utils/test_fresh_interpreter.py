"""Tests for the child interpreters the repository contracts start."""

from __future__ import annotations

from pathlib import Path

import pytest
from substrax.testing import ChildFailedError

from tests.utils.fresh_interpreter import (
    REPO_ROOT,
    run_repo_json,
    run_repo_python,
    WITHOUT_SEARCH_PATHS,
)


def test_run_repo_json_returns_the_last_json_line_printed_at_the_repository_root() -> None:
    code = "import json, os; print('warming up'); print(json.dumps({'cwd': os.getcwd()}))"

    assert run_repo_json(code) == {"cwd": str(REPO_ROOT)}


def test_run_repo_json_fails_when_the_child_fails() -> None:
    with pytest.raises(ChildFailedError):
        run_repo_json("raise SystemExit(2)")


def test_run_repo_python_runs_code_at_the_repository_root() -> None:
    result = run_repo_python("import os; print(os.getcwd())").check()

    assert result.stdout.strip() == str(REPO_ROOT)


def test_run_repo_python_returns_a_failed_script_run_without_raising(tmp_path: Path) -> None:
    script = tmp_path / "probe.py"
    script.write_text(
        "import json, os, sys\n"
        "print(json.dumps({'cwd': os.getcwd(), 'path': os.environ['PATH'], 'args': sys.argv[1:]}))\n"
        "sys.exit(3)\n",
        encoding="utf-8",
    )

    result = run_repo_python(script, "one", cwd=tmp_path, env=WITHOUT_SEARCH_PATHS)

    assert result.returncode == 3
    assert result.last_json() == {"cwd": str(tmp_path), "path": "", "args": ["one"]}
