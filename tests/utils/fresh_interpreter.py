"""Child interpreters for repository contracts that must not share the test process.

Contracts about import side effects, lazy exports, scripts and process configuration run in a
fresh interpreter through ``substrax.testing.run_python``, which starts it on the CPU backend
without the JAX settings exported in the developer's shell.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from substrax.testing import ChildResult, run_python


REPO_ROOT = Path(__file__).resolve().parents[2]

# The slowest child the repository contracts start, importing trimesh and artifex.benchmarks,
# takes about 2.4 s on a developer machine. 60 s leaves room for slower CI runners and still
# stops a child that hangs.
CHILD_TIMEOUT_SECONDS = 60.0

# Empty search paths, so a child finds no tools on PATH and no modules outside this interpreter.
WITHOUT_SEARCH_PATHS: Mapping[str, str] = MappingProxyType({"PATH": "", "PYTHONPATH": ""})


def run_repo_python(
    program: str | Path,
    *args: str,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ChildResult:
    """Run code, or a script given as a path, in a fresh interpreter at the repository root.

    Args:
        program: Code to run with ``-c``, or the path of a script.
        *args: Arguments passed to the program.
        cwd: The child's working directory; the repository root when ``None``.
        env: Variables set in the child over this process's environment.

    Returns:
        The child's exit code, output and duration; a non-zero exit is returned, not raised.
    """
    return run_python(program, *args, timeout=CHILD_TIMEOUT_SECONDS, cwd=cwd or REPO_ROOT, env=env)


def run_repo_json(code: str) -> Any:
    """Run code at the repository root and return the JSON its last line of output holds.

    Args:
        code: Code to run with ``-c``; its last line of output must be JSON.

    Returns:
        The decoded value.
    """
    return run_repo_python(code).check().last_json()
