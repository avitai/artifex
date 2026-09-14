"""Importing a module leaves the process alone: no logging configuration, no environment writes.

A module that calls ``logging.basicConfig`` at import changes logging for everything imported
after it, test collection included, and a test module that writes ``os.environ`` at import changes
the environment of every test collected after it. Entry points configure logging in ``main()`` or
under ``if __name__ == "__main__":``.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
_IMPORTABLE_ROOTS = ("src", "scripts", "examples", "tests")
_ENVIRONMENT_METHODS = frozenset({"update", "setdefault", "pop", "popitem", "clear"})


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _writes_environment(node: ast.AST) -> bool:
    """Whether one statement or expression changes the process environment."""
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
        targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
        return any(isinstance(t, ast.Subscript) and _is_os_environ(t.value) for t in targets)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        function = node.func
        if _is_os_environ(function.value) and function.attr in _ENVIRONMENT_METHODS:
            return True
        return (
            isinstance(function.value, ast.Name)
            and function.value.id == "os"
            and function.attr in {"putenv", "unsetenv"}
        )
    return False


def _configures_logging(node: ast.AST) -> bool:
    """Whether a node calls ``logging.basicConfig``."""
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    if isinstance(function, ast.Attribute):
        return (
            function.attr == "basicConfig"
            and isinstance(function.value, ast.Name)
            and function.value.id == "logging"
        )
    return isinstance(function, ast.Name) and function.id == "basicConfig"


def _is_main_guard(node: ast.AST) -> bool:
    """Whether ``node`` is ``if __name__ == "__main__":``, which runs only as a script."""
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def _module_level_nodes(path: Path, matches: Callable[[ast.AST], bool]) -> list[int]:
    """Line numbers of nodes that ``matches`` and that run when the module is imported.

    Function and class bodies and the ``__main__`` guard run only when called or run as a script,
    so they are not searched.
    """
    lines: list[int] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if _is_main_guard(node):
            return
        if isinstance(node, (ast.stmt, ast.expr)) and matches(node):
            lines.append(node.lineno)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in ast.parse(path.read_text(encoding="utf-8")).body:
        visit(statement)
    return lines


def _python_files(*roots: str, pattern: str = "*.py") -> list[Path]:
    return [
        path
        for root in roots
        for path in sorted((REPO_ROOT / root).rglob(pattern))
        if "example_data" not in path.parts
    ]


def test_scanner_reports_import_time_calls_and_skips_entry_points(tmp_path: Path) -> None:
    """Positive control: calls at import are found; ``main()`` and the guard are not searched."""
    module = tmp_path / "module.py"
    module.write_text(
        "import logging\n"
        "import os\n"
        "logging.basicConfig()\n"
        "os.environ['X'] = '1'\n"
        "def main():\n"
        "    logging.basicConfig()\n"
        "if __name__ == '__main__':\n"
        "    logging.basicConfig()\n"
        "    os.environ.update({})\n",
        encoding="utf-8",
    )

    assert _module_level_nodes(module, _configures_logging) == [3]
    assert _module_level_nodes(module, _writes_environment) == [4]


def test_no_module_configures_logging_at_import() -> None:
    """Importing a module must leave the root logger alone."""
    modules = _python_files(*_IMPORTABLE_ROOTS)
    configured = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in modules
        for line in _module_level_nodes(path, _configures_logging)
    ]

    assert REPO_ROOT / "scripts" / "validate_docs.py" in modules
    assert REPO_ROOT / "examples" / "generative_models" / "loss_examples.py" in modules
    assert configured == []


def test_no_test_module_writes_the_environment_at_import() -> None:
    """Environment a test needs belongs in that test (``monkeypatch.setenv``)."""
    modules = _python_files("tests", pattern="test_*.py")
    writes = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in modules
        for line in _module_level_nodes(path, _writes_environment)
    ]

    assert Path(__file__) in modules
    assert writes == []
