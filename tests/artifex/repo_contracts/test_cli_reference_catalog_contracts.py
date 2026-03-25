from __future__ import annotations

import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs" / "cli"

CURRENT_RUNTIME_PAGES = {
    "__main__.md": {
        "module": "artifex.cli.__main__",
        "source": "src/artifex/cli/__main__.py",
        "required": ["main", "main_callback", "app"],
    },
    "config.md": {
        "module": "artifex.cli.config",
        "source": "src/artifex/cli/config.py",
        "required": ["create", "validate", "show", "diff", "version", "list", "get"],
    },
}

RETIRED_COMMAND_PAGES = {
    "benchmark.md",
    "convert.md",
    "evaluate.md",
    "generate.md",
    "serve.md",
    "train.md",
}

DEAD_INTERNAL_PAGES = {
    "formatting.md": "artifex.cli.utils.formatting",
    "logging.md": "artifex.cli.utils.logging",
    "main.md": "artifex.cli.main",
    "progress.md": "artifex.cli.utils.progress",
}


def test_cli_reference_catalog_keeps_only_runtime_pages_and_retired_command_notices() -> None:
    """The CLI docs should keep only real runtime owners plus truthful command retirements."""
    assert (DOCS_ROOT / "index.md").exists()

    for page_name, expected_module in DEAD_INTERNAL_PAGES.items():
        assert not (DOCS_ROOT / page_name).exists()
        assert expected_module not in (DOCS_ROOT / "index.md").read_text(encoding="utf-8")

    for page_name, expected in CURRENT_RUNTIME_PAGES.items():
        contents = (DOCS_ROOT / page_name).read_text(encoding="utf-8")
        assert "**Status:** `Supported runtime CLI surface`" in contents
        assert expected["module"] in contents
        assert expected["source"] in contents
        for required in expected["required"]:
            assert required in contents

    for page_name in RETIRED_COMMAND_PAGES:
        contents = (DOCS_ROOT / page_name).read_text(encoding="utf-8")
        assert "not currently shipped in the runtime CLI" in contents
        assert "cli.commands." not in contents


def test_cli_reference_modules_are_importable_and_documented_callables_exist() -> None:
    """Retained CLI reference pages should map to importable modules and real callables."""
    cli_root = importlib.import_module("artifex.cli")
    entrypoint = importlib.import_module("artifex.cli.__main__")
    config = importlib.import_module("artifex.cli.config")

    assert hasattr(cli_root, "main")
    assert hasattr(cli_root, "app")
    assert hasattr(entrypoint, "app")
    assert hasattr(entrypoint, "main")
    assert hasattr(entrypoint, "main_callback")

    for command in ["create", "validate", "show", "diff", "version", "list_cmd", "get"]:
        assert hasattr(config, command)


def test_cli_nav_and_index_drop_dead_internal_module_pages() -> None:
    """MkDocs nav and the CLI index should not publish the dead internal module catalog."""
    index_contents = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")
    mkdocs_contents = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    cli_reference_block = mkdocs_contents.split("      - CLI Reference:\n", 1)[1].split(
        "      - Config Reference:\n", 1
    )[0]

    for page_name in DEAD_INTERNAL_PAGES:
        assert f"cli/{page_name}" not in cli_reference_block

    for page_name in CURRENT_RUNTIME_PAGES:
        assert f"cli/{page_name}" in cli_reference_block

    for page_name in RETIRED_COMMAND_PAGES:
        assert f"cli/{page_name}" in cli_reference_block

    for banned in [
        "artifex.cli.main",
        "artifex.cli.utils.formatting",
        "artifex.cli.utils.logging",
        "artifex.cli.utils.progress",
        "print_help",
    ]:
        assert banned not in (
            index_contents + (DOCS_ROOT / "__main__.md").read_text(encoding="utf-8")
        )

    for required in [
        "artifex.cli.__main__",
        "artifex.cli.config",
        "The shipped CLI owns configuration management only.",
    ]:
        assert required in index_contents + (DOCS_ROOT / "config.md").read_text(encoding="utf-8")
