"""The python blocks on the theory pages run, in page order, against the installed artifex.

A page's blocks build on each other, so each page runs as one program: its blocks joined in
order, in a fresh interpreter. A block that is not meant to run is fenced as ``text``.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
from tests.utils.fresh_interpreter import REPO_ROOT, run_repo_python


THEORY_PAGES = tuple(
    REPO_ROOT / "docs" / "user-guide" / "concepts" / f"{family}-explained.md"
    for family in ("autoregressive", "diffusion", "ebm", "flow", "gan", "vae")
)

# A fence at any indentation: blocks inside grid cards and list items render too.
#
# Searched before writing this (2026-11-24): Sybil 10.2.0, pytest-markdown-docs 0.9.2,
# pytest-codeblocks 0.18.0 and mktestdocs 0.2.5. None runs a page's blocks as one program in a
# fresh interpreter while maintained: Sybil and pytest-codeblocks `exec` each block in-process,
# pytest-markdown-docs re-runs every earlier block per item and needs a `continuation` mark on
# each later fence, and mktestdocs (which joins a page) last released 2025-07 and treats any line
# holding a fence marker as a boundary.
PYTHON_BLOCK = re.compile(
    r"^(?P<indent>[ \t]*)```python[^\n]*\n(?P<body>.*?)^(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def python_blocks(page: Path) -> list[str]:
    """Return the page's python blocks, each dedented by its fence's indentation."""
    return [
        textwrap.dedent(match.group("body"))
        for match in PYTHON_BLOCK.finditer(page.read_text(encoding="utf-8"))
    ]


def test_block_extractor_finds_indented_fences(tmp_path: Path) -> None:
    """A fence inside a list item or card is a block too, not skipped text."""
    sample = "- item\n\n    ```python\n    x = 1\n    ```\n\n```python\ny = x + 1\n```\n"
    page = tmp_path / "page.md"
    page.write_text(sample, encoding="utf-8")

    assert python_blocks(page) == ["x = 1\n", "y = x + 1\n"]


@pytest.mark.parametrize("page", THEORY_PAGES, ids=lambda page: page.stem)
def test_theory_page_python_blocks_run_in_page_order(page: Path) -> None:
    """Every python block on the page runs, after the blocks before it, without error."""
    blocks = python_blocks(page)
    assert blocks, f"{page.name} has no python block"

    result = run_repo_python("\n\n".join(blocks))

    assert result.returncode == 0, f"{page.name} failed:\n{result.stderr}"
