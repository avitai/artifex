"""Public documentation hygiene contracts."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

PUBLIC_AI_ASSISTANT_PATTERN = re.compile(r"\b(?:Anthropic|Claude|ChatGPT|Codex|Copilot)\b")
CONTEXTUAL_MODEL_OR_CITATION_PATTERN = re.compile(r"\b(?:OpenAI|Gemini)\b")
ALLOWED_CONTEXTUAL_MODEL_FILES = {
    Path("docs/user-guide/concepts/autoregressive-explained.md"),
    Path("docs/user-guide/concepts/diffusion-explained.md"),
    Path("docs/user-guide/concepts/generative-models-unified.md"),
}


def _public_text_files() -> list[Path]:
    roots = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in {".md", ".py", ".toml", ".yml", ".yaml"}
            and "memory-bank" not in path.parts
        )
    return sorted(files)


def test_public_docs_do_not_name_ai_assistants_or_coding_products() -> None:
    """Public docs should avoid assistant/vendor names unless explicitly allowed."""
    violations: list[str] = []
    for path in _public_text_files():
        relative = path.relative_to(REPO_ROOT)
        text = path.read_text(encoding="utf-8")
        if PUBLIC_AI_ASSISTANT_PATTERN.search(text):
            violations.append(str(relative))
        if (
            CONTEXTUAL_MODEL_OR_CITATION_PATTERN.search(text)
            and relative not in ALLOWED_CONTEXTUAL_MODEL_FILES
        ):
            violations.append(str(relative))

    assert violations == []
