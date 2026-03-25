from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs"


def _flatten_nav(nav: list[object]) -> list[str]:
    paths: list[str] = []
    for item in nav:
        if isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str):
                    paths.append(value)
                elif isinstance(value, list):
                    paths.extend(_flatten_nav(value))
        elif isinstance(item, str):
            paths.append(item)
    return paths


def _load_mkdocs_config() -> dict[str, object]:
    with (REPO_ROOT / "mkdocs.yml").open() as handle:
        return yaml.load(handle, Loader=yaml.BaseLoader)


def _slugify(heading: str) -> str:
    normalized = heading.strip().lower()
    normalized = re.sub(r"`", "", normalized)
    normalized = re.sub(r"[^\w\s-]", "", normalized)
    normalized = re.sub(r"[-\s]+", "-", normalized)
    return normalized.strip("-")


@lru_cache(maxsize=None)
def _heading_slugs(relative_path: str) -> set[str]:
    contents = (DOCS_ROOT / relative_path).read_text()
    headings = re.findall(r"^#{1,6}\s+(.+)$", contents, flags=re.MULTILINE)
    return {_slugify(heading) for heading in headings}


def test_every_markdown_doc_is_in_mkdocs_nav() -> None:
    """Every shipped markdown page should be reachable from the docs navigation."""
    config = _load_mkdocs_config()
    nav_paths = set(_flatten_nav(config["nav"]))

    docs_pages = {
        str(path.relative_to(DOCS_ROOT)).replace("\\", "/") for path in DOCS_ROOT.rglob("*.md")
    }

    assert docs_pages - nav_paths == set()


def test_paired_notebooks_are_accounted_for_in_docs_contract() -> None:
    """Notebook source artifacts should be either excluded or intentionally navigable."""
    config = _load_mkdocs_config()
    exclude_docs = config.get("exclude_docs", "")
    nav_paths = set(_flatten_nav(config["nav"]))

    assert "*.ipynb" in exclude_docs
    assert "examples/templates/example_template.ipynb" in nav_paths
    assert "getting-started/quickstart.ipynb" in nav_paths


def test_markdown_links_and_heading_anchors_resolve() -> None:
    """All internal markdown links should point to real docs files and headings."""
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    pages = sorted(DOCS_ROOT.rglob("*.md"))

    for page in pages:
        contents = page.read_text()
        for target in link_pattern.findall(contents):
            if target.startswith(("http://", "https://", "mailto:")):
                continue

            if target.startswith("#"):
                anchor = target[1:]
                if not anchor:
                    continue
                assert anchor in _heading_slugs(str(page.relative_to(DOCS_ROOT)).replace("\\", "/"))
                continue

            relative_target, _, anchor = target.partition("#")
            if not relative_target.endswith(".md"):
                continue

            target_path = (page.parent / relative_target).resolve()
            assert target_path.exists(), f"{page.relative_to(DOCS_ROOT)} -> {target}"

            if anchor:
                relative = str(target_path.relative_to(DOCS_ROOT)).replace("\\", "/")
                assert anchor in _heading_slugs(relative), (
                    f"{page.relative_to(DOCS_ROOT)} -> {target}"
                )
