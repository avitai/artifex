#!/usr/bin/env python3
"""Validate the curated MkDocs contract without silently rewriting it."""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

import yaml


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)
MKDOCSTRINGS_DIRECTIVE = re.compile(r"^::: ([\w\.]+)$", flags=re.MULTILINE)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One unresolved docs-validation failure."""

    issue_type: str
    message: str
    file: str | None = None
    target: str | None = None


class DocValidator:
    """Validate the repo's curated documentation surface."""

    def __init__(
        self,
        *,
        config_path: Path,
        docs_path: Path,
        src_path: Path,
        fix: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize the validator for one curated MkDocs surface."""
        self.config_path = config_path
        self.docs_path = docs_path
        self.src_path = src_path
        self.fix_mode = fix
        self.verbose = verbose
        self.issues: list[ValidationIssue] = []
        self.applied_fixes: list[str] = []
        self._mkdocs_config: dict[str, object] | None = None
        self.module_map = self._discover_modules()

    def _discover_modules(self) -> set[str]:
        """Discover Python module names under the configured source tree."""
        LOGGER.info("Discovering Python modules...")
        if not self.src_path.exists():
            return set()

        modules: set[str] = set()
        for python_file in self.src_path.rglob("*.py"):
            if "__pycache__" in python_file.parts:
                continue

            relative = python_file.relative_to(self.src_path).with_suffix("")
            parts = list(relative.parts)
            if parts and parts[-1] == "__init__":
                parts = parts[:-1]
            if not parts:
                continue
            modules.add(".".join(parts))

        if self.verbose:
            LOGGER.info("Discovered %s modules", len(modules))
        return modules

    def _load_mkdocs_config(self) -> dict[str, object]:
        """Load the configured MkDocs file without evaluating Python tags."""
        if self._mkdocs_config is None:
            with self.config_path.open(encoding="utf-8") as handle:
                loaded = yaml.load(handle, Loader=yaml.BaseLoader)
            if isinstance(loaded, dict):
                self._mkdocs_config = cast(dict[str, object], loaded)
            else:
                self._mkdocs_config = {}
        return self._mkdocs_config

    def _add_issue(
        self,
        issue_type: str,
        message: str,
        *,
        file: str | None = None,
        target: str | None = None,
    ) -> None:
        LOGGER.warning(message)
        self.issues.append(ValidationIssue(issue_type, message, file=file, target=target))

    def _check_module_exists(self, module_name: str) -> bool:
        """Check whether a fully qualified Python module exists."""
        for candidate in self._module_candidates(module_name):
            if candidate in self.module_map:
                return True
            try:
                if importlib.util.find_spec(candidate) is not None:
                    return True
            except (AttributeError, ModuleNotFoundError, ValueError):
                continue
        return False

    @staticmethod
    def _module_candidates(module_name: str) -> list[str]:
        """Return the module name plus progressively shorter dotted prefixes."""
        candidates = [module_name]
        while "." in module_name:
            module_name = module_name.rsplit(".", 1)[0]
            candidates.append(module_name)
        return candidates

    def _excluded_doc_patterns(self) -> list[str]:
        """Return docs patterns excluded from the built site."""
        config = self._load_mkdocs_config()
        patterns: list[str] = []
        exclude_docs = config.get("exclude_docs", "")
        if isinstance(exclude_docs, str):
            patterns.extend(line.strip() for line in exclude_docs.splitlines() if line.strip())

        plugins = config.get("plugins", [])
        if isinstance(plugins, list):
            for plugin in plugins:
                if not isinstance(plugin, dict):
                    continue
                include_exclude = plugin.get("include_exclude_files")
                if isinstance(include_exclude, dict):
                    exclude = include_exclude.get("exclude", [])
                    if isinstance(exclude, list):
                        patterns.extend(str(pattern) for pattern in exclude)
                    break
        return patterns

    def _iter_doc_pages(self) -> list[Path]:
        """Return all built docs pages under the configured docs tree."""
        pages: list[Path] = []
        excluded_patterns = self._excluded_doc_patterns()
        for page in self.docs_path.rglob("*"):
            if not page.is_file():
                continue
            if page.suffix not in {".md", ".ipynb"}:
                continue
            relative = str(page.relative_to(self.docs_path)).replace("\\", "/")
            if any(fnmatch.fnmatch(relative, pattern) for pattern in excluded_patterns):
                continue
            pages.append(page)
        return pages

    @staticmethod
    def _flatten_nav(entries: list[object]) -> set[str]:
        """Flatten a MkDocs nav tree into a set of docs-relative file paths."""
        paths: set[str] = set()
        for item in entries:
            if isinstance(item, dict):
                for value in item.values():
                    if isinstance(value, str):
                        paths.add(value)
                    elif isinstance(value, list):
                        paths.update(DocValidator._flatten_nav(value))
            elif isinstance(item, str):
                paths.add(item)
        return paths

    @staticmethod
    def _slugify(heading: str) -> str:
        """Generate the slug used by MkDocs-style markdown anchors.

        Mirrors mkdocs' default slugifier: lowercases ASCII, drops non-ASCII
        characters (e.g. Greek letters in headings), and collapses dashes /
        whitespace into single hyphens. A heading like ``β-TCVAE and ...``
        therefore renders an anchor with a leading hyphen, which the
        validator must match exactly.
        """
        normalized = heading.strip().lower().replace("`", "")
        normalized = re.sub(r"[^a-z0-9_\s-]", "", normalized)
        normalized = re.sub(r"[-\s]+", "-", normalized)
        return normalized.strip("-") if not normalized.startswith("-") else normalized.rstrip("-")

    @lru_cache(maxsize=None)
    def _heading_slugs(self, relative_path: str) -> set[str]:
        """Collect all anchor slugs for one docs page.

        Picks up both markdown headings (rendered as auto-generated slugs)
        and explicit HTML anchors of the form ``<a id="..."></a>`` /
        ``<a name="..."></a>`` (used for citation back-links).
        """
        contents = (self.docs_path / relative_path).read_text(encoding="utf-8")
        headings = re.findall(r"^#{1,6}\s+(.+)$", contents, flags=re.MULTILINE)
        slugs = {self._slugify(heading) for heading in headings}
        explicit_anchors = re.findall(
            r"<a\s+(?:id|name)\s*=\s*[\"']([^\"']+)[\"']", contents, flags=re.IGNORECASE
        )
        slugs.update(explicit_anchors)
        return slugs

    def _relative_to_docs(self, path: Path) -> str:
        """Convert a docs file path into a docs-relative string."""
        return str(path.relative_to(self.docs_path)).replace("\\", "/")

    def validate_config(self) -> None:
        """Validate the MkDocs config and nav structure."""
        LOGGER.info("Validating MkDocs configuration...")
        config = self._load_mkdocs_config()

        theme = config.get("theme")
        if isinstance(theme, dict):
            custom_dir = theme.get("custom_dir")
            if isinstance(custom_dir, str):
                resolved = (self.config_path.parent / custom_dir).resolve()
                if not resolved.exists():
                    if self.fix_mode:
                        resolved.mkdir(parents=True, exist_ok=True)
                        self.applied_fixes.append(f"Created theme custom_dir: {resolved}")
                    else:
                        self._add_issue(
                            "missing_theme_custom_dir",
                            f"Theme custom_dir does not exist: {custom_dir}",
                            file=custom_dir,
                        )

        nav = config.get("nav")
        if not isinstance(nav, list):
            self._add_issue("missing_nav", "MkDocs configuration does not define a valid nav list.")
            return

        LOGGER.info("Validating navigation structure...")
        self._validate_nav_structure(nav)
        self.validate_nav_coverage(nav)

    def _validate_nav_structure(self, nav: list[object]) -> None:
        """Validate that every nav entry points at a real curated docs page."""
        for item in nav:
            if not isinstance(item, dict):
                continue
            for path_or_subnav in item.values():
                if isinstance(path_or_subnav, str):
                    doc_path = self.docs_path / path_or_subnav
                    if not doc_path.exists():
                        self._add_issue(
                            "missing_nav_file",
                            f"Navigation references missing file: {path_or_subnav}",
                            file=path_or_subnav,
                        )
                elif isinstance(path_or_subnav, list):
                    self._validate_nav_structure(path_or_subnav)

    def validate_nav_coverage(self, nav: list[object]) -> None:
        """Validate that every built page is reachable from navigation."""
        LOGGER.info("Validating navigation coverage...")
        nav_paths = self._flatten_nav(nav)
        for page in self._iter_doc_pages():
            relative = self._relative_to_docs(page)
            if relative not in nav_paths:
                self._add_issue(
                    "orphan_doc",
                    f"Documentation page is not included in nav: {relative}",
                    file=relative,
                )

    def validate_internal_links(self) -> None:
        """Validate internal markdown links and heading anchors."""
        LOGGER.info("Validating internal links...")
        docs_root = self.docs_path.resolve()
        for markdown_file in [page for page in self._iter_doc_pages() if page.suffix == ".md"]:
            contents = self._strip_non_link_regions(markdown_file.read_text(encoding="utf-8"))
            relative_markdown = self._relative_to_docs(markdown_file)
            for target in MARKDOWN_LINK.findall(contents):
                if target.startswith(("http://", "https://", "mailto:")):
                    continue

                if target.startswith("#"):
                    anchor = target[1:]
                    if anchor and anchor not in self._heading_slugs(relative_markdown):
                        self._add_issue(
                            "broken_anchor",
                            f"Broken anchor in {relative_markdown}: {target}",
                            file=relative_markdown,
                            target=target,
                        )
                    continue

                relative_target, _, anchor = target.partition("#")
                if not relative_target:
                    continue

                target_path = (markdown_file.parent / relative_target).resolve()
                if docs_root not in target_path.parents and target_path != docs_root:
                    self._add_issue(
                        "escaped_docs_link",
                        f"Relative link escapes the docs tree in {relative_markdown}: {target}",
                        file=relative_markdown,
                        target=target,
                    )
                    continue

                if not target_path.exists():
                    self._add_issue(
                        "broken_link",
                        f"Broken link in {relative_markdown}: {target}",
                        file=relative_markdown,
                        target=target,
                    )
                    continue

                if not relative_target.endswith(".md"):
                    continue

                if not anchor:
                    continue
                target_relative = str(target_path.relative_to(self.docs_path.resolve())).replace(
                    "\\", "/"
                )
                if anchor not in self._heading_slugs(target_relative):
                    self._add_issue(
                        "broken_anchor",
                        f"Broken anchor in {relative_markdown}: {target}",
                        file=relative_markdown,
                        target=target,
                    )

    @staticmethod
    def _strip_non_link_regions(contents: str) -> str:
        """Remove fenced code, inline code, and math regions before link scanning."""
        stripped = re.sub(r"```.*?```", "", contents, flags=re.DOTALL)
        stripped = re.sub(r"\$\$.*?\$\$", "", stripped, flags=re.DOTALL)
        stripped = re.sub(r"`[^`]*`", "", stripped)
        return re.sub(r"(?<!\$)\$[^$\n]+\$(?!\$)", "", stripped)

    def validate_mkdocstrings_modules(self) -> None:
        """Validate mkdocstrings module directives against the configured source tree."""
        LOGGER.info("Validating mkdocstrings module references...")
        for markdown_file in [page for page in self._iter_doc_pages() if page.suffix == ".md"]:
            contents = markdown_file.read_text(encoding="utf-8")
            relative_markdown = self._relative_to_docs(markdown_file)
            for module_name in MKDOCSTRINGS_DIRECTIVE.findall(contents):
                if not self._check_module_exists(module_name):
                    self._add_issue(
                        "missing_module",
                        f"mkdocstrings reference does not resolve: {module_name}",
                        file=relative_markdown,
                        target=module_name,
                    )

    def report(self) -> None:
        """Log a summary of unresolved issues and applied fixes."""
        LOGGER.info("=" * 50)
        LOGGER.info("Documentation Validation Summary")
        LOGGER.info("=" * 50)

        if not self.issues:
            LOGGER.info("No issues found. Documentation contract is valid.")
        else:
            grouped: dict[str, list[ValidationIssue]] = defaultdict(list)
            for issue in self.issues:
                grouped[issue.issue_type].append(issue)

            LOGGER.info("Found %s unresolved issue(s).", len(self.issues))
            for issue_type, issues in grouped.items():
                LOGGER.info("%s: %s", issue_type.replace("_", " "), len(issues))
                if self.verbose:
                    for issue in issues:
                        LOGGER.info("  - %s", issue.message)

        if self.applied_fixes:
            LOGGER.info("Applied %s safe fix(es).", len(self.applied_fixes))
            for fix in self.applied_fixes:
                LOGGER.info("  - %s", fix)

    def run(self) -> int:
        """Run the full docs validation suite."""
        LOGGER.info("Starting documentation validation...")
        if not self.config_path.exists():
            self._add_issue(
                "missing_config",
                f"MkDocs configuration does not exist: {self.config_path}",
                file=str(self.config_path),
            )
        if not self.docs_path.exists():
            self._add_issue(
                "missing_docs_dir",
                f"Docs directory does not exist: {self.docs_path}",
                file=str(self.docs_path),
            )
        if not self.src_path.exists():
            self._add_issue(
                "missing_src_dir",
                f"Source directory does not exist: {self.src_path}",
                file=str(self.src_path),
            )

        if not self.issues:
            self.validate_config()
            self.validate_mkdocstrings_modules()
            self.validate_internal_links()

        self.report()
        return 0 if not self.issues else 1


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="Validate the curated MkDocs documentation tree.")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply safe non-content fixes when possible",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Disable fix mode explicitly and report unresolved issues only",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed issue output")
    parser.add_argument(
        "--config-path",
        default="mkdocs.yml",
        help="Path to the MkDocs configuration file (default: mkdocs.yml)",
    )
    parser.add_argument(
        "--docs-path",
        default="docs",
        help="Path to the curated documentation tree (default: docs)",
    )
    parser.add_argument(
        "--src-path",
        default="src",
        help="Path to the Python source tree used for mkdocstrings checks (default: src)",
    )
    return parser


def main() -> int:
    """Run the docs validator CLI."""
    args = build_parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    validator = DocValidator(
        config_path=Path(args.config_path),
        docs_path=Path(args.docs_path),
        src_path=Path(args.src_path),
        fix=args.fix and not args.check_only,
        verbose=args.verbose,
    )
    return validator.run()


if __name__ == "__main__":
    raise SystemExit(main())
