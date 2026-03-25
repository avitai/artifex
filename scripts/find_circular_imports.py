#!/usr/bin/env python3
"""Report repo-local circular import groups from the full Python import graph."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from artifex.generative_models.utils.code_analysis.dependency_analyzer import ModuleDependency


DEFAULT_SOURCE_DIR = Path("src") / "artifex"
DEFAULT_OUTPUT_FILE = Path("test_artifacts") / "code_analysis" / "circular_imports.txt"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory containing the Python modules to analyze",
    )
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Ignored repo-local path for the circular-import report",
    )
    parser.add_argument(
        "--fail-on-cycles",
        action="store_true",
        help="Exit non-zero when one or more circular dependency groups are detected",
    )
    return parser.parse_args(argv)


def _render_report(source_dir: Path, cycles: Sequence[Sequence[ModuleDependency]]) -> str:
    """Render the circular import report."""
    lines = [
        "# Circular Import Analysis",
        "",
        f"Source directory: {source_dir.resolve()}",
        f"Detected {len(cycles)} circular dependency group(s).",
        "",
    ]

    if not cycles:
        lines.append("No circular dependency groups detected.")
        lines.append("")
        return "\n".join(lines)

    for index, cycle in enumerate(cycles, start=1):
        lines.append(f"## Group {index}")
        lines.append("")
        for dep in sorted(
            cycle,
            key=lambda item: ((item.source_module or item.source), item.target),
        ):
            source_module = dep.source_module or dep.source
            lines.append(f"- {source_module} -> {dep.target}")
        lines.append("")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the circular-import analysis CLI."""
    args = parse_args(argv)

    from artifex.generative_models.utils.code_analysis.dependency_analyzer import (
        DependencyAnalyzer,
        detect_circular_dependencies,
    )

    source_dir = Path(args.source_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    analyzer = DependencyAnalyzer(str(source_dir))
    cycles = detect_circular_dependencies(analyzer.get_all_dependencies())
    output_file.write_text(_render_report(source_dir, cycles), encoding="utf-8")

    sys.stdout.write(f"Analyzed {len(analyzer.modules)} Python modules in {source_dir.resolve()}\n")
    sys.stdout.write(f"Detected {len(cycles)} circular dependency group(s)\n")
    sys.stdout.write(f"Wrote circular import report to {output_file}\n")

    if args.fail_on_cycles and cycles:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
