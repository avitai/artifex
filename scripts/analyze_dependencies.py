#!/usr/bin/env python3
"""Analyze repo-local Python import relationships with truthful module identities."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("test_artifacts") / "code_analysis" / "dependency_analysis"
DEFAULT_SOURCE_DIR = Path("src") / "artifex"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory containing the Python modules to analyze",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Ignored repo-local directory for rendered analysis artifacts",
    )
    parser.add_argument(
        "--report-file",
        default=None,
        help="Optional explicit path for the markdown report",
    )
    parser.add_argument(
        "--graph-format",
        default="svg",
        help="Graphviz output format for the rendered dependency graph",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dependency analyzer CLI."""
    args = parse_args(argv)

    from artifex.generative_models.utils.code_analysis.dependency_analyzer import (
        DependencyAnalyzer,
        detect_circular_dependencies,
        generate_dependency_report,
    )

    analyzer = DependencyAnalyzer(args.source_dir)
    dependencies = analyzer.get_all_dependencies()
    circular_dependencies = detect_circular_dependencies(dependencies)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_basename = output_dir / "dependencies"
    analyzer.generate_graph(str(graph_basename), format=args.graph_format)
    graph_path = graph_basename.with_suffix(f".{args.graph_format}")

    report_path = (
        Path(args.report_file) if args.report_file else output_dir / "dependency_report.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        generate_dependency_report(dependencies, circular_dependencies),
        encoding="utf-8",
    )

    sys.stdout.write(
        f"Analyzed {len(analyzer.modules)} Python modules in {Path(args.source_dir).resolve()}\n"
    )
    sys.stdout.write(f"Found {len(dependencies)} import edges\n")
    sys.stdout.write(f"Detected {len(circular_dependencies)} circular dependency group(s)\n")
    sys.stdout.write(f"Wrote dependency graph to {graph_path}\n")
    sys.stdout.write(f"Wrote dependency report to {report_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
