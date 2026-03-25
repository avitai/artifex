#!/usr/bin/env python3
# ruff: noqa: T201
"""Analyze test-to-source coverage without writing into curated docs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifex.generative_models.utils.code_analysis.test_structure_analyzer import (
    TestStructureAnalyzer,
)


DEFAULT_OUTPUT_DIR = Path("test_artifacts/code_analysis/test_structure")


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Analyze repo-local test structure against the current source tree."
    )
    parser.add_argument("--test-dir", default="tests", help="Path to the test directory")
    parser.add_argument("--src-dir", default="src/artifex", help="Path to the source directory")
    parser.add_argument(
        "--package-prefix",
        default="artifex",
        help="Package prefix used for source-module imports",
    )
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_OUTPUT_DIR / "test_structure.md"),
        help="Ignored repo-local path for the markdown analysis report",
    )
    parser.add_argument(
        "--json-output",
        default=str(DEFAULT_OUTPUT_DIR / "test_analysis.json"),
        help="Ignored repo-local path for the JSON analysis payload",
    )
    return parser


def main() -> int:
    """Run the test-structure analysis CLI."""
    args = build_parser().parse_args()
    analyzer = TestStructureAnalyzer(
        test_dir=args.test_dir,
        src_dir=args.src_dir,
        package_prefix=args.package_prefix,
    )

    print("Analyzing test structure...")
    print(f"Test directory: {args.test_dir}")
    print(f"Source directory: {args.src_dir}")
    print(f"Package prefix: {args.package_prefix}")

    test_files = analyzer.discover_test_files()
    source_modules = analyzer.discover_source_modules()
    print(f"Found {len(test_files)} test files")
    print(f"Found {len(source_modules)} source modules")

    analysis = analyzer.analyze()
    print("\nAnalysis results:")
    print(f"Total test files: {analysis.total_test_files}")
    print(f"Total source modules: {analysis.total_source_modules}")
    print(f"Source modules with tests: {analysis.source_modules_with_tests}")
    print(f"Coverage percentage: {analysis.calculate_coverage_percentage():.1f}%")
    print(f"Test files without source imports: {len(analysis.test_files_without_source_imports)}")

    report_path = Path(args.output_file)
    _write_text(report_path, analysis.generate_report())
    print(f"\nSaved report to {report_path}")

    if args.json_output:
        json_path = Path(args.json_output)
        _write_json(
            json_path,
            {
                "total_test_files": analysis.total_test_files,
                "total_source_modules": analysis.total_source_modules,
                "source_modules_with_tests": analysis.source_modules_with_tests,
                "test_files_without_imports": analysis.test_files_without_imports,
                "module_test_mapping": analysis.module_test_mapping,
                "untested_modules": analysis.untested_modules,
                "test_files_without_source_imports": analysis.test_files_without_source_imports,
            },
        )
        print(f"Saved raw analysis data to {json_path}")

    print("\nRecommendations:")
    for index, recommendation in enumerate(analysis.generate_recommendations(), start=1):
        print(f"{index}. {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
