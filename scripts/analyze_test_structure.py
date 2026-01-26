#!/usr/bin/env python3
"""
Script to analyze the test structure of the artifex package.

This script uses the TestStructureAnalyzer to analyze the relationship between
test files and source modules, generate reports, and identify areas for
improvement.
"""

import argparse
import os
import sys
from pathlib import Path


# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from artifex.generative_models.utils.code_analysis.test_structure_analyzer import (
    TestStructureAnalyzer,
)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Analyze test structure of the artifex package.")
    parser.add_argument(
        "--test-dir",
        default="tests",
        help="Path to the test directory",
    )
    parser.add_argument(
        "--src-dir",
        default="src/artifex",
        help="Path to the source directory",
    )
    parser.add_argument(
        "--package-prefix",
        default="artifex",
        help="Package prefix for source modules",
    )
    parser.add_argument(
        "--output-file",
        default="docs/test_structure.md",
        help="Path to save the analysis report",
    )
    parser.add_argument(
        "--json-output",
        default="docs/test_analysis.json",
        help="Path to save the raw analysis data in JSON format",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Initialize analyzer
    analyzer = TestStructureAnalyzer(
        test_dir=args.test_dir, src_dir=args.src_dir, package_prefix=args.package_prefix
    )

    print("Analyzing test structure...")
    print(f"Test directory: {args.test_dir}")
    print(f"Source directory: {args.src_dir}")
    print(f"Package prefix: {args.package_prefix}")

    # Discover test files and source modules
    test_files = analyzer.discover_test_files()
    source_modules = analyzer.discover_source_modules()

    print(f"Found {len(test_files)} test files")
    print(f"Found {len(source_modules)} source modules")

    # Analyze test structure
    analysis = analyzer.analyze()

    print("\nAnalysis results:")
    print(f"Total test files: {analysis.total_test_files}")
    print(f"Total source modules: {analysis.total_source_modules}")
    print(f"Source modules with tests: {analysis.source_modules_with_tests}")
    print(f"Coverage percentage: {analysis.calculate_coverage_percentage():.1f}%")
    print(f"Test files without source imports: {len(analysis.test_files_without_source_imports)}")

    # Generate report
    report = analysis.generate_report()

    with open(args.output_file, "w") as f:
        f.write(report)

    print(f"\nSaved report to {args.output_file}")

    # Save raw analysis data as JSON
    if args.json_output:
        import json

        # Convert analysis to dict
        analysis_dict = {
            "total_test_files": analysis.total_test_files,
            "total_source_modules": analysis.total_source_modules,
            "source_modules_with_tests": analysis.source_modules_with_tests,
            "test_files_without_imports": analysis.test_files_without_imports,
            "module_test_mapping": analysis.module_test_mapping,
            "untested_modules": analysis.untested_modules,
            "test_files_without_source_imports": analysis.test_files_without_source_imports,
        }

        with open(args.json_output, "w") as f:
            json.dump(analysis_dict, f, indent=2)

        print(f"Saved raw analysis data to {args.json_output}")

    # Print recommendations
    print("\nRecommendations:")
    for i, recommendation in enumerate(analysis.generate_recommendations(), 1):
        print(f"{i}. {recommendation}")


if __name__ == "__main__":
    main()
