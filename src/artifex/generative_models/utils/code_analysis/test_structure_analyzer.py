"""
Module for analyzing test structure and relationships to source modules.

This module provides tools for discovering, analyzing, and reporting
relationships between test files and source modules.
"""

import ast
import os
import re
from dataclasses import dataclass


@dataclass
class TestFileAnalysis:
    """
    Analysis results for test files and their relationships to source modules.
    """

    total_test_files: int
    """Total number of test files discovered."""

    total_source_modules: int
    """Total number of source modules discovered."""

    source_modules_with_tests: int
    """Number of source modules that have associated tests."""

    test_files_without_imports: int
    """Number of test files without any source module imports."""

    module_test_mapping: dict[str, list[str]]
    """Mapping from module names to test files that test them."""

    test_module_mapping: dict[str, list[str]]
    """Mapping from test files to modules they test."""

    untested_modules: list[str]
    """list of source modules that have no associated tests."""

    test_files_without_source_imports: list[str]
    """list of test files that don't import any source modules."""

    def calculate_coverage_percentage(self) -> float:
        """
        Calculate the percentage of source modules that have tests.

        Returns:
            Percentage of source modules with tests.
        """
        if self.total_source_modules == 0:
            return 0.0

        return (self.source_modules_with_tests / self.total_source_modules) * 100.0

    def generate_recommendations(self) -> list[str]:
        """
        Generate recommendations based on the analysis.

        Returns:
            list of recommendations.
        """
        recommendations = []

        # If coverage is low, prioritize creating tests
        if self.calculate_coverage_percentage() < 50.0:
            recommendations.append("Prioritize creating tests for untested modules")

        # If many test files don't import source modules, recommend fixing
        if self.test_files_without_imports > 0:
            if self.calculate_coverage_percentage() >= 50.0:
                # For good coverage but org issues, put import recommendation first
                recommendations.append(
                    "Ensure test files properly import their corresponding source modules"
                )
                recommendations.append("Standardize test file naming and organization")
            else:
                # For poor coverage, keep the standard order
                recommendations.append("Standardize test file naming and organization")
                recommendations.append(
                    "Ensure test files properly import their corresponding source modules"
                )

        # Always recommend improved organization
        recommendations.append("Organize tests to match the source module structure")

        # Always recommend fixtures for common setup
        recommendations.append("Create proper pytest fixtures for common setup operations")

        return recommendations

    def generate_report(self) -> str:
        """
        Generate a report of the test structure analysis.

        Returns:
            Formatted text report.
        """
        report = ["# Test Structure Analysis\n\n"]

        # Overview section
        report.append("## Overview\n\n")
        report.append(f"- Total test files: {self.total_test_files}\n")
        report.append(f"- Total source modules: {self.total_source_modules}\n")
        report.append(f"- Source modules with tests: {self.source_modules_with_tests}\n")
        report.append(f"- Coverage percentage: {self.calculate_coverage_percentage():.1f}%\n")
        report.append(
            f"- Test files without source imports: "
            f"{len(self.test_files_without_source_imports)}\n\n"
        )

        # Modules with tests
        report.append("## Modules With Tests\n\n")
        report.append("| Module | Test Files |\n")
        report.append("|--------|------------|\n")

        for module, test_files in sorted(self.module_test_mapping.items()):
            report.append(f"| {module} | {len(test_files)} |\n")

        # Untested modules
        if self.untested_modules:
            report.append("\n## Untested Modules\n\n")

            for module in sorted(self.untested_modules)[:20]:
                report.append(f"- {module}\n")

            if len(self.untested_modules) > 20:
                report.append(f"\n... and {len(self.untested_modules) - 20} more.\n")

        # Test files without source imports
        if self.test_files_without_source_imports:
            report.append("\n## Test Files Without Source Imports\n\n")

            for test_file in sorted(self.test_files_without_source_imports)[:20]:
                report.append(f"- {test_file}\n")

            if len(self.test_files_without_source_imports) > 20:
                report.append(
                    f"\n... and {len(self.test_files_without_source_imports) - 20} more.\n"
                )

        # Recommendations
        report.append("\n## Recommendations\n\n")

        for recommendation in self.generate_recommendations():
            report.append(f"- {recommendation}\n")

        return "".join(report)


def is_test_file(file_path: str) -> bool:
    """
    Check if a file is a test file based on naming patterns.

    Args:
        file_path: Path to the file.

    Returns:
        True if the file is a test file, False otherwise.
    """
    patterns = [
        r"^test_.*\.py$",
        r"^.*_test\.py$",
    ]

    file_name = os.path.basename(file_path)

    for pattern in patterns:
        if re.match(pattern, file_name):
            return True

    return False


def extract_imports(file_path: str) -> list[str]:
    """
    Extract imports from a Python file.

    Args:
        file_path: Path to the Python file.

    Returns:
        list of imported module names.
    """
    imports = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for name in node.names:
                        imports.append(f"{node.module}.{name.name}")
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

    return imports


def find_source_module_mappings(
    test_dir: str, src_dir: str, package_prefix: str
) -> dict[str, list[str]]:
    """
    Find mappings between test files and source modules.

    Args:
        test_dir: Path to the test directory.
        src_dir: Path to the source directory.
        package_prefix: Package prefix for source modules.

    Returns:
        Mapping from test files to lists of source modules they test.
    """
    test_files = []

    # Find all Python files in the test directory
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if is_test_file(file_path):
                    test_files.append(file_path)

    # Find source modules
    source_modules = []

    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, src_dir)
                module_path = (
                    package_prefix + "." + os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
                )
                source_modules.append((file_path, module_path))

    # Map test files to source modules
    test_to_source: dict[str, list[str]] = {}

    for test_file in test_files:
        imports = extract_imports(test_file)
        module_matches: list[str] = []

        for imp in imports:
            if imp.startswith(package_prefix):
                for _, module_path in source_modules:
                    if imp == module_path or imp.startswith(module_path + "."):
                        if module_path not in module_matches:
                            module_matches.append(module_path)

        if module_matches:
            test_to_source[test_file] = module_matches

    return test_to_source


class TestStructureAnalyzer:
    """
    Analyzes the structure of test files and their relationships to source modules.
    """

    def __init__(self, test_dir: str, src_dir: str, package_prefix: str):
        """
        Initialize the analyzer.

        Args:
            test_dir: Path to the test directory.
            src_dir: Path to the source directory.
            package_prefix: Package prefix for source modules.
        """
        self.test_dir = test_dir
        self.src_dir = src_dir
        self.package_prefix = package_prefix

    def discover_test_files(self) -> list[str]:
        """
        Discover all test files in the test directory.

        Returns:
            list of paths to test files.
        """
        test_files = []

        for root, _, files in os.walk(self.test_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    if is_test_file(file_path):
                        test_files.append(file_path)

        return test_files

    def discover_source_modules(self) -> list[tuple[str, str]]:
        """
        Discover all source modules in the source directory.

        Returns:
            list of tuples (file_path, module_path).
        """
        source_modules: list[tuple[str, str]] = []

        for root, _, files in os.walk(self.src_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.src_dir)
                    module_path = (
                        self.package_prefix
                        + "."
                        + os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
                    )
                    source_modules.append((file_path, module_path))

        return source_modules

    def analyze(self) -> TestFileAnalysis:
        """
        Analyze the test structure.

        Returns:
            TestFileAnalysis object with the analysis results.
        """
        # Discover test files and source modules
        test_files = self.discover_test_files()
        source_modules = self.discover_source_modules()

        # Build mappings
        test_to_source: dict[str, list[str]] = find_source_module_mappings(
            self.test_dir, self.src_dir, self.package_prefix
        )

        # Build reverse mapping (module to test files)
        module_to_test: dict[str, list[str]] = {}

        for test_file, modules in test_to_source.items():
            for module in modules:
                if module not in module_to_test:
                    module_to_test[module] = []
                module_to_test[module].append(test_file)

        # Find untested modules
        module_paths = [module_path for _, module_path in source_modules]
        untested_modules = [
            module_path for module_path in module_paths if module_path not in module_to_test
        ]

        # Find test files without source imports
        test_files_without_imports: list[str] = [
            test_file for test_file in test_files if test_file not in test_to_source
        ]

        # Create analysis result
        return TestFileAnalysis(
            total_test_files=len(test_files),
            total_source_modules=len(source_modules),
            source_modules_with_tests=len(module_to_test),
            test_files_without_imports=len(test_files_without_imports),
            module_test_mapping=module_to_test,
            test_module_mapping=test_to_source,
            untested_modules=untested_modules,
            test_files_without_source_imports=test_files_without_imports,
        )
