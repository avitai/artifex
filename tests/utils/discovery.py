"""Test discovery utilities for the artifex package.

This module provides functions to discover and filter tests based on different criteria,
such as module name, feature, or test pattern.
"""

import fnmatch
import os
import pathlib


def find_all_test_files(test_dir: str = "tests") -> list[str]:
    """Find all test files in the given directory.

    Args:
        test_dir: The directory to search for test files

    Returns:
        A list of paths to test files
    """
    test_files = []
    test_path = pathlib.Path(test_dir)

    for path in test_path.glob("**/*.py"):
        # Skip __init__.py files
        if path.name == "__init__.py":
            continue

        # Only include files that start with test_ or end with _test
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            test_files.append(str(path))

    return sorted(test_files)


def find_tests_by_module(module_name: str, test_dir: str = "tests") -> list[str]:
    """Find test files related to a specific module.

    Args:
        module_name: The name of the module to find tests for
        test_dir: The directory to search for test files

    Returns:
        A list of paths to test files for the specified module
    """
    # Get all test files
    all_tests = find_all_test_files(test_dir)

    # Convert module name to a path pattern
    module_parts = module_name.split(".")
    path_pattern = os.path.join("**", *module_parts)

    # Filter test files by module pattern
    module_tests = []
    for test_file in all_tests:
        test_path = pathlib.Path(test_file)

        # Check if the file path contains the module path pattern
        if fnmatch.fnmatch(str(test_path), f"*{path_pattern}*"):
            module_tests.append(test_file)

        # Also check file content for imports of the module
        try:
            with open(test_file, "r") as f:
                content = f.read()
                if f"import {module_name}" in content or f"from {module_name}" in content:
                    if test_file not in module_tests:
                        module_tests.append(test_file)
        except Exception:
            # Skip files that can't be read
            pass

    return sorted(module_tests)


def find_tests_by_pattern(pattern: str, test_dir: str = "tests") -> list[str]:
    """Find test files matching a specific pattern.

    Args:
        pattern: The pattern to match test files against
        test_dir: The directory to search for test files

    Returns:
        A list of paths to test files matching the pattern
    """
    all_tests = find_all_test_files(test_dir)

    matching_tests = []
    for test_file in all_tests:
        if fnmatch.fnmatch(test_file, f"*{pattern}*"):
            matching_tests.append(test_file)

    return sorted(matching_tests)


def find_tests_by_feature(feature: str, test_dir: str = "tests") -> list[str]:
    """Find test files related to a specific feature.

    Args:
        feature: The feature name to find tests for
        test_dir: The directory to search for test files

    Returns:
        A list of paths to test files for the specified feature
    """
    # Get all test files
    all_tests = find_all_test_files(test_dir)

    # Filter test files by feature name in path or content
    feature_tests = []
    for test_file in all_tests:
        # Check if feature is in the path
        if feature.lower() in test_file.lower():
            feature_tests.append(test_file)
            continue

        # Check file content for feature name
        try:
            with open(test_file, "r") as f:
                content = f.read().lower()
                if feature.lower() in content:
                    feature_tests.append(test_file)
        except Exception:
            # Skip files that can't be read
            pass

    return sorted(feature_tests)


def find_tests_by_type(test_type: str, test_dir: str = "tests") -> list[str]:
    """Find test files of a specific type (unit, integration, etc.).

    Args:
        test_type: The type of tests to find (unit, integration, etc.)
        test_dir: The directory to search for test files

    Returns:
        A list of paths to test files of the specified type
    """
    # Define test type markers
    type_markers = {
        "unit": ["unit", "unittest"],
        "integration": ["integration", "integrationtest"],
        "functional": ["functional", "functionaltest"],
        "benchmark": ["benchmark", "benchmarktest"],
        "e2e": ["e2e", "end2end", "endtoend"],
    }

    if test_type not in type_markers:
        raise ValueError(
            f"Unknown test type: {test_type}. Supported types: {', '.join(type_markers.keys())}"
        )

    # Get all test files
    all_tests = find_all_test_files(test_dir)

    # Filter test files by type markers in path or content
    markers = type_markers[test_type]
    type_tests = []

    for test_file in all_tests:
        # Check if any marker is in the path
        path_contains_marker = any(marker in test_file.lower() for marker in markers)
        if path_contains_marker:
            type_tests.append(test_file)
            continue

        # Check file content for type markers
        try:
            with open(test_file, "r") as f:
                content = f.read().lower()
                content_contains_marker = any(
                    f"@{marker}" in content
                    or f"# {marker}" in content
                    or f"_{marker}_" in content
                    or f"{marker}_test" in content
                    for marker in markers
                )
                if content_contains_marker:
                    type_tests.append(test_file)
        except Exception:
            # Skip files that can't be read
            pass

    return sorted(type_tests)


def get_pytest_args_for_tests(
    test_files: list[str], verbose: bool = False, capture_output: bool = True
) -> list[str]:
    """Generate pytest command arguments for the given test files.

    Args:
        test_files: list of test files to run
        verbose: Whether to run in verbose mode
        capture_output: Whether to capture test output

    Returns:
        A list of pytest command arguments
    """
    args = []

    # Add verbosity flag
    if verbose:
        args.append("-v")

    # Add no-capture flag if capture_output is False
    if not capture_output:
        args.append("-s")

    # Add test files
    args.extend(test_files)

    return args
