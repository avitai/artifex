"""Command-line interface for the test discovery module."""

import argparse
import os
import subprocess
import sys

from tests.utils.discovery import (
    find_all_test_files,
    find_tests_by_feature,
    find_tests_by_module,
    find_tests_by_pattern,
    find_tests_by_type,
    get_pytest_args_for_tests,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the test discovery CLI.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Test discovery and execution helper.")

    # Add arguments for different test discovery methods
    discovery_group = parser.add_mutually_exclusive_group(required=True)
    discovery_group.add_argument("-a", "--all", action="store_true", help="Run all tests")
    discovery_group.add_argument(
        "-m",
        "--module",
        type=str,
        help="Run tests for a specific module (e.g., artifex.generative_models.core)",
    )
    discovery_group.add_argument(
        "-p",
        "--pattern",
        type=str,
        help="Run tests matching a pattern (e.g., 'config' or 'transformer')",
    )
    discovery_group.add_argument(
        "-f",
        "--feature",
        type=str,
        help="Run tests for a specific feature (e.g., 'vae' or 'attention')",
    )
    discovery_group.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["unit", "integration", "functional", "benchmark", "e2e"],
        help="Run tests of a specific type (unit, integration, etc.)",
    )

    # Add options for controlling test execution
    parser.add_argument("-v", "--verbose", action="store_true", help="Run tests in verbose mode")
    parser.add_argument("-s", "--no-capture", action="store_true", help="Don't capture test output")
    parser.add_argument(
        "-d", "--dry-run", action="store_true", help="Show test files without running them"
    )
    parser.add_argument(
        "--test-dir", type=str, default="tests", help="Directory containing tests (default: tests)"
    )

    # Add options for test output format
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for test artifacts (default: test_artifacts)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show the test summary, not individual test results",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Don't generate HTML test reports",
    )

    return parser.parse_args()


def get_test_files(args: argparse.Namespace) -> list[str]:
    """Get test files based on command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        list of test files to run
    """
    test_dir = args.test_dir

    if args.all:
        return find_all_test_files(test_dir)
    elif args.module:
        return find_tests_by_module(args.module, test_dir)
    elif args.pattern:
        return find_tests_by_pattern(args.pattern, test_dir)
    elif args.feature:
        return find_tests_by_feature(args.feature, test_dir)
    elif args.type:
        return find_tests_by_type(args.type, test_dir)
    else:
        return []


def run_tests(test_files: list[str], args: argparse.Namespace) -> int:
    """Run tests with pytest.

    Args:
        test_files: list of test files to run
        args: Parsed command-line arguments

    Returns:
        Exit code from pytest
    """
    if not test_files:
        print("No tests found.")
        return 0

    # Get pytest arguments
    pytest_args = get_pytest_args_for_tests(
        test_files, verbose=args.verbose, capture_output=not args.no_capture
    )

    # Add output format options
    if args.output_dir:
        os.environ["PYTEST_ARTIFACTS_DIR"] = args.output_dir

    if args.summary_only:
        # Skip individual test output when using summary
        pytest_args.insert(0, "-q")

    if args.no_html:
        os.environ["PYTEST_NO_HTML"] = "1"

    # Execute pytest with the given arguments
    cmd = [sys.executable, "-m", "pytest", *pytest_args]
    result = subprocess.run(cmd)

    return result.returncode


def main() -> int:
    """Main entry point for the test discovery CLI.

    Returns:
        Exit code
    """
    args = parse_args()

    # Get test files based on command-line arguments
    test_files = get_test_files(args)

    print(f"Found {len(test_files)} test(s):")
    for test_file in test_files:
        print(f"  {test_file}")
    print()

    # If it's a dry run, don't actually run the tests
    if args.dry_run:
        return 0

    # Run the tests with pytest
    return run_tests(test_files, args)


if __name__ == "__main__":
    sys.exit(main())
