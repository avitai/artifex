"""Utilities for file operations."""

import sys
from pathlib import Path


def ensure_valid_output_path(
    path: str, base_dir: str | None = None, create_dir: bool = True
) -> str:
    """Ensure the output path is within a valid directory.

    Args:
        path: The proposed output path
        base_dir: The base directory to use (test_results or benchmark_results)
        create_dir: Whether to create the directory if it doesn't exist

    Returns:
        A validated path within the appropriate directory
    """
    if base_dir is None:
        # Determine if this is a test or benchmark based on calling context and path
        frame = sys._getframe(1)
        caller_name = frame.f_code.co_name
        module_name = frame.f_globals.get("__name__", "")

        # Also check the path itself for hints
        path_str = str(path)

        if "test" in caller_name or "test" in module_name or "test" in path_str:
            base_dir = "test_results"
        else:
            base_dir = "benchmark_results"

    # Convert to Path objects for easier manipulation
    path_obj = Path(path)
    base_dir_obj = Path(base_dir)

    # If path is not already within base_dir, place it there
    if base_dir not in str(path_obj):
        # Simply append the entire path to the base_dir
        path_obj = base_dir_obj / path_obj

    # Create directory if needed
    if create_dir:
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    return str(path_obj)


def get_valid_output_dir(
    output_dir: str | None = None, base_dir: str | None = None, create_dir: bool = True
) -> str:
    """Get a valid output directory path.

    Args:
        output_dir: The proposed output directory
        base_dir: The base directory to use (test_results or benchmark_results)
        create_dir: Whether to create the directory if it doesn't exist

    Returns:
        A validated directory path within the appropriate base directory
    """
    if base_dir is None:
        # Determine if this is a test or benchmark based on calling context and path
        frame = sys._getframe(1)
        caller_name = frame.f_code.co_name
        module_name = frame.f_globals.get("__name__", "")

        # Also check the path itself for hints
        path_str = str(output_dir) if output_dir else ""

        if "test" in caller_name or "test" in module_name or "test" in path_str:
            base_dir = "test_results"
        else:
            base_dir = "benchmark_results"

    # If no output_dir provided, use base_dir
    if output_dir is None:
        output_dir = base_dir

    # Convert to Path objects for easier manipulation
    output_dir_obj = Path(output_dir)
    base_dir_obj = Path(base_dir)

    # If output_dir is not already within base_dir, place it there
    if base_dir not in str(output_dir_obj):
        output_dir_obj = base_dir_obj / output_dir_obj

    # Create directory if needed
    if create_dir:
        output_dir_obj.mkdir(parents=True, exist_ok=True)

    return str(output_dir_obj)
