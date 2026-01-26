#!/usr/bin/env python3
"""
Fix imports in test files to use proper package imports without src prefix.

This script scans all test files for imports that use 'src.artifex' and
replaces them with 'artifex' imports. It does not modify system path
manipulations or other code.
"""

import re
from pathlib import Path


# Regex pattern to match imports using src.artifex
SRC_IMPORT_PATTERN = re.compile(r"(from|import)\s+src\.artifex")


def fix_imports_in_file(file_path):
    """Fix imports in a single file by replacing src.artifex with artifex."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if the file has src.artifex imports
    if not SRC_IMPORT_PATTERN.search(content):
        return False

    # Replace src.artifex with artifex in imports
    modified_content = SRC_IMPORT_PATTERN.sub(r"\1 artifex", content)

    # Write the modified content back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(modified_content)

    return True


def find_and_fix_test_files(base_dir="tests"):
    """Find all test files and fix their imports."""
    test_dir = Path(base_dir)
    fixed_files = []

    # Look for Python files in test directory
    for file_path in test_dir.glob("**/*.py"):
        if fix_imports_in_file(file_path):
            fixed_files.append(str(file_path))

    return fixed_files


if __name__ == "__main__":
    print("Fixing imports in test files...")
    fixed_files = find_and_fix_test_files()

    print(f"Fixed imports in {len(fixed_files)} files:")
    for file in fixed_files:
        print(f"  - {file}")
