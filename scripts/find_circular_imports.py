#!/usr/bin/env python3
"""Detect circular imports in Python modules."""

import ast
import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("circular_import_finder")


def find_imports(file_path: str) -> set[str]:
    """Find all imports in a Python file."""
    with open(file_path, "r") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            logger.warning(f"Syntax error in {file_path}")
            return set()

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def main():
    """Find circular imports in the artifex package."""
    # Find all Python files in the artifex package
    artifex_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "artifex")
    )

    # Get all Python files
    python_files: list[str] = []
    for root, _, files in os.walk(artifex_dir):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    logger.info(f"Found {len(python_files)} Python files in the artifex package")

    # Map each module to its imports
    module_imports: dict[str, set[str]] = {}
    file_to_module: dict[str, str] = {}

    for file_path in python_files:
        relative_path = os.path.relpath(file_path, os.path.dirname(artifex_dir))
        module_name = os.path.splitext(relative_path)[0].replace(os.path.sep, ".")
        if module_name.startswith("src."):
            module_name = module_name[4:]  # Remove "src." prefix
        file_to_module[file_path] = module_name
        module_imports[module_name] = find_imports(file_path)

    # Find circular imports
    circular_imports: list[tuple[str, str]] = []

    for module, imports in module_imports.items():
        for imported_module in imports:
            if (
                imported_module in module_imports
                and module in module_imports[imported_module]
                and module != imported_module
            ):
                circular_imports.append((module, imported_module))

    # Remove duplicates
    unique_circular_imports = set()
    for a, b in circular_imports:
        if (b, a) not in unique_circular_imports:
            unique_circular_imports.add((a, b))

    # Write results to file
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs",
    )
    output_path = os.path.join(output_dir, "circular_imports.txt")
    with open(output_path, "w") as f:
        f.write("# Circular Import Analysis\n\n")
        if circular_imports:
            f.write("## Circular Imports Detected\n\n")
            for module_a, module_b in sorted(unique_circular_imports):
                f.write(f"- `{module_a}` imports `{module_b}` and vice versa\n")
        else:
            f.write("No circular imports detected in the artifex package.\n")

    logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
