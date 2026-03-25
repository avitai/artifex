#!/usr/bin/env python3
# ruff: noqa: T201
"""Validate the curated docs tree, then build or serve it with MkDocs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_command(command: list[str], cwd: Path, description: str) -> int:
    """Run one CLI command and stream its output to the terminal."""
    print(description)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        print(f"Command failed: {' '.join(command)}")
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Validate curated docs, then build or serve them with MkDocs."
    )
    parser.add_argument(
        "--config-path",
        default="mkdocs.yml",
        help="Path to the MkDocs configuration file (default: mkdocs.yml)",
    )
    parser.add_argument(
        "--docs-path",
        default="docs",
        help="Path to the curated documentation tree (default: docs)",
    )
    parser.add_argument(
        "--src-path",
        default="src",
        help="Path to the Python source tree used for docs validation (default: src)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the validated docs after the strict build completes",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to use when serving documentation (default: 8000)",
    )
    return parser


def main() -> int:
    """Run the docs build workflow."""
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent.parent
    validator_script = project_root / "scripts" / "validate_docs.py"

    validation_command = [
        sys.executable,
        str(validator_script),
        "--config-path",
        args.config_path,
        "--docs-path",
        args.docs_path,
        "--src-path",
        args.src_path,
    ]
    if _run_command(validation_command, project_root, "Validating curated documentation...") != 0:
        return 1

    build_command = ["uv", "run", "mkdocs", "build", "--clean", "--strict", "-f", args.config_path]
    if _run_command(build_command, project_root, "Building documentation...") != 0:
        return 1

    if not args.serve:
        return 0

    serve_command = [
        "uv",
        "run",
        "mkdocs",
        "serve",
        "-f",
        args.config_path,
        "--dev-addr",
        f"0.0.0.0:{args.port}",
    ]
    return _run_command(serve_command, project_root, "Serving documentation...")


if __name__ == "__main__":
    raise SystemExit(main())
