#!/bin/bash
# Cache and Temporary Files Cleanup Script
# =========================================
#
# PURPOSE:
#   Removes all cache files, build artifacts, and temporary files from the
#   Artifex repository while preserving important configuration and data.
#
# USAGE:
#   ./scripts/clean_cache.sh
#
# WHAT IT CLEANS:
#   - Python bytecode cache (__pycache__, *.pyc, *.pyo)
#   - Testing artifacts (.pytest_cache, .coverage, htmlcov)
#   - Type checking cache (.mypy_cache)
#   - Linting cache (.ruff_cache)
#   - Benchmarking cache (.benchmarks)
#   - Build artifacts (dist/, build/, *.egg-info, site/)
#   - Temporary directories (temp/, test_artifacts/)
#   - IDE log files (but preserves settings)
#
# WHAT IT PRESERVES:
#   - Virtual environment (.venv/)
#   - Dependency lock file (uv.lock)
#   - Git repository (.git/)
#   - IDE configuration (.vscode/ settings)
#   - Environment configuration (.env)
#
# SAFE TO RUN:
#   This script is safe to run at any time and will not delete any source
#   code or important configuration files.
#
# Author: Artifex Team
# License: MIT

echo "🧹 Cleaning cache files from repository..."

# Remove Python cache files
echo "Removing Python bytecode cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Remove testing and coverage cache
echo "Removing test and coverage cache..."
rm -rf .pytest_cache/ 2>/dev/null || true
rm -rf .coverage 2>/dev/null || true
rm -rf htmlcov/ 2>/dev/null || true

# Remove type checking cache
echo "Removing type checking cache..."
rm -rf .mypy_cache/ 2>/dev/null || true

# Remove linting cache
echo "Removing linting cache..."
rm -rf .ruff_cache/ 2>/dev/null || true

# Remove benchmarking cache
echo "Removing benchmarking cache..."
rm -rf .benchmarks/ 2>/dev/null || true

# Remove build artifacts
echo "Removing build artifacts..."
rm -rf dist/ 2>/dev/null || true
rm -rf build/ 2>/dev/null || true
rm -rf ./*.egg-info/ 2>/dev/null || true
rm -rf site/ 2>/dev/null || true

# Remove temporary files
echo "Removing temporary files..."
rm -rf temp/ 2>/dev/null || true
rm -rf test_artifacts/ 2>/dev/null || true

# Remove IDE cache (but keep configuration)
echo "Removing IDE cache..."
find .vscode/ -name "*.log" -delete 2>/dev/null || true
# find .cursor/ -name "*.log" -delete 2>/dev/null || true

echo "✅ Cache cleanup completed!"
echo ""
echo "Kept the following (as they should be preserved):"
echo "  - .venv/ (virtual environment)"
echo "  - uv.lock (dependency lock file)"
echo "  - .git/ (git repository)"
echo "  - Configuration files (.vscode/ settings)"
