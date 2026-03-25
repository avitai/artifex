# Dependency Analyzer

**Status:** `Supported runtime utility`
**Module:** `artifex.generative_models.utils.code_analysis.dependency_analyzer`
**Source:** `src/artifex/generative_models/utils/code_analysis/dependency_analyzer.py`

This page documents the retained dependency-analysis utility used by the code-analysis tooling.
It is a package-local helper, not part of a broad top-level `artifex.utils.*` framework.

## Key Symbols

- `DependencyAnalyzer`
- `ModuleDependency`
- `detect_circular_dependencies(...)`
- `generate_dependency_report(...)`
- `get_module_dependencies(...)`

## Current Scope

Use this module for repository dependency inspection and reporting.
The broader historical utility families now live either with their owning package
or remain roadmap-only.
