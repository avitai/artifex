# Artifex Scripts Directory

This directory contains utility scripts for development, testing, and maintenance of the Artifex project.

## 📋 Table of Contents

- [Environment Setup](#-environment-setup)
- [Testing](#-testing)
- [Code Quality](#-code-quality)
- [Documentation](#-documentation)
- [Development Utilities](#development-utilities)
- [Maintenance](#-maintenance)

## 🚀 Environment Setup

### setup_env.py

**Purpose:** Python-based environment management utility complementing the main setup.sh script.

**Features:**

- Environment status checking
- Installation verification
- Configuration file generation
- Environment cleanup

**Usage:**

```bash
# Check environment status
python scripts/setup_env.py check

# Verify installation
python scripts/setup_env.py verify

# Clean environment
python scripts/setup_env.py clean [--deep]

# Create .env configuration only
python scripts/setup_env.py create-env [--cpu-only]
```

**Note:** For full environment setup, use `./setup.sh` in the project root.

## 🧪 Testing

### smart_test_runner.sh

**Purpose:** Intelligent test execution with automatic GPU detection and parallel optimization.

**Features:**

- Automatic GPU/CPU detection
- Smart parallelization strategy
- GPU test marker detection
- Environment variable loading

**Execution Strategies:**

- GPU tests on GPU: Sequential (prevents CUDA conflicts)
- GPU tests on CPU: Skip with warning
- CPU tests: Parallel execution
- Mixed: Automatic detection

**Usage:**

```bash
# Run all tests with smart detection
./scripts/smart_test_runner.sh tests/

# Run specific test types
./scripts/smart_test_runner.sh tests/ -k test_gpu    # GPU tests only
./scripts/smart_test_runner.sh tests/ -m "not gpu"   # CPU tests only

# Force execution mode
./scripts/smart_test_runner.sh tests/ --parallel     # Force parallel
./scripts/smart_test_runner.sh tests/ --sequential   # Force sequential
```

### blackjax_test_helper.py

**Purpose:** Helper for managing BlackJAX test execution.

**Features:**

- Enable/disable BlackJAX tests via environment variables
- Test filtering and selection
- Coverage report generation

**Usage:**

```bash
# Enable BlackJAX tests
python scripts/blackjax_test_helper.py --enable

# Check status
python scripts/blackjax_test_helper.py --status

# Run with BlackJAX tests
python scripts/blackjax_test_helper.py --enable --run
```

## 🎮 GPU/Hardware Verification

### verify_gpu_setup.py

**Purpose:** Comprehensive GPU setup verification and diagnostics.

**Features:**

- Hardware capability detection
- JAX device configuration verification
- Memory management testing
- CUDA library validation
- Performance analysis
- Detailed recommendations

**Usage:**

```bash
# Full verification suite
python scripts/verify_gpu_setup.py

# Quick critical tests only
python scripts/verify_gpu_setup.py --critical-only

# Configure before verification
python scripts/verify_gpu_setup.py --configure-first
```

### gpu_utils.py

**Purpose:** Utility functions for GPU detection and configuration.

**Features:**

- CUDA version detection
- GPU memory querying
- Device capability checking
- Environment configuration helpers

**Usage:**

```python
from gpu_utils import detect_cuda_version, get_gpu_memory

cuda_version = detect_cuda_version()
memory_info = get_gpu_memory()
```

## 📝 Code Quality

### analyze_dependencies.py

**Purpose:** Analyze module dependencies and detect circular imports.

**Features:**

- Dependency graph generation
- Circular dependency detection
- Module relationship analysis
- Visual dependency reports

**Usage:**

```bash
# Analyze dependencies
python scripts/analyze_dependencies.py

# Custom output directory
python scripts/analyze_dependencies.py --output-dir custom/path

# Specific source directory
python scripts/analyze_dependencies.py --source-dir src/artifex
```

### analyze_test_structure.py

**Purpose:** Analyze test coverage and structure.

**Features:**

- Test file discovery
- Source module mapping
- Coverage analysis
- Test structure recommendations

**Usage:**

```bash
# Analyze test structure
python scripts/analyze_test_structure.py

# Custom directories
python scripts/analyze_test_structure.py --test-dir tests --src-dir src/artifex

# JSON output
python scripts/analyze_test_structure.py --json-output analysis.json
```

### find_circular_imports.py

**Purpose:** Detect circular import dependencies.

**Features:**

- AST-based import analysis
- Circular dependency detection
- Detailed cycle reporting

**Usage:**

```bash
# Find circular imports
python scripts/find_circular_imports.py
```

### fix_imports.py

**Purpose:** Fix import statements in test files.

**Features:**

- Remove `src.` prefix from imports
- Batch processing of test files
- Safe import rewriting

**Usage:**

```bash
# Fix imports in all test files
python scripts/fix_imports.py
```

### check_jax_nnx_compatibility.py

**Purpose:** Verify JAX and Flax NNX version compatibility.

**Features:**

- Version checking for JAX ecosystem
- Compatibility matrix validation
- Recommended version reporting

**Usage:**

```bash
# Check compatibility
python scripts/check_jax_nnx_compatibility.py
```

## 📚 Documentation

### build_docs.py

**Purpose:** Comprehensive documentation build orchestration.

**Features:**

- Documentation generation from source
- MkDocs build integration
- Optional documentation serving
- Error handling and reporting

**Usage:**

```bash
# Build documentation
python scripts/build_docs.py

# Build and serve
python scripts/build_docs.py --serve [--port 8000]

# Skip generation step
python scripts/build_docs.py --skip-generation
```

### generate_docs.py

**Purpose:** Modern documentation generator with dynamic discovery.

**Features:**

- Dynamic project structure discovery
- Automatic module categorization
- Incremental generation support
- MkDocs navigation generation
- Progress tracking and error handling

**Usage:**

```bash
# Standard generation
python scripts/generate_docs.py

# Clean rebuild with verbose output
python scripts/generate_docs.py --clean --verbose

# Incremental build
python scripts/generate_docs.py --incremental
```

### validate_docs.py

**Purpose:** Validate and auto-fix documentation issues.

**Features:**

- Validates Python module references in mkdocstrings
- Auto-fixes incorrect module paths
- Checks for missing directories and files
- Validates navigation structure in mkdocs.yml
- Reports issues with detailed suggestions

**Usage:**

```bash
# Check for issues
python scripts/validate_docs.py --check-only

# Auto-fix issues
python scripts/validate_docs.py --fix

# Verbose checking with fixes
python scripts/validate_docs.py --fix --verbose
```

## 🧹 Maintenance

### clean_cache.sh

**Purpose:** Remove cache files and temporary artifacts.

**Cleans:**

- Python bytecode (`__pycache__`, `*.pyc`, `*.pyo`)
- Test artifacts (`.pytest_cache`, `.coverage`)
- Type checking cache (`.mypy_cache`)
- Linting cache (`.ruff_cache`)
- Build artifacts (`dist/`, `build/`, `*.egg-info`)
- Temporary files

**Preserves:**

- Virtual environment (`.venv/`)
- Lock files (`uv.lock`)
- Git repository (`.git/`)
- Configuration files

**Usage:**

```bash
# Clean all cache files
./scripts/clean_cache.sh
```

## 🔧 Script Guidelines

### Adding New Scripts

1. **Documentation**: Include comprehensive header documentation
2. **Error Handling**: Implement proper error handling and exit codes
3. **Logging**: Use colored output for better readability
4. **Dependencies**: Document all required dependencies
5. **Examples**: Provide usage examples in the script header

### Script Header Template

```bash
#!/bin/bash
# Script Name and Purpose
# =======================
#
# PURPOSE:
#   Brief description of what the script does
#
# USAGE:
#   ./scripts/script_name.sh [OPTIONS]
#
# OPTIONS:
#   --option1    Description of option1
#   --option2    Description of option2
#
# EXAMPLES:
#   ./scripts/script_name.sh --option1 value
#
# EXIT CODES:
#   0 - Success
#   1 - Error
#
# DEPENDENCIES:
#   - dependency1
#   - dependency2
#
# Author: Artifex Team
# License: MIT
```

### Python Script Template

```python
#!/usr/bin/env python
"""
Script Name and Purpose
=======================

PURPOSE:
    Brief description of what the script does

USAGE:
    python scripts/script_name.py [OPTIONS]

OPTIONS:
    --option1    Description of option1
    --option2    Description of option2

EXAMPLES:
    python scripts/script_name.py --option1 value

DEPENDENCIES:
    - dependency1
    - dependency2

Author: Artifex Team
License: MIT
"""
```

## 📊 Examples

Example scripts have been moved to the `examples/` directory:

- `examples/benchmarks/protein_benchmark_example.py` - Protein benchmark demonstration
- `examples/benchmarks/run_protein_benchmarks.py` - Protein benchmark runner
- `examples/verify_examples.py` - README examples verification

## 🚦 Quick Reference

| Task | Command |
|------|---------|
| Check environment | `python scripts/setup_env.py check` |
| Verify GPU setup | `python scripts/verify_gpu_setup.py` |
| Run all tests | `./scripts/smart_test_runner.sh tests/` |
| Clean caches | `./scripts/clean_cache.sh` |
| Build docs | `python scripts/build_docs.py` |
| Check dependencies | `python scripts/analyze_dependencies.py` |
| Fix imports | `python scripts/fix_imports.py` |

## 🤝 Contributing

When adding or modifying scripts:

1. Follow the documentation templates above
2. Test thoroughly on both CPU and GPU systems
3. Update this README with new script documentation
4. Ensure scripts are executable (`chmod +x script.sh`)
5. Use consistent error handling and logging

## 📝 License

All scripts in this directory are part of the Artifex project and are licensed under the MIT License.
