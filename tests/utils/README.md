# Test Discovery Helper

This tool helps you discover and run tests in the artifex project.

## Features

- Find tests by module, feature, or pattern
- Filter tests by type (unit, integration, etc.)
- Run tests with pytest
- Show test files without running them

## Usage

You can use the test discovery helper in two ways:

### 1. Using the `run_tests.py` script

```bash
# Run all tests
./scripts/run_tests.py -a

# Run tests for a specific module
./scripts/run_tests.py -m artifex.generative_models.core

# Run tests matching a pattern
./scripts/run_tests.py -p config

# Run tests for a specific feature
./scripts/run_tests.py -f transformer

# Run tests of a specific type
./scripts/run_tests.py -t unit

# Show tests without running them
./scripts/run_tests.py -m artifex.generative_models.core -d

# Run tests in verbose mode
./scripts/run_tests.py -a -v

# Don't capture test output
./scripts/run_tests.py -a -s
```

### 2. Importing the module in Python code

```python
from tests.utils.discovery import (
    find_all_test_files,
    find_tests_by_module,
    find_tests_by_pattern,
    find_tests_by_feature,
    find_tests_by_type,
    get_pytest_args_for_tests,
)

# Find all test files
test_files = find_all_test_files()

# Find tests for a specific module
module_tests = find_tests_by_module("artifex.generative_models.core")

# Find tests matching a pattern
pattern_tests = find_tests_by_pattern("config")

# Find tests for a specific feature
feature_tests = find_tests_by_feature("transformer")

# Find tests of a specific type
unit_tests = find_tests_by_type("unit")
```

## Options

```
usage: run_tests.py [-h] (-a | -m MODULE | -p PATTERN | -f FEATURE | -t {unit,integration,functional,benchmark,e2e})
                    [-v] [-s] [-d] [--test-dir TEST_DIR]

Test discovery and execution helper.

options:
  -h, --help            show this help message and exit
  -a, --all             Run all tests
  -m MODULE, --module MODULE
                        Run tests for a specific module (e.g., artifex.generative_models.core)
  -p PATTERN, --pattern PATTERN
                        Run tests matching a pattern (e.g., 'config' or 'transformer')
  -f FEATURE, --feature FEATURE
                        Run tests for a specific feature (e.g., 'vae' or 'attention')
  -t {unit,integration,functional,benchmark,e2e}, --type {unit,integration,functional,benchmark,e2e}
                        Run tests of a specific type (unit, integration, etc.)
  -v, --verbose         Run tests in verbose mode
  -s, --no-capture      Don't capture test output
  -d, --dry-run         Show test files without running them
  --test-dir TEST_DIR   Directory containing tests (default: tests)
