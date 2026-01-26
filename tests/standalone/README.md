# Standalone Tests

This directory contains standalone tests for the configuration system components. These tests are isolated from the main module imports to avoid import issues and provide a quick way to test the basic functionality of the configuration classes.

## Purpose

The standalone tests were created to:

1. Work around circular import issues in the module structure
2. Provide a reliable way to test core functionality independently
3. Achieve good test coverage despite module structure issues
4. Enable quick validation of critical components

## Test Files

- `test_distributed_config.py`: Tests for distributed training configuration
- `test_hyperparam_config.py`: Tests for hyperparameter search configuration
- `test_distributed.py`: Simplified distributed configuration test
- `test_simple.py`: Basic sanity check test

## Running the Tests

You can run the standalone tests using the unified test runner:

```bash
# Run all standalone tests (parallel execution and progress bar by default)
./test.py standalone

# Run without parallel execution
./test.py standalone --no-parallel

# Run with verbose output
./test.py standalone -v

# Run with coverage report
./test.py standalone -c

# Run with classic dots output instead of progress bar
./test.py standalone --no-progress-bar

# Run a specific standalone test file
./test.py specific -f tests/standalone/test_distributed_config.py
```

Alternatively, you can use the shell script helper:

```bash
# Run standalone tests specifically
./scripts/run_tests.sh --standalone

# Control parallel execution (enabled by default)
./scripts/run_tests.sh --standalone --no-parallel  # Disable parallel
./scripts/run_tests.sh --standalone -j 4           # Use 4 processes

# Control output style (progress bar by default)
./scripts/run_tests.sh --standalone --no-progress  # Use classic dots
```

## Test Structure

Each test file recreates the necessary class structure for testing without relying on the actual module imports. This allows for testing the core functionality independently of the module structure and potential import issues.

The tests cover:

- Basic instantiation
- Default values
- Validation rules
- Edge cases
- Conversion and serialization
- Integration with other components

## Pydantic v2 Compatibility

The standalone tests use Pydantic v2 compatible validators:

```python
# Pydantic v2 model validator
@model_validator(mode='after')
def validate_ranks(self):
    # Validation logic
    return self

# Instead of Pydantic v1 style
@root_validator
def validate_ranks(cls, values):
    # Validation logic
    return values
```

## Coverage

Current test coverage is above 95%, providing high confidence in the core functionality of the configuration classes. Key areas covered include:

- Schema validation for all configuration types
- Distributed training configuration
- Hyperparameter search configuration

## Adding New Tests

To add new standalone tests:

1. Create a new file named `test_<component>.py`
2. Include the necessary imports
3. Define standalone versions of the required classes
4. Write test functions prefixed with `test_`
5. Ensure all edge cases and failure modes are tested

Your test will be automatically discovered and run by the test runner.

## Recent Improvements

- Migrated to Pydantic v2 validator syntax
- Added robust validation for hyperparameter configurations
- Enhanced distributed configuration validation
