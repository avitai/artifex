# BlackJAX Integration Tests

The BlackJAX integration tests are slow and are disabled by default, but can be easily enabled.

## Disabling BlackJAX Tests

BlackJAX tests are disabled by default. To run with the tests disabled:

```bash
# Tests will run with BlackJAX tests disabled by default (no need to set anything)
pytest tests/

# Explicitly disable BlackJAX tests
SKIP_BLACKJAX_TESTS=1 pytest tests/
```

## Enabling BlackJAX Tests

To enable and run the BlackJAX tests:

```bash
# Method 1: Set ENABLE_BLACKJAX_TESTS to any non-empty value (preferred method)
ENABLE_BLACKJAX_TESTS=1 pytest tests/

# Method 2: Set SKIP_BLACKJAX_TESTS to empty string (legacy method)
SKIP_BLACKJAX_TESTS="" pytest tests/

# Method 3: Use the pytest marker to run ONLY BlackJAX tests
pytest tests/ -m blackjax
```

## BlackJAX Test Helper Script

The easiest way to manage BlackJAX tests is to use the helper script:

```bash
# Show current status of BlackJAX tests
./scripts/blackjax_test_helper.py --status

# Enable BlackJAX tests
./scripts/blackjax_test_helper.py --enable

# Disable BlackJAX tests
./scripts/blackjax_test_helper.py --disable

# Enable tests and run them immediately
./scripts/blackjax_test_helper.py --enable --run

# Run only BlackJAX tests in parallel (parallel is default)
./scripts/blackjax_test_helper.py --enable --run --only-blackjax

# Run only BlackJAX tests without parallel execution
./scripts/blackjax_test_helper.py --enable --run --only-blackjax --no-parallel
```

## Making BlackJAX Tests Run By Default

To make BlackJAX tests run by default in CI/CD or for all developers, you can:

1. **Add to CI/CD pipeline file**:

   ```yaml
   env:
     ENABLE_BLACKJAX_TESTS: "1"
   ```

2. **For shell rc files** (.bashrc, .zshrc, etc.):

   ```bash
   export ENABLE_BLACKJAX_TESTS=1
   ```

3. **For project .env file** (if used with python-dotenv):

   ```
   ENABLE_BLACKJAX_TESTS=1
   ```

4. **Reset the default in code**:
   You can edit the `should_skip_blackjax_tests()` function in the sampling module's `__init__.py`
   to change the default behavior.

## Selective Test Runs

You can be more selective about which tests to run:

```bash
# Run only the BlackJAX sampler tests
pytest tests/artifex/generative_models/core/sampling/test_blackjax_samplers.py -m blackjax

# Run all tests except BlackJAX tests
pytest tests/ -m "not blackjax"
```

## Using the Helper Script

The `scripts/run_tests.sh` helper script makes it easy to run tests with or without BlackJAX:

```bash
# Run without BlackJAX tests (default)
./scripts/run_tests.sh --fast

# Run with BlackJAX tests
./scripts/run_tests.sh --all

# Run only BlackJAX tests
./scripts/run_tests.sh --only-blackjax

# With verbose output
./scripts/run_tests.sh --all -v

# With coverage report
./scripts/run_tests.sh --all -c

# Disable parallel execution (parallel is default)
./scripts/run_tests.sh --all --no-parallel

# Use classic dots output instead of progress bar
./scripts/run_tests.sh --all --no-progress
```

## Parallel Testing with BlackJAX

Tests run in parallel by default using pytest-xdist, which is especially beneficial for BlackJAX tests:

```bash
# Run all tests including BlackJAX tests (parallel is default)
ENABLE_BLACKJAX_TESTS=1 pytest tests/

# Run with specific number of processes
ENABLE_BLACKJAX_TESTS=1 pytest tests/ -n4

# Disable parallel execution if needed
ENABLE_BLACKJAX_TESTS=1 pytest tests/ -o console_output_style=progress

# Using the main test script
./test.py all --enable-blackjax      # Parallel by default
./test.py all --enable-blackjax --no-parallel  # Disable parallel execution
```

When running BlackJAX tests in parallel, each process gets its own JAX runtime, which can help isolate the tests and prevent interference between them.

## Test Display Options

The tests now use a progress bar display by default instead of the traditional dots:

```bash
# Use the default progress bar
pytest tests/

# Switch to count-style output
pytest tests/ -o console_output_style=count

# Use classic dots display
pytest tests/ -o console_output_style=classic

# With the test script
./test.py all --output-style=count
./test.py all --no-progress-bar      # Use classic dots
```

## CI/CD Configuration

For CI/CD pipelines, it's recommended to include a dedicated job for BlackJAX tests that can run less frequently:

```yaml
jobs:
  fast_tests:
    # Run regular tests with BlackJAX tests disabled (default)
    run: pytest tests/

  blackjax_tests:
    # Only run this on scheduled jobs or specific branches
    if: github.event_name == 'schedule' || github.ref == 'refs/heads/main'
    run: ENABLE_BLACKJAX_TESTS=1 pytest tests/ -m blackjax
```

## Technical Details

The BlackJAX tests are marked using both:

1. A custom pytest marker (`@pytest.mark.blackjax`)
2. A conditional skip decorator that checks the `SKIP_BLACKJAX_TESTS` and `ENABLE_BLACKJAX_TESTS` environment variables

See the sampling module's `__init__.py` and `test_blackjax_samplers.py` for implementation details.
