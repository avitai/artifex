# BlackJAX Tests

This directory contains tests for BlackJAX sampler integration. BlackJAX is a first-class
dependency in Artifex, so these tests belong to the normal pytest contract.

## Running BlackJAX Tests

```bash
# Run the full suite
uv run pytest tests/

# Run only the BlackJAX-marked tests
uv run pytest tests/ -m blackjax

# Run this module only
uv run pytest tests/artifex/generative_models/core/sampling/test_blackjax_samplers.py -v
```

## Test Categories

The BlackJAX tests are organized into categories:

1. **Statistical validation tests**: These tests verify that samplers generate distributions with expected statistics. They may occasionally fail due to sampling variance and are marked with `@pytest.mark.xfail`.

2. **API and initialization tests**: These tests verify proper initialization and basic API functionality.

3. **Memory-intensive tests**: Tests for NUTS sampler are memory-intensive and may fail on systems with limited memory.

## Selection

If you need a focused local run that excludes these tests, use standard pytest selection:

```bash
uv run pytest tests/ -m "not blackjax"
```
