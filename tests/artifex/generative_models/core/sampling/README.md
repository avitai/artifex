# BlackJAX Tests

This directory contains tests for BlackJAX samplers integration. These tests are disabled by default due to their computational intensity and occasional statistical nature.

## Running BlackJAX Tests

By default, all BlackJAX tests are skipped. This behavior is controlled through environment variables:

1. Set `ENABLE_BLACKJAX_TESTS=1` to run all BlackJAX tests
2. Set `SKIP_BLACKJAX_TESTS=` (empty string) to also enable tests

## Helper Script

A helper script is provided to easily run all BlackJAX tests:

```bash
# Run all BlackJAX tests
./run_blackjax_tests.py

# Run specific BlackJAX test(s)
./run_blackjax_tests.py tests/artifex/generative_models/core/sampling/test_blackjax_samplers.py::test_hmc_sampling_normal_dist
```

## Test Categories

The BlackJAX tests are organized into categories:

1. **Statistical validation tests**: These tests verify that samplers generate distributions with expected statistics. They may occasionally fail due to sampling variance and are marked with `@pytest.mark.xfail`.

2. **API and initialization tests**: These tests verify proper initialization and basic API functionality.

3. **Memory-intensive tests**: Tests for NUTS sampler are memory-intensive and may fail on systems with limited memory.

## Temporarily Skipped Tests

Some tests have been additionally marked with `@pytest.mark.skip(reason="Temporarily skipped")` to bypass issues that are being addressed separately:

- `test_hmc_sampling_normal_dist`
- `test_mala_sampling_normal_dist`
- `test_nuts_sampling_normal_dist`

To run these tests, you'll need to remove the skip markers or run them with pytest's `--skip-skip` option in addition to setting `ENABLE_BLACKJAX_TESTS=1`.
