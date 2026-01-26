# Core Layers Tests

This directory contains tests for the core layers used in the generative models, including positional encodings and ResNet blocks.

## Known Issues

### 1. ResNet Forward Pass Tests

The following tests still cause segmentation faults and should remain skipped:

#### ResNet Tests

- `TestResNetBlock.test_forward`
- `TestResNetBlock.test_forward_with_stride`
- `TestResNetBlock.test_forward_without_norm`
- `TestBottleneckBlock.test_forward`
- `TestBottleneckBlock.test_forward_with_stride`
- `TestBottleneckBlock.test_forward_without_norm`

These tests fail with segmentation faults when attempting to run the forward pass of the ResNet blocks. Recent re-enablement attempts (August 15, 2025) confirmed that the issue persists. The problem appears to be with JAX's compilation of convolutional operations rather than with our implementation. The initialization tests for these blocks pass successfully.

### 2. Fixed Issues

The following tests have been fixed:

#### LearnedPositionalEncoding Tests

- Previously skipped due to using the deprecated `self.param()` API
- Updated to use the newer NNX API with `nnx.Param` class instead
- Modified to correctly handle NNX random number generators by calling `rngs.params()`
- Updated test_longer_sequence_than_max_length to expect either IndexError or ValueError

## Running Tests

You can run all tests (skipping the known failing ones) with:

```bash
python -m pytest tests/artifex/generative_models/core/layers
```

To run only positional encoding tests:

```bash
python -m pytest tests/artifex/generative_models/core/layers/test_positional.py
```

To run only ResNet initialization tests (which pass):

```bash
python -m pytest tests/artifex/generative_models/core/layers/test_resnet.py::TestResNetBlock::test_init tests/artifex/generative_models/core/layers/test_resnet.py::TestResNetBlock::test_init_with_stride tests/artifex/generative_models/core/layers/test_resnet.py::TestResNetBlock::test_init_with_layer_norm tests/artifex/generative_models/core/layers/test_resnet.py::TestResNetBlock::test_init_with_group_norm tests/artifex/generative_models/core/layers/test_resnet.py::TestBottleneckBlock::test_init tests/artifex/generative_models/core/layers/test_resnet.py::TestBottleneckBlock::test_init_with_stride tests/artifex/generative_models/core/layers/test_resnet.py::TestBottleneckBlock::test_init_with_layer_norm tests/artifex/generative_models/core/layers/test_resnet.py::TestBottleneckBlock::test_init_with_group_norm tests/artifex/generative_models/core/layers/test_resnet.py::TestBottleneckBlock::test_bottleneck_ratio
