# Command Line Interface

Artifex provides a powerful CLI for training, generating, evaluating, and serving generative models without writing code.

## Overview

<div class="grid cards" markdown>

- :material-console:{ .lg .middle } **Training**

    ---

    Train models with `artifex train` using YAML configs

- :material-creation:{ .lg .middle } **Generation**

    ---

    Generate samples with `artifex generate`

- :material-chart-bar:{ .lg .middle } **Evaluation**

    ---

    Benchmark models with `artifex evaluate`

- :material-server:{ .lg .middle } **Serving**

    ---

    Deploy models with `artifex serve`

</div>

## Installation

The CLI is installed automatically with Artifex:

```bash
# Verify installation
artifex --help

# Or run via uv
uv run artifex --help
```

## Quick Start

### Train a Model

```bash
# Train from config file
artifex train --config configs/vae_mnist.yaml

# Train with overrides
artifex train --config configs/vae_mnist.yaml \
    --batch-size 128 \
    --epochs 100 \
    --lr 1e-3

# Resume training from checkpoint
artifex train --config configs/vae_mnist.yaml \
    --resume checkpoints/vae_mnist/latest
```

### Generate Samples

```bash
# Generate samples from trained model
artifex generate --checkpoint checkpoints/vae_mnist/best \
    --num-samples 100 \
    --output generated/

# Generate with specific seed
artifex generate --checkpoint model.ckpt \
    --num-samples 50 \
    --seed 42 \
    --output samples/
```

### Evaluate Model

```bash
# Run evaluation
artifex evaluate --checkpoint checkpoints/vae_mnist/best \
    --dataset mnist \
    --metrics fid inception_score

# Evaluate with custom test set
artifex evaluate --checkpoint model.ckpt \
    --data-path /path/to/test/data \
    --output results/evaluation.json
```

### Serve Model

```bash
# Start inference server
artifex serve --checkpoint checkpoints/vae_mnist/best \
    --port 8000

# Serve with specific host
artifex serve --checkpoint model.ckpt \
    --host 0.0.0.0 \
    --port 8080
```

## Commands

### `artifex train`

Train generative models from configuration.

```bash
artifex train [OPTIONS]

Options:
  --config PATH          Path to training config file [required]
  --batch-size INT       Override batch size
  --epochs INT           Override number of epochs
  --lr FLOAT             Override learning rate
  --resume PATH          Resume from checkpoint
  --output-dir PATH      Output directory for checkpoints
  --seed INT             Random seed
  --device TEXT          Device to use (auto/gpu/cpu)
  --wandb                Enable Weights & Biases logging
  --mlflow               Enable MLflow logging
  --help                 Show help message
```

[:octicons-arrow-right-24: Train Command Reference](train.md)

### `artifex generate`

Generate samples from trained models.

```bash
artifex generate [OPTIONS]

Options:
  --checkpoint PATH      Path to model checkpoint [required]
  --num-samples INT      Number of samples to generate [default: 10]
  --output PATH          Output directory for samples
  --seed INT             Random seed for reproducibility
  --temperature FLOAT    Sampling temperature [default: 1.0]
  --batch-size INT       Batch size for generation
  --format TEXT          Output format (png/npy/pt)
  --help                 Show help message
```

[:octicons-arrow-right-24: Generate Command Reference](generate.md)

### `artifex evaluate`

Evaluate model performance with metrics.

```bash
artifex evaluate [OPTIONS]

Options:
  --checkpoint PATH      Path to model checkpoint [required]
  --dataset TEXT         Dataset name or path
  --data-path PATH       Custom data path
  --metrics TEXT         Metrics to compute (comma-separated)
  --output PATH          Output file for results
  --batch-size INT       Batch size for evaluation
  --help                 Show help message
```

[:octicons-arrow-right-24: Evaluate Command Reference](evaluate.md)

### `artifex serve`

Deploy model as REST API.

```bash
artifex serve [OPTIONS]

Options:
  --checkpoint PATH      Path to model checkpoint [required]
  --host TEXT            Host address [default: localhost]
  --port INT             Port number [default: 8000]
  --workers INT          Number of workers [default: 1]
  --help                 Show help message
```

[:octicons-arrow-right-24: Serve Command Reference](serve.md)

### `artifex benchmark`

Run comprehensive benchmarks.

```bash
artifex benchmark [OPTIONS]

Options:
  --config PATH          Benchmark configuration file
  --model PATH           Model checkpoint or config
  --suite TEXT           Benchmark suite to run
  --output PATH          Output directory for results
  --help                 Show help message
```

[:octicons-arrow-right-24: Benchmark Command Reference](benchmark.md)

### `artifex convert`

Convert between model formats.

```bash
artifex convert [OPTIONS]

Options:
  --input PATH           Input model path [required]
  --output PATH          Output model path [required]
  --format TEXT          Target format (onnx/tflite/safetensors)
  --help                 Show help message
```

[:octicons-arrow-right-24: Convert Command Reference](convert.md)

## Configuration Files

### Training Config Example

```yaml
# configs/vae_mnist.yaml
model:
  type: vae
  latent_dim: 32
  encoder:
    hidden_dims: [256, 128]
    activation: relu
  decoder:
    hidden_dims: [128, 256]
    activation: relu

training:
  batch_size: 128
  epochs: 100
  optimizer:
    type: adam
    learning_rate: 1e-3
  scheduler:
    type: cosine
    warmup_steps: 1000

data:
  dataset: mnist
  train_split: 0.9

logging:
  wandb: true
  project: artifex-experiments
  log_interval: 100
```

### Using Environment Variables

```bash
# Set defaults via environment
export ARTIFEX_DEVICE=gpu
export ARTIFEX_SEED=42
export ARTIFEX_OUTPUT_DIR=./outputs

# These become defaults for all commands
artifex train --config config.yaml
```

## Utility Modules

### Configuration Utilities

- [config](config.md) - Configuration loading and validation
- [formatting](formatting.md) - Output formatting helpers

### Logging Utilities

- [logging](logging.md) - Logging configuration
- [progress](progress.md) - Progress bar utilities

## Examples

### Complete Training Workflow

```bash
# 1. Train the model
artifex train --config configs/dcgan_cifar.yaml \
    --output-dir experiments/dcgan_001 \
    --wandb

# 2. Generate samples
artifex generate \
    --checkpoint experiments/dcgan_001/best.ckpt \
    --num-samples 1000 \
    --output experiments/dcgan_001/samples/

# 3. Evaluate quality
artifex evaluate \
    --checkpoint experiments/dcgan_001/best.ckpt \
    --dataset cifar10 \
    --metrics fid inception_score lpips \
    --output experiments/dcgan_001/metrics.json
```

### Hyperparameter Search

```bash
# Run multiple training jobs
for lr in 1e-4 1e-3 1e-2; do
    artifex train --config configs/vae.yaml \
        --lr $lr \
        --output-dir experiments/vae_lr_$lr
done
```

## Related Documentation

- [Training Guide](../user-guide/training/training-guide.md) - Detailed training documentation
- [Configuration System](../user-guide/training/configuration.md) - Config file format
- [Model Deployment](../examples/framework/model-deployment.md) - Deployment guide
