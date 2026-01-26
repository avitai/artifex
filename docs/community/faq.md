# Frequently Asked Questions

Common questions about Artifex and their answers.

## Installation

### Q: How do I install Artifex?

**A**: Using uv (recommended):

```bash
uv pip install artifex
```

For GPU support:

```bash
uv sync --extra cuda12  # For CUDA 12
```

See the [Installation Guide](../getting-started/installation.md) for more details.

### Q: Do I need a GPU?

**A**: No, Artifex works on CPU. However, training large models is much faster on GPU. Artifex automatically uses GPU if available.

### Q: Which Python version should I use?

**A**: Python 3.10 or later. We test on Python 3.10, 3.11, and 3.12.

## Models

### Q: Which models does Artifex support?

**A**: Artifex supports:

- VAE (Variational Autoencoders)
- GAN (Generative Adversarial Networks)
- Diffusion Models (DDPM, DDIM)
- Flow Models (RealNVP, Glow, Continuous Normalizing Flows)

See [Models Overview](../user-guide/models/vae-guide.md) for details.

### Q: Can I use custom architectures?

**A**: Yes! Artifex provides flexibility for custom models. See [Custom Architectures](../user-guide/advanced/architectures.md).

### Q: How do I load pre-trained models?

**A**: Use the checkpointing system:

```python
from artifex.generative_models.core.checkpointing import load_checkpoint

model, step = load_checkpoint(checkpoint_manager, model_template)
```

## Training

### Q: How do I train a model?

**A**: Basic training example:

```python
from artifex.generative_models.training.trainer import Trainer

trainer = Trainer(model_config=config, training_config=train_config)
trainer.train(train_dataset, val_dataset)
```

See [Training Guide](../user-guide/training/training-guide.md) for complete examples.

### Q: Training is slow. How can I speed it up?

**A**: Several options:

1. **Use GPU**: Much faster than CPU
2. **Increase batch size**: Better GPU utilization
3. **Use mixed precision**: `dtype=jnp.bfloat16`
4. **JIT compilation**: Happens automatically with JAX
5. **Data parallelism**: Distribute across multiple GPUs

See [Distributed Training](../user-guide/advanced/distributed.md).

### Q: My model's loss is NaN. What's wrong?

**A**: Common causes:

1. **Learning rate too high**: Try reducing it (e.g., 1e-4 instead of 1e-3)
2. **Gradient explosion**: Add gradient clipping
3. **Numerical instability**: Check for divisions by zero or log(0)
4. **Bad initialization**: Use proper weight initialization

### Q: How do I save checkpoints during training?

**A**: Checkpoints are saved automatically by the Trainer. Configure frequency:

```python
training_config = TrainingConfig(
    num_epochs=100,
    checkpoint_every=1000,  # Save every 1000 steps
)
```

## Data

### Q: What data formats does Artifex support?

**A**: Artifex supports:

- Images (PNG, JPG, NumPy arrays)
- Text (tokenized sequences)
- Audio (waveforms, spectrograms)
- Multi-modal (combined modalities)

See [Data Guide](../user-guide/data/data-guide.md).

### Q: How do I use my own dataset?

**A**: Create a custom dataset:

```python
from artifex.generative_models.modalities.base import BaseDataset

class MyDataset(BaseDataset):
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for sample in self.data:
            yield {"data": sample}
```

See [Custom Datasets](../user-guide/data/data-guide.md#custom-datasets).

### Q: Can I use data augmentation?

**A**: Yes! Artifex provides augmentation for all modalities:

```python
@jax.jit
def augment(batch, key):
    # Apply augmentation
    batch = random_flip(batch, key)
    batch = random_crop(batch, key)
    return batch
```

## Technical

### Q: Why JAX instead of PyTorch/TensorFlow?

**A**: JAX offers:

- Automatic differentiation
- JIT compilation for speed
- Functional programming style
- Easy parallelization with vmap/pmap
- Better for research and experimentation

### Q: Why Flax NNX specifically?

**A**: Flax NNX is:

- Modern and actively developed
- More Pythonic than Linen
- Better for complex architectures
- Easier state management
- Official Flax direction

### Q: Can I mix PyTorch/TensorFlow with Artifex?

**A**: No. Artifex uses JAX/Flax NNX exclusively. However, you can:

- Convert PyTorch/TensorFlow weights to JAX
- Use Artifex for training, export for PyTorch inference
- Integrate at the data loading level

### Q: How do I debug JAX code?

**A**: Debugging tips:

1. **Disable JIT**: Run without `@jax.jit` decorator
2. **Print values**: Use `jax.debug.print()`
3. **Check shapes**: Print intermediate shapes
4. **Use assertions**: Add shape checks
5. **Breakpoints**: Use `jax.debug.breakpoint()`

See JAX debugging docs for more details.

## Performance

### Q: How much memory do I need?

**A**: Depends on model size and batch size:

- Small models (VAE): 2-4GB GPU
- Medium models (StyleGAN): 8-16GB GPU
- Large models (large diffusion): 24-40GB GPU

Use gradient accumulation for larger models.

### Q: How many GPUs do I need?

**A**: One GPU is sufficient for most tasks. Multiple GPUs help with:

- Larger batch sizes (data parallelism)
- Larger models (model parallelism)
- Faster training (distributed training)

See [Distributed Training](../user-guide/advanced/distributed.md).

### Q: Can I train on CPU?

**A**: Yes, but it's slow. Recommended for:

- Testing and debugging
- Small models
- Limited datasets

Not recommended for production training.

## Deployment

### Q: How do I deploy a trained model?

**A**: Several options:

1. **Export model**: Save with checkpointing
2. **Create REST API**: Use Flask/FastAPI
3. **Containerize**: Use Docker
4. **Cloud deployment**: Deploy to cloud platforms

See [Deployment Guide](../user-guide/integrations/deployment.md).

### Q: Can I convert models to ONNX?

**A**: JAX models can be converted to ONNX for deployment, but it requires additional tools. See integration guides for details.

### Q: How do I optimize for inference?

**A**: Optimization techniques:

1. **JIT compilation**: Automatic with JAX
2. **Batching**: Process multiple samples together
3. **Mixed precision**: Use bfloat16
4. **Model pruning**: Remove unnecessary parameters

See [Optimization](../user-guide/integrations/deployment.md#optimization).

## Troubleshooting

### Q: I get "Out of Memory" errors

**A**: Solutions:

1. **Reduce batch size**: Smaller batches use less memory
2. **Gradient accumulation**: Simulate larger batches
3. **Gradient checkpointing**: Trade compute for memory
4. **Mixed precision**: Use bfloat16
5. **Model parallelism**: Split model across devices

### Q: Tests are failing

**A**: Common issues:

1. **Missing dependencies**: Run `uv sync --all-extras`
2. **CUDA not available**: Some tests require GPU
3. **Outdated code**: Pull latest changes
4. **Environment issues**: Create fresh virtual environment

### Q: Import errors

**A**: Check:

1. **Installation**: `uv pip list | grep artifex`
2. **Python path**: Verify artifex is installed
3. **Virtual environment**: Activate correct environment
4. **Dependencies**: Install with `uv sync`

## Contributing

### Q: How can I contribute?

**A**: Many ways to contribute:

- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation
- Help others in discussions

See [Contributing Guide](contributing.md).

### Q: I found a bug. What should I do?

**A**: Please open a GitHub issue with:

- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)

### Q: Can I request a feature?

**A**: Yes! Open a GitHub issue describing:

- Feature description
- Use case
- Why it's useful
- Possible implementation ideas

## Getting Help

### Q: Where can I get help?

**A**: Several resources:

- **Documentation**: Start here
- **GitHub Issues**: For bugs and features
- **GitHub Discussions**: For questions
- **Examples**: Check examples directory

### Q: Documentation is unclear

**A**: Please let us know! Open an issue describing:

- Which page is unclear
- What's confusing
- What would help

We appreciate feedback to improve docs.

## License

### Q: What's the license?

**A**: Artifex is open source under the MIT License. You can:

- Use commercially
- Modify the code
- Distribute
- Use privately

See LICENSE file for details.

### Q: Can I use Artifex in my company?

**A**: Yes! The MIT License allows commercial use.
