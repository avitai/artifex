# Roadmap

This document outlines the development status and planned improvements for Artifex.

## Current Status: Major Refactoring

Artifex is undergoing a significant architectural refactoring. The library is being restructured for better modularity, maintainability, and API consistency.

> **Note**: During this refactoring period:
>
> - APIs may change without deprecation warnings
> - Some tests may fail or be skipped
> - Documentation may not reflect current implementation

### Components Under Refactoring

- **Core Model Implementations**
  - VAE Family: VAE, Beta-VAE, VQ-VAE, Conditional VAE
  - GAN Family: DCGAN, WGAN, StyleGAN, CycleGAN, PatchGAN
  - Diffusion Models: DDPM, DDIM, Score-based, DiT, Latent Diffusion
  - Normalizing Flows: RealNVP, Glow, MAF, IAF, Neural Spline Flows
  - Energy-Based Models: Langevin dynamics, MCMC sampling
  - Autoregressive Models: PixelCNN, WaveNet, Transformer-based
  - Geometric Models: Point clouds, meshes, SE(3) molecular flows

- **Multi-Modal Support**
  - Image: Convolutional architectures, quality metrics
  - Text: Tokenization, language modeling
  - Audio: Spectral processing, WaveNet
  - Protein: Structure generation with physical constraints
  - Tabular: Mixed data types
  - Timeseries: Sequential patterns
  - Molecular: Chemical structure generation
  - Geometric: Point clouds, meshes, voxels

- **Infrastructure**
  - Unified frozen dataclass configuration system
  - Protocol-based architecture
  - Factory pattern for model creation
  - GPU/CPU device management
  - Composable loss functions

### In Progress

- **Test Suite Restructuring**
  - Reorganizing test hierarchy
  - Improving test isolation
  - Updating test fixtures
  - Coverage reporting improvements

- **API Stabilization**
  - Consistent method signatures
  - Clear public/private boundaries
  - Improved error messages
  - Type annotation completion

- **Documentation Updates**
  - Syncing docs with code changes
  - Updating code examples
  - API reference refresh

### Planned Features

- **Performance Optimizations**
  - JIT compilation improvements
  - Memory-efficient attention
  - Gradient checkpointing
  - Mixed precision training

- **Additional Model Variants**
  - Consistency models
  - Flow matching
  - Rectified flows
  - Additional transformer architectures

- **Extended Modality Support**
  - Video generation
  - 3D scene understanding
  - Multi-modal alignment

- **Training Improvements**
  - Advanced learning rate schedules
  - Automatic hyperparameter tuning
  - Training visualization dashboard

- **Scaling Features**
  - Multi-GPU training support
  - TPU compatibility
  - Gradient accumulation
  - Memory optimization

## Version Milestones

### v0.1.x (Current - Refactoring)

Focus: Architecture refactoring and API stabilization

- Restructuring codebase for better modularity
- Stabilizing public APIs
- Improving test coverage and reliability
- Updating documentation

### v0.2.x (Planned)

Focus: Stability and performance

- Stable, documented APIs
- Multi-GPU support
- Memory optimizations
- Extended benchmark suite
- Performance profiling tools

### v0.3.x (Planned)

Focus: Advanced features

- Additional model architectures
- Video and multi-modal support
- Advanced fine-tuning methods
- Production deployment tools

## Contributing

We welcome contributions in all areas. Priority areas during the refactoring period:

1. **Testing**: Helping stabilize and expand test coverage
2. **Documentation**: Keeping docs in sync with code changes
3. **Bug Reports**: Reporting issues encountered during refactoring
4. **Code Review**: Reviewing refactoring PRs

See the [GitHub Issues](https://github.com/avitai/artifex/issues) for current tasks and feature requests.

## See Also

- [Design Philosophy](philosophy.md) - Development principles
- [Testing Guide](testing.md) - How to run and write tests
