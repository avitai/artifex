# Artifex: A Comprehensive Generative Modeling Framework for Research and Production

**Authors:** Mahdi Shafiei
**Affiliation:** Avitai Bio
**Email:** <mahdi@avitai.bio>
**Date:** January 26, 2026
**Status:** Work in Progress - Active Development
**arXiv ID:** [To be assigned]
**Categories:** cs.LG, cs.AI, stat.ML

---

## Abstract

We present Artifex, a comprehensive generative modeling library built on JAX/Flax that provides unified implementations of state-of-the-art generative models across multiple modalities. Artifex aims to address the gap between research prototypes and deployable generative modeling systems by offering a modular, type-safe, and scalable framework. The library implements seven major generative modeling paradigms—Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Diffusion Models, Flow-based Models, Autoregressive Models, Energy-Based Models (EBMs), and Geometric Models—with support for eight data modalities: images, text, audio, tabular data, time series, protein structures, molecular data, and multi-modal representations. Artifex introduces architectural patterns including a protocol-based design system, hardware-aware optimization infrastructure, multi-dimensional parallelism strategies, and a unified configuration management system using frozen dataclasses. The framework is currently undergoing active development and refactoring, with APIs subject to change. Artifex provides researchers with a foundation for generative modeling research on JAX.

**Note:** This preprint describes ongoing development. The framework is not yet production-ready and specific performance claims require validation as the codebase stabilizes.

**Keywords:** Generative Models, JAX, Flax, Machine Learning Framework, Scalable AI, Multi-Modal Learning

---

## 1. Introduction

Generative modeling has emerged as one of the most transformative areas in artificial intelligence, enabling the creation of realistic content across diverse modalities including images, text, audio, and structured data. However, the field faces significant challenges in transitioning from research prototypes to deployable systems. Existing frameworks often lack the scalability, type safety, and comprehensive evaluation capabilities required for serious deployment, while research-focused libraries typically sacrifice robustness for experimental flexibility.

The Artifex library aims to address these challenges by providing a unified framework for generative modeling built on the JAX ecosystem with Flax NNX. Artifex combines the performance benefits of JAX's automatic differentiation and XLA compilation with modern neural network design patterns. The framework is currently in active development, with ongoing refactoring to establish strong architectural foundations.

**Development Status:** Artifex is undergoing major refactoring. APIs are subject to change, and users should expect breaking changes between versions. The framework prioritizes establishing correct foundations over backward compatibility.

### 1.1 Key Contributions

This work makes the following key contributions to the generative modeling ecosystem:

1. **Unified Generative Modeling Framework**: A library implementing seven major generative modeling paradigms with consistent interfaces and type-safe protocols.

2. **Multi-Modal Architecture**: Support for eight distinct data modalities with specialized adapters, evaluation metrics, and domain-specific constraints.

3. **Scaling Infrastructure (Experimental)**: Hardware-aware optimization and multi-dimensional parallelism strategies. Large-scale distributed training capabilities are under development.

4. **Protocol-Based Design System**: Type-safe interfaces using Python's Protocol system, enabling static and runtime type checking while maintaining implementation flexibility.

5. **Evaluation Framework (In Progress)**: Standardized metrics and benchmarking infrastructure being developed across supported modalities and model types.

6. **Modern JAX/Flax Integration**: Full compatibility with Flax NNX patterns, automatic differentiation, and JAX transformations.

### 1.2 Related Work

Several frameworks have attempted to address the challenges in generative modeling infrastructure. PyTorch-based libraries such as PyTorch Lightning and Hugging Face Transformers provide excellent research capabilities but often lack production scaling features. TensorFlow-based solutions offer better production support but suffer from API complexity and slower research iteration cycles. JAX-based frameworks like Flax and Haiku provide excellent performance but typically focus on specific model types or lack comprehensive evaluation capabilities.

Artifex aims to provide a unified framework that addresses the complete generative modeling pipeline while maintaining research flexibility. The library's protocol-based design enables seamless integration with existing JAX ecosystem tools while providing the type safety needed for reliable experimentation.

---

## 2. Framework Architecture

### 2.1 Core Design Principles

Artifex is built on four fundamental design principles that guide its architecture and implementation:

**Modularity**: The framework employs a modular design where components can be used independently or composed to create complex systems. This enables researchers to experiment with specific components while allowing practitioners to build production systems using proven patterns.

**Type Safety**: All interfaces are defined using Python's Protocol system, providing static type checking during development and runtime verification during execution. This ensures reliability and reduces debugging overhead in production environments.

**Performance-First Design**: Every component is designed with performance as a primary consideration, leveraging JAX's automatic differentiation, XLA compilation, and hardware-aware optimization to achieve optimal execution efficiency.

**Scalability**: The framework provides built-in support for scaling from single-device experimentation to multi-node distributed training, with automatic hardware detection and optimization strategies.

### 2.2 Generative Model Protocol

At the core of Artifex's architecture is the `GenerativeModelProtocol`, which defines a consistent interface for all generative models:

```python
@runtime_checkable
class GenerativeModelProtocol(Protocol):
    def __call__(self, x: Any, *, rngs: nnx.Rngs | None = None, **kwargs) -> dict[str, Any]:
        """Forward pass through the model."""
        ...

    def generate(self, n_samples: int = 1, *, rngs: nnx.Rngs | None = None, **kwargs) -> jax.Array:
        """Generate samples from the model."""
        ...

    def loss_fn(self, batch: Dict[str, Any], outputs: Dict[str, Any], **kwargs) -> Dict[str, Array]:
        """Compute loss function."""
        ...
```

This protocol-based approach enables:

- Static and runtime type checking for model implementations
- Common interfaces across diverse model architectures
- Generic training and evaluation code that works with any model
- Type-safe interaction with JAX transformations and optimizers

### 2.3 Model Registry and Factory System

Artifex implements a centralized model registry and factory system that enables dynamic model creation and configuration:

```python
@register_model("vae")
class VAEModel(GenerativeModel):
    """Variational Autoencoder implementation."""
    ...

# Dynamic model creation
model = create_model(config, rngs=rngs)
```

The factory system provides:

- Centralized model registration and discovery
- Type-safe model creation with configuration validation
- Support for model variants and parameter inheritance
- Integration with the unified configuration system

### 2.4 Configuration Management

Artifex employs a hierarchical configuration system built on frozen dataclasses that provides:

- **Type Safety**: All configuration parameters are type-annotated and immutable after creation
- **Immutability**: Frozen dataclasses prevent accidental configuration modification
- **Inheritance**: Configuration classes use inheritance for shared parameters
- **Validation**: Custom `__post_init__` methods validate configuration consistency

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class BaseConfig:
    """Base configuration class for all configs."""
    dtype: str = "float32"

@dataclass(frozen=True)
class VAEConfig(BaseConfig):
    """Configuration for VAE models."""
    latent_dim: int = 64
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    beta: float = 1.0
```

---

## 3. Generative Model Implementations

### 3.1 Model Categories

Artifex implements seven major categories of generative models, each with multiple variants and specialized implementations:

#### 3.1.1 Variational Autoencoders (VAEs)

- **Basic VAE**: Standard variational autoencoder with KL divergence regularization
- **Beta-VAE**: Enhanced disentanglement through β-parameter control
- **Conditional VAE**: Conditional generation with auxiliary information
- **Hierarchical VAE**: Multi-level latent representations

#### 3.1.2 Generative Adversarial Networks (GANs)

- **DCGAN**: Deep Convolutional GAN for image generation
- **WGAN**: Wasserstein GAN with gradient penalty
- **Conditional GAN**: Conditional generation capabilities
- **LSGAN**: Least Squares GAN for improved stability
- **CycleGAN**: Unpaired image-to-image translation
- **PatchGAN**: Multi-scale discriminator for high-resolution generation
- **StyleGAN3**: Alias-free generative adversarial network

#### 3.1.3 Diffusion Models

- **DDPM**: Denoising Diffusion Probabilistic Models
- **DDIM**: Denoising Diffusion Implicit Models for faster sampling *(in development)*
- **Score-based Models**: Score-based generative modeling
- **Latent Diffusion**: Diffusion in latent space for efficiency
- **DiT**: Diffusion Transformers for scalable generation
- **Guided Diffusion**: Classifier and classifier-free guidance
- **Stable Diffusion**: Text-to-image generation *(planned)*

#### 3.1.4 Flow-based Models

- **Glow**: Generative Flow with invertible 1×1 convolutions
- **MAF**: Masked Autoregressive Flow
- **IAF**: Inverse Autoregressive Flow
- **Neural Spline Flows**: Continuous normalizing flows with splines
- **Conditional Flows**: Conditional normalizing flows

#### 3.1.5 Autoregressive Models

- **PixelCNN**: Autoregressive image generation with causal convolutions
- **WaveNet**: Autoregressive audio generation with dilated convolutions
- **Transformer**: Autoregressive text generation with attention mechanisms
- **PixelCNN++**: Enhanced PixelCNN with mixture of logistics

#### 3.1.6 Energy-Based Models (EBMs)

- **Basic EBM**: Energy-based models with Langevin dynamics
- **Contrastive Divergence**: Efficient training with persistent contrastive divergence
- **MCMC Sampling**: Integration with BlackJAX for advanced sampling
- **Sample Buffers**: Efficient sample management for training

#### 3.1.7 Geometric Models

- **Point Cloud Models**: 3D point cloud generation with transformers
- **Mesh Models**: 3D mesh generation with template deformation
- **Voxel Models**: 3D voxel grid generation with 3D convolutions
- **Protein Models**: Specialized models for protein structure generation

### 3.2 Implementation Quality

Model implementations in Artifex aim to adhere to the following quality standards:

- **Test Coverage**: Test suite under active development; coverage expanding with refactoring
- **Type Safety**: Full type annotations with runtime validation via Protocol system
- **Documentation**: Docstrings and usage examples being developed
- **Performance**: Implementations optimized with JAX transformations
- **Modularity**: Clean separation of concerns with composable components

*Note: The codebase is undergoing major refactoring. Some tests are temporarily skipped while APIs stabilize.*

---

## 4. Multi-Modal Architecture

### 4.1 Modality Framework

Artifex implements a modality framework that separates model architectures from domain-specific data types. This design enables researchers to experiment with different architectures while maintaining domain-specific optimizations and constraints.

The modality framework supports eight distinct data types:

#### 4.1.1 Image Modality

- **Representations**: RGB, RGBA, grayscale, and multi-channel formats
- **Preprocessing**: Standardization, normalization, and augmentation pipelines
- **Evaluation Metrics**: FID, Inception Score, LPIPS, and perceptual metrics
- **Constraints**: Spatial consistency and visual quality preservation

#### 4.1.2 Text Modality

- **Representations**: Token-level, subword, and byte-level encodings
- **Tokenization**: Support for multiple tokenization strategies
- **Evaluation Metrics**: BLEU, ROUGE, perplexity, and diversity metrics
- **Constraints**: Semantic coherence and grammatical correctness

#### 4.1.3 Audio Modality

- **Representations**: Waveform, spectrogram, and mel-spectrogram formats
- **Preprocessing**: Audio normalization, filtering, and feature extraction
- **Evaluation Metrics**: Spectral distance, perceptual metrics, and quality scores
- **Constraints**: Temporal consistency and audio quality preservation

#### 4.1.4 Tabular Modality

- **Data Types**: Numerical, categorical, ordinal, and binary features
- **Preprocessing**: Feature scaling, encoding, and normalization
- **Evaluation Metrics**: Statistical distance, privacy scores, and utility metrics
- **Constraints**: Data type preservation and privacy protection

#### 4.1.5 Time Series Modality

- **Representations**: Univariate and multivariate time series
- **Preprocessing**: Temporal alignment, missing value handling, and normalization
- **Evaluation Metrics**: DTW distance, autocorrelation, and spectral metrics
- **Constraints**: Temporal consistency and trend preservation

#### 4.1.6 Protein Modality

- **Representations**: 3D coordinates, sequence, and structural features
- **Preprocessing**: Structural alignment, feature extraction, and normalization
- **Evaluation Metrics**: Structural similarity, energy scores, and biological metrics
- **Constraints**: Physical constraints, bond lengths, and angles

#### 4.1.7 Molecular Modality

- **Representations**: SMILES strings, molecular graphs, and 3D conformers
- **Preprocessing**: Molecule tokenization, graph construction, and conformer generation
- **Evaluation Metrics**: Validity scores, drug-likeness metrics, and molecular property prediction
- **Constraints**: Chemical validity, valence rules, and stereochemistry

#### 4.1.8 Multi-Modal Modality

- **Representations**: Combined embeddings from multiple modalities
- **Preprocessing**: Modality-specific encoders with shared latent spaces
- **Evaluation Metrics**: Cross-modal consistency and retrieval metrics
- **Constraints**: Alignment constraints across modalities

### 4.2 Modality Adapters

Each modality implements specialized adapters that customize generic models for specific data types:

```python
class ModelAdapter(Protocol):
    def adapt_model(self, model: GenerativeModel) -> GenerativeModel:
        """Adapt a generic model for specific modality requirements."""
        ...

    def preprocess(self, data: Any) -> Any:
        """Preprocess data for the specific modality."""
        ...

    def postprocess(self, outputs: Any) -> Any:
        """Postprocess model outputs for the specific modality."""
        ...
```

### 4.3 Multi-Modal Integration

Artifex supports multi-modal generation through specialized fusion strategies:

- **Concatenation**: Simple concatenation of modality representations
- **Attention-based Fusion**: Cross-modal attention mechanisms
- **Gated Fusion**: Learnable gating for modality combination
- **Hierarchical Fusion**: Multi-level fusion with different granularities

---

## 5. Scaling and Performance Infrastructure (Experimental)

*Note: The scaling infrastructure is under active development. Distributed training features are experimental and excluded from CI testing.*

### 5.1 Hardware-Aware Optimization

Artifex implements hardware detection and optimization capabilities:

```python
class HardwareDetector:
    def detect_hardware(self) -> HardwareSpecs:
        """Detect available hardware and capabilities."""
        ...

    def estimate_memory_usage(self, batch_size: int, model_size: int) -> float:
        """Estimate memory usage for given configuration."""
        ...

    def get_optimal_configuration(self, model: GenerativeModel) -> OptimizationConfig:
        """Get optimal configuration for detected hardware."""
        ...
```

The hardware detection system provides:

- **Automatic Platform Detection**: GPU, TPU, and CPU platform identification
- **Memory Estimation**: Accurate memory usage prediction for different model sizes
- **Performance Analysis**: Roofline analysis and bottleneck identification
- **Optimization Recommendations**: Hardware-specific optimization strategies

### 5.2 Multi-Dimensional Parallelism

Artifex implements sophisticated parallelism strategies for large-scale training:

#### 5.2.1 Data Parallelism

- **Standard Data Parallel**: Replicated model across devices with gradient synchronization
- **Fully Sharded Data Parallel (FSDP)**: Memory-efficient data parallelism with parameter sharding
- **Gradient Accumulation**: Support for large effective batch sizes with limited memory

#### 5.2.2 Tensor Parallelism

- **Model Parallelism**: Splitting model layers across devices
- **Pipeline Parallelism**: Sequential execution across device pipelines
- **Expert Parallelism**: Specialized parallelism for mixture-of-experts models

#### 5.2.3 Multi-Dimensional Strategies

```python
@dataclass
class ShardingConfig:
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    fsdp_enabled: bool = False
    fsdp_min_weight_size: int = 1024
```

The multi-dimensional parallelism system enables:

- **Automatic Configuration**: Optimal parallelism configuration based on model size and hardware
- **Dynamic Scaling**: Runtime adjustment of parallelism strategies
- **Memory Optimization**: Efficient memory usage across different parallelism dimensions
- **Performance Monitoring**: Real-time performance tracking and optimization

### 5.3 Device Mesh Management

Artifex provides sophisticated device mesh management for distributed training:

```python
class DeviceMeshManager:
    def create_mesh(self, mesh_shape: tuple[int, ...], axis_names: tuple[str, ...]) -> Mesh:
        """Create optimized device mesh for given configuration."""
        ...

    def optimize_for_transformer(self, device_count: int, model_size: str) -> tuple[int, ...]:
        """Optimize mesh shape for transformer workloads."""
        ...

    def validate_mesh_config(self, mesh_shape: tuple[int, ...], device_count: int) -> bool:
        """Validate mesh configuration for given hardware."""
        ...
```

The device mesh management system provides:

- **Topology Optimization**: Optimal device mesh shapes for different workloads
- **Hardware Validation**: Configuration validation against available hardware
- **Performance Prediction**: Expected performance for different mesh configurations
- **Automatic Optimization**: Hardware-aware mesh shape calculation

### 5.4 Production Optimization Pipeline (Planned)

Artifex includes a production optimization pipeline (under development):

```python
class ProductionOptimizer:
    def optimize_for_production(self, model: GenerativeModel,
                              target: OptimizationTarget,
                              sample_inputs: tuple) -> OptimizationResult:
        """Optimize model for production deployment."""
        ...
```

The production optimization pipeline provides:

- **JIT Compilation**: Just-in-time compilation for optimal performance
- **Quantization**: Model quantization for reduced memory usage
- **Pruning**: Structured and unstructured pruning for efficiency
- **Caching**: Intelligent caching strategies for repeated computations
- **Monitoring**: Real-time performance monitoring and optimization

---

## 6. Evaluation and Benchmarking Framework (In Progress)

*Note: The evaluation framework is being developed. Benchmark results should be considered preliminary and are being validated during the refactoring process.*

### 6.1 Metrics System

Artifex implements an evaluation framework with standardized metrics across modalities:

#### 6.1.1 Image Metrics

- **Fréchet Inception Distance (FID)**: Quality and diversity assessment
- **Inception Score**: Quality assessment using pre-trained classifiers
- **LPIPS**: Perceptual similarity using learned features
- **SSIM**: Structural similarity index
- **PSNR**: Peak signal-to-noise ratio

#### 6.1.2 Text Metrics

- **BLEU**: Bilingual evaluation understudy for translation quality
- **ROUGE**: Recall-oriented understudy for gisting evaluation
- **Perplexity**: Language model perplexity for coherence assessment
- **Diversity Metrics**: Lexical and semantic diversity measures

#### 6.1.3 Audio Metrics

- **Spectral Distance**: Frequency domain similarity
- **Perceptual Metrics**: Psychoacoustic quality assessment
- **Signal-to-Noise Ratio**: Audio quality measurement
- **Temporal Consistency**: Time-domain coherence assessment

#### 6.1.4 Tabular Metrics

- **Statistical Distance**: Distribution similarity measures
- **Privacy Scores**: Differential privacy and membership inference
- **Utility Metrics**: Downstream task performance
- **Data Quality**: Completeness and consistency assessment

#### 6.1.5 Time Series Metrics

- **Dynamic Time Warping (DTW)**: Temporal alignment similarity
- **Autocorrelation**: Temporal dependency preservation
- **Spectral Metrics**: Frequency domain characteristics
- **Trend Preservation**: Long-term pattern consistency

#### 6.1.6 Protein Metrics

- **Structural Similarity**: 3D structure comparison
- **Energy Scores**: Physical plausibility assessment
- **Biological Metrics**: Functional relevance measures
- **Constraint Satisfaction**: Physical constraint compliance

### 6.2 Benchmarking Infrastructure

Artifex provides a comprehensive benchmarking system:

```python
class BenchmarkSuite:
    def run_benchmark(self, model: GenerativeModel,
                     dataset: Dataset,
                     metrics: list[Metric]) -> BenchmarkResult:
        """Run comprehensive benchmark evaluation."""
        ...

    def compare_models(self, models: list[GenerativeModel]) -> ComparisonResult:
        """Compare multiple models across standardized benchmarks."""
        ...
```

The benchmarking infrastructure provides:

- **Standardized Datasets**: Pre-processed datasets for fair comparison
- **Automated Evaluation**: Automated metric computation and reporting
- **Performance Tracking**: Historical performance tracking and regression detection
- **Result Storage**: Persistent storage and analysis of benchmark results
- **CI/CD Integration**: Integration with continuous integration pipelines

### 6.3 Evaluation Pipeline

Artifex implements a flexible evaluation pipeline that automatically selects appropriate metrics:

```python
class EvaluationPipeline:
    def evaluate(self, model: GenerativeModel,
                data: ModalityData,
                modality: Modality) -> EvaluationResult:
        """Evaluate model with modality-specific metrics."""
        ...
```

The evaluation pipeline provides:

- **Automatic Metric Selection**: Appropriate metrics based on modality and model type
- **Batch Processing**: Efficient evaluation of large datasets
- **Parallel Evaluation**: Parallel metric computation for faster evaluation
- **Result Aggregation**: Statistical aggregation of evaluation results
- **Visualization**: Automatic generation of evaluation visualizations

---

## 7. Extension and Customization Framework

### 7.1 Extension Mechanism

Artifex provides a flexible extension mechanism for adding domain-specific functionality:

```python
class ModelExtension(Protocol):
    def apply(self, model: GenerativeModel,
              inputs: Any,
              outputs: Any) -> dict[str, Any]:
        """Apply extension to model outputs."""
        ...

    def compute_loss(self, batch: dict[str, Any],
                   outputs: dict[str, Any]) -> dict[str, jax.Array]:
        """Compute extension-specific loss terms."""
        ...
```

The extension system enables:

- **Domain-Specific Constraints**: Physical, biological, or domain-specific constraints
- **Custom Loss Functions**: Specialized loss functions for specific applications
- **Model Modifications**: Non-invasive model modifications and enhancements
- **Evaluation Extensions**: Custom evaluation metrics and procedures

### 7.2 Protein Extensions

Artifex includes specialized extensions for protein modeling:

```python
class BondLengthExtension(ModelExtension):
    """Extension for enforcing protein bond length constraints."""

    def compute_loss(self, batch: dict[str, Any],
                   outputs: dict[str, Any]) -> dict[str, jax.Array]:
        """Compute bond length constraint loss."""
        ...

class BondAngleExtension(ModelExtension):
    """Extension for enforcing protein bond angle constraints."""

    def compute_loss(self, batch: dict[str, Any],
                   outputs: dict[str, Any]) -> dict[str, jax.Array]:
        """Compute bond angle constraint loss."""
        ...
```

### 7.3 Custom Extension Development

The extension framework provides comprehensive support for custom extension development:

- **Base Classes**: Abstract base classes for different extension types
- **Type Safety**: Full type checking and validation for extensions
- **Documentation**: Comprehensive documentation and examples
- **Testing**: Testing utilities for extension validation
- **Integration**: Seamless integration with existing models and training pipelines

---

## 8. Implementation and Quality Assurance

*Note: Quality assurance practices are being established during the refactoring phase. Test coverage and documentation are expanding as APIs stabilize.*

### 8.1 Code Quality Standards

Artifex aims for the following code quality standards:

- **Test-Driven Development**: Test suite under active development; some tests temporarily skipped during refactoring
- **Type Safety**: Full type annotations with runtime validation via Protocol system
- **Documentation**: Docstrings and API documentation being developed
- **Code Style**: Consistent code formatting enforced via Ruff and pre-commit hooks
- **Performance**: Performance benchmarks being validated

### 8.2 Testing Infrastructure

The testing infrastructure includes:

- **Unit Tests**: Unit tests for components; coverage expanding
- **Integration Tests**: End-to-end integration testing being developed
- **Performance Tests**: Performance regression testing (experimental)
- **GPU Tests**: GPU-specific tests with automatic CPU fallback
- **Modality Tests**: Modality-specific functionality testing

*Current status: Some tests are marked as `skip` or `xfail` while APIs are being stabilized. Distributed training tests are excluded from CI.*

### 8.3 Continuous Integration

Artifex employs continuous integration:

- **Automated Testing**: Test execution on Ubuntu (CUDA) and macOS platforms
- **Code Quality**: Automated linting (Ruff) and type checking (Pyright)
- **Security Scanning**: Dependency vulnerability scanning via pip-audit
- **Platform Support**: Linux (CUDA), macOS (CPU/Metal) build verification

---

## 9. Experimental Results and Performance Analysis (Preliminary)

**Disclaimer:** The results in this section are preliminary and collected during development. All benchmark numbers require validation as the codebase stabilizes. Production performance claims have not been independently verified.

### 9.1 Model Performance Benchmarks (To Be Validated)

The following results are from development testing and should be considered preliminary:

#### 9.1.1 Image Generation Performance

- **DDPM on CIFAR-10**: FID score ~3-5 range (validation in progress)
- **VAE on MNIST**: Basic functionality verified; quantitative metrics pending
- **DCGAN on CelebA**: Training verified; FID/IS metrics being validated

#### 9.1.2 Text Generation Performance

- **Transformer models**: Basic training loop functional
- Performance benchmarks pending validation

#### 9.1.3 Audio Generation Performance

- **WaveNet**: Architecture implemented; benchmarks pending
- Audio quality metrics to be validated

#### 9.1.4 Protein Structure Generation

- **Point Cloud Models**: Physical constraint enforcement implemented
- Quantitative structure metrics pending validation

### 9.2 Scaling Performance (Experimental)

Scaling capabilities are under development:

- **Single GPU**: Basic training verified
- **Multi-GPU**: Data parallel training implemented; scaling tests pending
- **Distributed Training**: Experimental; excluded from CI testing
- **Memory Efficiency**: Optimizations in progress

### 9.3 Production Readiness

*Production deployment capabilities are not yet validated. The framework is in active development and not recommended for production use.*

---

## 10. Intended Use Cases and Applications

### 10.1 Research Applications

Artifex is designed for research applications including:

- **Novel Architecture Development**: Rapid prototyping of new generative model architectures
- **Multi-Modal Research**: Cross-modal generation and representation learning
- **Domain-Specific Applications**: Protein design, drug discovery, and materials science
- **Evaluation Studies**: Evaluation of generative model performance

### 10.2 Intended Production Applications

Once stabilized, the framework aims to support:

- **Data Augmentation**: Synthetic data generation for machine learning training
- **Scientific Computing**: Molecular and protein structure generation
- **Content Generation**: Automated content generation applications

*Note: Production deployment is not currently recommended while the framework is under active development.*

### 10.3 Educational Applications

Artifex can serve as an educational resource:

- **Graduate Courses**: Teaching generative modeling concepts with JAX/Flax
- **Workshops and Tutorials**: Hands-on exploration of generative models
- **Research Training**: Platform for learning modern generative modeling techniques

---

## 11. Future Directions and Development Roadmap

### 11.1 Current Priorities (Refactoring Phase)

The immediate development focus includes:

- **API Stabilization**: Establishing consistent interfaces across all model types
- **Test Coverage**: Expanding test suite and removing skipped tests
- **Documentation**: Completing API documentation and usage examples
- **CI/CD**: Strengthening continuous integration with comprehensive validation

### 11.2 Planned Enhancements

After stabilization, the roadmap includes:

- **DDIM Implementation**: Completing faster sampling for diffusion models
- **Stable Diffusion**: Text-to-image generation capabilities
- **Enhanced Scaling**: Validated distributed training support
- **macOS Metal Support**: Full GPU acceleration on Apple Silicon

### 11.3 Research Directions

The framework aims to enable research in:

- **Multi-Modal Fusion**: Advanced techniques for cross-modal generation
- **Efficient Sampling**: Improved sampling techniques for faster generation
- **Uncertainty Quantification**: Better uncertainty estimation in generative models
- **Interpretability**: Enhanced interpretability and explainability features

### 11.4 Community Contributions

Artifex welcomes community contributions:

- **Extension Development**: Domain-specific extensions and modalities
- **Model Implementations**: New model variants and architectures
- **Bug Fixes**: Issue reports and fixes welcome during refactoring phase
- **Documentation**: Improvements to documentation and examples

---

## 12. Conclusion

Artifex aims to provide a unified framework for generative modeling research built on JAX/Flax NNX. The framework implements seven major generative modeling paradigms with support for eight data modalities, offering researchers a platform for experimentation and development.

Current status of the Artifex framework:

1. **Unified Architecture**: Type-safe interfaces using Python Protocols across model types
2. **Multi-Modal Support**: Eight modalities including image, text, audio, tabular, time series, protein, molecular, and multi-modal
3. **JAX/Flax Foundation**: Built on modern Flax NNX patterns with JAX transformations
4. **Extensibility**: Extension mechanism for domain-specific constraints (e.g., protein modeling)
5. **Active Development**: Major refactoring in progress with APIs subject to change

The framework is currently in active development and not yet recommended for production use. APIs are being stabilized, test coverage is expanding, and performance benchmarks are being validated.

Future work focuses on completing the refactoring phase, expanding test coverage, validating performance claims, and building a community of contributors. The open-source nature of Artifex invites collaboration from the machine learning community.

---

## Acknowledgments

The authors thank the JAX and Flax development teams for their excellent frameworks. Special thanks to the broader machine learning community for inspiring the design patterns used in Artifex.

---

## References

[1] Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., ... & Wanderman-Milne, S. (2018). JAX: composable transformations of Python+NumPy programs. *arXiv preprint arXiv:1806.01572*.

[2] Heek, J., Levskaya, A., Oliver, A., Ritter, M., Rondepierre, B., Steiner, A., & van Zee, M. (2023). Flax: A neural network library and ecosystem for JAX. *GitHub repository*.

[3] Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*.

[4] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. *Advances in neural information processing systems*, 27.

[5] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33, 6840-6851.

[6] Rezende, D., & Mohamed, S. (2015). Variational inference with normalizing flows. *International conference on machine learning* (pp. 1530-1538).

[7] Van Den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016). Wavenet: A generative model for raw audio. *arXiv preprint arXiv:1609.03499*.

[8] LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006). A tutorial on energy-based learning. *Predicting structured data*, 1(0).

[9] Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). Pointnet: Deep learning on point sets for 3d classification and segmentation. *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 652-660).

[10] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.

---

## Appendix A: Installation and Quick Start

### A.1 Installation

```bash
# Clone the repository
git clone https://github.com/avitai/artifex.git
cd artifex

# Run setup script (recommended)
# For Linux with CUDA:
./setup.sh

# For macOS:
./setup.sh --cpu
# Or with Metal acceleration (Apple Silicon):
./setup.sh --metal

# Activate the environment
source activate.sh

# Alternative: Manual installation with uv
uv sync --extra cuda-dev  # Linux with CUDA
uv sync --extra all-cpu   # CPU only
uv sync --extra all-macos # macOS with Metal
```

### A.2 Quick Start Example

*Note: APIs are subject to change during development. Check the latest documentation for current usage.*

```python
import jax
from flax import nnx
from artifex.generative_models.models.vae import VAE
from artifex.generative_models.core.configuration.vae_config import (
    VAEConfig, EncoderConfig, DecoderConfig
)

# Create random number generator
key = jax.random.key(0)
rngs = nnx.Rngs(params=key)

# Create VAE configuration
encoder_config = EncoderConfig(
    hidden_dims=[32, 64],
    latent_dim=16,
)
decoder_config = DecoderConfig(
    hidden_dims=[64, 32],
    output_dim=(28, 28, 1),
)
config = VAEConfig(
    encoder=encoder_config,
    decoder=decoder_config,
    latent_dim=16,
)

# Initialize model (rngs passed only during initialization)
model = VAE(config, rngs=rngs)

# Forward pass (no rngs argument)
batch = jax.random.normal(key, (4, 28, 28, 1))
outputs = model(batch)

# Generate samples (no rngs argument)
samples = model.sample(n_samples=4)
print(f"Generated samples shape: {samples.shape}")
```

---

## Appendix B: API Reference

### B.1 Core Classes

#### GenerativeModelProtocol

The base protocol for all generative models in Artifex.

#### ModelConfiguration

Configuration class for model instantiation and parameter management.

#### ModelRegistry

Registry system for model discovery and creation.

### B.2 Modality Classes

#### Modality

Base protocol for data modality implementations.

#### ModelAdapter

Protocol for adapting generic models to specific modalities.

### B.3 Evaluation Classes

#### EvaluationPipeline

Main evaluation pipeline for model assessment.

#### MetricsRegistry

Registry for evaluation metrics.

#### BenchmarkSuite

Comprehensive benchmarking system.

---

## Appendix C: Hardware Requirements

### C.1 Supported Platforms

- **Linux**: Ubuntu 20.04+, CUDA 11.8+ for GPU acceleration
- **macOS**: macOS 12+ (Intel or Apple Silicon), optional Metal acceleration

### C.2 Hardware Requirements

- **Development**: 16GB RAM, CUDA-compatible GPU recommended
- **Training**: 32GB+ RAM, RTX 3080 or better for reasonable training times

### C.3 Benchmark Results

Benchmark validation is in progress. Results will be published as they are validated.

---

## Appendix D: Contributing Guidelines

### D.1 Development Setup

```bash
# Clone repository
git clone https://github.com/avitai/artifex.git
cd artifex

# Run setup script
./setup.sh

# Activate environment
source activate.sh

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest tests/ -v
```

### D.2 Code Style

Artifex follows these code style guidelines:

- Ruff formatting (replaces Black)
- Type annotations required (Pyright for type checking)
- Docstrings for public APIs
- Test coverage expanding with refactoring

### D.3 Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure all tests pass
5. Submit pull request

---

*This preprint describes Artifex, a generative modeling framework under active development. The source code is available at <https://github.com/avitai/artifex>. Note that APIs are subject to change and the framework is not yet recommended for production use.*
