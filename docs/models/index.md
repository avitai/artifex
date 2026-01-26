# Model Gallery

Artifex provides implementations of state-of-the-art generative models with 2025 research compliance, including Diffusion Transformers (DiT), SE(3) equivariant flows for molecular generation, and advanced MCMC sampling with BlackJAX integration.

<div class="grid cards" markdown>

- :material-brain:{ .lg .middle } **7 Model Families**

    ---

    VAE, GAN, Diffusion, Flow, EBM, Autoregressive, and Geometric models with 67+ implementations

- :material-rocket-launch:{ .lg .middle } **2025 Research Compliance**

    ---

    Latest architectures including DiT, StyleGAN3, SE(3) molecular flows, and score-based diffusion

- :material-lightning-bolt:{ .lg .middle } **Production Ready**

    ---

    Hardware-optimized, fully tested, type-safe implementations built on JAX/Flax NNX

- :material-puzzle:{ .lg .middle } **Multi-Modal**

    ---

    Native support for images, text, audio, proteins, molecules, and 3D geometric data

</div>

## Overview

All models in Artifex follow a unified interface and are built on JAX/Flax NNX for:

- **Hardware acceleration**: Automatic GPU/TPU support with XLA optimization
- **Type safety**: Full type annotations and protocol-based interfaces
- **Composability**: Mix and match components across model types
- **Reproducibility**: Deterministic RNG handling and comprehensive testing
- **Scalability**: Distributed training with data, model, and pipeline parallelism

## Model Families

### <span class="model-vae">VAE</span> Variational Autoencoders

Latent variable models with probabilistic encoding for representation learning and generation.

**Available Models:**

| Model | Description | Key Features | Use Cases |
|-------|-------------|--------------|-----------|
| **VAE** | Standard Variational Autoencoder | Gaussian latents, KL regularization | Representation learning, compression |
| **β-VAE** | Disentangled VAE | Controllable β parameter, beta annealing | Disentangled representations |
| **β-VAE with Capacity** | β-VAE with capacity control | Gradual capacity increase, controlled disentanglement | Balance reconstruction and disentanglement |
| **Conditional VAE** | Class-conditional VAE | Label conditioning, controlled generation | Supervised generation |
| **VQ-VAE** | Vector Quantized VAE | Discrete latent codes, codebook learning | Discrete representations, compression |

**Quick Start:**

```python
from artifex.generative_models.models.vae import VAE
from artifex.generative_models.core.configuration import VAEConfig, EncoderConfig, DecoderConfig
from flax import nnx

encoder = EncoderConfig(name="encoder", input_shape=(32, 32, 3), latent_dim=128, hidden_dims=(64, 128, 256), activation="relu")
decoder = DecoderConfig(name="decoder", latent_dim=128, output_shape=(32, 32, 3), hidden_dims=(256, 128, 64), activation="relu")
config = VAEConfig(name="vae", encoder=encoder, decoder=decoder, encoder_type="cnn", kl_weight=1.0)

model = VAE(config, rngs=nnx.Rngs(0))
```

**Documentation:**

- [VAE User Guide](../user-guide/models/vae-guide.md) - Complete guide with examples
- [VAE API Reference](../api/models/vae.md) - Detailed API documentation

---

### <span class="model-gan">GAN</span> Generative Adversarial Networks

Adversarial training for high-quality image generation and image-to-image translation.

**Available Models:**

| Model | Description | Key Features | Use Cases |
|-------|-------------|--------------|-----------|
| **DCGAN** | Deep Convolutional GAN | Convolutional architecture, stable training | Image generation baseline |
| **WGAN** | Wasserstein GAN | Wasserstein distance, critic training | Stable training, mode coverage |
| **LSGAN** | Least Squares GAN | Least squares loss, improved stability | Image generation with stable training |
| **StyleGAN** | Style-based GAN | Style mixing, AdaIN layers | High-quality face generation |
| **StyleGAN3** | Alias-free StyleGAN | Translation/rotation equivariance | Alias-free high-quality generation |
| **CycleGAN** | Cycle-consistent GAN | Unpaired translation, cycle loss | Image-to-image translation |
| **PatchGAN** | Patch-based discriminator | Local image patches, texture detail | Image-to-image tasks, super-resolution |
| **Conditional GAN** | Class-conditional GAN | Label conditioning | Controlled generation |

**Quick Start:**

```python
from artifex.generative_models.models.gan import DCGAN
from artifex.generative_models.core.configuration import DCGANConfig, GeneratorConfig, DiscriminatorConfig
from flax import nnx

generator = GeneratorConfig(name="generator", latent_dim=100, features=(512, 256, 128, 64))
discriminator = DiscriminatorConfig(name="discriminator", features=(64, 128, 256, 512))
config = DCGANConfig(name="dcgan", image_shape=(3, 64, 64), generator=generator, discriminator=discriminator)

model = DCGAN(config, rngs=nnx.Rngs(0))
```

**Documentation:**

- [GAN User Guide](../user-guide/models/gan-guide.md) - Complete guide with training tips
- [GAN API Reference](../api/models/gan.md) - Detailed API documentation

---

### <span class="model-diffusion">Diffusion</span> Diffusion Models

State-of-the-art denoising diffusion models for high-quality generation.

**Available Models:**

| Model | Description | Key Features | Use Cases |
|-------|-------------|--------------|-----------|
| **DDPM** | Denoising Diffusion Probabilistic Models | Gaussian diffusion, noise prediction | Image generation, baseline |
| **DDIM** | Denoising Diffusion Implicit Models | Deterministic sampling, faster inference | Fast high-quality generation |
| **Score-based** | Score-based generative models | Score matching, SDE/ODE solvers | Flexible sampling strategies |
| **Latent Diffusion** | Latent space diffusion | VAE encoder/decoder, efficient training | High-resolution generation |
| **DiT** | Diffusion Transformer | Transformer backbone, scalable | Large-scale image generation |
| **Stable Diffusion** | Text-to-image diffusion | CLIP conditioning, latent diffusion | Text-to-image generation |

**Quick Start:**

```python
from artifex.generative_models.models.diffusion import DDPMModel
from artifex.generative_models.core.configuration import DDPMConfig, UNetBackboneConfig, NoiseScheduleConfig
from flax import nnx

backbone = UNetBackboneConfig(name="backbone", in_channels=3, out_channels=3, base_channels=128, channel_mults=(1, 2, 4))
noise_schedule = NoiseScheduleConfig(name="schedule", schedule_type="linear", num_timesteps=1000, beta_start=1e-4, beta_end=2e-2)
config = DDPMConfig(name="ddpm", input_shape=(3, 32, 32), backbone=backbone, noise_schedule=noise_schedule)

model = DDPMModel(config, rngs=nnx.Rngs(0))
```

**Documentation:**

- [Diffusion User Guide](../user-guide/models/diffusion-guide.md) - Complete guide with sampling methods
- [Diffusion API Reference](diffusion.md) - Detailed API documentation
- [DiT Architecture](dit.md) - Diffusion Transformer details

---

### <span class="model-flow">Flow</span> Normalizing Flows

Invertible transformations with tractable likelihoods for exact density estimation.

**Available Models:**

| Model | Description | Key Features | Use Cases |
|-------|-------------|--------------|-----------|
| **RealNVP** | Real-valued Non-Volume Preserving | Affine coupling layers, multi-scale | Density estimation baseline |
| **Glow** | Generative Flow | Invertible 1x1 convolutions, ActNorm | High-quality image generation |
| **MAF** | Masked Autoregressive Flow | Autoregressive coupling, parallel training | Flexible density estimation |
| **IAF** | Inverse Autoregressive Flow | Fast sampling, parallel generation | Variational inference |
| **Neural Spline Flow** | Spline-based coupling | Smooth transformations, expressive | High-quality density estimation |
| **SE(3) Molecular Flow** | Equivariant molecular flows | SE(3) symmetry, molecular generation | Drug design, molecular modeling |
| **Conditional Flow** | Class-conditional flows | Label conditioning | Controlled generation |

**Quick Start:**

```python
from artifex.generative_models.models.flow import NormalizingFlow
from artifex.generative_models.core.configuration import FlowConfig
from flax import nnx

config = FlowConfig(name="realnvp", input_shape=(32, 32, 3), num_flows=8, coupling_type="affine")

model = NormalizingFlow(config, rngs=nnx.Rngs(0))
```

**Documentation:**

- [Flow User Guide](../user-guide/models/flow-guide.md) - Complete guide with coupling layers
- [Flow API Reference](../api/models/flow.md) - Detailed API documentation
- [SE(3) Molecular Flows](se3_molecular.md) - Equivariant flows for molecules

---

### <span class="model-ebm">EBM</span> Energy-Based Models

Energy function learning with MCMC sampling for compositional generation.

**Available Models:**

| Model | Description | Key Features | Use Cases |
|-------|-------------|--------------|-----------|
| **EBM** | Energy-based model | Energy function learning, flexible | Compositional generation |
| **EBM with MCMC** | EBM with MCMC sampling | Langevin dynamics, HMC, NUTS | High-quality sampling |
| **Persistent CD** | Persistent Contrastive Divergence | Persistent chains, efficient training | Stable EBM training |

**MCMC Samplers** (via BlackJAX integration):

- **HMC**: Hamiltonian Monte Carlo
- **NUTS**: No-U-Turn Sampler (adaptive HMC)
- **MALA**: Metropolis-Adjusted Langevin Algorithm
- **Langevin Dynamics**: First-order gradient-based sampling

**Quick Start:**

```python
from artifex.generative_models.models.energy import EBM
from artifex.generative_models.core.configuration import EBMConfig, EnergyNetworkConfig, MCMCConfig
from flax import nnx

energy_network = EnergyNetworkConfig(name="energy_net", hidden_dims=(512, 256, 128), activation="swish")
mcmc = MCMCConfig(name="mcmc", n_steps=100, step_size=0.01)
config = EBMConfig(name="ebm", input_dim=784, energy_network=energy_network, mcmc=mcmc)

model = EBM(config, rngs=nnx.Rngs(0))
```

**Documentation:**

- [EBM Guide](ebm.md) - Energy-based model details
- [MCMC Sampling](mcmc.md) - Sampling algorithms

---

### <span class="model-autoregressive">AR</span> Autoregressive Models

Sequential generation with explicit likelihood for ordered data.

**Available Models:**

| Model | Description | Key Features | Use Cases |
|-------|-------------|--------------|-----------|
| **PixelCNN** | Autoregressive image model | Masked convolutions, pixel-by-pixel | Image generation with likelihood |
| **WaveNet** | Autoregressive audio model | Dilated convolutions, long context | Audio generation, TTS |
| **Transformer** | Transformer-based AR | Self-attention, parallel training | Text, structured sequences |

**Quick Start:**

```python
from artifex.generative_models.models.autoregressive import PixelCNN
from artifex.generative_models.core.configuration import PixelCNNConfig
from flax import nnx

config = PixelCNNConfig(
    name="pixelcnn",
    image_shape=(32, 32, 3),
    num_layers=8,
    hidden_channels=128,
    kernel_size=3,
    num_classes=256
)

model = PixelCNN(config, rngs=nnx.Rngs(0))
```

**Documentation:**

- [PixelCNN API](pixel_cnn.md) - Image autoregressive models
- [WaveNet API](wavenet.md) - Audio autoregressive models
- [Transformer API](transformer.md) - Transformer-based models

---

### <span class="model-geometric">Geometric</span> Geometric Models

3D structure generation with physical constraints and equivariance.

**Available Models:**

| Model | Description | Key Features | Use Cases |
|-------|-------------|--------------|-----------|
| **Point Cloud Generator** | 3D point cloud generation | Permutation invariance | 3D object generation |
| **Mesh Generator** | 3D mesh generation | Vertex/face generation, deformation | 3D modeling |
| **Protein Graph** | Protein structure generation | Residue graphs, amino acid features | Protein design |
| **Protein Point Cloud** | Protein backbone generation | Cα coordinates, backbone geometry | Protein structure prediction |
| **Voxel Generator** | Voxel-based 3D generation | Regular 3D grid | 3D shape generation |
| **Graph Generator** | Graph-based generation | Node/edge features, message passing | Molecular graphs |

**Quick Start:**

```python
from artifex.generative_models.models.geometric import PointCloudModel
from artifex.generative_models.core.configuration import PointCloudConfig, PointCloudNetworkConfig
from flax import nnx

network = PointCloudNetworkConfig(name="network", embed_dim=256, num_heads=8, num_layers=6)
config = PointCloudConfig(name="point_cloud", num_points=128, point_dim=3, network=network)

model = PointCloudModel(config, rngs=nnx.Rngs(0))
```

**Documentation:**

- [Protein Models](protein_graph.md) - Protein structure generation
- [Point Cloud Models](point_cloud.md) - 3D point cloud generation
- [Graph Models](graph.md) - Graph-based generation

---

## Model Comparison

Choose the right model for your task:

| Model Type | Sample Quality | Training Stability | Speed | Exact Likelihood | Best For |
|------------|---------------|-------------------|-------|------------------|----------|
| <span class="model-vae">VAE</span> | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ (Lower bound) | Representation learning, fast sampling |
| <span class="model-gan">GAN</span> | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ❌ | High-quality images, style transfer |
| <span class="model-diffusion">Diffusion</span> | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ | State-of-the-art generation, controllability |
| <span class="model-flow">Flow</span> | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | Density estimation, exact inference |
| <span class="model-ebm">EBM</span> | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ❌ (unnormalized) | Compositional generation, flexibility |
| <span class="model-autoregressive">AR</span> | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ | Sequences, explicit probabilities |
| <span class="model-geometric">Geometric</span> | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Varies | 3D structures, physical constraints |

## Common Backbones

Artifex provides reusable backbone architectures used across model types:

### U-Net

Widely used in diffusion models and image-to-image tasks:

```python
from artifex.generative_models.models.common.unet import UNet

unet = UNet(
    in_channels=3,
    out_channels=3,
    channels=[128, 256, 512, 1024],
    num_res_blocks=2,
    attention_resolutions=[16, 8],
    rngs=nnx.Rngs(0)
)
```

**Documentation:** [U-Net API](unet.md)

### Diffusion Transformer (DiT)

Transformer-based backbone for diffusion models:

```python
from artifex.generative_models.models.diffusion.dit import DiT

dit = DiT(
    input_size=32,
    patch_size=2,
    in_channels=3,
    hidden_size=768,
    depth=12,
    num_heads=12,
    rngs=nnx.Rngs(0)
)
```

**Documentation:** [DiT API](dit.md)

### Encoders & Decoders

Modular encoder/decoder architectures:

- **MLP Encoder/Decoder**: Fully-connected networks
- **CNN Encoder/Decoder**: Convolutional networks for images
- **Conditional Encoder/Decoder**: Class-conditional variants
- **ResNet Encoder/Decoder**: Residual connections

**Documentation:** [Encoders](encoders.md) | [Decoders](decoders.md)

## Conditioning Methods

Artifex supports multiple conditioning strategies across models:

| Method | Description | Supported Models | Use Cases |
|--------|-------------|------------------|-----------|
| **Class Conditioning** | One-hot encoded labels | VAE, GAN, Diffusion, Flow | Supervised generation |
| **Text Conditioning** | CLIP embeddings | Diffusion, GAN | Text-to-image |
| **Image Conditioning** | Concatenation, cross-attention | GAN, Diffusion | Image-to-image, inpainting |
| **Embedding Conditioning** | Learned embeddings | All models | Flexible conditioning |

**Documentation:** [Conditioning Guide](conditioning.md)

## Model Registry

All models are registered in a global registry for easy instantiation:

```python
from artifex.generative_models.models.registry import (
    list_models,
    get_model_class,
    register_model
)

# List all available models
available = list_models()
print(f"Available models: {len(available)}")

# Get model class by name
vae_class = get_model_class("vae")

# Register custom model
from my_models import CustomVAE
register_model("custom_vae", CustomVAE)
```

**Documentation:** [Model Registry](registry.md)

## Training

All models follow a unified training interface:

```python
from artifex.generative_models.training import Trainer
from artifex.generative_models.core.configuration import TrainingConfig

training_config = TrainingConfig(
    batch_size=128,
    num_epochs=100,
    optimizer={"type": "adam", "learning_rate": 1e-3},
    scheduler={"type": "cosine", "warmup_steps": 1000}
)

trainer = Trainer(
    model=model,
    training_config=training_config,
    train_dataset=train_data,
    val_dataset=val_data
)

trainer.train()
```

**Documentation:**

- [Training Guide](../user-guide/training/training-guide.md) - Complete training guide
- [Distributed Training](../user-guide/advanced/distributed.md) - Multi-GPU/TPU training

## Evaluation

Evaluate models with modality-specific metrics:

```python
from artifex.benchmarks import EvaluationFramework

framework = EvaluationFramework(
    model=model,
    modality="image",
    metrics=["fid", "inception_score", "lpips"]
)

results = framework.evaluate(test_dataset)
print(results)
```

**Documentation:** [Benchmarks](../benchmarks/index.md)

## Examples

Hands-on examples for each model family:

- [VAE on MNIST](../examples/basic/vae-mnist.md) - Basic VAE training
- [GAN on MNIST](../examples/basic/simple-gan.md) - Image generation
- [Diffusion on MNIST](../examples/basic/diffusion-mnist.md) - Image generation
- [Flow on MNIST](../examples/basic/flow-mnist.md) - Density estimation
- [Protein generation](../examples/protein/protein-diffusion-example.md) - Geometric models

## Contributing

Add new models to Artifex:

1. Implement model following protocols in `core/protocols.py`
2. Add to appropriate directory (vae/, gan/, diffusion/, etc.)
3. Register in model registry
4. Add comprehensive tests
5. Document API and usage

**Documentation:** [Contributing Guide](../community/contributing.md)

## API Statistics

Current model coverage:

- **Total modules**: 67
- **Total classes**: 135
- **Total functions**: 482
- **Model families**: 7
- **Conditioning methods**: 4
- **Sampling methods**: 15+

---

<div class="grid cards" markdown>

- :material-book-open-variant:{ .lg .middle } **User Guides**

    ---

    Deep dive into each model family with examples and best practices

    [:octicons-arrow-right-24: Browse guides](../user-guide/models/vae-guide.md)

- :material-api:{ .lg .middle } **API Reference**

    ---

    Complete API documentation for all models and components

    [:octicons-arrow-right-24: API docs](base.md)

- :material-school:{ .lg .middle } **Tutorials**

    ---

    Step-by-step tutorials for common tasks and workflows

    [:octicons-arrow-right-24: Start learning](../getting-started/quickstart.md)

- :material-flask:{ .lg .middle } **Examples**

    ---

    Working code examples for all model types

    [:octicons-arrow-right-24: See examples](../examples/index.md)

</div>
