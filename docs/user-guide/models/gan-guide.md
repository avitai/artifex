# GAN User Guide

This guide provides practical instructions for training and using Generative Adversarial Networks (GANs) in Artifex. We cover all GAN variants, training strategies, common issues, and best practices.

## Quick Start

Here's a minimal example to get you started:

```python
import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.network_configs import (
    GeneratorConfig,
    DiscriminatorConfig,
)
from artifex.generative_models.models.gan import Generator, Discriminator
from artifex.generative_models.core.losses.adversarial import (
    vanilla_generator_loss,
    vanilla_discriminator_loss,
)

# Initialize RNGs (discriminator needs dropout stream)
gen_rngs = nnx.Rngs(params=0, sample=1, dropout=2)
disc_rngs = nnx.Rngs(params=3, dropout=4)

# Create generator config
gen_config = GeneratorConfig(
    name="simple_generator",
    hidden_dims=(256, 512),         # Use tuples, not lists
    output_shape=(28, 28, 1),       # MNIST shape (H, W, C)
    latent_dim=100,
    activation="relu",
    batch_norm=True,
    dropout_rate=0.0,
)

# Create discriminator config
disc_config = DiscriminatorConfig(
    name="simple_discriminator",
    input_shape=(28, 28, 1),
    hidden_dims=(512, 256),
    activation="leaky_relu",
    batch_norm=False,
    dropout_rate=0.3,
)

# Create models
generator = Generator(config=gen_config, rngs=gen_rngs)
discriminator = Discriminator(config=disc_config, rngs=disc_rngs)

# Generate samples
z = jax.random.normal(jax.random.key(0), (16, 100))
generator.eval()  # Set to evaluation mode
samples = generator(z)
print(f"Generated samples shape: {samples.shape}")  # (16, 28, 28, 1)
```

## Creating GAN Components

Artifex uses a **config-based API** for creating GAN components. Define configurations first, then create models from them.

### Basic Generator

The generator transforms random noise into data samples:

```python
from artifex.generative_models.core.configuration.network_configs import GeneratorConfig
from artifex.generative_models.models.gan import Generator

# Create generator configuration
gen_config = GeneratorConfig(
    name="image_generator",
    hidden_dims=(128, 256, 512),      # Hidden layer sizes (use tuples)
    output_shape=(32, 32, 3),         # Output: 32x32 RGB images (H, W, C)
    latent_dim=100,                    # Latent space dimension
    activation="relu",                 # Activation function
    batch_norm=True,                   # Use batch normalization
    dropout_rate=0.0,                  # Dropout rate (usually 0 for generator)
)

# Create generator from config
rngs = nnx.Rngs(params=0, sample=1)
generator = Generator(config=gen_config, rngs=rngs)

# Generate samples from random noise
z = jax.random.normal(jax.random.key(0), (batch_size, 100))
generator.train()  # Set to training mode
fake_samples = generator(z)
```

**Key Parameters:**

- `hidden_dims`: Tuple of hidden layer dimensions (progressively increases capacity)
- `output_shape`: Target data shape (height, width, channels)
- `latent_dim`: Size of input latent vector (typically 64-512)
- `batch_norm`: Stabilizes training (recommended for generator)
- `activation`: "relu" for generator, "leaky_relu" for discriminator

### Basic Discriminator

The discriminator classifies samples as real or fake:

```python
from artifex.generative_models.core.configuration.network_configs import DiscriminatorConfig
from artifex.generative_models.models.gan import Discriminator

# Create discriminator configuration
disc_config = DiscriminatorConfig(
    name="image_discriminator",
    input_shape=(32, 32, 3),          # Input image shape (H, W, C)
    hidden_dims=(512, 256, 128),      # Hidden layer sizes (often mirrors generator)
    activation="leaky_relu",           # LeakyReLU prevents dying neurons
    batch_norm=False,                  # Usually False for discriminator
    dropout_rate=0.3,                  # Dropout to prevent overfitting
)

# Create discriminator from config
disc_rngs = nnx.Rngs(params=2)
discriminator = Discriminator(config=disc_config, rngs=disc_rngs)

# Classify samples
discriminator.train()  # Set to training mode
real_data = jnp.ones((batch_size, 32, 32, 3))
fake_data = generator(z)

real_scores = discriminator(real_data)  # Should be close to 1
fake_scores = discriminator(fake_data)  # Should be close to 0
```

**Key Parameters:**

- `hidden_dims`: Usually mirrors generator in reverse
- `activation`: "leaky_relu" is standard (slope 0.2)
- `batch_norm`: Usually False (can cause training issues)
- `dropout_rate`: 0.3-0.5 helps prevent overfitting

## GAN Variants

### 1. Vanilla GAN

The original GAN formulation with binary cross-entropy loss:

```python
from artifex.generative_models.core.configuration.network_configs import (
    GeneratorConfig,
    DiscriminatorConfig,
)
from artifex.generative_models.core.configuration.gan_config import GANConfig
from artifex.generative_models.models.gan import GAN

# Create nested configs
gen_config = GeneratorConfig(
    name="vanilla_generator",
    hidden_dims=(256, 512),
    output_shape=(28, 28, 1),
    latent_dim=100,
    activation="relu",
    batch_norm=True,
)

disc_config = DiscriminatorConfig(
    name="vanilla_discriminator",
    input_shape=(28, 28, 1),
    hidden_dims=(512, 256),
    activation="leaky_relu",
    batch_norm=False,
    dropout_rate=0.3,
)

# Create GAN config
gan_config = GANConfig(
    name="vanilla_gan",
    generator=gen_config,
    discriminator=disc_config,
    loss_type="vanilla",
    generator_lr=0.0002,
    discriminator_lr=0.0002,
)

# Create GAN
rngs = nnx.Rngs(params=0, sample=1, dropout=2)
gan = GAN(config=gan_config, rngs=rngs)

# Generate samples
samples = gan.generate(n_samples=16)
```

**When to use:**

- Learning GANs for the first time
- Simple datasets (MNIST, simple shapes)
- Proof-of-concept experiments

**Pros:** Simple, well-understood
**Cons:** Training instability, mode collapse

### 2. Deep Convolutional GAN (DCGAN)

Uses convolutional architecture for images:

```python
from artifex.generative_models.core.configuration.network_configs import (
    ConvGeneratorConfig,
    ConvDiscriminatorConfig,
)
from artifex.generative_models.core.configuration.gan_config import DCGANConfig
from artifex.generative_models.models.gan import DCGAN

# Create convolutional generator config
conv_gen_config = ConvGeneratorConfig(
    name="dcgan_generator",
    latent_dim=100,
    hidden_dims=(512, 256, 128, 64),  # Feature map progression
    output_shape=(64, 64, 3),         # 64x64 RGB images (H, W, C)
    activation="relu",
    batch_norm=True,
    kernel_size=(4, 4),
    stride=(2, 2),
    padding="SAME",
)

# Create convolutional discriminator config
conv_disc_config = ConvDiscriminatorConfig(
    name="dcgan_discriminator",
    hidden_dims=(64, 128, 256, 512),  # Mirrored progression
    input_shape=(64, 64, 3),
    activation="leaky_relu",
    batch_norm=True,
    kernel_size=(4, 4),
    stride=(2, 2),
    padding="SAME",
)

# Create DCGAN config with nested network configs
dcgan_config = DCGANConfig(
    name="dcgan_64",
    generator=conv_gen_config,
    discriminator=conv_disc_config,
    loss_type="vanilla",
    generator_lr=0.0002,
    discriminator_lr=0.0002,
    beta1=0.5,
    beta2=0.999,
)

# Create DCGAN
rngs = nnx.Rngs(params=0, sample=1, dropout=2)
dcgan = DCGAN(config=dcgan_config, rngs=rngs)

# Generate high-quality images
dcgan.eval()  # Set to evaluation mode
samples = dcgan.generate(n_samples=64)
```

**DCGAN Architecture Guidelines:**

1. Replace pooling with strided convolutions
2. Use batch normalization (except discriminator input and generator output)
3. Remove fully connected layers (except for latent projection)
4. Use ReLU in generator, LeakyReLU in discriminator
5. Use Tanh activation in generator output

**When to use:**

- Image generation tasks
- 64×64 to 128×128 resolution
- More stable training than vanilla GAN

**Pros:** More stable, better image quality
**Cons:** Still can suffer from mode collapse

### 3. Wasserstein GAN (WGAN)

Uses Wasserstein distance for more stable training:

```python
from artifex.generative_models.models.gan import WGAN, compute_gradient_penalty
from artifex.generative_models.core.configuration.gan_config import WGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    ConvGeneratorConfig,
    ConvDiscriminatorConfig,
)

# Create WGAN with proper typed configs
generator_config = ConvGeneratorConfig(
    name="wgan_generator",
    latent_dim=100,
    hidden_dims=(512, 256, 128, 64),
    output_shape=(3, 64, 64),
    activation="relu",
    batch_norm=True,
    kernel_size=(4, 4),
    stride=(2, 2),
    padding="SAME",
)

discriminator_config = ConvDiscriminatorConfig(
    name="wgan_critic",
    hidden_dims=(64, 128, 256, 512),
    input_shape=(3, 64, 64),
    activation="leaky_relu",
    kernel_size=(4, 4),
    stride=(2, 2),
    padding="SAME",
    use_instance_norm=True,
)

wgan_config = WGANConfig(
    name="wgan",
    generator=generator_config,
    discriminator=discriminator_config,
    critic_iterations=5,               # Update critic 5x per generator
    use_gradient_penalty=True,
    gradient_penalty_weight=10.0,      # Lambda for gradient penalty
)

wgan = WGAN(config=wgan_config, rngs=rngs)

# Training loop for WGAN
def train_wgan_step(wgan, real_samples, rngs, n_critic=5):
    """Train WGAN with proper critic/generator balance."""

    # Train critic n_critic times
    for _ in range(n_critic):
        z = jax.random.normal(rngs.sample(), (real_samples.shape[0], wgan.latent_dim))
        fake_samples = wgan.generator(z)

        # Compute discriminator loss with gradient penalty
        disc_loss = wgan.discriminator_loss(real_samples, fake_samples, rngs)

        # Update discriminator (use nnx.Optimizer in practice)

    # Train generator once
    z = jax.random.normal(rngs.sample(), (real_samples.shape[0], wgan.latent_dim))
    fake_samples = wgan.generator(z)
    gen_loss = wgan.generator_loss(fake_samples)

    return {"disc_loss": disc_loss, "gen_loss": gen_loss}
```

**Key Differences from Vanilla GAN:**

1. **Critic instead of discriminator** (no sigmoid at output)
2. **Wasserstein distance** instead of JS divergence
3. **Gradient penalty** enforces Lipschitz constraint
4. **Multiple critic updates** per generator update (5:1 ratio)
5. **Instance normalization** instead of batch norm in critic

**When to use:**

- Need stable training
- Want meaningful loss metric
- High-resolution images
- Research experiments

**Pros:** Very stable, meaningful loss, better mode coverage
**Cons:** Slower training, more complex

### 4. Least Squares GAN (LSGAN)

Uses least squares loss for smoother gradients:

```python
from artifex.generative_models.models.gan import LSGAN
from artifex.generative_models.core.configuration.gan_config import LSGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    ConvGeneratorConfig,
    ConvDiscriminatorConfig,
)

# Create LSGAN with typed configs
generator_config = ConvGeneratorConfig(
    name="lsgan_generator",
    latent_dim=100,
    hidden_dims=(512, 256, 128, 64),
    output_shape=(3, 64, 64),
    activation="relu",
    batch_norm=True,
    kernel_size=(4, 4),
    stride=(2, 2),
    padding="SAME",
)

discriminator_config = ConvDiscriminatorConfig(
    name="lsgan_discriminator",
    hidden_dims=(64, 128, 256, 512),
    input_shape=(3, 64, 64),
    activation="leaky_relu",
    kernel_size=(4, 4),
    stride=(2, 2),
    padding="SAME",
)

lsgan_config = LSGANConfig(
    name="lsgan",
    generator=generator_config,
    discriminator=discriminator_config,
    a=0.0,   # Target for fake samples in discriminator
    b=1.0,   # Target for real samples in discriminator
    c=1.0,   # Target for fake samples in generator
)

lsgan = LSGAN(config=lsgan_config, rngs=rngs)
```

**Key Difference:**

Loss function uses squared error instead of log loss:

- **Generator:** Minimize $(D(G(z)) - 1)^2$
- **Discriminator:** Minimize $(D(x) - 1)^2 + D(G(z))^2$

**When to use:**

- Want smoother gradients than vanilla GAN
- Need more stable training than vanilla
- Image generation with less training instability

**Pros:** More stable than vanilla, penalizes far-from-boundary samples
**Cons:** Still can mode collapse

### 5. Conditional GAN (cGAN)

Conditions generation on labels or other information:

```python
from artifex.generative_models.models.gan import (
    ConditionalGAN,
    ConditionalGenerator,
    ConditionalDiscriminator,
)
from artifex.generative_models.core.configuration.gan_config import ConditionalGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    ConditionalParams,
    ConditionalGeneratorConfig,
    ConditionalDiscriminatorConfig,
)

# Shared conditional parameters
cond_params = ConditionalParams(num_classes=10, embedding_dim=50)

# Create conditional generator config
gen_config = ConditionalGeneratorConfig(
    name="cgan_generator",
    latent_dim=100,
    hidden_dims=(512, 256, 128, 64),
    output_shape=(1, 28, 28),
    activation="relu",
    batch_norm=True,
    conditional=cond_params,
)

# Create conditional discriminator config
disc_config = ConditionalDiscriminatorConfig(
    name="cgan_discriminator",
    hidden_dims=(64, 128, 256, 512),
    input_shape=(1, 28, 28),
    activation="leaky_relu",
    conditional=cond_params,
)

# Create the conditional GAN
cgan_config = ConditionalGANConfig(
    name="conditional_gan",
    generator=gen_config,
    discriminator=disc_config,
)
cgan = ConditionalGAN(config=cgan_config, rngs=rngs)

# Generate conditioned on class labels (one-hot encoded)
labels = jax.nn.one_hot(jnp.arange(10), 10)  # One of each digit
z = jax.random.normal(rngs.sample(), (10, 100))

# Generate specific digits
cgan.eval()
samples = cgan.generator(z, labels)

# Discriminate with labels
cgan.train()
real_scores = cgan.discriminator(real_data, real_labels)
fake_scores = cgan.discriminator(samples, labels)
```

**Key Features:**

- **Controlled generation**: Specify what to generate
- **Class conditioning**: Generate specific categories
- **Embedding layer**: Maps labels to high-dimensional space
- **Concatenation**: Combines embeddings with features

**When to use:**

- Need to control generation (class, attributes)
- Have labeled data
- Want to generate specific categories
- Image-to-image translation with labels

**Pros:** Controlled generation, useful for labeled datasets
**Cons:** Requires labels, more complex

### 6. CycleGAN

Unpaired image-to-image translation:

```python
from artifex.generative_models.models.gan import CycleGAN
from artifex.generative_models.core.configuration.gan_config import CycleGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    CycleGANGeneratorConfig,
    PatchGANDiscriminatorConfig,
)

# Create CycleGAN for domain transfer (e.g., horse ↔ zebra)
cyclegan_config = CycleGANConfig(
    name="cyclegan_horse2zebra",
    generator={
        "a_to_b": CycleGANGeneratorConfig(
            name="horse_to_zebra",
            latent_dim=0,
            hidden_dims=(64, 128, 256),
            output_shape=(256, 256, 3),
            input_shape=(256, 256, 3),
            n_residual_blocks=6,
            activation="relu",
        ),
        "b_to_a": CycleGANGeneratorConfig(
            name="zebra_to_horse",
            latent_dim=0,
            hidden_dims=(64, 128, 256),
            output_shape=(256, 256, 3),
            input_shape=(256, 256, 3),
            n_residual_blocks=6,
            activation="relu",
        ),
    },
    discriminator={
        "disc_a": PatchGANDiscriminatorConfig(
            name="horse_discriminator",
            hidden_dims=(64, 128, 256, 512),
            input_shape=(256, 256, 3),
            activation="leaky_relu",
        ),
        "disc_b": PatchGANDiscriminatorConfig(
            name="zebra_discriminator",
            hidden_dims=(64, 128, 256, 512),
            input_shape=(256, 256, 3),
            activation="leaky_relu",
        ),
    },
    input_shape_a=(256, 256, 3),
    input_shape_b=(256, 256, 3),
    lambda_cycle=10.0,
    lambda_identity=0.5,
)

cyclegan = CycleGAN(config=cyclegan_config, rngs=rngs)

# Training step
def train_cyclegan_step(cyclegan, batch_x, batch_y):
    """Train CycleGAN with cycle consistency."""
    cyclegan.train()

    # Forward cycle: A -> B -> A
    fake_b = cyclegan.generator_a_to_b(batch_x)
    reconstructed_a = cyclegan.generator_b_to_a(fake_b)

    # Backward cycle: B -> A -> B
    fake_a = cyclegan.generator_b_to_a(batch_y)
    reconstructed_b = cyclegan.generator_a_to_b(fake_a)

    # Adversarial losses
    disc_b_real = cyclegan.discriminator_b(batch_y)
    disc_b_fake = cyclegan.discriminator_b(fake_b)
    disc_a_real = cyclegan.discriminator_a(batch_x)
    disc_a_fake = cyclegan.discriminator_a(fake_a)

    # Cycle consistency losses
    cycle_loss_a = jnp.mean(jnp.abs(reconstructed_a - batch_x))
    cycle_loss_b = jnp.mean(jnp.abs(reconstructed_b - batch_y))
    total_cycle_loss = cyclegan.lambda_cycle * (cycle_loss_a + cycle_loss_b)

    # Identity losses (helps preserve color)
    identity_a = cyclegan.generator_b_to_a(batch_x)
    identity_b = cyclegan.generator_a_to_b(batch_y)
    identity_loss_a = jnp.mean(jnp.abs(identity_a - batch_x))
    identity_loss_b = jnp.mean(jnp.abs(identity_b - batch_y))
    total_identity_loss = cyclegan.lambda_identity * (identity_loss_a + identity_loss_b)

    return {
        "cycle_loss": total_cycle_loss,
        "identity_loss": total_identity_loss,
    }
```

**Key Features:**

- **Two generators**: G: X→Y and F: Y→X
- **Two discriminators**: D_X and D_Y
- **Cycle consistency**: x → G(x) → F(G(x)) ≈ x
- **No paired data needed**

**When to use:**

- Image-to-image translation without paired data
- Style transfer (photo ↔ painting)
- Domain adaptation (synthetic ↔ real)
- Seasonal changes (summer ↔ winter)

**Pros:** No paired data needed, flexible
**Cons:** Computationally expensive (4 networks), can fail if domains too different

### 7. PatchGAN

Discriminator operates on image patches:

```python
from artifex.generative_models.models.gan import (
    PatchGANDiscriminator,
    MultiScalePatchGANDiscriminator,
)

# Single-scale PatchGAN
from artifex.generative_models.core.configuration.network_configs import (
    PatchGANDiscriminatorConfig,
)

patch_config = PatchGANDiscriminatorConfig(
    name="patchgan_disc",
    hidden_dims=(64, 128, 256, 512),
    input_shape=(256, 256, 3),
    activation="leaky_relu",
)

patch_discriminator = PatchGANDiscriminator(config=patch_config, rngs=rngs)

# Returns list of intermediate features with final output last
patch_features = patch_discriminator(images)

# Multi-scale PatchGAN (better for high-resolution)
multiscale_discriminator = MultiScalePatchGANDiscriminator(
    config=patch_config,
    num_scales=3,
    rngs=rngs,
)

# Returns (outputs_per_scale, features_per_scale)
outputs, features = multiscale_discriminator(images)
```

**Key Features:**

- **Patch-based**: Classifies overlapping patches
- **Local texture**: Better for texture quality
- **Efficient**: Fewer parameters than full-image discriminator
- **Multi-scale**: Can combine predictions at different resolutions

**When to use:**

- High-resolution images (>256×256)
- Image-to-image translation (Pix2Pix)
- Focus on local texture quality
- With CycleGAN for better results

**Pros:** Efficient, good for textures, scales well
**Cons:** May miss global structure issues

## Training GANs

### Basic Training Loop

Here's a complete training loop for a vanilla GAN:

```python
import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.models.gan import GAN

# Create model
gan = GAN(config, rngs=nnx.Rngs(params=0, dropout=1, sample=2))

# Create optimizers (separate for generator and discriminator)
# wrt=nnx.Param required in NNX 0.11.0+
gen_optimizer = nnx.Optimizer(
    gan.generator,
    optax.adam(learning_rate=0.0002, b1=0.5, b2=0.999),
    wrt=nnx.Param
)

disc_optimizer = nnx.Optimizer(
    gan.discriminator,
    optax.adam(learning_rate=0.0002, b1=0.5, b2=0.999),
    wrt=nnx.Param
)

# Training step
@nnx.jit
def train_step(gan, gen_opt, disc_opt, batch, rngs):
    """Single training step for vanilla GAN."""

    # Discriminator update
    def disc_loss_fn(disc):
        # Get generator samples (stop gradient to not update generator)
        z = jax.random.normal(rngs.sample(), (batch.shape[0], gan.latent_dim))
        fake_samples = gan.generator(z, training=True)
        fake_samples = jax.lax.stop_gradient(fake_samples)

        # Discriminator scores
        real_scores = disc(batch, training=True)
        fake_scores = disc(fake_samples, training=True)

        # Vanilla GAN discriminator loss
        real_loss = -jnp.log(jnp.clip(real_scores, 1e-7, 1.0))
        fake_loss = -jnp.log(jnp.clip(1.0 - fake_scores, 1e-7, 1.0))

        return jnp.mean(real_loss + fake_loss)

    # Compute discriminator loss and update
    disc_loss, disc_grads = nnx.value_and_grad(disc_loss_fn)(gan.discriminator)
    disc_opt.update(disc_grads)

    # Generator update
    def gen_loss_fn(gen):
        # Generate samples
        z = jax.random.normal(rngs.sample(), (batch.shape[0], gan.latent_dim))
        fake_samples = gen(z, training=True)

        # Get discriminator scores (stop gradient on discriminator)
        disc = jax.lax.stop_gradient(gan.discriminator)
        fake_scores = disc(fake_samples, training=True)

        # Non-saturating generator loss
        return -jnp.mean(jnp.log(jnp.clip(fake_scores, 1e-7, 1.0)))

    # Compute generator loss and update
    gen_loss, gen_grads = nnx.value_and_grad(gen_loss_fn)(gan.generator)
    gen_opt.update(gen_grads)

    return {
        "disc_loss": disc_loss,
        "gen_loss": gen_loss,
    }

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Preprocess: scale to [-1, 1] for tanh output
        batch = (batch / 127.5) - 1.0

        # Training step
        metrics = train_step(gan, gen_optimizer, disc_optimizer, batch, rngs)

        # Log metrics
        if step % log_interval == 0:
            print(f"Epoch {epoch}, Step {step}")
            print(f"  Discriminator Loss: {metrics['disc_loss']:.4f}")
            print(f"  Generator Loss: {metrics['gen_loss']:.4f}")

        # Generate samples for visualization
        if step % sample_interval == 0:
            samples = gan.generate(n_samples=16, rngs=rngs)
            save_images(samples, f"samples_step_{step}.png")
```

### WGAN Training Loop

WGAN requires multiple discriminator updates per generator update:

```python
@nnx.jit
def train_wgan_step(wgan, gen_opt, critic_opt, batch, rngs, n_critic=5):
    """Training step for WGAN-GP."""

    # Train critic n_critic times
    critic_losses = []
    for i in range(n_critic):
        def critic_loss_fn(critic):
            # Generate fake samples
            z = jax.random.normal(rngs.sample(), (batch.shape[0], wgan.latent_dim))
            fake_samples = wgan.generator(z, training=True)
            fake_samples = jax.lax.stop_gradient(fake_samples)

            # Get critic outputs
            real_validity = critic(batch, training=True)
            fake_validity = critic(fake_samples, training=True)

            # Wasserstein loss
            wasserstein_distance = jnp.mean(fake_validity) - jnp.mean(real_validity)

            # Gradient penalty
            alpha = jax.random.uniform(
                rngs.sample(),
                shape=(batch.shape[0], 1, 1, 1),
                minval=0.0,
                maxval=1.0
            )
            interpolated = alpha * batch + (1 - alpha) * fake_samples

            def critic_interp_fn(x):
                return jnp.sum(critic(x, training=True))

            gradients = jax.grad(critic_interp_fn)(interpolated)
            gradients = jnp.reshape(gradients, (batch.shape[0], -1))
            gradient_norm = jnp.sqrt(jnp.sum(gradients**2, axis=1) + 1e-12)
            gradient_penalty = jnp.mean((gradient_norm - 1.0) ** 2) * 10.0

            return wasserstein_distance + gradient_penalty

        # Update critic
        critic_loss, critic_grads = nnx.value_and_grad(critic_loss_fn)(wgan.discriminator)
        critic_opt.update(critic_grads)
        critic_losses.append(critic_loss)

    # Train generator once
    def gen_loss_fn(gen):
        z = jax.random.normal(rngs.sample(), (batch.shape[0], wgan.latent_dim))
        fake_samples = gen(z, training=True)

        critic = jax.lax.stop_gradient(wgan.discriminator)
        fake_validity = critic(fake_samples, training=True)

        # WGAN generator loss: maximize critic output
        return -jnp.mean(fake_validity)

    gen_loss, gen_grads = nnx.value_and_grad(gen_loss_fn)(wgan.generator)
    gen_opt.update(gen_grads)

    return {
        "critic_loss": jnp.mean(jnp.array(critic_losses)),
        "gen_loss": gen_loss,
    }
```

### Two-Timescale Update Rule (TTUR)

Use different learning rates for generator and discriminator:

```python
# Generator: slower learning rate (wrt=nnx.Param required in NNX 0.11.0+)
gen_optimizer = nnx.Optimizer(
    gan.generator,
    optax.adam(learning_rate=0.0001, b1=0.5, b2=0.999),  # lr = 0.0001
    wrt=nnx.Param
)

# Discriminator: faster learning rate
disc_optimizer = nnx.Optimizer(
    gan.discriminator,
    optax.adam(learning_rate=0.0004, b1=0.5, b2=0.999),  # lr = 0.0004
    wrt=nnx.Param
)
```

**Why it works:**

- Discriminator needs to stay ahead to provide useful signal
- Prevents generator from overwhelming discriminator
- More stable training dynamics

## Generation and Sampling

### Basic Generation

```python
# Generate samples
n_samples = 64
samples = gan.generate(n_samples=n_samples, rngs=rngs)

# Samples are in [-1, 1] range (from Tanh)
# Convert to [0, 255] for visualization
samples = ((samples + 1) / 2 * 255).astype(jnp.uint8)
```

### Latent Space Interpolation

Smoothly interpolate between two points in latent space:

```python
def interpolate_latent(gan, z1, z2, num_steps=10, rngs=None):
    """Interpolate between two latent vectors."""
    # Create interpolation weights
    alphas = jnp.linspace(0, 1, num_steps)

    # Interpolate
    interpolated_samples = []
    for alpha in alphas:
        z_interp = alpha * z2 + (1 - alpha) * z1
        sample = gan.generator(z_interp[None, :], training=False)
        interpolated_samples.append(sample[0])

    return jnp.stack(interpolated_samples)

# Generate two random latent vectors
z1 = jax.random.normal(rngs.sample(), (latent_dim,))
z2 = jax.random.normal(rngs.sample(), (latent_dim,))

# Interpolate
interpolated = interpolate_latent(gan, z1, z2, num_steps=20)
```

### Latent Space Exploration

Explore the latent space by varying dimensions:

```python
def explore_latent_dimension(gan, dim_idx, num_samples=10, range_scale=3.0):
    """Explore a specific latent dimension."""
    # Fixed random vector
    z_base = jax.random.normal(rngs.sample(), (latent_dim,))

    # Vary single dimension
    values = jnp.linspace(-range_scale, range_scale, num_samples)

    samples = []
    for value in values:
        z = z_base.at[dim_idx].set(value)
        sample = gan.generator(z[None, :], training=False)
        samples.append(sample[0])

    return jnp.stack(samples)

# Explore dimension 0
samples_dim0 = explore_latent_dimension(gan, dim_idx=0, num_samples=10)
```

### Conditional Generation

For conditional GANs, specify the condition:

```python
# Generate specific digits (MNIST)
labels = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
z = jax.random.normal(rngs.sample(), (10, latent_dim))

samples = cond_generator(z, labels, training=False)
# Each sample corresponds to its label
```

## Evaluation and Monitoring

### Visual Inspection

The most important evaluation method for GANs:

```python
import matplotlib.pyplot as plt

def visualize_samples(samples, nrow=8, title="Generated Samples"):
    """Visualize a grid of samples."""
    n_samples = samples.shape[0]
    ncol = (n_samples + nrow - 1) // nrow

    # Convert from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2

    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n_samples:
            # Transpose from (C, H, W) to (H, W, C)
            img = jnp.transpose(samples[i], (1, 2, 0))
            # Handle grayscale
            if img.shape[-1] == 1:
                img = img[:, :, 0]
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
        ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Generate and visualize
samples = gan.generate(n_samples=64, rngs=rngs)
visualize_samples(samples)
```

### Loss Monitoring

Track both generator and discriminator losses:

```python
# During training
history = {
    "gen_loss": [],
    "disc_loss": [],
    "real_scores": [],
    "fake_scores": [],
}

for epoch in range(num_epochs):
    for batch in dataloader:
        metrics = train_step(gan, gen_opt, disc_opt, batch, rngs)

        history["gen_loss"].append(float(metrics["gen_loss"]))
        history["disc_loss"].append(float(metrics["disc_loss"]))

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(history["gen_loss"], label="Generator Loss")
plt.plot(history["disc_loss"], label="Discriminator Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.legend()
plt.title("GAN Training Losses")
plt.show()
```

**Healthy training signs:**

- Both losses decrease initially then stabilize
- Losses oscillate but don't diverge
- Real scores stay around 0.7-0.9
- Fake scores start low, gradually increase
- Visual quality improves over time

**Warning signs:**

- Discriminator loss → 0 (too strong)
- Generator loss → ∞ (gradient vanishing)
- Mode collapse (all samples look same)
- Training instability (wild oscillations)

### Inception Score (IS)

Measures quality and diversity:

```python
from artifex.generative_models.core.evaluation.metrics.image import InceptionScore

# Create IS metric with a classifier function
is_metric = InceptionScore(classifier=inception_classifier_fn)

# Compute IS
results = is_metric.compute(generated_samples, splits=10)
print(f"Inception Score: {results['is_mean']:.2f} ± {results['is_std']:.2f}")
```

**Higher is better** (good models: 8-10 for ImageNet)

### Fréchet Inception Distance (FID)

Measures similarity to real data:

```python
from artifex.generative_models.core.evaluation.metrics.image import FrechetInceptionDistance

# Create FID metric with a feature extractor function
fid_metric = FrechetInceptionDistance(feature_extractor=feature_extractor_fn)

# Compute FID between real and generated images
results = fid_metric.compute(real_images, generated_samples)
print(f"FID Score: {results['fid']:.2f}")
```

**Lower is better** (good models: < 50, excellent: < 10)

## Common Issues and Solutions

### Mode Collapse

**Symptom:** Generator produces limited variety of samples.

**Detection:**

```python
# Check sample diversity
samples = gan.generate(n_samples=100, rngs=rngs)
samples_flat = samples.reshape(samples.shape[0], -1)

# Compute pairwise distances
from scipy.spatial.distance import pdist
distances = pdist(samples_flat)

if jnp.mean(distances) < threshold:
    print("Warning: Possible mode collapse detected!")
```

**Solutions:**

1. **Use WGAN or LSGAN:**

```python
config.loss_type = "wasserstein"  # or "least_squares"
```

2. **Minibatch discrimination:**

```python
# Add minibatch statistics to discriminator
def minibatch_stddev(x):
    """Compute standard deviation across batch."""
    batch_std = jnp.std(x, axis=0, keepdims=True)
    return jnp.mean(batch_std)
```

3. **Add noise to discriminator inputs:**

```python
# Gradually decay noise
noise_std = 0.1 * (1 - epoch / num_epochs)
noisy_real = real_data + jax.random.normal(key, real_data.shape) * noise_std
noisy_fake = fake_data + jax.random.normal(key, fake_data.shape) * noise_std
```

4. **Use feature matching:**

```python
# Match discriminator feature statistics
def feature_matching_loss(real_features, fake_features):
    return jnp.mean((jnp.mean(real_features, axis=0) -
                     jnp.mean(fake_features, axis=0)) ** 2)
```

### Training Instability

**Symptom:** Losses oscillate wildly, training doesn't converge.

**Solutions:**

1. **Use spectral normalization:**

```python
discriminator = Discriminator(
    hidden_dims=[512, 256, 128],
    use_spectral_norm=True,  # Enable spectral norm
    rngs=rngs,
)
```

2. **Two-timescale update rule:**

```python
# Different learning rates
gen_lr = 0.0001
disc_lr = 0.0004
```

3. **Gradient penalty (WGAN-GP):**

```python
# Use WGAN with gradient penalty
wgan_config.gradient_penalty_weight = 10.0
```

4. **Label smoothing:**

```python
# Smooth labels for discriminator
real_labels = jnp.ones((batch_size, 1)) * 0.9  # Instead of 1.0
fake_labels = jnp.zeros((batch_size, 1)) + 0.1  # Instead of 0.0
```

### Vanishing Gradients

**Symptom:** Generator loss stops decreasing, samples don't improve.

**Solutions:**

1. **Use non-saturating loss:**

```python
# Instead of: -log(1 - D(G(z)))
# Use: -log(D(G(z)))
gen_loss = -jnp.mean(jnp.log(jnp.clip(fake_scores, 1e-7, 1.0)))
```

2. **Reduce discriminator capacity:**

```python
# Make discriminator weaker
config.discriminator.hidden_dims = [256, 128]  # Smaller than [512, 256]
```

3. **Update discriminator less frequently:**

```python
# Update discriminator every 2 generator updates
if step % 2 == 0:
    disc_loss = train_discriminator(...)
gen_loss = train_generator(...)
```

### Poor Sample Quality

**Symptom:** Blurry or unrealistic samples.

**Solutions:**

1. **Use DCGAN architecture:**

```python
# Replace MLP with convolutional architecture
from artifex.generative_models.models.gan import DCGAN
gan = DCGAN(config, rngs=rngs)
```

2. **Increase model capacity:**

```python
config.generator.hidden_dims = [512, 1024, 2048]  # Larger
```

3. **Train longer:**

```python
num_epochs = 200  # GANs need many epochs
```

4. **Better data preprocessing:**

```python
# Normalize to [-1, 1] for Tanh
data = (data / 127.5) - 1.0

# Ensure consistent shape
data = jnp.transpose(data, (0, 3, 1, 2))  # NHWC → NCHW
```

## Best Practices

### DO

✅ **Use DCGAN guidelines** for image generation:

```python
# Strided convolutions, batch norm, LeakyReLU
generator = DCGANGenerator(...)
discriminator = DCGANDiscriminator(...)
```

✅ **Scale data to [-1, 1]** for Tanh output:

```python
data = (data / 127.5) - 1.0
```

✅ **Use Adam optimizer** with β₁=0.5:

```python
optimizer = nnx.adam(learning_rate=0.0002, b1=0.5, b2=0.999)
```

✅ **Monitor both losses and samples**:

```python
if step % 100 == 0:
    visualize_samples(gan.generate(16, rngs=rngs))
```

✅ **Use two-timescale updates** (TTUR):

```python
gen_lr = 0.0001
disc_lr = 0.0004
```

✅ **Start with WGAN** for stable training:

```python
config.loss_type = "wasserstein"
```

✅ **Save checkpoints regularly**:

```python
if epoch % 10 == 0:
    nnx.save_checkpoint(gan, f"checkpoints/gan_epoch_{epoch}")
```

### DON'T

❌ **Don't use batch norm in discriminator input**:

```python
# BAD
discriminator.layers[0] = BatchNorm(...)

# GOOD
discriminator.batch_norm = False  # Or skip first layer
```

❌ **Don't use same learning rate** for G and D:

```python
# BAD
gen_lr = disc_lr = 0.0002

# GOOD
gen_lr = 0.0001
disc_lr = 0.0004  # Discriminator learns faster
```

❌ **Don't forget to scale data**:

```python
# BAD
data = data / 255.0  # [0, 1] doesn't match Tanh [-1, 1]

# GOOD
data = (data / 127.5) - 1.0  # [-1, 1] matches Tanh
```

❌ **Don't ignore mode collapse warnings**:

```python
# Check diversity regularly
if jnp.std(samples) < 0.1:
    print("Warning: Possible mode collapse!")
```

❌ **Don't use too small batch sizes**:

```python
# BAD
batch_size = 8  # Too small, unstable

# GOOD
batch_size = 64  # Better stability
```

## Summary

This guide covered:

- **Creating GANs**: Generators, discriminators, and full GAN models
- **Variants**: Vanilla, DCGAN, WGAN, LSGAN, cGAN, CycleGAN, PatchGAN
- **Training**: Basic loops, WGAN training, two-timescale updates
- **Generation**: Basic sampling, interpolation, conditional generation
- **Evaluation**: Visual inspection, IS, FID
- **Troubleshooting**: Mode collapse, instability, vanishing gradients
- **Best practices**: What to do and what to avoid

## Next Steps

- **Theory**: See [GAN Concepts](../concepts/gan-explained.md) for mathematical foundations
- **API Reference**: Check [GAN API Documentation](../../api/models/gan.md) for detailed specifications
- **Example**: Follow [MNIST GAN Tutorial](../../examples/basic/simple-gan.md) for hands-on training
- **Advanced**: Explore StyleGAN and Progressive GAN for state-of-the-art results
