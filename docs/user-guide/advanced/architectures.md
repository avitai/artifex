# Custom Architectures

Build custom model architectures using Flax NNX in Artifex. This guide covers advanced architectural patterns, custom layers, and integration with Artifex's training and evaluation systems.

<div class="grid cards" markdown>

- :material-cube-outline:{ .lg .middle } **Custom Layers**

    ---

    Create custom neural network layers with Flax NNX

    [:octicons-arrow-right-24: Learn more](#custom-layers)

- :material-vector-polygon:{ .lg .middle } **Custom Models**

    ---

    Build complete custom generative models

    [:octicons-arrow-right-24: Learn more](#custom-models)

- :material-connection:{ .lg .middle } **Architecture Patterns**

    ---

    Common architectural patterns and best practices

    [:octicons-arrow-right-24: Learn more](#architecture-patterns)

- :material-puzzle:{ .lg .middle } **Integration**

    ---

    Integrate custom models with Artifex's systems

    [:octicons-arrow-right-24: Learn more](#artifex-integration)

</div>

## Overview

Artifex provides flexibility to create custom architectures while maintaining compatibility with the training, evaluation, and deployment infrastructure.

### Why Custom Architectures?

Build custom architectures when:

- **Research**: Implementing novel architectural ideas
- **Domain-Specific**: Specialized requirements (proteins, molecules, etc.)
- **Optimization**: Custom operations for performance
- **Experimentation**: Rapid prototyping of new ideas

## Custom Layers

Create custom neural network layers using Flax NNX.

### Basic Custom Layer

```python
import jax
import jax.numpy as jnp
from flax import nnx

class CustomLinear(nnx.Module):
    """Custom linear layer with additional features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        weight_init: callable = nnx.initializers.lecun_normal(),
        bias_init: callable = nnx.initializers.zeros_init(),
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # Initialize weight
        self.weight = nnx.Param(
            weight_init(rngs.params(), (in_features, out_features), dtype)
        )

        # Initialize bias if needed
        if use_bias:
            self.bias = nnx.Param(
                bias_init(rngs.params(), (out_features,), dtype)
            )
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            Output tensor (..., out_features)
        """
        # Matrix multiplication
        output = x @ self.weight.value

        # Add bias
        if self.use_bias:
            output = output + self.bias.value

        return output


# Usage
layer = CustomLinear(
    in_features=784,
    out_features=256,
    rngs=nnx.Rngs(0)
)

x = jnp.ones((32, 784))
output = layer(x)
print(f"Output shape: {output.shape}")  # (32, 256)
```

### Advanced Custom Layer with Regularization

```python
class RegularizedLinear(nnx.Module):
    """Linear layer with built-in regularization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dropout_rate: float = 0.0,
        weight_decay: float = 0.0,
        spectral_norm: bool = False,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.weight_decay = weight_decay
        self.spectral_norm = spectral_norm

        # Weight initialization
        self.weight = nnx.Param(
            nnx.initializers.lecun_normal()(
                rngs.params(),
                (in_features, out_features)
            )
        )

        self.bias = nnx.Param(jnp.zeros(out_features))

        # Dropout
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def _apply_spectral_norm(self, weight: jax.Array) -> jax.Array:
        """Apply spectral normalization to weight."""
        # Compute largest singular value
        u, s, vh = jnp.linalg.svd(weight, full_matrices=False)

        # Normalize by largest singular value
        weight_normalized = weight / s[0]

        return weight_normalized

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Forward pass with regularization.

        Args:
            x: Input tensor
            deterministic: If True, disable dropout

        Returns:
            Output tensor
        """
        # Get weight
        weight = self.weight.value

        # Apply spectral normalization
        if self.spectral_norm:
            weight = self._apply_spectral_norm(weight)

        # Linear transformation
        output = x @ weight + self.bias.value

        # Apply dropout
        if self.dropout is not None and not deterministic:
            output = self.dropout(output)

        return output

    def get_regularization_loss(self) -> jax.Array:
        """Compute regularization loss for this layer."""
        if self.weight_decay > 0:
            # L2 regularization
            return self.weight_decay * jnp.sum(self.weight.value ** 2)
        return 0.0


# Usage
layer = RegularizedLinear(
    in_features=784,
    out_features=256,
    dropout_rate=0.1,
    weight_decay=1e-4,
    spectral_norm=True,
    rngs=nnx.Rngs(0)
)

# Forward pass
x = jnp.ones((32, 784))
output = layer(x, deterministic=False)

# Get regularization loss
reg_loss = layer.get_regularization_loss()
```

### Attention Layer

```python
class MultiHeadAttention(nnx.Module):
    """Multi-head attention layer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        dropout_rate: float = 0.0,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Q, K, V projections
        self.q_proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

        # Output projection
        self.out_proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

        # Dropout
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Multi-head attention forward pass.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            mask: Optional attention mask (batch, seq_len, seq_len)
            deterministic: If True, disable dropout

        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, hidden_size)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: (batch, num_heads, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        scale = jnp.sqrt(self.head_dim)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / scale

        # Apply mask if provided
        if mask is not None:
            # Expand mask for heads: (batch, 1, seq_len, seq_len)
            mask = mask[:, None, :, :]
            scores = jnp.where(mask, scores, -1e9)

        # Softmax
        attention_weights = nnx.softmax(scores, axis=-1)

        # Apply dropout
        if self.dropout is not None and not deterministic:
            attention_weights = self.dropout(attention_weights)

        # Attend to values
        context = jnp.einsum("bhqk,bhkd->bhqd", attention_weights, v)

        # Reshape back: (batch, seq_len, hidden_size)
        context = jnp.transpose(context, (0, 2, 1, 3))
        context = context.reshape(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.out_proj(context)

        return output


# Usage
attention = MultiHeadAttention(
    hidden_size=512,
    num_heads=8,
    dropout_rate=0.1,
    rngs=nnx.Rngs(0)
)

x = jnp.ones((2, 10, 512))  # (batch=2, seq_len=10, hidden=512)
output = attention(x, deterministic=False)
print(f"Output shape: {output.shape}")  # (2, 10, 512)
```

### Residual Block

```python
class ResidualBlock(nnx.Module):
    """Residual block with normalization and activation."""

    def __init__(
        self,
        channels: int,
        *,
        stride: int = 1,
        downsample: bool = False,
        activation: callable = nnx.relu,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.activation = activation

        # Main path
        self.conv1 = nnx.Conv(
            in_features=channels,
            out_features=channels,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding="SAME",
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(num_features=channels, rngs=rngs)

        self.conv2 = nnx.Conv(
            in_features=channels,
            out_features=channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(num_features=channels, rngs=rngs)

        # Shortcut path
        if downsample:
            self.shortcut = nnx.Conv(
                in_features=channels,
                out_features=channels,
                kernel_size=(1, 1),
                strides=(stride, stride),
                padding="VALID",
                rngs=rngs,
            )
            self.shortcut_bn = nnx.BatchNorm(num_features=channels, rngs=rngs)
        else:
            self.shortcut = None

    def __call__(
        self,
        x: jax.Array,
        *,
        use_running_average: bool = False,
    ) -> jax.Array:
        """Forward pass through residual block.

        Args:
            x: Input tensor (batch, height, width, channels)
            use_running_average: Use running stats for batch norm

        Returns:
            Output tensor (batch, height, width, channels)
        """
        # Save input for residual connection
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out, use_running_average=use_running_average)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out, use_running_average=use_running_average)

        # Shortcut path
        if self.shortcut is not None:
            identity = self.shortcut(identity)
            identity = self.shortcut_bn(
                identity,
                use_running_average=use_running_average
            )

        # Residual connection
        out = out + identity
        out = self.activation(out)

        return out


# Usage
block = ResidualBlock(
    channels=64,
    stride=2,
    downsample=True,
    rngs=nnx.Rngs(0)
)

x = jnp.ones((2, 32, 32, 64))
output = block(x, use_running_average=False)
print(f"Output shape: {output.shape}")  # (2, 16, 16, 64)
```

## Custom Models

Build complete custom generative models.

### Custom VAE Architecture

```python
from artifex.generative_models.core.protocols import GenerativeModel
from flax import nnx
import jax
import jax.numpy as jnp

class CustomVAE(nnx.Module):
    """Custom VAE with flexible architecture."""

    def __init__(
        self,
        input_shape: tuple,
        latent_dim: int,
        encoder_layers: list[int],
        decoder_layers: list[int],
        *,
        activation: callable = nnx.relu,
        use_batch_norm: bool = True,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.activation = activation

        # Flatten input size
        self.input_dim = int(jnp.prod(jnp.array(input_shape)))

        # Encoder
        self.encoder = self._build_encoder(
            encoder_layers,
            use_batch_norm,
            rngs
        )

        # Latent projections
        self.mean_layer = nnx.Linear(
            in_features=encoder_layers[-1],
            out_features=latent_dim,
            rngs=rngs,
        )
        self.logvar_layer = nnx.Linear(
            in_features=encoder_layers[-1],
            out_features=latent_dim,
            rngs=rngs,
        )

        # Decoder
        self.decoder = self._build_decoder(
            decoder_layers,
            use_batch_norm,
            rngs
        )

        # Output layer
        self.output_layer = nnx.Linear(
            in_features=decoder_layers[-1],
            out_features=self.input_dim,
            rngs=rngs,
        )

    def _build_encoder(
        self,
        layers: list[int],
        use_batch_norm: bool,
        rngs: nnx.Rngs,
    ) -> list:
        """Build encoder layers."""
        encoder_layers = []

        # Input layer
        encoder_layers.append(nnx.Linear(self.input_dim, layers[0], rngs=rngs))
        if use_batch_norm:
            encoder_layers.append(nnx.BatchNorm(layers[0], rngs=rngs))

        # Hidden layers
        for i in range(len(layers) - 1):
            encoder_layers.append(
                nnx.Linear(layers[i], layers[i + 1], rngs=rngs)
            )
            if use_batch_norm:
                encoder_layers.append(nnx.BatchNorm(layers[i + 1], rngs=rngs))

        return encoder_layers

    def _build_decoder(
        self,
        layers: list[int],
        use_batch_norm: bool,
        rngs: nnx.Rngs,
    ) -> list:
        """Build decoder layers."""
        decoder_layers = []

        # Input layer (from latent)
        decoder_layers.append(nnx.Linear(self.latent_dim, layers[0], rngs=rngs))
        if use_batch_norm:
            decoder_layers.append(nnx.BatchNorm(layers[0], rngs=rngs))

        # Hidden layers
        for i in range(len(layers) - 1):
            decoder_layers.append(
                nnx.Linear(layers[i], layers[i + 1], rngs=rngs)
            )
            if use_batch_norm:
                decoder_layers.append(nnx.BatchNorm(layers[i + 1], rngs=rngs))

        return decoder_layers

    def encode(
        self,
        x: jax.Array,
        *,
        use_running_average: bool = False,
    ) -> dict[str, jax.Array]:
        """Encode input to latent distribution.

        Args:
            x: Input tensor (batch, *input_shape)
            use_running_average: Use running stats for batch norm

        Returns:
            Dictionary with 'mean' and 'logvar'
        """
        # Flatten input
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        # Forward through encoder
        for layer in self.encoder:
            if isinstance(layer, nnx.BatchNorm):
                x = layer(x, use_running_average=use_running_average)
            else:
                x = layer(x)
            x = self.activation(x)

        # Latent parameters
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)

        return {"mean": mean, "logvar": logvar}

    def reparameterize(
        self,
        mean: jax.Array,
        logvar: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Reparameterization trick.

        Args:
            mean: Mean of latent distribution
            logvar: Log variance of latent distribution
            rngs: RNG for sampling

        Returns:
            Sampled latent vector
        """
        if rngs is not None and "sample" in rngs:
            key = rngs.sample()
        else:
            key = jax.random.key(0)

        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mean.shape)
        z = mean + eps * std

        return z

    def decode(
        self,
        z: jax.Array,
        *,
        use_running_average: bool = False,
    ) -> jax.Array:
        """Decode latent vector to reconstruction.

        Args:
            z: Latent vector (batch, latent_dim)
            use_running_average: Use running stats for batch norm

        Returns:
            Reconstruction (batch, *input_shape)
        """
        x = z

        # Forward through decoder
        for layer in self.decoder:
            if isinstance(layer, nnx.BatchNorm):
                x = layer(x, use_running_average=use_running_average)
            else:
                x = layer(x)
            x = self.activation(x)

        # Output layer
        x = self.output_layer(x)
        x = nnx.sigmoid(x)  # Normalize to [0, 1]

        # Reshape to input shape
        batch_size = z.shape[0]
        x = x.reshape(batch_size, *self.input_shape)

        return x

    def __call__(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
        use_running_average: bool = False,
    ) -> dict[str, jax.Array]:
        """Full forward pass (encode-reparameterize-decode).

        Args:
            x: Input tensor (batch, *input_shape)
            rngs: RNG for sampling
            use_running_average: Use running stats for batch norm

        Returns:
            Dictionary with 'reconstruction', 'mean', 'logvar', 'latent'
        """
        # Encode
        latent_params = self.encode(x, use_running_average=use_running_average)

        # Reparameterize
        z = self.reparameterize(
            latent_params["mean"],
            latent_params["logvar"],
            rngs=rngs
        )

        # Decode
        reconstruction = self.decode(z, use_running_average=use_running_average)

        return {
            "reconstruction": reconstruction,
            "mean": latent_params["mean"],
            "logvar": latent_params["logvar"],
            "latent": z,
        }


# Create custom VAE
model = CustomVAE(
    input_shape=(28, 28, 1),  # MNIST-like
    latent_dim=20,
    encoder_layers=[512, 256, 128],
    decoder_layers=[128, 256, 512],
    use_batch_norm=True,
    rngs=nnx.Rngs(0),
)

# Forward pass
x = jnp.ones((32, 28, 28, 1))
output = model(x, rngs=nnx.Rngs(1))

print(f"Reconstruction shape: {output['reconstruction'].shape}")  # (32, 28, 28, 1)
print(f"Latent shape: {output['latent'].shape}")  # (32, 20)
```

### Custom GAN with Advanced Techniques

```python
class CustomGenerator(nnx.Module):
    """Custom generator with self-attention."""

    def __init__(
        self,
        latent_dim: int,
        output_shape: tuple,
        hidden_dims: list[int],
        *,
        use_attention: bool = True,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.use_attention = use_attention

        # Build generator network
        self.layers = []

        # Initial projection
        self.layers.append(nnx.Linear(latent_dim, hidden_dims[0], rngs=rngs))
        self.layers.append(nnx.BatchNorm(hidden_dims[0], rngs=rngs))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                nnx.Linear(hidden_dims[i], hidden_dims[i + 1], rngs=rngs)
            )
            self.layers.append(nnx.BatchNorm(hidden_dims[i + 1], rngs=rngs))

            # Add self-attention at middle layer
            if use_attention and i == len(hidden_dims) // 2:
                self.attention = MultiHeadAttention(
                    hidden_size=hidden_dims[i + 1],
                    num_heads=4,
                    rngs=rngs,
                )

        # Output layer
        output_dim = int(jnp.prod(jnp.array(output_shape)))
        self.output_layer = nnx.Linear(hidden_dims[-1], output_dim, rngs=rngs)

    def __call__(
        self,
        z: jax.Array,
        *,
        use_running_average: bool = False,
    ) -> jax.Array:
        """Generate samples from noise.

        Args:
            z: Noise vector (batch, latent_dim)
            use_running_average: Use running stats for batch norm

        Returns:
            Generated samples (batch, *output_shape)
        """
        x = z

        # Forward through layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nnx.BatchNorm):
                x = layer(x, use_running_average=use_running_average)
            else:
                x = layer(x)
                x = nnx.relu(x)

            # Apply self-attention if available
            if self.use_attention and hasattr(self, "attention"):
                if i == len(self.layers) // 2:
                    # Reshape for attention (add sequence dimension)
                    batch_size = x.shape[0]
                    x = x.reshape(batch_size, 1, -1)
                    x = self.attention(x)
                    x = x.reshape(batch_size, -1)

        # Output layer with tanh activation
        x = self.output_layer(x)
        x = nnx.tanh(x)

        # Reshape to output shape
        batch_size = z.shape[0]
        x = x.reshape(batch_size, *self.output_shape)

        return x


class CustomDiscriminator(nnx.Module):
    """Custom discriminator with spectral normalization."""

    def __init__(
        self,
        input_shape: tuple,
        hidden_dims: list[int],
        *,
        spectral_norm: bool = True,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.spectral_norm = spectral_norm

        input_dim = int(jnp.prod(jnp.array(input_shape)))

        # Build discriminator network
        self.layers = []

        # Input layer
        if spectral_norm:
            self.layers.append(
                RegularizedLinear(
                    input_dim,
                    hidden_dims[0],
                    spectral_norm=True,
                    rngs=rngs,
                )
            )
        else:
            self.layers.append(nnx.Linear(input_dim, hidden_dims[0], rngs=rngs))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            if spectral_norm:
                self.layers.append(
                    RegularizedLinear(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        spectral_norm=True,
                        rngs=rngs,
                    )
                )
            else:
                self.layers.append(
                    nnx.Linear(hidden_dims[i], hidden_dims[i + 1], rngs=rngs)
                )

        # Output layer (binary classification)
        if spectral_norm:
            self.output_layer = RegularizedLinear(
                hidden_dims[-1], 1,
                spectral_norm=True,
                rngs=rngs,
            )
        else:
            self.output_layer = nnx.Linear(hidden_dims[-1], 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Discriminate real vs fake samples.

        Args:
            x: Input samples (batch, *input_shape)

        Returns:
            Logits (batch, 1)
        """
        # Flatten input
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        # Forward through layers
        for layer in self.layers:
            x = layer(x)
            x = nnx.leaky_relu(x, negative_slope=0.2)

        # Output layer (no activation, return logits)
        logits = self.output_layer(x)

        return logits


# Create custom GAN
generator = CustomGenerator(
    latent_dim=100,
    output_shape=(28, 28, 1),
    hidden_dims=[256, 512, 1024],
    use_attention=True,
    rngs=nnx.Rngs(0),
)

discriminator = CustomDiscriminator(
    input_shape=(28, 28, 1),
    hidden_dims=[1024, 512, 256],
    spectral_norm=True,
    rngs=nnx.Rngs(1),
)

# Generate samples
z = jax.random.normal(jax.random.key(0), (32, 100))
fake_samples = generator(z)
print(f"Generated shape: {fake_samples.shape}")  # (32, 28, 28, 1)

# Discriminate
real_samples = jnp.ones((32, 28, 28, 1))
real_logits = discriminator(real_samples)
fake_logits = discriminator(fake_samples)
print(f"Real logits shape: {real_logits.shape}")  # (32, 1)
```

## Architecture Patterns

Common architectural patterns and best practices.

### Residual Connections

```python
class ResidualNetwork(nnx.Module):
    """Network with residual connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nnx.Linear(input_dim, hidden_dim, rngs=rngs)

        # Residual blocks
        self.blocks = [
            self._create_residual_block(hidden_dim, rngs)
            for _ in range(num_blocks)
        ]

        # Output projection
        self.output_proj = nnx.Linear(hidden_dim, input_dim, rngs=rngs)

    def _create_residual_block(
        self,
        hidden_dim: int,
        rngs: nnx.Rngs,
    ) -> list:
        """Create a single residual block."""
        return [
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.LayerNorm(hidden_dim, rngs=rngs),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.LayerNorm(hidden_dim, rngs=rngs),
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with residual connections."""
        # Input projection
        x = self.input_proj(x)

        # Residual blocks
        for block_layers in self.blocks:
            residual = x

            # Forward through block layers
            x = block_layers[0](x)  # Linear
            x = nnx.relu(x)
            x = block_layers[1](x)  # LayerNorm

            x = block_layers[2](x)  # Linear
            x = block_layers[3](x)  # LayerNorm

            # Residual connection
            x = x + residual
            x = nnx.relu(x)

        # Output projection
        x = self.output_proj(x)

        return x
```

### Skip Connections (U-Net Style)

```python
class UNetEncoder(nnx.Module):
    """U-Net style encoder with skip connections."""

    def __init__(
        self,
        channels_list: list[int],
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.down_blocks = []

        for i in range(len(channels_list) - 1):
            in_channels = channels_list[i]
            out_channels = channels_list[i + 1]

            # Downsampling block
            block = [
                nnx.Conv(
                    in_features=in_channels,
                    out_features=out_channels,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="SAME",
                    rngs=rngs,
                ),
                nnx.BatchNorm(out_channels, rngs=rngs),
            ]
            self.down_blocks.append(block)

    def __call__(
        self,
        x: jax.Array,
        *,
        use_running_average: bool = False,
    ) -> tuple[jax.Array, list[jax.Array]]:
        """Forward pass with skip connections.

        Args:
            x: Input tensor
            use_running_average: Use running stats for batch norm

        Returns:
            (encoded, skip_connections)
        """
        skip_connections = []

        for block in self.down_blocks:
            # Save for skip connection
            skip_connections.append(x)

            # Downsample
            x = block[0](x)  # Conv
            x = block[1](x, use_running_average=use_running_average)  # BatchNorm
            x = nnx.relu(x)

        return x, skip_connections


class UNetDecoder(nnx.Module):
    """U-Net style decoder with skip connections."""

    def __init__(
        self,
        channels_list: list[int],
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.up_blocks = []

        for i in range(len(channels_list) - 1):
            in_channels = channels_list[i]
            out_channels = channels_list[i + 1]

            # Upsampling block
            block = [
                nnx.ConvTranspose(
                    in_features=in_channels,
                    out_features=out_channels,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="SAME",
                    rngs=rngs,
                ),
                nnx.BatchNorm(out_channels, rngs=rngs),
            ]
            self.up_blocks.append(block)

    def __call__(
        self,
        x: jax.Array,
        skip_connections: list[jax.Array],
        *,
        use_running_average: bool = False,
    ) -> jax.Array:
        """Forward pass with skip connections.

        Args:
            x: Encoded tensor
            skip_connections: Skip connections from encoder
            use_running_average: Use running stats for batch norm

        Returns:
            Decoded tensor
        """
        for i, block in enumerate(self.up_blocks):
            # Upsample
            x = block[0](x)  # ConvTranspose
            x = block[1](x, use_running_average=use_running_average)  # BatchNorm
            x = nnx.relu(x)

            # Add skip connection
            if i < len(skip_connections):
                skip = skip_connections[-(i + 1)]
                x = jnp.concatenate([x, skip], axis=-1)

        return x
```

### Dense Connections (DenseNet Style)

```python
class DenseBlock(nnx.Module):
    """Dense block with concatenated connections."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        num_layers: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.growth_rate = growth_rate

        self.layers = []
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate

            layer = [
                nnx.BatchNorm(layer_in_channels, rngs=rngs),
                nnx.Conv(
                    in_features=layer_in_channels,
                    out_features=growth_rate,
                    kernel_size=(3, 3),
                    padding="SAME",
                    rngs=rngs,
                ),
            ]
            self.layers.append(layer)

    def __call__(
        self,
        x: jax.Array,
        *,
        use_running_average: bool = False,
    ) -> jax.Array:
        """Forward pass with dense connections.

        Args:
            x: Input tensor
            use_running_average: Use running stats for batch norm

        Returns:
            Output tensor with all features concatenated
        """
        features = [x]

        for layer in self.layers:
            # BatchNorm + ReLU + Conv
            out = layer[0](x, use_running_average=use_running_average)
            out = nnx.relu(out)
            out = layer[1](out)

            # Concatenate with previous features
            features.append(out)
            x = jnp.concatenate(features, axis=-1)

        return x
```

## Artifex Integration

Integrate custom models with Artifex's systems.

### Implementing the GenerativeModel Protocol

```python
from artifex.generative_models.core.protocols import GenerativeModel
from flax import nnx
import jax
import jax.numpy as jnp

class MyCustomGenerativeModel(nnx.Module):
    """Custom model implementing Artifex's GenerativeModel protocol."""

    def __init__(
        self,
        config: dict,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        # Extract config
        self.latent_dim = config.get("latent_dim", 20)
        self.input_shape = config.get("input_shape", (28, 28, 1))

        # Build architecture (your custom design)
        self.encoder = CustomVAE(
            input_shape=self.input_shape,
            latent_dim=self.latent_dim,
            encoder_layers=[512, 256],
            decoder_layers=[256, 512],
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs
    ) -> dict[str, jax.Array]:
        """Forward pass returning Artifex-compatible output.

        Must return dictionary with at least:
        - 'loss': scalar loss for training
        - Model-specific outputs (e.g., 'reconstruction', 'samples')
        """
        # Forward pass
        output = self.encoder(x, rngs=rngs, **kwargs)

        # Compute loss
        reconstruction_loss = jnp.mean((x - output["reconstruction"]) ** 2)
        kl_loss = -0.5 * jnp.mean(
            1 + output["logvar"] - output["mean"] ** 2 - jnp.exp(output["logvar"])
        )
        total_loss = reconstruction_loss + kl_loss

        # Return Artifex-compatible output
        return {
            "loss": total_loss,
            "reconstruction": output["reconstruction"],
            "latent": output["latent"],
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def sample(
        self,
        num_samples: int,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs
    ) -> jax.Array:
        """Generate samples (required for GenerativeModel protocol)."""
        if rngs is not None and "sample" in rngs:
            key = rngs.sample()
        else:
            key = jax.random.key(0)

        # Sample from prior
        z = jax.random.normal(key, (num_samples, self.latent_dim))

        # Decode
        samples = self.encoder.decode(z, **kwargs)

        return samples


# Usage with Artifex
from artifex.generative_models.training.trainer import Trainer

model = MyCustomGenerativeModel(
    config={"latent_dim": 20, "input_shape": (28, 28, 1)},
    rngs=nnx.Rngs(0),
)

# Integrate with Artifex's trainer
trainer = Trainer(
    model=model,
    # ... other config
)

# Training and evaluation work automatically
# trainer.train(train_dataset, val_dataset)
```

### Custom Loss Functions

```python
def custom_vae_loss(
    model: nnx.Module,
    batch: dict[str, jax.Array],
    *,
    beta: float = 1.0,
    **kwargs
) -> tuple[jax.Array, dict]:
    """Custom VAE loss with β-weighting.

    Args:
        model: The VAE model
        batch: Batch dictionary with 'data' key
        beta: Weight for KL divergence term

    Returns:
        (total_loss, metrics_dict)
    """
    # Forward pass
    output = model(batch["data"], **kwargs)

    # Reconstruction loss
    recon_loss = jnp.mean(
        (batch["data"] - output["reconstruction"]) ** 2
    )

    # KL divergence
    kl_loss = -0.5 * jnp.mean(
        1 + output["logvar"]
        - output["mean"] ** 2
        - jnp.exp(output["logvar"])
    )

    # Total loss with β-weighting
    total_loss = recon_loss + beta * kl_loss

    # Metrics for logging
    metrics = {
        "loss": total_loss,
        "reconstruction_loss": recon_loss,
        "kl_loss": kl_loss,
        "beta": beta,
    }

    return total_loss, metrics


# Use custom loss in training
@jax.jit
def train_step(model_state, batch, optimizer_state):
    """Training step with custom loss."""
    model = nnx.merge(model_graphdef, model_state)

    # Compute loss and gradients
    (loss, metrics), grads = nnx.value_and_grad(
        lambda m: custom_vae_loss(m, batch, beta=2.0),
        has_aux=True
    )(model)

    # Update parameters
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    model_state = optax.apply_updates(model_state, updates)

    return model_state, optimizer_state, metrics
```

## Best Practices

### DO

- ✅ **Follow Flax NNX patterns** - use `nnx.Module`, `nnx.Param`
- ✅ **Call `super().__init__()`** - always in module constructors
- ✅ **Use proper RNG handling** - check if key exists, provide fallback
- ✅ **Implement protocols** - match Artifex's interface expectations
- ✅ **Return dictionaries** - structured outputs for logging
- ✅ **Use type hints** - document input/output shapes
- ✅ **Test components separately** - unit test layers before integration
- ✅ **Profile performance** - measure speed and memory
- ✅ **Document architecture** - explain design choices
- ✅ **Version your models** - track architectural changes

### DON'T

- ❌ **Don't use Flax Linen** - only use Flax NNX
- ❌ **Don't forget super().**init**()** - causes initialization issues
- ❌ **Don't use numpy inside modules** - use `jax.numpy` instead
- ❌ **Don't mix PyTorch/TensorFlow** - stay in JAX ecosystem
- ❌ **Don't hardcode shapes** - make them configurable
- ❌ **Don't skip validation** - verify outputs are correct
- ❌ **Don't ignore memory** - monitor GPU usage
- ❌ **Don't over-engineer** - start simple, add complexity as needed
- ❌ **Don't skip documentation** - explain architecture decisions
- ❌ **Don't forget batch dimensions** - always handle batched inputs

## Summary

Custom architectures in Artifex:

1. **Custom Layers**: Build reusable components with Flax NNX
2. **Custom Models**: Create complete generative models
3. **Architecture Patterns**: Residual, skip, dense connections
4. **Artifex Integration**: Implement protocols for seamless integration

Key principles:

- Use Flax NNX exclusively
- Follow Artifex's protocol interfaces
- Return structured outputs (dictionaries)
- Document architecture choices
- Test and profile before deploying

## Next Steps

<div class="grid cards" markdown>

- :material-file-document-multiple:{ .lg .middle } **Advanced Examples**

    ---

    See complete examples of custom architectures in action

    [:octicons-arrow-right-24: View examples](../../examples/advanced/advanced-vae.md)

- :material-chart-line:{ .lg .middle } **Distributed Training**

    ---

    Scale custom models with distributed training

    [:octicons-arrow-right-24: Distributed guide](distributed.md)

- :material-ab-testing:{ .lg .middle } **Model Parallelism**

    ---

    Parallelize large custom models

    [:octicons-arrow-right-24: Parallelism guide](parallelism.md)

- :material-speedometer:{ .lg .middle } **Training Guide**

    ---

    Train custom models with Artifex's trainer

    [:octicons-arrow-right-24: Training guide](../training/training-guide.md)

</div>
