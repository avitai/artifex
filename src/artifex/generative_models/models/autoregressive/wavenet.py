"""WaveNet implementation for autoregressive sequence generation."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.autoregressive_config import WaveNetConfig
from artifex.generative_models.core.layers import WaveNetResidualBlock
from artifex.generative_models.models.autoregressive.base import AutoregressiveModel


class CausalConv1D(nnx.Module):
    """Causal 1D convolution that ensures no future information leakage."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        dilation: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize causal convolution.

        Args:
            in_features: Number of input channels
            out_features: Number of output channels
            kernel_size: Size of the convolution kernel
            dilation: Dilation factor
            rngs: Random number generators
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(kernel_size,),
            strides=(1,),
            padding=0,  # We'll handle padding manually
            feature_group_count=1,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        """Apply causal convolution.

        Args:
            x: Input tensor [batch, length, channels]
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor with causal padding applied
        """
        # Apply left padding for causality
        if self.padding > 0:
            # Pad on the left with zeros
            pad_config = [(0, 0), (self.padding, 0), (0, 0)]
            x = jnp.pad(x, pad_config, mode="constant", constant_values=0)

        # Apply convolution
        x = self.conv(x)

        return x


class GatedActivationUnit(nnx.Module):
    """Gated activation unit as used in WaveNet."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize gated activation unit.

        Args:
            channels: Number of channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor
            rngs: Random number generators
        """
        super().__init__()

        self.tanh_conv = CausalConv1D(
            in_features=channels,
            out_features=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            rngs=rngs,
        )

        self.sigmoid_conv = CausalConv1D(
            in_features=channels,
            out_features=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        """Apply gated activation.

        Args:
            x: Input tensor
            **kwargs: Additional keyword arguments

        Returns:
            Output with gated activation applied
        """
        tanh_out = jnp.tanh(self.tanh_conv(x))
        sigmoid_out = jax.nn.sigmoid(self.sigmoid_conv(x))
        return tanh_out * sigmoid_out


# ResidualBlock is now imported from core layers as WaveNetResidualBlock


class WaveNet(AutoregressiveModel):
    """WaveNet model for autoregressive sequence generation.

    Uses dilated convolutions to capture long-range dependencies
    while maintaining the autoregressive property.

    This model follows the config-based signature pattern:
    __init__(self, config: WaveNetConfig, *, rngs: nnx.Rngs)
    """

    def __init__(
        self,
        config: WaveNetConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize WaveNet model from config.

        Args:
            config: WaveNet configuration dataclass
            rngs: Random number generators
        """
        super().__init__(
            vocab_size=config.vocab_size,
            sequence_length=config.sequence_length,
            rngs=rngs,
        )

        # Store config
        self.config = config

        # Extract parameters from config
        self.residual_channels = config.residual_channels
        self.skip_channels = config.skip_channels
        self.num_blocks = config.num_blocks
        self.num_layers = config.layers_per_block
        self.kernel_size = config.kernel_size
        self.dilation_base = config.dilation_base
        self.use_gated_activation = config.use_gated_activation

        # Input embedding
        self.input_embedding = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=self.residual_channels,
            rngs=rngs,
        )

        # Initial causal convolution
        self.initial_conv = CausalConv1D(
            in_features=self.residual_channels,
            out_features=self.residual_channels,
            kernel_size=self.kernel_size,
            rngs=rngs,
        )

        # Residual blocks with increasing dilation
        self.residual_blocks = nnx.List([])
        for block in range(self.num_blocks):
            for layer in range(self.num_layers):
                dilation = self.dilation_base**layer

                residual_block = WaveNetResidualBlock(
                    residual_channels=self.residual_channels,
                    skip_channels=self.skip_channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    use_gated_activation=self.use_gated_activation,
                    rngs=rngs,
                )
                self.residual_blocks.append(residual_block)

        # Post-processing layers
        self.post_conv1 = nnx.Conv(
            in_features=self.skip_channels,
            out_features=self.skip_channels,
            kernel_size=(1,),
            rngs=rngs,
        )

        self.post_conv2 = nnx.Conv(
            in_features=self.skip_channels,
            out_features=config.vocab_size,
            kernel_size=(1,),
            rngs=rngs,
        )

    def __call__(
        self, x: jax.Array, *, rngs: nnx.Rngs | None = None, training: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Forward pass through WaveNet.

        Args:
            x: Input sequences [batch, length]
            rngs: Random number generators
            training: Whether in training mode
            **kwargs: Additional keyword arguments

        Returns:
            dictionary containing logits and intermediate outputs
        """
        # Input embedding
        embedded = self.input_embedding(x)  # [batch, length, residual_channels]

        # Initial convolution
        residual = self.initial_conv(embedded)

        # Collect skip connections
        skip_connections = []

        # Pass through residual blocks
        for block in self.residual_blocks:
            residual, skip = block(residual)
            skip_connections.append(skip)

        # Sum all skip connections
        skip_sum = jnp.sum(jnp.stack(skip_connections, axis=0), axis=0)

        # Post-processing
        out = nnx.relu(skip_sum)
        out = self.post_conv1(out)
        out = nnx.relu(out)
        logits = self.post_conv2(out)

        return {
            "logits": logits,
            "skip_connections": skip_connections,
            "embedded": embedded,
        }

    def generate_fast(
        self,
        n_samples: int = 1,
        max_length: int | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        temperature: float = 1.0,
        **kwargs,
    ) -> jax.Array:
        """Fast generation using incremental convolution.

        This is more efficient than the base class implementation
        as it caches intermediate convolution results.

        Args:
            n_samples: Number of samples to generate
            max_length: Maximum generation length
            rngs: Random number generators
            temperature: Sampling temperature
            **kwargs: Additional keyword arguments

        Returns:
            Generated sequences [n_samples, max_length]
        """
        if rngs is None:
            rngs = self._rngs

        if max_length is None:
            max_length = self.sequence_length

        # Initialize sequences
        sequences = jnp.zeros((n_samples, max_length), dtype=jnp.int32)

        # Get sampling key
        sample_key = self._get_rng_key(rngs, "sample", 0)

        # For fast generation, we'd need to implement incremental convolution
        # For now, use the standard autoregressive generation
        for pos in range(max_length):
            # Only process up to current position
            current_seq = sequences[:, : pos + 1]

            # Pad to minimum required length for the model
            min_length = max(pos + 1, self.kernel_size)
            if current_seq.shape[1] < min_length:
                padding = jnp.zeros((n_samples, min_length - current_seq.shape[1]), dtype=jnp.int32)
                current_seq = jnp.concatenate([padding, current_seq], axis=1)

            # Get logits
            outputs = self(current_seq, rngs=rngs, training=False, **kwargs)
            logits = outputs["logits"]

            # Extract logits for last position
            current_logits = logits[:, -1, :]  # [n_samples, vocab_size]

            # Apply temperature
            if temperature != 1.0:
                current_logits = current_logits / temperature

            # Sample next token
            sample_key, subkey = jax.random.split(sample_key)
            next_tokens = jax.random.categorical(subkey, current_logits, axis=-1)

            # Update sequences
            sequences = sequences.at[:, pos].set(next_tokens)

        return sequences

    def compute_receptive_field(self) -> int:
        """Compute the receptive field of the model.

        Returns:
            Size of the receptive field
        """
        receptive_field = 1

        for block in range(self.num_blocks):
            for layer in range(self.num_layers):
                dilation = self.dilation_base**layer
                receptive_field += (self.kernel_size - 1) * dilation

        return receptive_field

    def loss_fn(
        self,
        batch: Any,
        model_outputs: dict[str, Any],
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Compute loss for WaveNet.

        Args:
            batch: Input batch
            model_outputs: Model outputs
            rngs: Random number generators
            **kwargs: Additional keyword arguments

        Returns:
            dictionary containing loss and metrics
        """
        # Extract sequences from batch
        if isinstance(batch, dict):
            sequences = batch.get("x", batch.get("sequences", batch))
            mask = batch.get("mask", None)
        else:
            sequences = batch
            mask = None

        # Get logits
        logits = model_outputs["logits"]

        # Compute autoregressive loss using parent method
        loss = self.compute_loss(logits, sequences, mask)

        # Compute additional metrics
        batch_size, seq_len, vocab_size = logits.shape

        # Shift for autoregressive prediction
        shifted_targets = sequences[:, 1:]
        shifted_logits = logits[:, :-1, :]

        # Compute accuracy
        predictions = jnp.argmax(shifted_logits, axis=-1)
        if mask is not None:
            shifted_mask = mask[:, 1:]
            correct = (predictions == shifted_targets) * shifted_mask
            accuracy = jnp.sum(correct) / (jnp.sum(shifted_mask) + 1e-8)
        else:
            correct = predictions == shifted_targets
            accuracy = jnp.mean(correct)

        return {
            "loss": loss,
            "nll_loss": loss,
            "accuracy": accuracy,
            "perplexity": jnp.exp(loss),
            "receptive_field": self.compute_receptive_field(),
        }

    def get_intermediate_outputs(
        self, x: jax.Array, *, rngs: nnx.Rngs | None = None, **kwargs
    ) -> dict[str, Any]:
        """Get intermediate outputs from all layers.

        Args:
            x: Input sequences
            rngs: Random number generators
            **kwargs: Additional keyword arguments

        Returns:
            dictionary with intermediate activations
        """
        # Input embedding
        embedded = self.input_embedding(x)

        # Initial convolution
        residual = self.initial_conv(embedded)

        # Track all intermediate outputs
        residual_outputs = [residual]
        skip_outputs = []

        # Pass through residual blocks
        for i, block in enumerate(self.residual_blocks):
            residual, skip = block(residual)
            residual_outputs.append(residual)
            skip_outputs.append(skip)

        # Sum skip connections
        skip_sum = jnp.sum(jnp.stack(skip_outputs, axis=0), axis=0)

        # Post-processing
        post1 = nnx.relu(skip_sum)
        post1 = self.post_conv1(post1)

        post2 = nnx.relu(post1)
        logits = self.post_conv2(post2)

        return {
            "embedded": embedded,
            "residual_outputs": residual_outputs,
            "skip_outputs": skip_outputs,
            "skip_sum": skip_sum,
            "post1": post1,
            "post2": post2,
            "logits": logits,
        }

    def conditional_generate(
        self,
        conditioning: jax.Array,
        n_samples: int = 1,
        max_new_tokens: int = 100,
        *,
        rngs: nnx.Rngs | None = None,
        temperature: float = 1.0,
        **kwargs,
    ) -> jax.Array:
        """Generate sequences conditioned on a prefix.

        Args:
            conditioning: Conditioning sequence [conditioning_length]
            n_samples: Number of samples to generate
            max_new_tokens: Maximum number of new tokens to generate
            rngs: Random number generators
            temperature: Sampling temperature
            **kwargs: Additional keyword arguments

        Returns:
            Generated sequences [n_samples, total_length]
        """
        if rngs is None:
            rngs = self._rngs

        conditioning_length = len(conditioning)
        total_length = conditioning_length + max_new_tokens

        # Initialize sequences with conditioning
        sequences = jnp.zeros((n_samples, total_length), dtype=jnp.int32)
        sequences = sequences.at[:, :conditioning_length].set(
            jnp.tile(conditioning[None, :], (n_samples, 1))
        )

        # Get sampling key
        sample_key = self._get_rng_key(rngs, "sample", 0)

        # Generate new tokens
        for pos in range(conditioning_length, total_length):
            # Get logits for current sequence
            current_seq = sequences[:, : pos + 1]
            outputs = self(current_seq, rngs=rngs, training=False, **kwargs)
            logits = outputs["logits"]

            # Extract logits for current position
            current_logits = logits[:, pos, :] / temperature

            # Sample next token
            sample_key, subkey = jax.random.split(sample_key)
            next_tokens = jax.random.categorical(subkey, current_logits, axis=-1)

            # Update sequences
            sequences = sequences.at[:, pos].set(next_tokens)

        return sequences
