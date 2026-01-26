"""WaveNet implementation for autoregressive audio generation."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from .base import AudioModelConfig, BaseAudioModel


@dataclass
class WaveNetConfig(AudioModelConfig):
    """Configuration for WaveNet model.

    Args:
        modality_config: Audio modality configuration
        n_dilated_blocks: Number of dilated convolution blocks
        n_residual_channels: Number of residual channels
        n_skip_channels: Number of skip connection channels
        n_end_channels: Number of final convolution channels
        kernel_size: Convolution kernel size
        dilation_rates: List of dilation rates for each block
        quantization_levels: Number of quantization levels for mu-law
    """

    n_dilated_blocks: int = 10
    n_residual_channels: int = 64
    n_skip_channels: int = 64
    n_end_channels: int = 128
    kernel_size: int = 2
    dilation_rates: list[int] | None = None
    quantization_levels: int = 256

    def __post_init__(self):
        """Set default dilation rates if not provided."""
        if self.dilation_rates is None:
            # Standard WaveNet dilation pattern: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
            self.dilation_rates = [2**i for i in range(self.n_dilated_blocks)]


class DilatedConv1D(nnx.Module):
    """Dilated causal convolution layer for WaveNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize dilated convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            dilation: Dilation rate
            rngs: Random number generators
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Standard convolution - we'll implement causal padding manually
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size,),
            strides=(1,),
            padding="VALID",  # No padding - we handle it manually
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply dilated causal convolution.

        Args:
            x: Input tensor of shape (batch, time, channels)

        Returns:
            Output tensor of same temporal length
        """
        # Apply causal padding
        # For dilation d and kernel k, need padding of (k-1)*d
        padding_amount = (self.kernel_size - 1) * self.dilation

        # Pad on the left (causal)
        x_padded = jnp.pad(x, ((0, 0), (padding_amount, 0), (0, 0)), mode="constant")

        # Apply dilated convolution by strided slicing
        if self.dilation > 1:
            # Implement dilation by sub-sampling
            dilated_x = x_padded[:, :: self.dilation, :]
            output = self.conv(dilated_x)

            # Interpolate back to original temporal resolution
            # Simple nearest neighbor upsampling
            batch_size, time_dilated, channels = output.shape
            time_original = x.shape[1]

            # Create indices for upsampling
            indices = jnp.arange(time_original) // self.dilation
            indices = jnp.clip(indices, 0, time_dilated - 1)

            output = output[:, indices, :]
        else:
            output = self.conv(x_padded)

        # Ensure output has same temporal length as input
        output = output[:, : x.shape[1], :]

        return output


class ResidualBlock(nnx.Module):
    """Residual block with gated activation for WaveNet."""

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize residual block.

        Args:
            residual_channels: Number of residual channels
            skip_channels: Number of skip channels
            kernel_size: Convolution kernel size
            dilation: Dilation rate
            rngs: Random number generators
        """
        super().__init__()

        # Dilated convolution for filter and gate
        self.filter_conv = DilatedConv1D(
            residual_channels, residual_channels, kernel_size, dilation, rngs=rngs
        )
        self.gate_conv = DilatedConv1D(
            residual_channels, residual_channels, kernel_size, dilation, rngs=rngs
        )

        # 1x1 convolutions for residual and skip connections
        self.residual_conv = nnx.Conv(
            in_features=residual_channels,
            out_features=residual_channels,
            kernel_size=(1,),
            rngs=rngs,
        )
        self.skip_conv = nnx.Conv(
            in_features=residual_channels,
            out_features=skip_channels,
            kernel_size=(1,),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply residual block.

        Args:
            x: Input tensor

        Returns:
            Tuple of (residual_output, skip_output)
        """
        # Gated activation
        filter_output = nnx.tanh(self.filter_conv(x))
        gate_output = nnx.sigmoid(self.gate_conv(x))
        gated = filter_output * gate_output

        # Residual connection
        residual = self.residual_conv(gated)
        residual_output = x + residual

        # Skip connection
        skip_output = self.skip_conv(gated)

        return residual_output, skip_output


class WaveNetAudioModel(BaseAudioModel):
    """WaveNet model for autoregressive audio generation."""

    def __init__(
        self,
        config: WaveNetConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize WaveNet model.

        Args:
            config: WaveNet configuration
            rngs: Random number generators
        """
        super().__init__(config, rngs=rngs)
        self.wavenet_config = config

        # Input embedding for quantized audio
        self.input_embedding = nnx.Embed(
            num_embeddings=config.quantization_levels,
            features=config.n_residual_channels,
            rngs=rngs,
        )

        # Initial causal convolution
        self.initial_conv = nnx.Conv(
            in_features=config.n_residual_channels,
            out_features=config.n_residual_channels,
            kernel_size=(1,),
            rngs=rngs,
        )

        # Residual blocks
        self.residual_blocks = nnx.List([])
        for dilation in config.dilation_rates:
            block = ResidualBlock(
                residual_channels=config.n_residual_channels,
                skip_channels=config.n_skip_channels,
                kernel_size=config.kernel_size,
                dilation=dilation,
                rngs=rngs,
            )
            self.residual_blocks.append(block)

        # Final convolutions
        self.final_conv1 = nnx.Conv(
            in_features=config.n_skip_channels,
            out_features=config.n_end_channels,
            kernel_size=(1,),
            rngs=rngs,
        )
        self.final_conv2 = nnx.Conv(
            in_features=config.n_end_channels,
            out_features=config.quantization_levels,
            kernel_size=(1,),
            rngs=rngs,
        )

    def mu_law_encode(self, audio: jnp.ndarray) -> jnp.ndarray:
        """Apply mu-law encoding to audio.

        Args:
            audio: Raw audio in [-1, 1] range

        Returns:
            Quantized audio indices
        """
        mu = self.wavenet_config.quantization_levels - 1

        # Mu-law encoding
        encoded = jnp.sign(audio) * jnp.log1p(mu * jnp.abs(audio)) / jnp.log1p(mu)

        # Quantize to discrete levels
        quantized = jnp.round((encoded + 1) * mu / 2)
        quantized = jnp.clip(quantized, 0, mu).astype(jnp.int32)

        return quantized

    def mu_law_decode(self, quantized: jnp.ndarray) -> jnp.ndarray:
        """Decode mu-law encoded audio.

        Args:
            quantized: Quantized audio indices

        Returns:
            Raw audio in [-1, 1] range
        """
        mu = self.wavenet_config.quantization_levels - 1

        # Convert back to [-1, 1] range
        encoded = (quantized.astype(jnp.float32) * 2 / mu) - 1

        # Mu-law decoding
        decoded = jnp.sign(encoded) * (jnp.expm1(jnp.abs(encoded) * jnp.log1p(mu)) / mu)

        return decoded

    def __call__(
        self, x: jnp.ndarray, *, training: bool = False, **kwargs
    ) -> dict[str, jnp.ndarray]:
        """Forward pass through WaveNet.

        Args:
            x: Input audio (raw or quantized)
            training: Whether in training mode
            **kwargs: Additional arguments

        Returns:
            Dictionary with model outputs
        """
        # Handle different input types
        if x.dtype == jnp.float32:
            # Raw audio - quantize first
            x_quantized = self.mu_law_encode(x)
        else:
            # Already quantized
            x_quantized = x

        # Embed quantized input
        if x_quantized.ndim == 2:
            # Add channel dimension if needed
            x_quantized = x_quantized[..., None]

        # Flatten last two dimensions for embedding
        batch_size, time_steps, channels = x_quantized.shape
        x_flat = x_quantized.reshape(-1)

        # Apply embedding
        embedded = self.input_embedding(x_flat)
        embedded = embedded.reshape(batch_size, time_steps, -1)

        # Initial convolution
        h = self.initial_conv(embedded)

        # Residual blocks with skip connections
        skip_outputs = []
        for block in self.residual_blocks:
            h, skip = block(h)
            skip_outputs.append(skip)

        # Sum skip connections
        skip_sum = jnp.sum(jnp.stack(skip_outputs), axis=0)

        # Final convolutions with ReLU activations
        output = nnx.relu(skip_sum)
        output = nnx.relu(self.final_conv1(output))
        logits = self.final_conv2(output)

        return {
            "logits": logits,
            "predictions": logits,
        }

    @nnx.jit
    def _generate_step(
        self,
        context: jnp.ndarray,
        t: jax.Array,
        subkey: jax.Array,
    ) -> jnp.ndarray:
        """JIT-compiled single generation step with fixed-size context.

        Args:
            context: Full audio context (n_samples, n_timesteps)
            t: Current timestep index
            subkey: PRNG key for sampling

        Returns:
            Next audio sample values
        """
        outputs = self(context, training=False)
        logits = outputs["logits"]

        # Sample next token at position t
        next_logits = logits[:, t, :]
        next_sample = jax.random.categorical(subkey, next_logits)

        # Convert to audio using mu-law decode
        next_audio = self.mu_law_decode(next_sample[..., None])
        return next_audio.squeeze(-1)

    def generate(
        self,
        n_samples: int = 1,
        duration: float | None = None,
        seed_audio: jnp.ndarray | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> jnp.ndarray:
        """Generate audio samples autoregressively.

        Uses @nnx.jit compiled forward passes with fixed-size context for
        efficient generation. WaveNet's causal convolutions ensure that
        positions > t don't affect predictions at position t.

        Args:
            n_samples: Number of samples to generate
            duration: Duration of generated audio
            seed_audio: Optional seed audio to condition on
            rngs: Random number generators
            **kwargs: Additional generation parameters

        Returns:
            Generated audio samples
        """
        duration = duration or self.modality_config.duration
        n_timesteps = int(self.sample_rate * duration)

        key = (rngs or self.rngs).sample()

        # Initialize with seed or zeros
        if seed_audio is not None:
            seed_length = min(seed_audio.shape[-1], n_timesteps // 4)
            generated = jnp.zeros((n_samples, n_timesteps))
            generated = generated.at[:, :seed_length].set(seed_audio[:, :seed_length])
            start_idx = seed_length
        else:
            generated = jnp.zeros((n_samples, n_timesteps))
            start_idx = 0

        # Autoregressive generation with JIT-compiled steps
        # Use full fixed-size context at each step — causal convolutions
        # ensure positions > t don't affect output at position t
        for t in range(start_idx, n_timesteps):
            # Split key for this step
            key, subkey = jax.random.split(key)

            # JIT-compiled forward pass with fixed-size context
            # Pass t as array to keep it traced (not static)
            t_arr = jnp.array(t)
            next_audio = self._generate_step(generated, t_arr, subkey)

            # Store generated sample
            generated = generated.at[:, t].set(next_audio)

        return self.postprocess_audio(generated)

    def loss_fn(
        self,
        batch: dict[str, jnp.ndarray],
        model_outputs: dict[str, jnp.ndarray],
        **kwargs,  # noqa: ARG002 - reserved for future loss customization
    ) -> jax.Array:
        """Compute WaveNet training loss.

        Args:
            batch: Training batch with audio
            model_outputs: Model predictions
            **kwargs: Additional loss parameters (reserved for future use)

        Returns:
            Cross-entropy loss
        """
        del kwargs  # Unused but part of interface
        target_audio = batch["audio"]
        logits = model_outputs["logits"]

        # Quantize target audio
        target_quantized = self.mu_law_encode(target_audio)

        # Reshape for loss computation
        if target_quantized.ndim == 2:
            target_quantized = target_quantized[..., None]

        # Flatten for cross-entropy loss
        logits_flat = logits.reshape(-1, self.wavenet_config.quantization_levels)
        targets_flat = target_quantized.reshape(-1)

        # Cross-entropy loss
        log_probs = jax.nn.log_softmax(logits_flat)
        loss = -jnp.mean(log_probs[jnp.arange(targets_flat.shape[0]), targets_flat])

        return loss
