"""Positional encoding implementations for transformers - CORRECTED VERSION.

This module provides various positional encoding implementations that can be
used with transformer models using Flax NNX.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx


class PositionalEncoding(nnx.Module):
    """Base class for positional encodings.

    This class defines the common interface for all positional encodings.
    Subclasses should implement the `__call__` method.

    Attributes:
        dim: Dimension of the positional encoding.
        max_len: Maximum sequence length.
        dropout_rate: Dropout probability.
        dropout: Dropout layer if dropout_rate > 0.
    """

    def __init__(
        self,
        dim: int,
        max_len: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the positional encoding.

        Args:
            dim: Dimension of the encoding.
            max_len: Maximum sequence length.
            dropout_rate: Dropout probability.
            rngs: Random number generators.
        """
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.dropout_rate = dropout_rate

        # Validate that rngs is provided when dropout_rate > 0
        if self.dropout_rate > 0 and rngs is None:
            raise ValueError("rngs must be provided when dropout_rate > 0")

        # Initialize dropout if needed
        if self.dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=self.dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Apply positional encoding to input. To be implemented by subclasses.

        Args:
            x: Input tensor of shape [batch, length, dim].
            deterministic: Whether to apply dropout. If True, dropout is not applied.
            rngs: Random number generators for dropout.

        Returns:
            Output tensor with positional encoding applied.
        """
        raise NotImplementedError("Subclasses must implement this method")


class SinusoidalPositionalEncoding(PositionalEncoding):
    """Sinusoidal positional encoding.

    This implementation follows the original transformer paper, using
    sine and cosine functions of different frequencies.

    Attributes:
        pe: Non-trainable parameter storing the precomputed sinusoidal encodings.
    """

    def __init__(
        self,
        dim: int,
        max_len: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the sinusoidal positional encoding.

        Args:
            dim: Dimension of the encoding.
            max_len: Maximum sequence length.
            dropout_rate: Dropout probability.
            rngs: Random number generators, primarily for dropout if enabled.
                  Not used for the direct calculation of sinusoidal encodings.
        """
        super().__init__(dim=dim, max_len=max_len, dropout_rate=dropout_rate, rngs=rngs)

        # Initialize the positional encoding matrix
        position = jnp.arange(max_len)[:, None]  # max_len, 1
        div_term = jnp.exp(jnp.arange(0, dim, 2) * (-jnp.log(10000.0) / dim))  # dim/2

        pe = jnp.zeros((max_len, dim))

        # Apply sin to even indices in the array; 2i
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        # Apply cos to odd indices in the array; 2i+1
        if dim % 2 == 1:
            # Handle odd dimension case
            pe = pe.at[:, 1::2].set(jnp.cos(position * div_term[:-1]))
        else:
            pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        # Store PE as a non-trainable parameter (buffer)
        self.pe = nnx.Param(pe, trainable=False)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Apply sinusoidal positional encoding to input.

        Args:
            x: Input tensor of shape [batch, length, dim].
            deterministic: Whether to apply dropout.
            rngs: Random number generators for dropout.

        Returns:
            Output tensor with positional encoding added.
        """
        # Add positional encoding
        # Input x has shape [batch, length, dim]
        # self.pe.value has shape [max_len, dim]
        # We slice self.pe.value to the actual sequence length of x
        seq_len = x.shape[1]

        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")

        # The positional encoding is added to the input embeddings.
        # Broadcasting takes care of the batch dimension.
        x = x + self.pe.value[:seq_len, :]

        # Apply dropout if not in deterministic mode and dropout_rate > 0
        if self.dropout_rate > 0 and not deterministic:
            if self.dropout is None:
                raise RuntimeError("Dropout layer not initialized")
            x = self.dropout(x, deterministic=deterministic, rngs=rngs)

        return x


class LearnedPositionalEncoding(PositionalEncoding):
    """Learned positional encoding.

    Instead of using fixed sinusoidal functions, this encoding learns
    the position embeddings during training.

    Attributes:
        pe: Trainable parameter storing the learned positional embeddings.
        kernel_init: Initialization function for the embedding.
    """

    def __init__(
        self,
        dim: int,
        max_len: int,
        dropout_rate: float = 0.0,
        kernel_init: Callable = nnx.initializers.normal(stddev=0.02),
        *,
        rngs: nnx.Rngs,  # Made required since we need it for initialization
    ):
        """Initialize the learned positional encoding.

        Args:
            dim: Dimension of the encoding.
            max_len: Maximum sequence length.
            dropout_rate: Dropout probability.
            kernel_init: Initialization function for the embedding.
            rngs: Random number generators. Required for embedding initialization
                  and if dropout_rate > 0.
        """
        # Call super with rngs
        super().__init__(dim=dim, max_len=max_len, dropout_rate=dropout_rate, rngs=rngs)

        self.kernel_init = kernel_init

        # Get a JAX PRNGKey for parameter initialization from nnx.Rngs
        # 'params' is a conventional key name for parameter initialization RNGs
        key = rngs.params()

        # Initialize the learned positional encoding using Param class
        # The shape is (max_len, dim) as we learn one embedding vector per position.
        self.pe = nnx.Param(self.kernel_init(key, (max_len, dim)), trainable=True)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Apply learned positional encoding to input.

        Args:
            x: Input tensor of shape [batch, length, dim].
            deterministic: Whether to apply dropout.
            rngs: Random number generators for dropout.

        Returns:
            Output tensor with positional encoding added.
        """
        # Add positional encoding
        seq_len = x.shape[1]

        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")

        # Add the learned positional embeddings (sliced to the current sequence length)
        # to the input embeddings.
        x = x + self.pe.value[:seq_len, :]

        # Apply dropout if not in deterministic mode and dropout_rate > 0
        if self.dropout_rate > 0 and not deterministic:
            if self.dropout is None:
                raise RuntimeError("Dropout layer not initialized")
            x = self.dropout(x, deterministic=deterministic, rngs=rngs)

        return x


class RotaryPositionalEncoding(PositionalEncoding):
    """Rotary positional encoding (RoPE).

    RoPE performs rotation of the embedding features, where the amount of
    rotation depends on the position and feature dimension.

    Attributes:
        base: Base for frequency computation.
        sin: Non-trainable precomputed sine values for rotation.
        cos: Non-trainable precomputed cosine values for rotation.
    """

    def __init__(
        self,
        dim: int,
        max_len: int,
        base: int = 10000,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the rotary positional encoding.

        Args:
            dim: Dimension of the encoding (must be even).
            max_len: Maximum sequence length.
            base: Base for frequency computation.
            dropout_rate: Dropout probability.
            rngs: Random number generators, primarily for dropout if enabled.
                  Not used for the direct calculation of RoPE sin/cos values.
        """
        # Check that dimension is even, as RoPE operates on pairs of features.
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even for RoPE, got {dim}")

        super().__init__(dim=dim, max_len=max_len, dropout_rate=dropout_rate, rngs=rngs)

        self.base = base

        # Compute frequencies (theta_i in the RoPE paper)
        # inv_freq shape: (dim / 2,)
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2) / self.dim))

        # Position indices m: (max_len,)
        position = jnp.arange(max_len)

        # Outer product to get m * theta_i: (max_len, dim / 2)
        sinusoid_inp = jnp.einsum("i,j->ij", position, inv_freq)

        # Compute sin and cos values
        # sin and cos will have shape (max_len, dim / 2)
        sin_val = jnp.sin(sinusoid_inp)
        cos_val = jnp.cos(sinusoid_inp)

        # Store as non-trainable parameters (buffers)
        self.sin = nnx.Param(sin_val, trainable=False)
        self.cos = nnx.Param(cos_val, trainable=False)

    def _rotate_half(self, x: jax.Array) -> jax.Array:
        """Helper function to apply rotation to one half of the features.
        x_even, x_odd -> -x_odd, x_even
        """
        x_part1 = x[..., : x.shape[-1] // 2]
        x_part2 = x[..., x.shape[-1] // 2 :]
        return jnp.concatenate((-x_part2, x_part1), axis=-1)

    def __call__(
        self,
        x: jax.Array,  # Input shape: [batch, length, dim]
        *,
        deterministic: bool = False,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Apply rotary positional encoding to input.

        Args:
            x: Input tensor of shape [batch, length, dim].
            deterministic: Whether to apply dropout.
            rngs: Random number generators for dropout.

        Returns:
            Output tensor with rotary positional encoding applied.
        """
        seq_len = x.shape[1]

        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")

        # Get sin and cos values for the current sequence length
        # Sliced sin/cos shapes: [seq_len, dim/2]
        sin_pos = self.sin.value[:seq_len, :]
        cos_pos = self.cos.value[:seq_len, :]

        # Split input into two halves along the last dimension
        # x_left, x_right shapes: [batch, seq_len, dim/2]
        x_left, x_right = jnp.split(x, 2, axis=-1)

        # Apply rotary transformation
        # This implements the rotation matrix multiplication:
        # R(m, theta) @ [x_left; x_right] =
        #   [x_left * cos - x_right * sin; x_left * sin + x_right * cos]
        x_left_rotated = x_left * cos_pos - x_right * sin_pos
        x_right_rotated = x_left * sin_pos + x_right * cos_pos

        # Concatenate the rotated parts
        x_rotated = jnp.concatenate([x_left_rotated, x_right_rotated], axis=-1)

        # Apply dropout if not in deterministic mode and dropout_rate > 0
        if self.dropout_rate > 0 and not deterministic:
            if self.dropout is None:
                raise RuntimeError("Dropout layer not initialized")
            x_rotated = self.dropout(x_rotated, deterministic=deterministic, rngs=rngs)

        return x_rotated
