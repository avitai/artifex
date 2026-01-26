"""Energy-based model base classes and utilities.

This module provides base classes for implementing energy-based models (EBMs)
using Flax NNX. EBMs learn a data distribution by modeling an energy function
E(x) where p(x) ∝ exp(-E(x)).
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel


class EnergyFunction(nnx.Module):
    """Base class for energy functions.

    An energy function E(x) maps input data to scalar energy values.
    Lower energy values correspond to higher probability under the model.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the energy function.

        Args:
            rngs: Random number generators
        """
        super().__init__()
        self.rngs = rngs or nnx.Rngs()

    def __call__(self, x: jax.Array) -> jax.Array:
        """Compute the energy of input data.

        Args:
            x: Input data of shape (batch_size, ...)

        Returns:
            Energy values of shape (batch_size,)
        """
        raise NotImplementedError("Subclasses must implement __call__")


class EnergyBasedModel(GenerativeModel):
    """Base class for Energy-Based Models.

    Energy-based models learn a probability distribution p(x) ∝ exp(-E(x))
    where E(x) is an energy function. Training uses contrastive divergence
    to maximize the likelihood of data points while minimizing the likelihood
    of generated samples.
    """

    def __init__(
        self,
        energy_fn: EnergyFunction,
        *,
        rngs: nnx.Rngs | None = None,
        precision: jax.lax.Precision | None = None,
    ):
        """Initialize the energy-based model.

        Args:
            energy_fn: Energy function E(x)
            rngs: Random number generators
            precision: Numerical precision for computations
        """
        super().__init__(rngs=rngs or nnx.Rngs(), precision=precision)
        self.energy_fn = energy_fn

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute energy E(x) for input data.

        Args:
            x: Input data of shape (batch_size, ...)

        Returns:
            Energy values of shape (batch_size,)
        """
        return self.energy_fn(x)

    def energy_outputs(self, x: jax.Array) -> dict[str, jax.Array]:
        """Compute energy function outputs including energy and unnormalized log prob.

        Args:
            x: Input data of shape (batch_size, ...)

        Returns:
            Dictionary containing:
                - "energy": Energy values of shape (batch_size,)
                - "unnormalized_log_prob": Unnormalized log probabilities -E(x)
        """
        energy_values = self.energy(x)
        return {
            "energy": energy_values,
            "unnormalized_log_prob": -energy_values,
        }

    def unnormalized_log_prob(self, x: jax.Array) -> jax.Array:
        """Compute unnormalized log probability -E(x).

        Args:
            x: Input data of shape (batch_size, ...)

        Returns:
            Unnormalized log probabilities of shape (batch_size,)
        """
        return -self.energy(x)

    def score(self, x: jax.Array) -> jax.Array:
        """Compute the score function ∇_x log p(x) = -∇_x E(x).

        Args:
            x: Input data of shape (batch_size, ...)

        Returns:
            Score values with same shape as x
        """
        # Set eval mode to disable dropout during traced operations
        # This prevents TraceContextError when nnx.Dropout tries to mutate RNG
        self.energy_fn.eval()

        def energy_single(x_single):
            return self.energy(x_single[None])[0]

        result = -nnx.vmap(nnx.grad(energy_single))(x)

        # Restore train mode
        self.energy_fn.train()

        return result

    def __call__(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Forward pass through the energy-based model.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            x: Input data of shape (batch_size, ...)
            rngs: Random number generators (unused)
            **kwargs: Additional keyword arguments

        Returns:
            dictionary containing:
            - "energy": Energy values E(x)
            - "unnormalized_log_prob": Unnormalized log probabilities -E(x)
            - "score": Score function values ∇_x log p(x)
        """
        energy_vals = self.energy(x)
        log_prob_vals = -energy_vals
        score_vals = self.score(x)

        return {
            "energy": energy_vals,
            "unnormalized_log_prob": log_prob_vals,
            "score": score_vals,
        }

    def contrastive_divergence_loss(
        self,
        real_data: jax.Array,
        fake_data: jax.Array,
        alpha: float = 0.01,
    ) -> dict[str, jax.Array]:
        """Compute contrastive divergence loss.

        The loss consists of:
        1. Contrastive divergence: E(fake) - E(real)
        2. Regularization: α * (E(real)² + E(fake)²)

        Args:
            real_data: Real data samples
            fake_data: Generated/fake data samples
            alpha: Regularization strength

        Returns:
            dictionary containing loss components
        """
        real_energy = self.energy(real_data)
        fake_energy = self.energy(fake_data)

        # Contrastive divergence loss
        cd_loss = real_energy.mean() - fake_energy.mean()

        # Regularization loss to keep energies in reasonable range
        reg_loss = alpha * (real_energy**2 + fake_energy**2).mean()

        # Total loss
        total_loss = cd_loss + reg_loss

        return {
            "loss": total_loss,
            "contrastive_divergence": cd_loss,
            "regularization": reg_loss,
            "real_energy_mean": real_energy.mean(),
            "fake_energy_mean": fake_energy.mean(),
        }

    def loss_fn(
        self,
        batch: dict[str, jax.Array],
        model_outputs: dict[str, Any],
        *,
        rngs: nnx.Rngs | None = None,
        fake_samples: jax.Array | None = None,
        alpha: float = 0.01,
        **kwargs,
    ) -> dict[str, jax.Array]:
        """Compute loss for energy-based model training.

        Args:
            batch: Training batch containing real data
            model_outputs: Outputs from forward pass (unused)
            rngs: Random number generators (unused)
            fake_samples: Generated samples for contrastive divergence
            alpha: Regularization strength
            **kwargs: Additional arguments

        Returns:
            dictionary containing loss and metrics

        Raises:
            ValueError: If fake_samples is not provided
        """
        if fake_samples is None:
            raise ValueError("fake_samples must be provided for EBM training")

        # Extract real data from batch
        real_data = self._extract_data_from_batch(batch)

        # Compute contrastive divergence loss
        return self.contrastive_divergence_loss(real_data, fake_samples, alpha)

    def generate(
        self,
        n_samples: int = 1,
        *,
        shape: tuple[int, ...] | None = None,
        n_steps: int = 100,
        step_size: float = 0.01,
        noise_scale: float = 0.005,
        **kwargs,
    ) -> jax.Array:
        """Generate samples using Langevin dynamics MCMC.

        Args:
            n_samples: Number of samples to generate
            shape: Shape of each sample (excluding batch dimension)
            n_steps: Number of MCMC steps
            step_size: Step size for Langevin dynamics
            noise_scale: Standard deviation of noise added at each step
            **kwargs: Additional arguments

        Returns:
            Generated samples of shape (n_samples, *shape)

        Raises:
            ValueError: If shape is not provided
        """
        if shape is None:
            raise ValueError("shape must be provided for sample generation")

        # Extract raw JAX key from internal rngs BEFORE any traced operations
        sample_key = self.rngs.sample()

        # Initialize samples from uniform noise
        key, subkey = jax.random.split(sample_key)
        samples = jax.random.uniform(subkey, (n_samples, *shape), minval=-1.0, maxval=1.0)

        # Set energy function to eval mode to disable dropout during MCMC
        self.energy_fn.eval()

        # Run Langevin dynamics
        from artifex.generative_models.models.energy.mcmc import langevin_dynamics

        samples = langevin_dynamics(
            energy_fn=self.energy,
            initial_samples=samples,
            n_steps=n_steps,
            step_size=step_size,
            noise_scale=noise_scale,
            rng_key=key,
        )

        return samples


class MLPEnergyFunction(EnergyFunction):
    """MLP-based energy function.

    A simple energy function implemented as a multi-layer perceptron
    that maps input data to scalar energy values.
    """

    def __init__(
        self,
        hidden_dims: list[int],
        *,
        input_dim: int,
        activation: Callable = nnx.gelu,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize MLP energy function.

        Args:
            hidden_dims: List of hidden layer dimensions
            input_dim: Input dimension
            activation: Activation function
            use_bias: Whether to use bias in linear layers
            dropout_rate: Dropout rate (0.0 to disable)
            rngs: Random number generators
        """
        super().__init__(rngs=rngs or nnx.Rngs())

        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate

        # Build layers
        self.layers = nnx.List([])
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(
                nnx.Linear(
                    in_features=in_dim,
                    out_features=hidden_dim,
                    use_bias=use_bias,
                    rngs=rngs,
                )
            )
            in_dim = hidden_dim

        # Output layer (single scalar output)
        self.output_layer = nnx.Linear(
            in_features=in_dim,
            out_features=1,
            use_bias=use_bias,
            rngs=rngs,
        )

        # Dropout layer
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        """Forward pass through MLP energy function.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            x: Input data of shape (batch_size, input_dim)

        Returns:
            Energy values of shape (batch_size,)
        """
        # Flatten input if needed
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        # Forward pass through hidden layers
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

            # Apply dropout using auto mode from model.train()/eval()
            if self.dropout is not None:
                x = self.dropout(x)

        # Output layer
        x = self.output_layer(x)

        # Return scalar energies (squeeze last dimension)
        return x.squeeze(-1)


class CNNEnergyFunction(EnergyFunction):
    """CNN-based energy function for image data.

    A convolutional energy function suitable for image data that
    uses smooth activation functions (like Swish) for better gradients.
    """

    def __init__(
        self,
        hidden_dims: list[int],
        *,
        input_channels: int = 1,
        kernel_size: int = 3,
        activation: Callable = nnx.silu,  # Swish activation
        use_bias: bool = True,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize CNN energy function.

        Args:
            hidden_dims: List of channel dimensions for conv layers
            input_channels: Number of input channels
            kernel_size: Kernel size for convolutions
            activation: Activation function (use smooth functions like silu/swish)
            use_bias: Whether to use bias in conv layers
            rngs: Random number generators

        """
        super().__init__(rngs=rngs or nnx.Rngs())

        self.hidden_dims = hidden_dims
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias

        # Build convolutional layers
        self.conv_layers = nnx.List([])
        in_channels = input_channels

        for out_channels in hidden_dims:
            self.conv_layers.append(
                nnx.Conv(
                    in_features=in_channels,
                    out_features=out_channels,
                    kernel_size=(kernel_size, kernel_size),
                    strides=(2, 2),  # Downsample
                    padding="SAME",
                    use_bias=use_bias,
                    rngs=rngs,
                )
            )
            in_channels = out_channels

        # Global average pooling and final linear layer
        self.final_linear = nnx.Linear(
            in_features=hidden_dims[-1],
            out_features=1,
            use_bias=use_bias,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        """Forward pass through CNN energy function.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.

        Args:
            x: Input images of shape (batch_size, channels, height, width)

        Returns:
            Energy values of shape (batch_size,)
        """
        # Forward pass through conv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.activation(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # Average over spatial dimensions

        # Final linear layer to get scalar energy
        x = self.final_linear(x)

        return x.squeeze(-1)
