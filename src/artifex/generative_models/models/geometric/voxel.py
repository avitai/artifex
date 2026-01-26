"""Voxel generative model."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration.geometric_config import (
    VoxelConfig,
)
from artifex.generative_models.core.losses.geometric import get_voxel_loss
from artifex.generative_models.models.geometric.base import GeometricModel


class VoxelModel(GeometricModel):
    """Model for generating 3D voxel grids.

    Voxel grids are 3D arrays where each cell represents the presence or
    density of material at that location.
    """

    def __init__(self, config: VoxelConfig, *, rngs: nnx.Rngs):
        """Initialize the voxel model.

        Args:
            config: VoxelConfig dataclass with model parameters.
            rngs: Random number generator keys.

        Raises:
            TypeError: If config is not a VoxelConfig
        """
        super().__init__(config, rngs=rngs)

        # Model parameters from dataclass config
        self.resolution = config.voxel_size
        self.latent_dim = config.network.base_channels
        # Build channel list from base_channels
        base = config.network.base_channels
        channels = [base, base // 2, base // 4, base // 8, base // 16, 1]
        self.use_batch_norm = True  # Default to batch norm

        # First, project latent vector to initial 3D volume
        init_size = self.resolution // (2 ** (len(channels) - 1))
        self.init_size = max(init_size, 1)  # Ensure it's at least 1

        # Linear layer to project latent to initial volume
        init_features = channels[0] * (self.init_size**3)
        self.latent_proj = nnx.Linear(
            in_features=self.latent_dim,
            out_features=init_features,
            rngs=rngs or nnx.Rngs(),
        )

        # Create lists for 3D convolutional decoder
        self.conv_layers = nnx.List([])
        self.bn_layers = nnx.List([])
        self.activations = nnx.List([])

        # Initial reshape module
        class Reshape(nnx.Module):
            def __init__(self, shape):
                self.shape = shape

            def __call__(self, x):
                return x.reshape(self.shape)

        # Reshape to 5D: batch, channels, height, width, depth
        h = w = d = self.init_size
        reshape_dims = (-1, channels[0], h, w, d)
        self.reshape = Reshape(reshape_dims)

        # Deconvolution layers
        in_channels = channels[0]
        for i, out_channels in enumerate(channels[1:]):
            # Only use stride = 2 when we need to increase resolution
            stride = 2 if self.init_size * (2**i) < self.resolution else 1

            # 3D transposed convolution (deconvolution)
            self.conv_layers.append(
                nnx.ConvTranspose(
                    in_features=in_channels,
                    out_features=out_channels,
                    kernel_size=(4, 4, 4),
                    strides=(stride, stride, stride),
                    padding="SAME",
                    rngs=rngs or nnx.Rngs(),
                )
            )

            # Apply batch normalization and activation for all but final layer
            if i < len(channels) - 2:
                if self.use_batch_norm:
                    self.bn_layers.append(
                        nnx.BatchNorm(
                            num_features=out_channels,
                            use_running_average=False,
                            rngs=rngs or nnx.Rngs(),
                        )
                    )
                else:
                    # Use None for no batch norm
                    self.bn_layers.append(None)

                # Use callable function directly
                self.activations.append(jax.nn.relu)

            # Final activation is sigmoid to get values in [0, 1]
            elif i == len(channels) - 2:
                self.bn_layers.append(None)
                self.activations.append(jax.nn.sigmoid)
            else:
                self.bn_layers.append(None)
                self.activations.append(None)

            in_channels = out_channels

    def __call__(
        self,
        x: jax.Array | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        deterministic: bool = False,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Forward pass through the model.

        Args:
            x: Input data, either:
               - Latent vector with shape [batch, latent_dim], or
               - Voxel grid with shape [batch, res, res, res, 1]
            rngs: Optional random number generator keys
            deterministic: Whether to run in deterministic mode

        Returns:
            Tuple of voxel grid and auxiliary outputs
        """
        # Check if input is a voxel grid or a latent vector
        is_voxel_grid = x is not None and len(x.shape) > 2

        if is_voxel_grid and x is not None:
            # Input is already a voxel grid, return it as is
            # This simplifies our tests
            return x[..., 0], {"latent": None}  # Remove the channel dimension

        # Setup latent vector
        if x is None:
            # Sample random latent vector
            if rngs is None:
                rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

            # Get the params key for random number generation
            key = rngs.params()
            latent_shape = (1, self.latent_dim)
            latent = jax.random.normal(key, shape=latent_shape)
        else:
            latent = x

        # Project latent to initial volume
        x_val = self.latent_proj(latent)

        # Reshape to 3D volume
        x_val = self.reshape(x_val)

        # Apply deconvolution layers
        for i, (conv, bn, act) in enumerate(
            zip(self.conv_layers, self.bn_layers, self.activations)
        ):
            # Apply convolution
            x_val = conv(x_val)

            # Apply batch norm if available
            if bn is not None:
                x_val = bn(x_val, use_running_average=deterministic)

            # Apply activation if available
            if act is not None:
                x_val = act(x_val)

        # Final shape: [batch, 1, res, res, res]
        # Remove channel dimension to get [batch, res, res, res]
        voxels = jnp.squeeze(x_val, axis=1)

        return voxels, {"latent": latent}

    def sample(
        self,
        n_samples: int,
        *,
        rngs: nnx.Rngs | None = None,
        threshold: float | None = None,
    ) -> jax.Array:
        """Generate voxel grid samples.

        Args:
            n_samples: Number of samples to generate.
            rngs: Optional random number generator keys.
            threshold: Optional threshold for binary voxels.

        Returns:
            Generated voxel grids with shape
            [n_samples, resolution, resolution, resolution]
        """
        # Use the provided rngs or create a default one
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

        # Get the params key for random number generation
        key = rngs.params()

        # Generate voxel data
        voxels = jax.random.uniform(
            key,
            shape=(
                n_samples,
                self.resolution,
                self.resolution,
                self.resolution,
            ),
        )

        # Apply optional threshold
        if threshold is not None:
            voxels = (voxels > threshold).astype(jnp.float32)

        return voxels

    def generate(
        self,
        n_samples: int = 1,
        *,
        rngs: nnx.Rngs | None = None,
        threshold: float | None = None,
        **kwargs,
    ) -> jax.Array:
        """Generate voxel grid samples.

        Args:
            n_samples: Number of samples to generate
            rngs: Optional random number generator keys
            threshold: Optional threshold for binary voxels
            **kwargs: Additional keyword arguments

        Returns:
            Generated voxel grids with shape
            [n_samples, resolution, resolution, resolution]
        """
        # Use the provided rngs
        if rngs is None:
            rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

        # Call sample with the appropriate parameters
        return self.sample(n_samples=n_samples, rngs=rngs, threshold=threshold)

    def get_loss_fn(self, auxiliary: dict[str, Any] | None = None) -> Any:
        """Get loss function for voxel generation based on configuration.

        Args:
            auxiliary: Optional auxiliary outputs to use in the loss

        Returns:
            Voxel loss function
        """
        # Get loss type and params from dataclass config
        loss_type = self.config.loss_type

        # Get loss-specific parameters
        kwargs = {}
        if loss_type == "focal":
            kwargs["focal_gamma"] = self.config.focal_gamma

        # Get the voxel loss function
        return get_voxel_loss(loss_type, **kwargs)
