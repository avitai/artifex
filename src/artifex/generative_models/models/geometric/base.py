"""Base class for geometric generative models."""

import dataclasses
from typing import Any

import jax
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel
from artifex.generative_models.core.configuration.geometric_config import (
    GeometricConfig,
)


class GeometricModel(GenerativeModel):
    """Base class for geometric generative models.

    Geometric models generate 3D structures like point clouds, meshes, or
    voxels.
    """

    def __init__(
        self,
        config: GeometricConfig,
        *,
        extensions: dict[str, nnx.Module] | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize the geometric model.

        Args:
            config: GeometricConfig dataclass with model parameters.
            extensions: Optional dictionary of extension modules.
            rngs: Random number generator keys.

        Raises:
            TypeError: If config is not a dataclass config
        """
        # Validate config is a dataclass
        if not dataclasses.is_dataclass(config):
            raise TypeError(
                f"config must be a dataclass config (GeometricConfig or subclass), "
                f"got {type(config).__name__}"
            )
        self.config = config

        super().__init__(rngs=rngs)
        self.extensions = extensions or nnx.Dict({})

        # Initialize extension_modules as nnx.Dict for Flax NNX 0.12.0+ compatibility
        extension_modules_dict = {}

        # Initialize extensions
        for name, extension in self.extensions.items():
            extension_modules_dict[name] = extension

        self.extension_modules = nnx.Dict(extension_modules_dict)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        """Forward pass through the model.

        Args:
            x: Input array.
            deterministic: Whether to run in deterministic mode.

        Returns:
            dictionary of model outputs.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def sample(self, n_samples: int, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        """Generate samples from the model.

        Args:
            n_samples: Number of samples to generate.
            rngs: Optional random number generator keys.

        Returns:
            Generated samples.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def get_loss_fn(self, auxiliary: dict[str, Any] | None = None) -> Any:
        """Get the loss function for the model.

        Args:
            auxiliary: Optional auxiliary outputs to use in the loss.

        Returns:
            Loss function.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def apply_extensions(self, inputs: Any, outputs: Any, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Apply all registered extensions to the outputs.

        Args:
            inputs: Original inputs to the model.
            outputs: Outputs from the model.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of processed outputs and extension outputs dictionary.
        """
        processed_outputs = outputs
        extension_outputs = {}

        for name, extension in self.extension_modules.items():
            ext_result = extension(inputs, processed_outputs, **kwargs)
            extension_outputs[name] = ext_result

            # Update outputs if extension modifies them
            if hasattr(extension, "project"):
                processed_outputs = extension.project(processed_outputs)

        return processed_outputs, extension_outputs

    def get_extension_losses(
        self, batch: dict[str, Any], model_outputs: Any, **kwargs
    ) -> dict[str, jax.Array]:
        """Get losses from all registered extensions.

        Args:
            batch: Batch of data.
            model_outputs: Outputs from the model.
            **kwargs: Additional keyword arguments.

        Returns:
            dictionary of extension losses.
        """
        extension_losses = {}

        for name, extension in self.extension_modules.items():
            if hasattr(extension, "loss_fn"):
                extension_losses[name] = extension.loss_fn(batch, model_outputs, **kwargs)

        return extension_losses
