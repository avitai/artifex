"""Adapters for image models.

This module provides adapters that adapt generative models to work with
image data, following the unified configuration and factory patterns.
"""

from typing import Any

from flax import nnx

from artifex.generative_models.modalities.base import ModelAdapter


class ImageModalityAdapter(ModelAdapter):
    """Adapter for image models.

    This adapter provides a pass-through implementation for image models
    since most generative models already work with image data natively.
    Image-specific adaptations can be added here as needed.
    """

    def __init__(self):
        """Initialize the image modality adapter."""
        self.name = "image_adapter"
        self.modality = "image"

    def adapt(self, model: Any, config: Any) -> Any:
        """Adapt a model for image modality.

        Args:
            model: The model instance to adapt
            config: Model configuration (dataclass config)

        Returns:
            The adapted model (currently returns model unchanged)
        """
        # For now, image models don't need special adaptation
        # as most generative models already work with images
        # Future enhancements could include:
        # - Adding image-specific preprocessing layers
        # - Adding image augmentation capabilities
        # - Adding resolution adaptation layers
        return model

    def create(self, config: Any, *, rngs: nnx.Rngs, **kwargs) -> Any:
        """Create an image model from configuration.

        This is not typically used as the factory creates the base model,
        and this adapter just adapts it.

        Args:
            config: Model configuration (dataclass config)
            rngs: Random number generators
            **kwargs: Additional arguments

        Returns:
            Created model

        Raises:
            NotImplementedError: This method is not used in the current flow
        """
        raise NotImplementedError(
            "ImageModalityAdapter.create() is not implemented. "
            "Models should be created by the factory and then adapted."
        )
