"""Guidance techniques for diffusion models.

This module implements various guidance methods including classifier-free guidance,
classifier guidance, and other conditional generation techniques.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.models.diffusion.base import DiffusionModel


class ClassifierFreeGuidance:
    """Classifier-free guidance for conditional diffusion models.

    This implements the classifier-free guidance technique that allows
    trading off between sample diversity and adherence to conditioning.
    """

    def __init__(self, guidance_scale: float = 7.5, unconditional_conditioning: Any | None = None):
        """Initialize classifier-free guidance.

        Args:
            guidance_scale: Guidance strength (higher = more conditioning)
            unconditional_conditioning: Unconditional conditioning token/embedding
        """
        self.guidance_scale = guidance_scale
        self.unconditional_conditioning = unconditional_conditioning

    def __call__(
        self,
        model: DiffusionModel,
        x: jax.Array,
        t: jax.Array,
        conditioning: Any,
        **kwargs,
    ) -> jax.Array:
        """Apply classifier-free guidance.

        Args:
            model: Diffusion model
            x: Noisy input
            t: Timesteps
            conditioning: Conditional information
            **kwargs: Additional model arguments

        Returns:
            Guided noise prediction

        Note:
            NNX models store RNGs at init time, no need to pass rngs.
        """
        # Get conditional prediction
        cond_output = model(x, t, conditioning=conditioning, **kwargs)
        if isinstance(cond_output, dict):
            cond_noise = cond_output.get("predicted_noise", next(iter(cond_output.values())))
        else:
            cond_noise = cond_output

        # Get unconditional prediction
        uncond_output = model(x, t, conditioning=self.unconditional_conditioning, **kwargs)
        if isinstance(uncond_output, dict):
            uncond_noise = uncond_output.get("predicted_noise", next(iter(uncond_output.values())))
        else:
            uncond_noise = uncond_output

        # Apply classifier-free guidance
        guided_noise = uncond_noise + self.guidance_scale * (cond_noise - uncond_noise)

        return guided_noise


class ClassifierGuidance:
    """Classifier guidance for diffusion models.

    Uses a pre-trained classifier to guide the generation process
    towards desired classes.
    """

    def __init__(
        self, classifier: nnx.Module, guidance_scale: float = 1.0, class_label: int | None = None
    ):
        """Initialize classifier guidance.

        Args:
            classifier: Pre-trained classifier model
            guidance_scale: Guidance strength
            class_label: Target class label for guidance
        """
        self.classifier = classifier
        self.guidance_scale = guidance_scale
        self.class_label = class_label

    def __call__(
        self,
        model: DiffusionModel,
        x: jax.Array,
        t: jax.Array,
        *,
        class_label: int | None = None,
        **kwargs,
    ) -> jax.Array:
        """Apply classifier guidance.

        Args:
            model: Diffusion model
            x: Noisy input
            t: Timesteps
            class_label: Target class (overrides self.class_label)
            **kwargs: Additional model arguments

        Returns:
            Guided noise prediction

        Note:
            NNX models store RNGs at init time, no need to pass rngs.
        """
        target_class = class_label if class_label is not None else self.class_label

        if target_class is None:
            raise ValueError("No target class specified for classifier guidance")

        # Get model prediction
        model_output = model(x, t, **kwargs)
        if isinstance(model_output, dict):
            noise_pred = model_output.get("predicted_noise", next(iter(model_output.values())))
        else:
            noise_pred = model_output

        # Compute classifier gradient
        def classifier_fn(x_input: jax.Array) -> jax.Array:
            # Scale input to classifier's expected range
            x_scaled = self._scale_for_classifier(x_input)
            logits = self.classifier(x_scaled)
            return logits[..., target_class]

        # Get gradient of classifier with respect to input
        grad_fn = nnx.grad(lambda x_input: jnp.mean(classifier_fn(x_input)))
        classifier_grad = grad_fn(x)

        # Apply guidance
        guided_noise = noise_pred - self.guidance_scale * classifier_grad

        return guided_noise

    def _scale_for_classifier(self, x: jax.Array) -> jax.Array:
        """Scale input for classifier (typically to [0, 1] range).

        Args:
            x: Input tensor (typically in [-1, 1] range)

        Returns:
            Scaled tensor for classifier
        """
        return (x + 1.0) / 2.0


class GuidedDiffusionModel(DiffusionModel):
    """Diffusion model with built-in guidance support.

    This extends the base diffusion model to support various guidance
    techniques during generation.

    Uses the polymorphic backbone system - backbone type is determined
    by config.backbone.backbone_type discriminator.
    """

    def __init__(
        self,
        config: Any,
        *,
        rngs: nnx.Rngs,
        guidance_method: str | None = None,
        guidance_scale: float = 7.5,
        classifier: nnx.Module | None = None,
    ):
        """Initialize guided diffusion model.

        Args:
            config: Model configuration with nested BackboneConfig.
                    The backbone is created based on backbone_type.
            rngs: Random number generators
            guidance_method: Type of guidance ("classifier_free", "classifier", None)
            guidance_scale: Guidance strength
            classifier: Classifier for classifier guidance
        """
        super().__init__(config, rngs=rngs)

        self.guidance_method = guidance_method
        self.guidance_scale = guidance_scale

        # Initialize guidance
        if guidance_method == "classifier_free":
            self.guidance = ClassifierFreeGuidance(
                guidance_scale=guidance_scale,
                unconditional_conditioning=getattr(config, "unconditional_token", None),
            )
        elif guidance_method == "classifier":
            if classifier is None:
                raise ValueError("Classifier required for classifier guidance")
            self.guidance = ClassifierGuidance(classifier=classifier, guidance_scale=guidance_scale)
        else:
            self.guidance = None

    def guided_sample_step(
        self,
        x: jax.Array,
        t: jax.Array,
        conditioning: Any | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Single sampling step with guidance.

        Args:
            x: Current sample
            t: Timesteps
            conditioning: Conditioning information
            **kwargs: Additional arguments

        Returns:
            Guided noise prediction

        Note:
            NNX models store RNGs at init time, no need to pass rngs.
        """
        if self.guidance is None:
            # No guidance - use standard model
            output = self(x, t, **kwargs)
            if isinstance(output, dict):
                return output.get("predicted_noise", next(iter(output.values())))
            return output

        elif isinstance(self.guidance, ClassifierFreeGuidance):
            return self.guidance(self, x, t, conditioning, **kwargs)

        elif isinstance(self.guidance, ClassifierGuidance):
            return self.guidance(self, x, t, **kwargs)

        else:
            raise ValueError(f"Unknown guidance type: {type(self.guidance)}")

    def generate(
        self,
        n_samples: int = 1,
        *,
        rngs: nnx.Rngs | None = None,
        conditioning: Any | None = None,
        shape: tuple[int, ...] | None = None,
        clip_denoised: bool = True,
        **kwargs: Any,
    ) -> jax.Array:
        """Generate samples with guidance.

        Args:
            n_samples: Number of samples to generate
            rngs: Random number generators
            conditioning: Conditioning information for guided generation
            shape: Sample shape
            clip_denoised: Whether to clip denoised samples
            **kwargs: Additional keyword arguments

        Returns:
            Generated samples
        """
        # Determine shape of samples
        if shape is None:
            shape = getattr(self.config, "data_shape", (32, 32, 3))

        # Initialize noise
        sample_key = (rngs or self.rngs).sample()

        # Generate initial noise
        img = jax.random.normal(sample_key, (n_samples, *shape))

        # Get number of timesteps
        num_timesteps = getattr(self.config, "num_timesteps", 1000)

        # Iterate through timesteps in reverse
        for t in range(num_timesteps - 1, -1, -1):
            # Broadcast timestep to batch dimension
            timesteps = jnp.full((n_samples,), t, dtype=jnp.int32)

            # Get guided model output
            guided_noise = self.guided_sample_step(
                img, timesteps, conditioning, rngs=rngs, **kwargs
            )

            # Sample from p(x_{t-1} | x_t) using guided prediction
            img = self.p_sample(
                guided_noise, img, timesteps, rngs=rngs, clip_denoised=clip_denoised
            )

        return img


class ConditionalDiffusionMixin:
    """Mixin for adding conditional generation capabilities.

    This can be mixed into any diffusion model to add conditioning support.

    Follows Principle #4: Methods take configs, NOT individual parameters.
    The config must have a conditioning_dim field (e.g., ConditionalDiffusionConfig).
    """

    def __init__(self, config, *, rngs: nnx.Rngs):
        """Initialize conditional diffusion capabilities.

        Args:
            config: Configuration with conditioning_dim field (e.g., ConditionalDiffusionConfig)
            rngs: Random number generators
        """
        super().__init__(config, rngs=rngs)
        self.conditioning_dim = config.conditioning_dim

        # Initialize conditioning projection if needed
        if self.conditioning_dim is not None and hasattr(self, "backbone"):
            # Add conditioning projection to backbone if it doesn't have one
            if not hasattr(self.backbone, "conditioning_proj"):
                # This would need to be adapted based on the specific backbone
                pass

    def __call__(
        self,
        x: jax.Array,
        timesteps: jax.Array,
        *,
        conditioning: jax.Array | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Forward pass with conditioning.

        Note: Use model.train() for training mode and model.eval() for evaluation mode.
              NNX models store RNGs at init time, no need to pass rngs.

        Args:
            x: Input data
            timesteps: Diffusion timesteps
            conditioning: Conditioning information
            **kwargs: Additional arguments

        Returns:
            Model outputs
        """
        return super().__call__(x, timesteps, conditioning=conditioning, **kwargs)


def apply_guidance(
    noise_pred_cond: jax.Array, noise_pred_uncond: jax.Array, guidance_scale: float
) -> jax.Array:
    """Apply classifier-free guidance to noise predictions.

    Args:
        noise_pred_cond: Conditional noise prediction
        noise_pred_uncond: Unconditional noise prediction
        guidance_scale: Guidance strength

    Returns:
        Guided noise prediction
    """
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


def linear_guidance_schedule(
    step: int, total_steps: int, start_scale: float = 1.0, end_scale: float = 7.5
) -> float:
    """Linear guidance scale schedule.

    Args:
        step: Current step
        total_steps: Total number of steps
        start_scale: Starting guidance scale
        end_scale: Ending guidance scale

    Returns:
        Guidance scale for current step
    """
    alpha = step / total_steps
    return start_scale + alpha * (end_scale - start_scale)


def cosine_guidance_schedule(
    step: int, total_steps: int, start_scale: float = 1.0, end_scale: float = 7.5
) -> float:
    """Cosine guidance scale schedule.

    Args:
        step: Current step
        total_steps: Total number of steps
        start_scale: Starting guidance scale
        end_scale: Ending guidance scale

    Returns:
        Guidance scale for current step
    """
    alpha = 0.5 * (1 + jnp.cos(jnp.pi * step / total_steps))
    return end_scale + alpha * (start_scale - end_scale)
