"""Base classes for model extensions.

This module provides the core extension interfaces used across different
domains to enable extension of model functionality without modifying core
implementations.

Extension Hierarchy:
- Extension (base)
  - ModelExtension (processes outputs, provides loss)
    - ConstraintExtension (validates/projects outputs)
  - AugmentationExtension (data augmentation)
  - SamplingExtension (modifies generation)
  - LossExtension (modular loss terms)
  - EvaluationExtension (metrics computation)
  - CallbackExtension (training lifecycle hooks)
  - ModalityExtension (modality-specific preprocessing)
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import (
    AugmentationExtensionConfig,
    CallbackExtensionConfig,
    ConstraintExtensionConfig,
    EvaluationExtensionConfig,
    ExtensionConfig,
    LossExtensionConfig,
    ModalityExtensionConfig,
    SamplingExtensionConfig,
)


# =============================================================================
# Base Extension Class
# =============================================================================


class Extension(nnx.Module):
    """Base class for all extensions.

    Extensions provide modular functionality that can be attached to models
    without modifying core implementations. All extensions inherit from
    nnx.Module for JAX/NNX compatibility.

    Attributes:
        config: Extension configuration (frozen dataclass).
    """

    def __init__(self, config: ExtensionConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize the extension.

        Args:
            config: Extension configuration (must be ExtensionConfig dataclass).
            rngs: Random number generator keys.

        Raises:
            TypeError: If config is not an ExtensionConfig dataclass.
        """
        super().__init__()

        # Validate config is a frozen dataclass ExtensionConfig
        if not isinstance(config, ExtensionConfig):
            raise TypeError(
                f"config must be ExtensionConfig dataclass, got {type(config).__name__}"
            )

        self.config = config
        self._rngs = rngs

    def is_enabled(self) -> bool:
        """Check if the extension is enabled.

        Returns:
            True if extension is enabled, False otherwise.
        """
        return self.config.enabled

    @property
    def enabled(self) -> bool:
        """Get the enabled state of the extension.

        Returns:
            True if extension is enabled, False otherwise.
        """
        return self.config.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set the enabled state of the extension.

        Note: This modifies the config, which is a frozen dataclass.
        To enable/disable, we replace the config with an updated version.

        Args:
            value: New enabled state.
        """
        import dataclasses

        # Replace config with updated version (configs are frozen dataclasses)
        self.config = dataclasses.replace(self.config, enabled=value)

    @property
    def weight(self) -> float:
        """Get the extension weight.

        Returns:
            Weight for the extension's contribution.
        """
        return self.config.weight


# =============================================================================
# Model Extension
# =============================================================================


class ModelExtension(Extension):
    """Extension that processes model outputs and provides auxiliary loss.

    ModelExtensions can:
    - Process model inputs/outputs
    - Contribute additional loss terms to training
    - Be enabled/disabled dynamically
    """

    def __call__(self, inputs: Any, model_outputs: Any, **kwargs: Any) -> dict[str, Any]:
        """Process model inputs/outputs.

        Args:
            inputs: Original inputs to the model.
            model_outputs: Outputs from the model.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary of extension outputs.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def loss_fn(self, batch: dict[str, Any], model_outputs: Any, **kwargs: Any) -> jax.Array:
        """Calculate extension-specific loss.

        This method should be pure JAX operations for JIT compatibility.

        Args:
            batch: Batch of data.
            model_outputs: Outputs from the model.
            **kwargs: Additional keyword arguments.

        Returns:
            Extension-specific loss (scalar JAX array).
        """
        if not self.is_enabled():
            return jnp.array(0.0)
        return jnp.array(0.0)


# =============================================================================
# Constraint Extension
# =============================================================================


class ConstraintExtension(ModelExtension):
    """Extension that enforces constraints on model outputs.

    ConstraintExtensions add physical or domain-specific constraints
    through loss terms and optional output projection.

    Requires ConstraintExtensionConfig for tolerance and projection settings.
    """

    def __init__(self, config: ConstraintExtensionConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize the constraint extension.

        Args:
            config: Constraint extension configuration.
            rngs: Random number generator keys.
        """
        # Accept both ExtensionConfig and ConstraintExtensionConfig
        # for flexibility in subclasses
        super().__init__(config, rngs=rngs)

    def validate(self, outputs: Any) -> dict[str, jax.Array]:
        """Validate outputs against constraints.

        Args:
            outputs: Model outputs to validate.

        Returns:
            Dictionary of validation metrics (e.g., violation counts).

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def project(self, outputs: Any) -> Any:
        """Project outputs to satisfy constraints.

        Args:
            outputs: Model outputs to project.

        Returns:
            Projected outputs that satisfy constraints.
        """
        if not self.is_enabled():
            return outputs
        # Default: no projection
        return outputs


# =============================================================================
# Augmentation Extension
# =============================================================================


class AugmentationExtension(Extension):
    """Extension for data augmentation.

    AugmentationExtensions transform data during training to improve
    model robustness and generalization.
    """

    def __init__(self, config: AugmentationExtensionConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize the augmentation extension.

        Args:
            config: Augmentation extension configuration.
            rngs: Random number generator keys.
        """
        super().__init__(config, rngs=rngs)

    def __call__(
        self, data: jax.Array, *, deterministic: bool = False, key: jax.Array | None = None
    ) -> jax.Array:
        """Apply augmentation to data.

        Args:
            data: Input data to augment.
            deterministic: If True, apply deterministic augmentation.
            key: Optional random key for stochastic augmentation.

        Returns:
            Augmented data.
        """
        if deterministic or not self.is_enabled():
            return data
        return self.augment(data, key=key, deterministic=deterministic)

    def augment(
        self, data: jax.Array, *, key: jax.Array | None = None, deterministic: bool = False
    ) -> jax.Array:
        """Apply augmentation transformation.

        Args:
            data: Input data to augment.
            key: Random key for stochastic augmentation.
            deterministic: If True, apply deterministic augmentation.

        Returns:
            Augmented data.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError("Must be implemented by subclass")


# =============================================================================
# Sampling Extension
# =============================================================================


class SamplingExtension(Extension):
    """Extension that modifies the sampling/generation process.

    SamplingExtensions can:
    - Modify score/noise predictions during diffusion
    - Filter generated samples
    - Post-process samples
    """

    def __init__(self, config: SamplingExtensionConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize the sampling extension.

        Args:
            config: Sampling extension configuration.
            rngs: Random number generator keys.
        """
        super().__init__(config, rngs=rngs)

    def modify_score(
        self, score: jax.Array, timestep: jax.Array, context: dict[str, Any]
    ) -> jax.Array:
        """Modify the score/noise prediction during sampling.

        Args:
            score: Score or noise prediction from model.
            timestep: Current timestep in sampling process.
            context: Additional context (e.g., conditioning, unconditional score).

        Returns:
            Modified score.
        """
        return score

    def filter_samples(
        self, samples: jax.Array, context: dict[str, Any]
    ) -> tuple[jax.Array, jax.Array]:
        """Filter generated samples based on criteria.

        Args:
            samples: Generated samples to filter.
            context: Additional context for filtering.

        Returns:
            Tuple of (filtered_samples, validity_mask).
        """
        return samples, jnp.ones(samples.shape[0])

    def post_process_sample(self, sample: jax.Array, context: dict[str, Any]) -> jax.Array:
        """Post-process a generated sample.

        Args:
            sample: Generated sample.
            context: Additional context.

        Returns:
            Post-processed sample.
        """
        return sample


# =============================================================================
# Loss Extension
# =============================================================================


class LossExtension(Extension):
    """Extension providing modular loss components.

    LossExtensions enable composable loss functions with:
    - Weight scheduling during training
    - Warmup periods
    - Dynamic loss computation
    """

    def __init__(self, config: LossExtensionConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize the loss extension.

        Args:
            config: Loss extension configuration.
            rngs: Random number generator keys.
        """
        super().__init__(config, rngs=rngs)

    def compute_loss(
        self, predictions: Any, targets: Any, context: dict[str, Any]
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Compute the extension loss.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            context: Additional context.

        Returns:
            Tuple of (loss_value, metrics_dict).

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def get_weight_at_step(self, step: int) -> float:
        """Get the loss weight at a given training step.

        Implements weight scheduling based on config.

        Args:
            step: Current training step.

        Returns:
            Weight value at this step.
        """
        config = self.config
        if not isinstance(config, LossExtensionConfig):
            return self.weight

        schedule = config.weight_schedule
        warmup = config.warmup_steps

        if schedule == "constant" or warmup == 0:
            return self.weight

        if step >= warmup:
            return self.weight

        progress = step / max(1, warmup)

        if schedule == "linear":
            return self.weight * progress
        elif schedule == "cosine":
            import math

            return self.weight * (1 - math.cos(math.pi * progress)) / 2
        elif schedule == "exponential":
            return self.weight * (1 - (1 - progress) ** 2)

        return self.weight


# =============================================================================
# Evaluation Extension
# =============================================================================


class EvaluationExtension(Extension):
    """Extension for domain-specific evaluation metrics.

    EvaluationExtensions compute custom metrics during training/evaluation.
    """

    def __init__(self, config: EvaluationExtensionConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize the evaluation extension.

        Args:
            config: Evaluation extension configuration.
            rngs: Random number generator keys.
        """
        super().__init__(config, rngs=rngs)

    def compute_metrics(
        self, generated: jax.Array, reference: jax.Array | None = None
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        Args:
            generated: Generated samples to evaluate.
            reference: Optional reference samples for comparison.

        Returns:
            Dictionary of metric names to values.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError("Must be implemented by subclass")


# =============================================================================
# Callback Extension
# =============================================================================


class CallbackExtension(Extension):
    """Extension that hooks into training lifecycle events.

    CallbackExtensions enable custom behavior at various training stages:
    - train_begin/end
    - epoch_begin/end
    - batch_begin/end
    """

    def __init__(self, config: CallbackExtensionConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize the callback extension.

        Args:
            config: Callback extension configuration.
            rngs: Random number generator keys.
        """
        super().__init__(config, rngs=rngs)

    def on_train_begin(self, trainer: Any) -> None:
        """Called at the start of training.

        Args:
            trainer: The trainer instance.
        """
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training.

        Args:
            trainer: The trainer instance.
        """
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Called at the start of each epoch.

        Args:
            trainer: The trainer instance.
            epoch: Current epoch number.
        """
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        """Called at the end of each epoch.

        Args:
            trainer: The trainer instance.
            epoch: Current epoch number.
            logs: Metrics from the epoch.
        """
        pass

    def on_batch_begin(self, trainer: Any, batch_idx: int) -> None:
        """Called at the start of each batch.

        Args:
            trainer: The trainer instance.
            batch_idx: Current batch index.
        """
        pass

    def on_batch_end(self, trainer: Any, batch_idx: int, logs: dict[str, Any]) -> None:
        """Called at the end of each batch.

        Args:
            trainer: The trainer instance.
            batch_idx: Current batch index.
            logs: Metrics from the batch.
        """
        pass


# =============================================================================
# Modality Extension
# =============================================================================


class ModalityExtension(Extension):
    """Extension for modality-specific preprocessing and encoding.

    ModalityExtensions handle the conversion between raw data and
    model-compatible representations.
    """

    def __init__(self, config: ModalityExtensionConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize the modality extension.

        Args:
            config: Modality extension configuration.
            rngs: Random number generator keys.
        """
        super().__init__(config, rngs=rngs)

    def preprocess(self, raw_data: Any) -> dict[str, jax.Array]:
        """Convert raw data to model inputs.

        Args:
            raw_data: Raw input data.

        Returns:
            Dictionary of processed tensors.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def postprocess(self, model_output: Any) -> Any:
        """Convert model output back to domain representation.

        Args:
            model_output: Output from the model.

        Returns:
            Processed output in domain format.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def get_input_spec(self) -> dict[str, Any]:
        """Get the input specification for this modality.

        Returns:
            Dictionary describing expected input format.
        """
        return {}


# =============================================================================
# Extension Dictionary with Fixed __contains__
# =============================================================================


class ExtensionDict(nnx.Dict):
    """A dictionary for extensions that properly implements __contains__.

    This class extends nnx.Dict to fix a bug where the `in` operator raises
    AttributeError for missing keys instead of returning False. This is because
    nnx.Dict inherits from MutableMapping which uses __getitem__ in its
    __contains__ implementation, but nnx.Dict's __getitem__ raises AttributeError
    for missing keys instead of KeyError.

    Usage:
        extensions = ExtensionDict({"key1": ext1, "key2": ext2})
        if "missing_key" not in extensions:  # Works correctly, returns True
            print("Key not found")

    Note:
        This class maintains full compatibility with nnx.Dict and can be used
        as a drop-in replacement wherever nnx.Dict is used for extensions.
    """

    def __contains__(self, key: object) -> bool:
        """Check if key exists in the dictionary.

        This method properly handles the case where the key doesn't exist
        by checking against vars(self) directly, using the same filtering
        logic as nnx.Dict.__iter__ (which filters out '_pytree__' prefixed keys).

        Args:
            key: The key to check for.

        Returns:
            True if the key exists, False otherwise.
        """
        if not isinstance(key, str):
            return False
        return key in vars(self) and not key.startswith("_pytree__")
