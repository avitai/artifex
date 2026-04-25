"""Trainer for NNX-based generative models."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast, TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx

from artifex.generative_models.core.configuration import (
    SchedulerConfig,
    TrainingConfig,
)
from artifex.generative_models.training.callbacks import CallbackList
from artifex.generative_models.training.loops.streaming import create_data_pipeline
from artifex.generative_models.training.optimizers import create_optimizer
from artifex.generative_models.training.schedulers import create_scheduler
from artifex.generative_models.utils.logging import Logger, MetricsLogger


if TYPE_CHECKING:
    from artifex.generative_models.extensions.base import (
        Extension,
    )


TrainerLossFn = Callable[
    [nnx.Module, dict[str, Any], jax.Array, jax.Array],
    tuple[jax.Array, dict[str, Any]],
]


class Trainer:
    """Low-level trainer for NNX-based generative models.

    This trainer is designed to work with Flax NNX modules, using nnx.value_and_grad
    for proper gradient computation and state management.

    Callers must provide an explicit objective with signature:
        loss_fn(model: nnx.Module, batch: dict, rng: jax.Array, step: jax.Array)
        -> tuple[float, dict]

    The model is passed directly (not extracted params), allowing the objective
    to call model methods that may use internal state like rngs.
    """

    def __init__(
        self,
        model: nnx.Module,
        training_config: TrainingConfig,
        optimizer: optax.GradientTransformation | None = None,
        train_data_loader: Callable | None = None,
        val_data_loader: Callable | None = None,
        workdir: str | None = None,
        rng: jax.Array | None = None,
        *,
        loss_fn: TrainerLossFn,
        metrics_logger: MetricsLogger | None = None,
        logger: Logger | None = None,
        checkpoint_dir: str | None = None,
        save_interval: int = 1000,
        log_callback: Callable | None = None,
        callbacks: CallbackList | None = None,
        extensions: dict[str, Extension] | None = None,
    ):
        """Initialize the trainer.

        Args:
            model: The NNX model to train (must be nnx.Module).
            training_config: Configuration for training (must be TrainingConfig).
            optimizer: The optax optimizer to use.
            train_data_loader: Function to load training data.
            val_data_loader: Function to load validation data.
            workdir: Working directory for outputs.
            rng: JAX random number generator key.
            loss_fn: Explicit objective function. Signature:
                     loss_fn(model, batch, rng, step) -> (loss, metrics_dict)
            metrics_logger: Logger for training metrics.
            logger: Artifex logger for general logging.
            checkpoint_dir: Directory to save checkpoints.
            save_interval: Interval to save checkpoints.
            log_callback: Callback function for logging.
            callbacks: CallbackList for training lifecycle hooks.
            extensions: Dictionary mapping extension names to Extension instances.
                       Extensions can provide auxiliary losses and callbacks.

        Raises:
            TypeError: If model is not an nnx.Module or training_config is invalid.
        """
        # Validate model is an NNX module
        if not isinstance(model, nnx.Module):
            raise TypeError(
                f"model must be an nnx.Module, got {type(model).__name__}. "
                "The Trainer only supports Flax NNX modules."
            )

        # Validate training config
        if training_config is None:
            raise TypeError("training_config is required")
        if not isinstance(training_config, TrainingConfig):
            raise TypeError(
                f"training_config must be a TrainingConfig, got {type(training_config).__name__}"
            )

        self.model = model
        self.training_config = training_config
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.workdir = workdir
        self.rng = rng if rng is not None else jax.random.PRNGKey(0)
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable")

        self.loss_fn = loss_fn
        self.metrics_logger = metrics_logger
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir or (workdir if workdir else "checkpoints")
        self.save_interval = save_interval
        self.log_callback = log_callback
        self.callbacks = callbacks
        self.extensions: dict[str, Extension] = extensions if extensions is not None else {}

        # Store training metrics for convenience
        self.train_metrics: list[dict[str, Any]] = []
        self.val_metrics: list[dict[str, Any]] = []

        # Set up steps per epoch (needed before optimizer creation for scheduler)
        if train_data_loader is not None:
            self.steps_per_epoch = getattr(training_config, "steps_per_epoch", 100)
        else:
            self.steps_per_epoch = 100

        # Create the optimizer if not provided
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer

        # Create checkpoint directory if it doesn't exist
        if self.checkpoint_dir is not None:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize optimizer state with model parameters
        # Cast: nnx.state returns GraphState which is pytree-compatible with optax Params
        self.opt_state = self.optimizer.init(cast(optax.Params, nnx.state(self.model, nnx.Param)))

        # Training state (step counter and rng)
        self.step = 0

    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer from training config.

        Delegates to the centralized optimizer factory for all optimizer types.
        The factory supports: adam, adamw, sgd, rmsprop, adagrad, lamb, radam, nadam.

        Returns:
            Optax gradient transformation (optimizer).
        """
        opt_config = self.training_config.optimizer
        base_lr = opt_config.learning_rate

        # Create learning rate schedule if configured
        if self.training_config.scheduler is not None:
            schedule = self._create_schedule(self.training_config.scheduler, base_lr)
        else:
            schedule = None

        # Delegate to optimizer factory
        return create_optimizer(opt_config, schedule=schedule)

    def _create_schedule(self, scheduler_config: SchedulerConfig, base_lr: float) -> Any:
        """Create learning rate schedule from configuration.

        Delegates to the centralized scheduler factory for all schedule types.
        The factory supports: constant, linear, cosine, exponential, polynomial,
        step, multistep, cyclic, and one_cycle schedules.

        Args:
            scheduler_config: Configuration for the learning rate schedule.
            base_lr: Base learning rate.

        Returns:
            An optax Schedule or callable that maps step -> learning rate.
        """
        # For schedules that need total_steps, compute it if not provided
        if scheduler_config.total_steps is None:
            # Compute from training config
            computed_total_steps = self.training_config.num_epochs * self.steps_per_epoch

            # Create a modified config with computed total_steps
            # Only for scheduler types that need it
            if scheduler_config.scheduler_type in ("linear", "polynomial", "one_cycle"):
                # Create new config with computed total_steps
                scheduler_config = SchedulerConfig(
                    name=scheduler_config.name,
                    scheduler_type=scheduler_config.scheduler_type,
                    warmup_steps=scheduler_config.warmup_steps,
                    min_lr_ratio=scheduler_config.min_lr_ratio,
                    total_steps=computed_total_steps,
                    decay_steps=scheduler_config.decay_steps,
                    decay_rate=scheduler_config.decay_rate,
                    step_size=scheduler_config.step_size,
                    gamma=scheduler_config.gamma,
                    milestones=scheduler_config.milestones,
                    cycle_length=scheduler_config.cycle_length,
                )

        # For cosine scheduler, use cycle_length or compute from training config
        if scheduler_config.scheduler_type == "cosine" and scheduler_config.cycle_length is None:
            computed_cycle_length = self.training_config.num_epochs * self.steps_per_epoch
            scheduler_config = SchedulerConfig(
                name=scheduler_config.name,
                scheduler_type=scheduler_config.scheduler_type,
                warmup_steps=scheduler_config.warmup_steps,
                min_lr_ratio=scheduler_config.min_lr_ratio,
                total_steps=scheduler_config.total_steps,
                decay_steps=scheduler_config.decay_steps,
                decay_rate=scheduler_config.decay_rate,
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma,
                milestones=scheduler_config.milestones,
                cycle_length=computed_cycle_length,
            )

        return create_scheduler(scheduler_config, base_lr)

    @staticmethod
    def _average_metrics(metric_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Average a list of metric dictionaries, excluding the step counter."""
        if not metric_history:
            return {}

        averaged: dict[str, Any] = {}
        for key in metric_history[0]:
            if key == "step":
                continue
            values = [metrics[key] for metrics in metric_history if key in metrics]
            averaged[key] = sum(values) / len(values)

        return averaged

    @staticmethod
    def _prefix_validation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        """Prefix validation metrics so callbacks can monitor `val_*` keys."""
        prefixed: dict[str, Any] = {}
        for key, value in metrics.items():
            if key.startswith("val_"):
                prefixed[key] = value
            else:
                prefixed[f"val_{key}"] = value
        return prefixed

    def _callbacks_should_stop(self) -> bool:
        """Return whether any registered callback has requested an early stop."""
        if self.callbacks is None:
            return False
        return any(getattr(callback, "should_stop", False) is True for callback in self.callbacks)

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Execute a single training step using NNX transforms.

        This method uses nnx.value_and_grad to properly handle NNX module state
        during gradient computation. Extension losses are aggregated with the
        base model loss.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary of training metrics including loss and extension losses
        """
        if self.callbacks is not None:
            self.callbacks.on_batch_begin(self, self.step)

        for _ext_name, ext in self.extensions.items():
            on_batch_begin = getattr(ext, "on_batch_begin", None)
            if on_batch_begin is not None:
                on_batch_begin(self, self.step)

        # Split RNG for this step
        self.rng, step_rng = jax.random.split(self.rng)

        # Get model outputs for extensions (computed once, reused)
        # Define loss function that includes extension losses
        def loss_fn(model: nnx.Module) -> tuple[jax.Array, dict[str, Any]]:
            # Compute base loss
            base_loss, base_metrics = self.loss_fn(model, batch, step_rng, jnp.array(self.step))

            # Compute extension losses
            ext_losses: dict[str, jax.Array] = {}
            total_ext_loss = jnp.array(0.0)

            # Get model outputs for extensions
            model_outputs = None
            if self.extensions:
                # Get outputs from model if it has a __call__ method
                if hasattr(model, "__call__") and "input" in batch:
                    model_outputs = model(batch["input"])
                else:
                    encode_fn = getattr(model, "encode", None)
                    if encode_fn is not None and "input" in batch:
                        model_outputs = encode_fn(batch["input"])

            for ext_name, ext in self.extensions.items():
                if ext.is_enabled():
                    ext_loss_fn = getattr(ext, "loss_fn", None)
                    if ext_loss_fn is not None:
                        ext_loss = ext_loss_fn(batch, model_outputs)
                        weighted_loss = ext.weight * ext_loss
                        ext_losses[f"{ext_name}_loss"] = weighted_loss
                        total_ext_loss = total_ext_loss + weighted_loss

            # Combine base loss with extension losses
            total_loss = base_loss + total_ext_loss

            # Merge metrics
            metrics = {**base_metrics, **ext_losses}

            return total_loss, metrics

        # Compute loss and gradients using NNX transform
        # nnx.value_and_grad handles the model state properly
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(self.model)

        # Extract parameter gradients and params as pytree-compatible types
        param_grads = cast(optax.Updates, nnx.state(grads, nnx.Param))
        params = cast(optax.Params, nnx.state(self.model, nnx.Param))

        # Compute parameter updates
        updates, self.opt_state = self.optimizer.update(param_grads, self.opt_state, params)

        # Apply updates to model parameters in-place
        updated_params = optax.apply_updates(params, updates)
        nnx.update(self.model, updated_params)

        # Update step counter
        self.step += 1

        # Add loss and step to metrics
        metrics = {**metrics, "loss": float(loss), "step": self.step}

        # Log metrics if callback is provided
        if self.log_callback is not None:
            self.log_callback(self.step, metrics, prefix="train")

        # Store metrics
        self.train_metrics.append(metrics)

        # Call callbacks (after step completes, minimal overhead path)
        if self.callbacks is not None:
            self.callbacks.on_batch_end(self, self.step, metrics)

        # Call extension callbacks
        for _ext_name, ext in self.extensions.items():
            on_batch_end = getattr(ext, "on_batch_end", None)
            if on_batch_end is not None:
                on_batch_end(self, self.step, metrics)

        return metrics

    def validate_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Execute a single validation step.

        Args:
            batch: Batch of validation data

        Returns:
            Dictionary of validation metrics
        """
        # Split RNG for validation
        self.rng, val_rng = jax.random.split(self.rng)

        # Compute validation loss (no gradients needed)
        loss, metrics = self.loss_fn(self.model, batch, val_rng, jnp.array(self.step))

        metrics = {**metrics, "loss": float(loss), "step": self.step}

        # Log metrics if callback is provided
        if self.log_callback is not None:
            self.log_callback(self.step, metrics, prefix="val")

        # Store metrics
        self.val_metrics.append(metrics)

        return metrics

    def train_epoch(self) -> dict[str, Any]:
        """Train for one epoch.

        Returns:
            Average metrics for the epoch
        """
        if self.train_data_loader is None:
            raise ValueError("train_data_loader is required for train_epoch")

        data_iter = self.train_data_loader(self.training_config.batch_size)
        epoch_metrics: list[dict[str, Any]] = []

        for _ in range(self.steps_per_epoch):
            batch = next(data_iter)
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)

            if self.step % self.training_config.save_frequency == 0:
                self.save_checkpoint()

        return self._average_metrics(epoch_metrics)

    def train(
        self,
        train_data: dict[str, Any],
        num_epochs: int,
        batch_size: int,
        val_data: dict[str, Any] | None = None,
        val_interval: int = 100,
    ) -> dict[str, Any]:
        """Train the model for multiple epochs.

        Args:
            train_data: Training data dictionary
            num_epochs: Number of epochs to train
            batch_size: Batch size
            val_data: Optional validation data
            val_interval: Steps between validation

        Returns:
            Final metrics after training
        """
        if self.logger:
            self.logger.log_text(
                "training", f"Training for {num_epochs} epochs with batch_size={batch_size}"
            )

        metrics: dict[str, Any] = {}
        if self.callbacks is not None:
            self.callbacks.on_train_begin(self)
        for _ext_name, ext in self.extensions.items():
            on_train_begin = getattr(ext, "on_train_begin", None)
            if on_train_begin is not None:
                on_train_begin(self)

        try:
            for epoch in range(num_epochs):
                if self.callbacks is not None:
                    self.callbacks.on_epoch_begin(self, epoch)

                epoch_metrics: list[dict[str, Any]] = []

                # Create a shuffled datarax pipeline for this epoch
                self.rng, shuffle_rng = jax.random.split(self.rng)
                epoch_seed = int(jax.random.randint(shuffle_rng, (), 0, 2**31 - 1))
                source = MemorySource(
                    MemorySourceConfig(shuffle=True),
                    train_data,
                    rngs=nnx.Rngs(epoch_seed),
                )
                pipeline = create_data_pipeline(source, batch_size=batch_size)

                for batch_view in pipeline:
                    batch = batch_view.get_data()

                    # Train step
                    metrics = self.train_step(batch)
                    epoch_metrics.append(metrics)

                    # Log metrics
                    if self.metrics_logger:
                        self.metrics_logger.log_training_metrics(metrics, step=self.step)

                    # Validate periodically
                    if val_data is not None and self.step % val_interval == 0:
                        val_metrics = self.evaluate(val_data, batch_size)
                        if self.metrics_logger:
                            self.metrics_logger.log_validation_metrics(val_metrics, step=self.step)

                    # Save checkpoint
                    if self.checkpoint_dir and self.step % self.save_interval == 0:
                        self.save_checkpoint()

                    # Log progress
                    if self.step % 100 == 0 and self.logger:
                        self.logger.log_text(
                            "progress",
                            f"Epoch {epoch + 1}/{num_epochs}, "
                            f"Step {self.step}, "
                            f"Loss: {metrics['loss']:.4f}",
                        )

                epoch_logs = self._average_metrics(epoch_metrics)

                if val_data is not None:
                    if self.callbacks is not None:
                        self.callbacks.on_validation_begin(self)

                    val_metrics = self.evaluate(val_data, batch_size)

                    if self.metrics_logger:
                        self.metrics_logger.log_validation_metrics(val_metrics, step=self.step)

                    if self.callbacks is not None:
                        self.callbacks.on_validation_end(self, val_metrics)

                    epoch_logs.update(self._prefix_validation_metrics(val_metrics))

                if self.callbacks is not None:
                    self.callbacks.on_epoch_end(self, epoch, epoch_logs)

                if self._callbacks_should_stop():
                    break

            # Final validation
            if val_data is not None:
                final_metrics = self.evaluate(val_data, batch_size)
                if self.metrics_logger:
                    self.metrics_logger.log_test_metrics(final_metrics)
                return final_metrics

            return metrics
        finally:
            if self.callbacks is not None:
                self.callbacks.on_train_end(self)
            for _ext_name, ext in self.extensions.items():
                on_train_end = getattr(ext, "on_train_end", None)
                if on_train_end is not None:
                    on_train_end(self)

    def evaluate(self, data: dict[str, Any], batch_size: int) -> dict[str, Any]:
        """Evaluate the model on data.

        Args:
            data: Evaluation data dictionary
            batch_size: Batch size

        Returns:
            Average evaluation metrics
        """
        source = MemorySource(
            MemorySourceConfig(shuffle=False),
            data,
            rngs=nnx.Rngs(0),
        )
        pipeline = create_data_pipeline(source, batch_size=batch_size)
        all_metrics: list[dict[str, Any]] = []

        for batch_view in pipeline:
            batch = batch_view.get_data()
            self.rng, eval_rng = jax.random.split(self.rng)
            loss, metrics = self.loss_fn(self.model, batch, eval_rng, jnp.array(self.step))
            metrics["loss"] = float(loss)
            all_metrics.append(metrics)

        if not all_metrics:
            return {}

        # Average metrics
        avg_metrics: dict[str, Any] = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = sum(values) / len(values)

        return avg_metrics

    def save_checkpoint(self, path: str | None = None) -> None:
        """Save a checkpoint.

        Args:
            path: Path to save the checkpoint (uses default if None)
        """
        if path is None and self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory specified.")

        if path is None:
            path = str(Path(self.checkpoint_dir) / f"checkpoint_{self.step}.pkl")

        import pickle  # nosec B403

        checkpoint = {
            "step": self.step,
            "opt_state": self.opt_state,
            "model_state": nnx.state(self.model),
            "rng": self.rng,
            "extensions_state": {name: nnx.state(ext) for name, ext in self.extensions.items()},
        }

        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

        if self.logger:
            self.logger.log_text("checkpoint", f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint.

        Args:
            path: Path to load the checkpoint from
        """
        import pickle  # nosec B403

        with open(path, "rb") as f:
            checkpoint = pickle.load(f)  # nosec B301

        self.step = checkpoint["step"]
        self.opt_state = checkpoint["opt_state"]
        self.rng = checkpoint["rng"]

        # Restore model state
        nnx.update(self.model, checkpoint["model_state"])

        # Restore extension state
        if "extensions_state" in checkpoint:
            for name, ext_state in checkpoint["extensions_state"].items():
                if name in self.extensions:
                    nnx.update(self.extensions[name], ext_state)

        if self.logger:
            self.logger.log_text("checkpoint", f"Loaded checkpoint from {path}")
