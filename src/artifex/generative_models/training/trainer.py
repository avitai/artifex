"""Trainer for NNX-based generative models."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from importlib.metadata import version
from itertools import islice
from pathlib import Path
from typing import Any, cast, NamedTuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from datarax.core.spec import batch_length
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx
from substrax.checkpoint import OrbaxCheckpointStore, Producer, resolve_checkpoint_dir

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
    from datarax.pipeline import Pipeline

    from artifex.generative_models.extensions.base import (
        Extension,
    )


TrainerLossFn = Callable[
    [nnx.Module, dict[str, Any], jax.Array, jax.Array],
    tuple[jax.Array, dict[str, Any]],
]


# The record's producer: this package and its installed version.
_PRODUCER = Producer(name="artifex", version=version("avitai-artifex"))

# The schedule field the run supplies when the configuration leaves it unset.
_HORIZON_FIELDS = {
    "linear": "total_steps",
    "polynomial": "total_steps",
    "one_cycle": "total_steps",
    "cosine": "cycle_length",
}


class _EpochContext(NamedTuple):
    """What one training epoch needs beyond the pipeline it iterates."""

    epoch: int
    num_epochs: int
    val_data: dict[str, Any] | None
    batch_size: int
    val_interval: int


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
            workdir: Working directory for outputs; when the configuration names no
                ``checkpoint_dir``, its ``checkpoints`` subdirectory holds the checkpoints.
            rng: JAX random number generator key.
            loss_fn: Explicit objective function. Signature:
                     loss_fn(model, batch, rng, step) -> (loss, metrics_dict).
                     It runs inside the compiled step: ``step`` is a traced
                     int32 scalar (use ``jnp`` arithmetic on it, never
                     ``int()``), and Python side effects run at trace time only.
            metrics_logger: Logger for training metrics.
            logger: Artifex logger for general logging.
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
        # The configuration names the directory, else the run directory holds it; without
        # either the trainer checkpoints nowhere, so it never writes into the working
        # directory on its own. The store creates the directory on the first save.
        configured_dir = training_config.checkpoint_dir
        self.checkpoint_dir: Path | None = (
            None
            if configured_dir is None and workdir is None
            else resolve_checkpoint_dir(configured_dir, None if workdir is None else Path(workdir))
        )
        self.log_callback = log_callback
        self.callbacks = callbacks
        self.extensions: dict[str, Extension] = extensions if extensions is not None else {}

        # Store training metrics for convenience
        self.train_metrics: list[dict[str, Any]] = []
        self.val_metrics: list[dict[str, Any]] = []

        # The batches per epoch and the schedule horizon are what the data holds, so they
        # are known once ``train`` builds its pipeline, not guessed at construction.
        self.steps_per_epoch: int | None = None
        self.schedule: optax.Schedule | None = None
        self.schedule_horizon: int | None = None

        self.optimizer: optax.GradientTransformation | None
        self.opt_state: Any
        self._set_up_optimizer(optimizer)

        # Training state (step counter and rng)
        self.step = 0
        # The compiled gradient step, built on first use (see _build_compiled_step).
        self._compiled_step: Callable[..., tuple[jax.Array, dict[str, Any], Any]] | None = None

    def _set_up_optimizer(self, optimizer: optax.GradientTransformation | None) -> None:
        """Take the caller's optimizer, or build one now unless its schedule needs the run.

        A schedule whose horizon is configured (or none at all) builds at construction; one
        whose horizon must be derived from the run waits for ``train``.
        """
        self.optimizer = optimizer
        self.opt_state = None
        if optimizer is None and self._horizon_field_to_derive() is None:
            self.optimizer = self._create_optimizer(horizon=None)
        if self.optimizer is not None:
            self.opt_state = self._initial_opt_state(self.optimizer)

    def _initial_opt_state(self, optimizer: optax.GradientTransformation) -> Any:
        # Cast: nnx.state returns GraphState which is pytree-compatible with optax Params
        return optimizer.init(cast(optax.Params, nnx.state(self.model, nnx.Param)))

    def _horizon_field_to_derive(self) -> str | None:
        """The schedule field the run must supply, or ``None`` when the schedule needs none.

        A linear, polynomial or one-cycle schedule spans ``total_steps`` and a cosine
        schedule ``cycle_length``; when the configuration leaves that field unset, the
        horizon is the run's length, ``num_epochs`` times the batches per epoch.
        """
        scheduler = self.training_config.scheduler
        if scheduler is None:
            return None
        field = _HORIZON_FIELDS.get(scheduler.scheduler_type)
        if field is None or getattr(scheduler, field) is not None:
            return None
        return field

    def _horizon_message(self, field: str) -> str:
        scheduler_type = cast(SchedulerConfig, self.training_config.scheduler).scheduler_type
        return (
            f"the {scheduler_type} schedule needs {field}; set it in the scheduler "
            "configuration, or call train(), which derives it from the run"
        )

    def _require_optimizer(self) -> optax.GradientTransformation:
        """The optimizer, which a derived-horizon schedule has only once ``train`` ran.

        Returns:
            The optimizer in use.

        Raises:
            ValueError: If the schedule's horizon is neither configured nor derived yet.
        """
        if self.optimizer is None:
            field = self._horizon_field_to_derive()
            raise ValueError(self._horizon_message(field if field is not None else "a horizon"))
        return self.optimizer

    def _ensure_optimizer(self, horizon: int) -> None:
        """Build the optimizer from the run's horizon when construction deferred it."""
        if self.optimizer is None:
            self.optimizer = self._create_optimizer(horizon=horizon)
            self.opt_state = self._initial_opt_state(self.optimizer)

    def _create_optimizer(self, *, horizon: int | None) -> optax.GradientTransformation:
        """Create the optimizer from the training config.

        Delegates to the shared optimizer factory, which builds through ``substrax.optim``
        with the configured schedule as the optimizer's learning rate.

        Args:
            horizon: The run's length in steps, for a schedule whose horizon is not
                configured; ``None`` when the schedule needs none or configures its own.

        Returns:
            Optax gradient transformation (optimizer).
        """
        opt_config = self.training_config.optimizer
        scheduler = self.training_config.scheduler
        self.schedule = (
            None
            if scheduler is None
            else self._create_schedule(scheduler, opt_config.learning_rate, horizon)
        )
        return create_optimizer(self.model, opt_config, schedule=self.schedule)

    def _create_schedule(
        self, scheduler_config: SchedulerConfig, base_lr: float, horizon: int | None
    ) -> optax.Schedule:
        """Create the learning-rate schedule, filling an unset horizon from the run.

        Delegates to the centralized scheduler factory for all schedule types.
        The factory supports: constant, linear, cosine, exponential, polynomial,
        step, multistep, cyclic, and one_cycle schedules.

        Args:
            scheduler_config: Configuration for the learning rate schedule.
            base_lr: Base learning rate.
            horizon: The run's length in steps, used when the configuration leaves the
                schedule's horizon field unset.

        Returns:
            An optax schedule mapping the step to the learning rate.

        Raises:
            ValueError: If the schedule needs a horizon that is neither configured nor given.
        """
        field = _HORIZON_FIELDS.get(scheduler_config.scheduler_type)
        if field is not None and getattr(scheduler_config, field) is None:
            if horizon is None:
                raise ValueError(self._horizon_message(field))
            scheduler_config = dataclasses.replace(scheduler_config, **{field: horizon})
        self.schedule_horizon = None if field is None else getattr(scheduler_config, field)
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

    def _build_compiled_step(self) -> Callable[..., tuple[jax.Array, dict[str, Any], Any]]:
        """Compile the gradient step once: loss, gradients, optimizer update.

        The compiled function takes the model, the extensions, the optimizer
        state, a batch and ``(step_rng, step_index)``, updates the model in
        place and returns ``(loss, metrics, new_opt_state)``. Everything with a
        Python side effect (callbacks, logging, metric history) stays outside.
        """
        loss_fn = self.loss_fn
        optimizer = self._require_optimizer()

        @nnx.jit
        def compiled_step(
            model: nnx.Module,
            extensions: dict[str, Extension],
            opt_state: Any,
            batch: dict[str, Any],
            step_state: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, dict[str, Any], Any]:
            step_rng, step_index = step_state

            def total_loss_fn(model: nnx.Module) -> tuple[jax.Array, dict[str, Any]]:
                base_loss, base_metrics = loss_fn(model, batch, step_rng, step_index)
                ext_losses: dict[str, jax.Array] = {}
                total_ext_loss = jnp.array(0.0)
                model_outputs = None
                if extensions:
                    if hasattr(model, "__call__") and "input" in batch:
                        model_outputs = model(batch["input"])
                    else:
                        encode_fn = getattr(model, "encode", None)
                        if encode_fn is not None and "input" in batch:
                            model_outputs = encode_fn(batch["input"])
                for ext_name, ext in extensions.items():
                    if ext.is_enabled():
                        ext_loss_fn = getattr(ext, "loss_fn", None)
                        if ext_loss_fn is not None:
                            weighted_loss = ext.weight * ext_loss_fn(batch, model_outputs)
                            ext_losses[f"{ext_name}_loss"] = weighted_loss
                            total_ext_loss = total_ext_loss + weighted_loss
                return base_loss + total_ext_loss, {**base_metrics, **ext_losses}

            (loss, metrics), grads = nnx.value_and_grad(total_loss_fn, has_aux=True)(model)
            param_grads = cast(optax.Updates, nnx.state(grads, nnx.Param))
            params = cast(optax.Params, nnx.state(model, nnx.Param))
            updates, new_opt_state = optimizer.update(param_grads, opt_state, params)
            nnx.update(model, optax.apply_updates(params, updates))
            return loss, metrics, new_opt_state

        return compiled_step

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Execute a single training step through the compiled gradient step.

        The step (base loss plus enabled extension losses, gradients, optimizer
        update) is compiled once per batch shape; callbacks, logging and the
        metric history run around it in Python.

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

        self.rng, step_rng = jax.random.split(self.rng)
        if self._compiled_step is None:
            self._compiled_step = self._build_compiled_step()
        loss, metrics, self.opt_state = self._compiled_step(
            self.model,
            self.extensions,
            self.opt_state,
            batch,
            (step_rng, jnp.asarray(self.step, dtype=jnp.int32)),
        )

        self.step += 1
        metrics = {**metrics, "loss": float(loss), "step": self.step}

        if self.log_callback is not None:
            self.log_callback(self.step, metrics, prefix="train")

        self.train_metrics.append(metrics)

        if self.callbacks is not None:
            self.callbacks.on_batch_end(self, self.step, metrics)

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

    def train_epoch(self, steps: int | None = None) -> dict[str, Any]:
        """Train for one epoch over ``train_data_loader``.

        The epoch runs until the loader's iterator is exhausted, or for ``steps`` batches.

        Args:
            steps: The number of batches to train on, or ``None`` for the whole iterator.

        Returns:
            Average metrics for the epoch

        Raises:
            ValueError: If the trainer has no ``train_data_loader``.
        """
        if self.train_data_loader is None:
            raise ValueError("train_data_loader is required for train_epoch")

        data_iter = self.train_data_loader(self.training_config.batch_size)
        epoch_metrics: list[dict[str, Any]] = []

        for batch in islice(data_iter, steps):
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)

            if self._checkpoint_is_due():
                self.save_checkpoint()

        return self._average_metrics(epoch_metrics)

    def train(  # noqa: DOC502 - datarax's Pipeline raises the ValueError while being built
        self,
        train_data: dict[str, Any],
        num_epochs: int,
        batch_size: int,
        val_data: dict[str, Any] | None = None,
        val_interval: int = 100,
    ) -> dict[str, Any]:
        """Train the model for multiple epochs.

        The epoch's ragged final batch is dropped (PyTorch's rule), so every step sees
        ``batch_size`` real rows and the steps per epoch are what the data holds; a
        schedule whose horizon is not configured spans ``num_epochs`` times that count.

        Args:
            train_data: Training data dictionary
            num_epochs: Number of epochs to train
            batch_size: Batch size
            val_data: Optional validation data
            val_interval: Steps between validation

        Returns:
            Final metrics after training

        Raises:
            ValueError: If ``train_data`` holds fewer records than one batch.
        """
        if self.logger:
            self.logger.log_text(
                "training", f"Training for {num_epochs} epochs with batch_size={batch_size}"
            )

        # One shuffled pipeline for the whole call: each epoch after the first
        # starts with ``reset()``, which serves a new permutation of the data.
        self.rng, shuffle_rng = jax.random.split(self.rng)
        shuffle_seed = int(jax.random.randint(shuffle_rng, (), 0, 2**31 - 1))
        source = MemorySource(
            MemorySourceConfig(shuffle=True),
            train_data,
            rngs=nnx.Rngs(shuffle_seed),
        )
        pipeline = create_data_pipeline(
            source, batch_size=batch_size, rngs=nnx.Rngs(shuffle_seed), drop_last=True
        )
        # datarax refuses a drop_last pipeline that holds no full batch, so an epoch has at
        # least one step.
        self.steps_per_epoch = len(pipeline)
        self._ensure_optimizer(num_epochs * self.steps_per_epoch)

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

                if epoch > 0:
                    pipeline.reset()
                epoch_metrics = self._run_epoch(
                    pipeline,
                    _EpochContext(
                        epoch=epoch,
                        num_epochs=num_epochs,
                        val_data=val_data,
                        batch_size=batch_size,
                        val_interval=val_interval,
                    ),
                )
                if epoch_metrics:
                    metrics = epoch_metrics[-1]

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

    def _run_epoch(self, pipeline: Pipeline, context: _EpochContext) -> list[dict[str, Any]]:
        """Train over one pass of ``pipeline``; return the per-step metrics.

        Periodic validation, checkpointing and progress logging happen here,
        keyed on the global step.
        """
        epoch_metrics: list[dict[str, Any]] = []
        for batch in pipeline:
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)

            if self.metrics_logger:
                self.metrics_logger.log_training_metrics(metrics, step=self.step)

            if context.val_data is not None and self.step % context.val_interval == 0:
                val_metrics = self.evaluate(context.val_data, context.batch_size)
                if self.metrics_logger:
                    self.metrics_logger.log_validation_metrics(val_metrics, step=self.step)

            if self._checkpoint_is_due():
                self.save_checkpoint()

            if self.step % 100 == 0 and self.logger:
                self.logger.log_text(
                    "progress",
                    f"Epoch {context.epoch + 1}/{context.num_epochs}, "
                    f"Step {self.step}, "
                    f"Loss: {metrics['loss']:.4f}",
                )
        return epoch_metrics

    def evaluate(self, data: dict[str, Any], batch_size: int) -> dict[str, Any]:
        """Evaluate the model on every record of ``data``.

        One pass serves every record once; the last batch holds the records left, which may be
        fewer than ``batch_size``. Each batch's metrics weigh in by the records it holds, so
        the result is the per-record average over ``data``.

        Args:
            data: Evaluation data dictionary
            batch_size: Batch size

        Returns:
            Record-weighted average of the objective's metrics and its loss
        """
        source = MemorySource(
            MemorySourceConfig(shuffle=False),
            data,
            rngs=nnx.Rngs(0),
        )
        pipeline = create_data_pipeline(source, batch_size=batch_size)
        weighted: list[tuple[dict[str, Any], int]] = []

        for batch in pipeline:
            records = batch_length(batch)
            if not records:
                continue
            self.rng, eval_rng = jax.random.split(self.rng)
            loss, metrics = self.loss_fn(self.model, batch, eval_rng, jnp.array(self.step))
            weighted.append(({**metrics, "loss": float(loss)}, records))

        if not weighted:
            return {}

        total = sum(records for _, records in weighted)
        return {
            key: sum(metrics[key] * records for metrics, records in weighted) / total
            for key in weighted[0][0]
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """Return the checkpoint items, for storing trainer state beside application state.

        The items are substrax's ``model``, ``optimizer``, ``rng`` and ``extensions``,
        as :meth:`save_checkpoint` writes them; the step is the record's. The outer
        dictionary is new on every call, but the ``model`` and ``extensions`` entries
        are live ``nnx.State`` views whose Variables belong to the trainer, so
        assigning to a leaf changes the trainer. Copy the leaves first, for example
        with ``jax.tree.map(jnp.array, tree)``, when a detached snapshot is needed.

        Pass the items as the restore templates to a checkpoint store, validate the
        restored record, then hand the restored items to
        :meth:`apply_checkpoint_state`.

        Returns:
            The checkpoint items: model, optimizer, RNG and extension state.
        """
        self._require_optimizer()
        return {
            "model": nnx.state(self.model),
            "optimizer": self.opt_state,
            "rng": self.rng,
            "extensions": {name: nnx.state(ext) for name, ext in self.extensions.items()},
        }

    def _checkpoint_is_due(self) -> bool:
        """Whether the loop saves at this step: a directory is set and the cadence lands."""
        return (
            self.checkpoint_dir is not None and self.step % self.training_config.save_frequency == 0
        )

    def _checkpoint_store(self) -> OrbaxCheckpointStore:
        """Open the store under ``checkpoint_dir``; it keeps ``max_checkpoints`` steps.

        Returns:
            The store over ``checkpoint_dir``.

        Raises:
            ValueError: If the trainer has no checkpoint directory.
        """
        if self.checkpoint_dir is None:
            raise ValueError(
                "no checkpoint directory: set TrainingConfig.checkpoint_dir or give the "
                "trainer a workdir"
            )
        return OrbaxCheckpointStore(
            self.checkpoint_dir, max_to_keep=self.training_config.max_checkpoints
        )

    def save_checkpoint(self, step: int | None = None) -> Path:
        """Save the model, optimizer, RNG and extension state under ``step``.

        Args:
            step: Checkpoint step; defaults to the trainer's current step.

        Returns:
            The directory of the saved checkpoint.
        """
        step = self.step if step is None else step
        with self._checkpoint_store() as store:
            path = store.save(step, self.checkpoint_state(), producer=_PRODUCER)
        if self.logger:
            self.logger.log_text("checkpoint", f"Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, step: int | None = None) -> None:
        """Restore the model, optimizer, RNG and extension state from ``step``.

        A ``step`` the directory holds no checkpoint at propagates the store's
        ``CheckpointNotFoundError``, a ``FileNotFoundError``.

        Args:
            step: Checkpoint step; defaults to the latest one in ``checkpoint_dir``.

        Raises:
            FileNotFoundError: If ``step`` is ``None`` and the directory holds no checkpoint.
        """
        with self._checkpoint_store() as store:
            if step is None:
                step = store.latest_step()
                if step is None:
                    raise FileNotFoundError(f"no checkpoint under {self.checkpoint_dir}")
            checkpoint = store.restore(step, templates=self.checkpoint_state())

        self.apply_checkpoint_state(checkpoint.items, step=checkpoint.step)

        if self.logger:
            self.logger.log_text("checkpoint", f"Loaded checkpoint from step {checkpoint.step}")

    def apply_checkpoint_state(self, payload: dict[str, Any], *, step: int) -> None:
        """Apply restored checkpoint items to this trainer and set its step.

        This is the state-application half of :meth:`load_checkpoint`, so an
        application can check the checkpoint's record before any live state
        changes. The payload must come from a restore that used
        :meth:`checkpoint_state` as the templates. Entries are applied one at a
        time and the call is not transactional: validate before calling it, since
        a malformed payload can leave the trainer partly updated.

        Args:
            payload: Checkpoint items restored against :meth:`checkpoint_state`.
            step: Checkpoint step the restored state belongs to.
        """
        nnx.update(self.model, payload["model"])
        self.opt_state = payload["optimizer"]
        self.rng = payload["rng"]
        for name, extension_state in payload["extensions"].items():
            if name in self.extensions:
                nnx.update(self.extensions[name], extension_state)
        self.step = step
