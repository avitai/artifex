"""Trainer for NNX-based generative models."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from itertools import islice
from pathlib import Path
from sys import exception
from typing import Any, cast, NamedTuple, Self, TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from calibrax.metrics.stateful import MeanAccumulator
from datarax.core.spec import batch_length
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx
from substrax.callbacks import CallbackList
from substrax.checkpoint import resolve_checkpoint_dir
from substrax.optim import create_optimizer as create_native_optimizer
from substrax.rng import fold_in_name
from substrax.tracking import Logger

from artifex.generative_models.core.configuration import (
    SchedulerConfig,
    TrainingConfig,
)
from artifex.generative_models.training.checkpointing import (
    Checkpointer,
    CheckpointPolicy,
    RunState,
)
from artifex.generative_models.training.loops.streaming import create_data_pipeline
from artifex.generative_models.training.optimizers.factory import optimizer_spec
from artifex.generative_models.training.schedulers import create_scheduler
from artifex.generative_models.training.step import GradientStep, StepContext
from artifex.generative_models.utils.logging.metrics import MetricsLogger


if TYPE_CHECKING:
    from datarax.pipeline import Pipeline

    from artifex.generative_models.extensions.base import (
        Extension,
    )


TrainerLossFn = Callable[
    [nnx.Module, dict[str, Any], jax.Array, jax.Array],
    tuple[jax.Array, dict[str, Any]],
]


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


def _validate_trainer(
    model: object, config: object, objective: object, cadence: int | None
) -> None:
    """Validate construction before modes, optimizer state or storage are created.

    Args:
        model: Candidate native model.
        config: Candidate typed training configuration.
        objective: Candidate loss and metric callable.
        cadence: Optional positive attempted-step observation cadence.

    Raises:
        TypeError: If the model, training configuration or objective has the wrong type.
        ValueError: If the finite-observation cadence is invalid.
    """
    if not isinstance(model, nnx.Module):
        raise TypeError(f"model must be an nnx.Module, got {type(model).__name__}")
    if config is None:
        raise TypeError("training_config is required")
    if not isinstance(config, TrainingConfig):
        raise TypeError(f"training_config must be a TrainingConfig, got {type(config).__name__}")
    if not callable(objective):
        raise TypeError("loss_fn must be callable")
    if cadence is not None and (type(cadence) is not int or cadence < 1):
        raise ValueError("check_finite_every must be a positive integer or None")


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
        optimizer: optax.GradientTransformation | nnx.Optimizer | None = None,
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
        check_finite_every: int | None = 1,
    ) -> None:
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
            check_finite_every: Observe accumulated failure before every this many
                attempts. ``1`` reports before the next attempt; ``None`` defers to
                driver, callback, checkpoint and close boundaries.

        Construction propagates ``TypeError`` for an invalid model, configuration
        or objective, and ``ValueError`` for an invalid finite-observation cadence.
        """
        _validate_trainer(model, training_config, loss_fn, check_finite_every)
        self.check_finite_every = check_finite_every

        self.model = model
        self._train_view = nnx.view(
            model,
            only=nnx.Module,
            deterministic=False,
            use_running_average=False,
            raise_if_not_found=False,
        )
        self._eval_view = nnx.view(
            model,
            only=nnx.Module,
            deterministic=True,
            use_running_average=True,
            raise_if_not_found=False,
        )
        self.training_config = training_config
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.workdir = workdir
        self.rng = rng if rng is not None else jax.random.PRNGKey(0)
        self._evaluation_rng = nnx.Rngs(evaluation=fold_in_name(self.rng, "evaluation"))
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

        self._failure = nnx.Variable(jnp.asarray(False))
        self._gradient_step: GradientStep | None = None
        self._set_up_optimizer(optimizer)

        # Training state (step counter and rng)
        self.step = 0
        self.epoch = 0
        self._checkpointer: Checkpointer | None = None
        self._closed = False
        # The compiled gradient step, built on first use (see _build_compiled_step).
        self._compiled_step: Callable[..., dict[str, jax.Array]] | None = None

    def _set_up_optimizer(
        self, optimizer: optax.GradientTransformation | nnx.Optimizer | None
    ) -> None:
        """Take the caller's optimizer, or build one now unless its schedule needs the run.

        A schedule whose horizon is configured (or none at all) builds at construction; one
        whose horizon must be derived from the run waits for ``train``.
        """
        native = (
            optimizer
            if optimizer is None or isinstance(optimizer, nnx.Optimizer)
            else nnx.Optimizer(
                self.model,
                optax.apply_if_finite(
                    optimizer, max_consecutive_errors=int(jnp.iinfo(jnp.int32).max)
                ),
                wrt=nnx.Param,
            )
        )
        if optimizer is None and self._horizon_field_to_derive() is None:
            native = self._create_optimizer(horizon=None)
        if native is not None:
            self._gradient_step = GradientStep(
                native, _TrainingObjective(self.loss_fn, self.extensions)
            )

    @property
    def optimizer(self) -> nnx.Optimizer | None:
        """The optimizer owned by the native step, absent only before a derived horizon."""
        return None if self._gradient_step is None else self._gradient_step.optimizer

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

    def _require_optimizer(self) -> nnx.Optimizer:
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
            self._set_up_optimizer(self._create_optimizer(horizon=horizon))

    def _create_optimizer(self, *, horizon: int | None) -> nnx.Optimizer:
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
        return create_native_optimizer(self.model, optimizer_spec(opt_config, self.schedule))

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

    def _build_compiled_step(self) -> Callable[..., dict[str, jax.Array]]:
        """Bind the live model and step state through donated native NNX compilation."""
        self._require_optimizer()

        def run(
            bundle: tuple[nnx.Module, GradientStep, nnx.Variable],
            batch: dict[str, Any],
            context: StepContext,
        ) -> dict[str, jax.Array]:
            model, step, failure = bundle
            metrics = step(model, batch, context)
            finite = getattr(step.optimizer.opt_state, "last_finite", None)
            rejected = False if finite is None else ~finite[...]
            failure[...] = failure[...] | ~jnp.isfinite(metrics["loss"]) | rejected
            return metrics

        return nnx.jit_partial(
            run,
            (self._train_view, self._gradient_step, self._failure),
            graph=True,
            graph_updates=False,
            donate_argnums=0,
        )

    def check_finite(self) -> None:
        """Observe accumulated failed losses or skipped updates before another attempt.

        Raises:
            FloatingPointError: If a previous attempt failed numerical admission.
        """
        if bool(self._failure[...]):
            raise FloatingPointError("training encountered a non-finite loss or gradient")

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Execute a single training step through the compiled gradient step.

        The step (base loss plus enabled extension losses, gradients, optimizer
        update) is compiled once per batch shape. Host callbacks, logging and
        history belong to the training drivers, not this numerical interface.

        Args:
            batch: Batch of training data

        Returns:
            Device metrics, loss, gradient norm and attempted-update count. Hooks
            observe accumulated failure before receiving progress.

        Raises:
            RuntimeError: If the Trainer lifecycle is closed.
        """
        if self._closed:
            raise RuntimeError("trainer is closed")
        if self.check_finite_every is not None and self.step % self.check_finite_every == 0:
            self.check_finite()
        if self._compiled_step is None:
            self._compiled_step = self._build_compiled_step()
        metrics = self._compiled_step(batch, StepContext.at(self.rng, self.step))

        self.step += 1
        return {**metrics, "step": self.step}

    def _driver_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Observe numerical progress before host notifications and history retention."""
        self.check_finite()
        index = self.step
        if self.callbacks is not None:
            self.callbacks.on_batch_begin(self, index)
        self._extension_hook("on_batch_begin", index)
        metrics = self.train_step(batch)
        self.check_finite()
        if self.log_callback is not None:
            self.log_callback(self.step, metrics, prefix="train")
        self.train_metrics.append(metrics)
        if self.callbacks is not None:
            self.callbacks.on_batch_end(self, index, metrics)
        self._extension_hook("on_batch_end", index, metrics)
        return metrics

    def _extension_hook(self, name: str, *args: Any) -> None:
        """Run the same registered extension lifecycle at each host boundary."""
        for extension in self.extensions.values():
            hook = getattr(extension, name, None)
            if hook is not None:
                hook(self, *args)

    def validate_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Execute a single validation step.

        Args:
            batch: Batch of validation data

        Returns:
            Dictionary of validation metrics
        """
        # Split RNG for validation
        val_rng = self._evaluation_rng.evaluation()

        # Compute validation loss (no gradients needed)
        loss, metrics = self.loss_fn(self._eval_view, batch, val_rng, jnp.array(self.step))

        self._admit_evaluation_loss(float(loss))
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
            metrics = self._driver_step(batch)
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
        shuffle_rng = fold_in_name(self.rng, "shuffle")
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
        try:
            self._begin_training()
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
                self.epoch = epoch + 1
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
            self._finish_training()

    def _begin_training(self) -> None:
        """Enter registered host lifecycles under the run's error-cleanup boundary."""
        if self.callbacks is not None:
            self.callbacks.on_train_begin(self)
        self._extension_hook("on_train_begin")

    def _finish_training(self) -> None:
        """Observe final numerical failure and persistence before leaving a run."""
        try:
            if exception() is None:
                self.check_finite()
        finally:
            try:
                if self.callbacks is not None:
                    self.callbacks.on_train_end(self)
                self._extension_hook("on_train_end")
            finally:
                if exception() is None:
                    self.wait_until_finished()
                else:
                    self._close_storage()

    def _run_epoch(self, pipeline: Pipeline, context: _EpochContext) -> list[dict[str, Any]]:
        """Train over one pass of ``pipeline``; return the per-step metrics.

        Periodic validation, checkpointing and progress logging happen here,
        keyed on the global step.
        """
        epoch_metrics: list[dict[str, Any]] = []
        for batch in pipeline:
            metrics = self._driver_step(batch)
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

        Raises:
            ValueError: If the population exceeds native metric count capacity.
        """
        source = MemorySource(
            MemorySourceConfig(shuffle=False),
            data,
            rngs=nnx.Rngs(0),
        )
        pipeline = create_data_pipeline(source, batch_size=batch_size)
        population = batch_length(data)
        if population is None or population > jnp.iinfo(jnp.int32).max:
            raise ValueError("evaluation requires a population within native metric count capacity")
        accumulator: MeanAccumulator | None = None

        for batch in pipeline:
            records = batch_length(batch)
            if not records:
                continue
            eval_rng = self._evaluation_rng.evaluation()
            loss, metrics = self.loss_fn(self._eval_view, batch, eval_rng, jnp.array(self.step))
            values = {**metrics, "loss": loss}
            if accumulator is None:
                accumulator = MeanAccumulator(tuple(values))
            _accumulate_metrics(accumulator, values, jnp.asarray(records, jnp.int32))

        if accumulator is None:
            return {}
        result = {name: float(value) for name, value in accumulator.compute().items()}
        self._admit_evaluation_loss(result["loss"])
        return result

    def _admit_evaluation_loss(self, loss: float) -> None:
        """Retain failed evaluation evidence before any selection callback can run.

        Args:
            loss: The reduced, host-observed validation loss.

        Raises:
            FloatingPointError: If the observed validation loss is nonfinite.
        """
        if not math.isfinite(loss):
            self._failure[...] = jnp.asarray(True)
            raise FloatingPointError("evaluation encountered a non-finite loss")

    def run_state(self) -> RunState:
        """Expose the complete native checkpoint owner for composed application saves."""
        self._require_optimizer()
        assert self._gradient_step is not None
        return RunState(
            model=self.model,
            step=self._gradient_step,
            base_key=self.rng,
            epoch=self.epoch,
            extra_state={
                "evaluation_rng": self._evaluation_rng,
                **({} if self.callbacks is None else {"callbacks": self.callbacks}),
            },
        )

    def checkpoint_state(self) -> dict[str, Any]:
        """Return live native items; detach arrays before retaining them across donation.

        The complete step includes optimizer, objective and auxiliary-loss state.
        This layout replaces the separate raw optimizer checkpoint item.
        """
        return self.run_state().items()

    def _checkpoint_is_due(self) -> bool:
        """Whether the configured checkpoint cadence lands on this completed attempt."""
        return (
            self.checkpoint_dir is not None
            and self.step > 0
            and (self.step % self.training_config.save_frequency == 0)
        )

    def _require_checkpointer(self) -> Checkpointer:
        """Return the run's single persistent native checkpoint lifecycle."""
        if self._closed:
            raise RuntimeError("trainer is closed")
        if self.checkpoint_dir is None:
            raise ValueError("no checkpoint directory: configure one or give a workdir")
        if self._checkpointer is None:
            self._checkpointer = Checkpointer(
                CheckpointPolicy(
                    directory=self.checkpoint_dir,
                    every_steps=self.training_config.save_frequency,
                    keep=self.training_config.max_checkpoints,
                )
            )
        return self._checkpointer

    def save_async(self) -> Path:
        """Admit complete state and submit native capture before another donated update.

        The returned path is not publication evidence. Wait for completion or close.
        """
        self.check_finite()
        return self._require_checkpointer().save_async(self.step, self.run_state())

    def save_checkpoint(self, step: int | None = None) -> Path:
        """Save complete current state and observe completed persistence.

        A checkpoint address must match its actual attempted-update count.
        """
        if step is not None and step != self.step:
            raise ValueError("checkpoint step differs from the actual trainer step")
        path = self.save_async()
        self.wait_until_finished()
        return path

    def wait_until_finished(self) -> None:
        """Observe completion errors without ending the Trainer's lifecycle."""
        if self._checkpointer is not None:
            self._checkpointer.wait_until_finished()

    def load_checkpoint(self, step: int | None = None) -> None:
        """Restore the complete native run, clearing its failed observation window."""
        result = self._require_checkpointer().restore(self.run_state(), step=step)
        self.rng = result.base_key
        self.epoch = result.epoch
        self.step = result.step
        self._failure[...] = jnp.asarray(False)

    def apply_checkpoint_state(self, payload: dict[str, Any], *, step: int) -> None:
        """Apply identity-validated native items after detached numerical admission.

        Host extensions are applied sequentially. Reconstruct if one fails; this is
        not a rollback API. Source and record checks remain the caller's obligation.
        """
        StepContext.at(payload["rng"], step)
        self.run_state().apply_items(payload)
        self.rng = payload["rng"]
        self.step = step
        self._failure[...] = jnp.asarray(False)

    def close(self) -> None:
        """Drain native persistence and release its resources, including error exits."""
        if not self._closed:
            try:
                self.check_finite()
            finally:
                self._close_storage()

    def _close_storage(self) -> None:
        """Release persistence on every exit without replacing an active numerical error."""
        try:
            if self._checkpointer is not None:
                self._checkpointer.close()
        finally:
            self._closed = True

    def __enter__(self) -> Self:
        """Enter an explicit Trainer lifecycle."""
        if self._closed:
            raise RuntimeError("trainer is closed")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Close on success or exception, propagating persistence failures."""
        self.close()


class _TrainingObjective(nnx.Module):
    """Compose the configured objective and enabled auxiliary losses for one step."""

    def __init__(self, objective: TrainerLossFn, extensions: dict[str, Any]) -> None:
        self.objective = objective
        self.extensions = nnx.data(extensions)

    def __call__(
        self, model: nnx.Module, batch: Mapping[str, Any], context: StepContext
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        loss, metrics = self.objective(model, dict(batch), context.key, context.step)
        outputs = None
        if self.extensions and "input" in batch:
            forward = getattr(model, "__call__", None) or getattr(model, "encode", None)
            if forward is not None:
                outputs = forward(batch["input"])
        metrics = dict(metrics)
        for name, extension in self.extensions.items():
            loss_fn = getattr(extension, "loss_fn", None)
            if extension.is_enabled() and loss_fn is not None:
                value = extension.weight * loss_fn(batch, outputs)
                loss = loss + value
                metrics[f"{name}_loss"] = value
        return loss, metrics


@nnx.jit
def _accumulate_metrics(
    accumulator: MeanAccumulator, metrics: dict[str, jax.Array], records: jax.Array
) -> None:
    """Keep the shared record-weighted reduction on device until observation."""
    accumulator.update(metrics, records)
