"""Checkpointing utilities for model persistence via Orbax."""

import logging
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp
from flax import nnx


logger = logging.getLogger(__name__)


def setup_checkpoint_manager(
    base_dir: str | Path,
) -> tuple[ocp.CheckpointManager, str]:
    """Set up an Orbax checkpoint manager.

    Args:
        base_dir: Base directory for storing checkpoints.

    Returns:
        A tuple of (checkpoint_manager, absolute_path_string).

    Raises:
        FileNotFoundError: If the directory cannot be created.
        PermissionError: If the directory is not writable.
        OSError: On other filesystem errors.
        ValueError: If the manager configuration is invalid.
        TypeError: If the manager configuration is invalid.
    """
    try:
        base_dir_abs = str(Path(base_dir).resolve())
        logger.info("Setting up checkpoint manager in absolute path: %s...", base_dir_abs)

        Path(base_dir_abs).mkdir(parents=True, exist_ok=True)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=5,
            create=True,
        )

        checkpoint_manager = ocp.CheckpointManager(
            directory=base_dir_abs,
            options=options,
        )

        logger.info("Successfully created checkpoint manager in %s", base_dir_abs)
        return checkpoint_manager, base_dir_abs
    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.exception("Error with checkpoint directory: %s", e)
        raise
    except (ValueError, TypeError) as e:
        logger.exception("Error in checkpoint manager configuration: %s", e)
        raise


def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    model: nnx.Module | nnx.GraphDef,
    step: int,
) -> ocp.CheckpointManager:
    """Save model checkpoint using Orbax.

    Args:
        checkpoint_manager: The Orbax CheckpointManager instance.
        model: The NNX model whose state will be saved.
        step: The training step number for this checkpoint.

    Returns:
        The checkpoint manager (for chaining).

    Raises:
        TypeError: If *model* is not an ``nnx.Module`` or ``nnx.GraphDef``.
        OSError: On filesystem write errors.
        ValueError: On configuration or serialization errors.
    """
    logger.info("Attempting to save checkpoint for step %d...", step)

    try:
        if not isinstance(model, (nnx.Module, nnx.GraphDef)):
            raise TypeError(f"Expected model to be nnx.Module or nnx.GraphDef, got {type(model)}")

        model_state = nnx.state(model)
        logger.debug("Successfully extracted model state.")
    except (TypeError, ValueError, AttributeError) as e:
        logger.exception("Error getting model state: %s", e)
        raise

    try:
        save_args = ocp.args.Composite(
            model=ocp.args.StandardSave(model_state),
        )

        checkpoint_manager.save(step, args=save_args)
        checkpoint_manager.wait_until_finished()

        logger.info(
            "Successfully saved checkpoint for step %d to %s",
            step,
            checkpoint_manager.directory,
        )

    except (OSError, IOError, FileNotFoundError, PermissionError) as e:
        logger.exception("Error writing checkpoint to disk: %s", e)
        raise
    except (ValueError, TypeError, AttributeError) as e:
        logger.exception("Error in checkpoint_manager configuration or usage: %s", e)
        raise

    return checkpoint_manager


def load_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    target_model_template: nnx.Module | nnx.GraphDef | None = None,
    step: int | None = None,
) -> tuple[Any | None, int | None]:
    """Load model checkpoint using Orbax.

    Args:
        checkpoint_manager: The Orbax CheckpointManager instance.
        target_model_template: An optional NNX Module instance or GraphDef
            with the same structure as the saved model.  If provided, the
            loaded state will be applied to this template.  If ``None``, the
            raw state dict is returned.
        step: The specific step to restore.  If ``None``, restores the
            latest step.

    Returns:
        A tuple of (restored_model_or_state, step).  Returns ``(None, None)``
        if no checkpoint is found.

    Raises:
        TypeError: If *target_model_template* has an unexpected type.
        OSError: On filesystem read errors.
        ValueError: On deserialization or structure-mismatch errors.
    """
    try:
        if step is None:
            step = checkpoint_manager.latest_step()
            if step is None:
                logger.info(
                    "No checkpoints found in %s to restore.",
                    checkpoint_manager.directory,
                )
                return None, None

        logger.info(
            "Attempting to restore checkpoint from step %d in %s...",
            step,
            checkpoint_manager.directory,
        )

        if target_model_template:
            if not isinstance(target_model_template, (nnx.Module, nnx.GraphDef)):
                raise TypeError(
                    f"Expected target_model_template to be nnx.Module or "
                    f"nnx.GraphDef, got {type(target_model_template)}"
                )
            target_state = nnx.state(target_model_template)
            restore_args = ocp.args.Composite(
                model=ocp.args.StandardRestore(target_state),
            )
            logger.debug("Restoring checkpoint into provided model template.")
            restored_data = checkpoint_manager.restore(step, args=restore_args)
            nnx.update(target_model_template, restored_data["model"])

            logger.info(
                "Successfully restored checkpoint from step %d into the template.",
                step,
            )
            return target_model_template, step

        # Restore raw state dictionary if no template is given
        restore_args = ocp.args.Composite(
            model=ocp.args.StandardRestore(),
        )
        logger.debug("Restoring checkpoint as raw state dictionary.")
        restored_data = checkpoint_manager.restore(step, args=restore_args)
        logger.info("Successfully restored raw checkpoint data from step %d", step)

        if "model" not in restored_data:
            logger.warning("'model' key not found in restored data dictionary.")
            return None, step
        return restored_data["model"], step

    except (FileNotFoundError, IOError, OSError, PermissionError) as e:
        logger.exception("Error accessing checkpoint files: %s", e)
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.exception("Error in checkpoint data or model structure: %s", e)
        raise


def save_checkpoint_with_optimizer(
    checkpoint_manager: ocp.CheckpointManager,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    step: int,
) -> ocp.CheckpointManager:
    """Save both model and optimizer state to checkpoint.

    Args:
        checkpoint_manager: The Orbax CheckpointManager instance.
        model: The NNX model to save.
        optimizer: The NNX Optimizer to save.
        step: The step number for this checkpoint.

    Returns:
        The checkpoint manager (for chaining).
    """
    logger.info("Saving checkpoint with optimizer at step %d...", step)

    try:
        model_state = nnx.state(model)
        optimizer_state = nnx.state(optimizer)

        save_args = ocp.args.Composite(
            model=ocp.args.StandardSave(model_state),
            optimizer=ocp.args.StandardSave(optimizer_state),
        )

        checkpoint_manager.save(step, args=save_args)
        checkpoint_manager.wait_until_finished()

        logger.info("Successfully saved checkpoint with optimizer at step %d", step)
        return checkpoint_manager

    except (OSError, IOError, FileNotFoundError, PermissionError) as e:
        logger.exception("Error writing checkpoint to disk: %s", e)
        raise
    except (ValueError, TypeError, AttributeError) as e:
        logger.exception("Error in checkpoint configuration: %s", e)
        raise


def load_checkpoint_with_optimizer(
    checkpoint_manager: ocp.CheckpointManager,
    model_template: nnx.Module,
    optimizer_template: nnx.Optimizer,
    step: int | None = None,
) -> tuple[nnx.Module | None, nnx.Optimizer | None, int | None]:
    """Load both model and optimizer state from checkpoint.

    Args:
        checkpoint_manager: The Orbax CheckpointManager instance.
        model_template: An NNX Module with the same structure as the saved model.
        optimizer_template: An NNX Optimizer with the same structure as saved.
        step: The specific step to restore. If None, restores the latest step.

    Returns:
        A tuple containing:
            - The restored model (or None if not found).
            - The restored optimizer (or None if not found).
            - The step number restored from (or None if not found).
    """
    try:
        if step is None:
            step = checkpoint_manager.latest_step()
            if step is None:
                logger.info(
                    "No checkpoints found in %s",
                    checkpoint_manager.directory,
                )
                return None, None, None

        logger.info("Loading checkpoint with optimizer from step %d...", step)

        model_state = nnx.state(model_template)
        optimizer_state = nnx.state(optimizer_template)

        restore_args = ocp.args.Composite(
            model=ocp.args.StandardRestore(model_state),
            optimizer=ocp.args.StandardRestore(optimizer_state),
        )

        restored_data = checkpoint_manager.restore(step, args=restore_args)

        nnx.update(model_template, restored_data["model"])
        nnx.update(optimizer_template, restored_data["optimizer"])

        logger.info("Successfully loaded checkpoint with optimizer from step %d", step)
        return model_template, optimizer_template, step

    except (FileNotFoundError, IOError, OSError, PermissionError) as e:
        logger.exception("Error accessing checkpoint files: %s", e)
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.exception("Error in checkpoint data or structure: %s", e)
        raise


def validate_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    model: nnx.Module,
    step: int,
    validation_data: Any,
    tolerance: float = 1e-5,
) -> bool:
    """Validate that a checkpoint loads correctly and produces consistent outputs.

    Args:
        checkpoint_manager: The Orbax CheckpointManager instance.
        model: The current model whose state was saved.
        step: The step number to validate.
        validation_data: Input data to test model outputs.
        tolerance: Maximum allowed difference between outputs.

    Returns:
        True if checkpoint is valid, False otherwise.
    """
    try:
        output_before = model(validation_data)

        all_steps = checkpoint_manager.all_steps()
        if step not in all_steps:
            logger.warning("Checkpoint at step %d does not exist", step)
            return False

        model_state = nnx.state(model)

        restore_args = ocp.args.Composite(model=ocp.args.StandardRestore(model_state))

        restored_data = checkpoint_manager.restore(step, args=restore_args)

        import jax.numpy as jnp

        nnx.update(model, restored_data["model"])
        output_after = model(validation_data)

        max_diff = float(jnp.max(jnp.abs(output_before - output_after)))

        if max_diff > tolerance:
            logger.warning("Checkpoint validation failed! Max diff: %s", max_diff)
            return False

        logger.info("Checkpoint at step %d validated successfully", step)
        return True

    except (OSError, ValueError, TypeError, KeyError) as e:
        logger.warning("Checkpoint validation failed with error: %s", e)
        return False


def recover_from_corruption(
    checkpoint_dir: str,
    model_template: nnx.Module,
) -> tuple[nnx.Module | None, int | None]:
    """Attempt to recover from corrupted checkpoints.

    Tries loading checkpoints from newest to oldest until one succeeds.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        model_template: An NNX Module with the same structure as saved model.

    Returns:
        A tuple containing:
            - The recovered model (or None if recovery failed).
            - The step number recovered from (or None if failed).
    """
    checkpoint_manager, _ = setup_checkpoint_manager(checkpoint_dir)

    all_steps = checkpoint_manager.all_steps()

    if not all_steps:
        logger.info("No checkpoints available for recovery")
        return None, None

    for step in sorted(all_steps, reverse=True):
        try:
            logger.info("Attempting to recover from step %d...", step)

            restored_model, loaded_step = load_checkpoint(
                checkpoint_manager,
                model_template,
                step=step,
            )

            if restored_model is not None:
                logger.info("Successfully recovered from step %d", loaded_step)
                return restored_model, loaded_step

        # Recovery tries every checkpoint — many failure modes are possible
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
            logger.warning("Failed to load step %d: %s", step, e)
            continue

    logger.info("Could not recover any checkpoint")
    return None, None
