"""The trainer's checkpoint is substrax's format 3: named items beside a metadata record."""

from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from substrax.checkpoint import CheckpointNotFoundError, OrbaxCheckpointStore

from artifex.generative_models.core.configuration import (
    ExtensionConfig,
    OptimizerConfig,
    TrainingConfig,
)
from artifex.generative_models.extensions.base import Extension
from artifex.generative_models.training.trainer import Trainer


ITEMS = {"model", "optimizer", "rng", "extensions"}


def objective(model, batch, rng, step):
    """A small differentiable objective exercising native optimizer state."""
    del rng, step
    loss = jnp.mean(jnp.square(model(batch["x"]) - batch["y"]))
    return loss, {"loss": loss}


def assert_same(left, right):
    """Compare full pytree structure and values, including typed RNG keys."""
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for first, second in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        assert jnp.array_equal(first, second)


def config(checkpoint_dir: Path | None = None, **fields) -> TrainingConfig:
    """A training configuration for adam, checkpointing where ``checkpoint_dir`` says."""
    return TrainingConfig(
        name="test",
        optimizer=OptimizerConfig(name="adam", optimizer_type="adam", learning_rate=1e-3),
        checkpoint_dir=checkpoint_dir,
        **fields,
    )


def build_trainer(
    checkpoint_dir: Path | None = None, *, workdir: Path | None = None, **kwargs
) -> Trainer:
    """A linear model under adam, checkpointing where the configuration says."""
    return Trainer(
        nnx.Linear(2, 1, rngs=nnx.Rngs(3)),
        kwargs.pop("training_config", None) or config(checkpoint_dir),
        loss_fn=objective,
        workdir=None if workdir is None else str(workdir),
        **kwargs,
    )


BATCH = {"x": jnp.ones((2, 2)), "y": jnp.ones((2, 1))}
DATA = {"x": jnp.ones((8, 2)), "y": jnp.ones((8, 1))}


@pytest.mark.parametrize("with_extension", [False, True])
def test_detached_restore_then_apply_preserves_next_training_step(tmp_path, with_extension):
    """Inspect the record without mutation, then resume the exact optimizer trajectory."""
    extension = Extension(ExtensionConfig(name="test"), rngs=nnx.Rngs(7))
    trainer = build_trainer(
        tmp_path / "native", extensions={"test": extension} if with_extension else {}
    )
    trainer.train_step(BATCH)
    saved = trainer.checkpoint_state()
    assert set(saved) == ITEMS
    with OrbaxCheckpointStore(tmp_path / "composed") as store:
        store.save(1, saved, extra={"request": "expected"})
        trainer.train_step(BATCH)
        expected_next = trainer.checkpoint_state()
        checkpoint = store.restore(1, templates=trainer.checkpoint_state())
        assert checkpoint.metadata.extra["request"] == "expected"
        assert trainer.step == 2
        assert_same(trainer.checkpoint_state(), expected_next)
        trainer.apply_checkpoint_state(checkpoint.items, step=checkpoint.step)
    assert trainer.step == 1
    assert_same(trainer.checkpoint_state(), saved)
    trainer.train_step(BATCH)
    assert_same(trainer.checkpoint_state(), expected_next)
    trainer.save_checkpoint()
    trainer.train_step(BATCH)
    trainer.load_checkpoint(step=2)
    assert trainer.step == 2
    assert_same(trainer.checkpoint_state(), expected_next)


def test_save_checkpoint_writes_the_items_and_names_artifex_as_the_producer(tmp_path):
    trainer = build_trainer(tmp_path / "ckpt")
    trainer.train_step(BATCH)

    path = trainer.save_checkpoint()

    assert isinstance(path, Path)
    assert path.is_relative_to(tmp_path / "ckpt")
    with OrbaxCheckpointStore(tmp_path / "ckpt") as store:
        metadata = store.read_metadata(1)
    assert set(metadata.items) == ITEMS
    assert metadata.step == 1
    assert metadata.producer is not None
    assert (metadata.producer.name, metadata.producer.version) == (
        "artifex",
        version("avitai-artifex"),
    )


def test_load_checkpoint_of_a_missing_step_raises_the_store_error(tmp_path):
    trainer = build_trainer(tmp_path / "ckpt")
    trainer.train_step(BATCH)
    trainer.save_checkpoint()

    with pytest.raises(CheckpointNotFoundError):
        trainer.load_checkpoint(step=7)


def test_load_checkpoint_without_any_checkpoint_is_a_file_not_found(tmp_path):
    """The store's missing-step error is a FileNotFoundError, and so is an empty directory."""
    trainer = build_trainer(tmp_path / "empty")

    with pytest.raises(FileNotFoundError, match="empty"):
        trainer.load_checkpoint()
    assert issubclass(CheckpointNotFoundError, FileNotFoundError)


class TestCheckpointDirectory:
    """The configuration names the directory, substrax's resolver fills the default, and
    nothing creates it before a save."""

    def test_the_configured_directory_wins(self, tmp_path):
        trainer = build_trainer(tmp_path / "explicit", workdir=tmp_path / "run")

        assert trainer.checkpoint_dir == (tmp_path / "explicit").resolve()

    def test_the_run_directory_holds_a_checkpoints_subdirectory(self, tmp_path):
        trainer = build_trainer(workdir=tmp_path / "run")

        assert trainer.checkpoint_dir == (tmp_path / "run" / "checkpoints").resolve()

    def test_neither_given_means_no_checkpointing(self, monkeypatch, tmp_path):
        """Without a directory or a workdir the trainer saves nothing and leaves the cwd alone."""
        monkeypatch.chdir(tmp_path)
        trainer = build_trainer(training_config=config(save_frequency=1))
        assert trainer.checkpoint_dir is None

        trainer.train(DATA, num_epochs=1, batch_size=2)

        assert not (tmp_path / "checkpoints").exists()
        assert trainer.step == 4
        with pytest.raises(ValueError, match="no checkpoint directory"):
            trainer.save_checkpoint()
        with pytest.raises(ValueError, match="no checkpoint directory"):
            trainer.load_checkpoint()

    def test_nothing_is_created_before_the_first_save(self, tmp_path):
        trainer = build_trainer(tmp_path / "later")

        assert not (tmp_path / "later").exists()
        trainer.train_step(BATCH)
        trainer.save_checkpoint()
        assert (tmp_path / "later").is_dir()

    def test_the_trainer_takes_no_directory_or_cadence_of_its_own(self, tmp_path):
        """The configuration is the one owner of where and how often checkpoints are written."""
        for keyword in ({"checkpoint_dir": str(tmp_path)}, {"save_interval": 5}):
            with pytest.raises(TypeError, match=next(iter(keyword))):
                build_trainer(tmp_path / "ckpt", **keyword)


class TestConfiguredCadenceAndRetention:
    """``save_frequency`` and ``max_checkpoints`` from the configuration drive the store."""

    def test_train_saves_every_save_frequency_steps(self, tmp_path):
        trainer = build_trainer(training_config=config(tmp_path / "ckpt", save_frequency=2))

        trainer.train(DATA, num_epochs=1, batch_size=2)  # four steps

        with OrbaxCheckpointStore(tmp_path / "ckpt") as store:
            assert store.list_steps() == [2, 4]

    def test_train_epoch_saves_every_save_frequency_steps(self, tmp_path):
        trainer = build_trainer(
            training_config=config(tmp_path / "ckpt", save_frequency=3, batch_size=2),
            train_data_loader=lambda batch_size: iter(
                [{"x": jnp.ones((batch_size, 2)), "y": jnp.ones((batch_size, 1))}] * 6
            ),
        )

        trainer.train_epoch()

        with OrbaxCheckpointStore(tmp_path / "ckpt") as store:
            assert store.list_steps() == [3, 6]

    def test_the_store_keeps_max_checkpoints(self, tmp_path):
        trainer = build_trainer(training_config=config(tmp_path / "ckpt", max_checkpoints=2))
        for _ in range(3):
            trainer.train_step(BATCH)
            trainer.save_checkpoint()

        with OrbaxCheckpointStore(tmp_path / "ckpt") as store:
            assert store.list_steps() == [2, 3]
