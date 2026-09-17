"""The final batch of an epoch, under datarax 0.1.12.

A random-access source serves every batch at ``batch_size``; the rows past the end of an
epoch are padding, marked in the batch's ``valid_mask``. Training drops that batch
(``drop_last=True``, PyTorch's rule), so no padded row reaches a gradient step and the
steps per epoch are what the data holds; evaluation keeps it and scores the real rows only;
a learning-rate schedule whose horizon is not configured takes it from the run,
``num_epochs`` times the batches per epoch.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx

from artifex.generative_models.core.configuration import (
    DataConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from artifex.generative_models.training.loops import create_data_pipeline
from artifex.generative_models.training.trainer import Trainer


_N = 70
_BATCH = 16  # four full batches and a six-row tail
_FULL_BATCHES = _N // _BATCH
_LEARNING_RATE = 1e-3
_MIN_LR_RATIO = 0.1


class _Linear(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.dense = nnx.Linear(in_features=4, out_features=2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.dense(x)


def _data() -> dict[str, jax.Array]:
    return {
        "input": jax.random.normal(jax.random.key(1), (_N, 4)),
        "id": jnp.arange(_N, dtype=jnp.int32),
    }


def _source(*, shuffle: bool = False) -> MemorySource:
    return MemorySource(MemorySourceConfig(shuffle=shuffle), _data(), rngs=nnx.Rngs(0, shuffle=0))


def _config(scheduler: SchedulerConfig | None = None, num_epochs: int = 100) -> TrainingConfig:
    return TrainingConfig(
        name="final-batch",
        batch_size=_BATCH,
        num_epochs=num_epochs,
        optimizer=OptimizerConfig(name="adam", optimizer_type="adam", learning_rate=_LEARNING_RATE),
        scheduler=scheduler,
        save_frequency=10_000,
    )


def _loss_fn(
    model: nnx.Module, batch: dict[str, jax.Array], _rng: jax.Array, _step: jax.Array
) -> tuple[jax.Array, dict[str, jax.Array]]:
    prediction = model(batch["input"])
    return jnp.mean(prediction**2), {"ids": batch["id"]}


def _id_mean_loss_fn(
    _model: nnx.Module, batch: dict[str, jax.Array], _rng: jax.Array, _step: jax.Array
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """A loss that is the mean record id, so padding shows up in the average."""
    return jnp.mean(batch["id"].astype(jnp.float32)), {"rows": jnp.asarray(batch["id"].shape[0])}


def _trainer(
    tmp_path: Path,
    *,
    config: TrainingConfig | None = None,
    loss_fn: Any = _loss_fn,
    train_data_loader: Any = None,
) -> Trainer:
    return Trainer(
        _Linear(rngs=nnx.Rngs(0)),
        dataclasses.replace(config or _config(), checkpoint_dir=tmp_path / "ckpt"),
        train_data_loader=train_data_loader,
        rng=jax.random.key(0),
        loss_fn=loss_fn,
    )


def _cosine(cycle_length: int | None = None) -> SchedulerConfig:
    return SchedulerConfig(
        name="cosine",
        scheduler_type="cosine",
        min_lr_ratio=_MIN_LR_RATIO,
        cycle_length=cycle_length,
    )


def _ids_per_epoch(trainer: Trainer) -> list[list[int]]:
    ids = [np.asarray(m["ids"]).tolist() for m in trainer.train_metrics]
    steps = trainer.steps_per_epoch
    assert steps is not None
    return [
        [i for batch in ids[e * steps : (e + 1) * steps] for i in batch]
        for e in range(len(ids) // steps)
    ]


class TestCreateDataPipeline:
    def test_the_default_serves_the_padded_batch_with_its_mask(self) -> None:
        pipeline = create_data_pipeline(_source(), batch_size=_BATCH)

        batches = list(pipeline)

        assert len(batches) == _FULL_BATCHES + 1
        assert all("valid_mask" in batch for batch in batches)
        assert sum(int(batch["valid_mask"].sum()) for batch in batches) == _N

    def test_drop_last_serves_only_full_batches(self) -> None:
        pipeline = create_data_pipeline(_source(), batch_size=_BATCH, drop_last=True)

        assert len(pipeline) == _FULL_BATCHES
        batches = list(pipeline)
        assert len(batches) == _FULL_BATCHES
        assert all(bool(batch["valid_mask"].all()) for batch in batches)


class TestTrainingDropsTheRaggedBatch:
    def test_no_padded_row_reaches_a_step_and_the_steps_per_epoch_are_recorded(
        self, tmp_path: Path
    ) -> None:
        trainer = _trainer(tmp_path)

        trainer.train(_data(), num_epochs=2, batch_size=_BATCH)

        assert trainer.steps_per_epoch == _FULL_BATCHES
        assert trainer.step == 2 * _FULL_BATCHES
        for epoch_ids in _ids_per_epoch(trainer):
            assert len(epoch_ids) == _FULL_BATCHES * _BATCH
            assert len(set(epoch_ids)) == len(epoch_ids)  # no row twice: nothing wrapped

    def test_the_steps_per_epoch_are_unknown_before_training(self, tmp_path: Path) -> None:
        assert _trainer(tmp_path).steps_per_epoch is None

    def test_fewer_records_than_a_batch_is_refused(self, tmp_path: Path) -> None:
        trainer = _trainer(tmp_path)
        data = {key: value[: _BATCH - 1] for key, value in _data().items()}

        with pytest.raises(ValueError, match="fewer than one batch"):
            trainer.train(data, num_epochs=1, batch_size=_BATCH)


class TestScheduleHorizon:
    def test_a_derived_horizon_is_the_run_length(self, tmp_path: Path) -> None:
        """``num_epochs`` of the run times the batches per epoch, not a guessed constant."""
        trainer = _trainer(tmp_path, config=_config(scheduler=_cosine()))

        trainer.train(_data(), num_epochs=3, batch_size=_BATCH)

        assert trainer.schedule_horizon == 3 * _FULL_BATCHES
        assert trainer.schedule is not None
        assert float(trainer.schedule(0)) == pytest.approx(_LEARNING_RATE)
        assert float(trainer.schedule(trainer.schedule_horizon)) == pytest.approx(
            _LEARNING_RATE * _MIN_LR_RATIO
        )

    def test_a_step_before_training_with_a_derived_horizon_is_refused(self, tmp_path: Path) -> None:
        trainer = _trainer(tmp_path, config=_config(scheduler=_cosine()))
        batch = next(iter(create_data_pipeline(_source(), batch_size=_BATCH, drop_last=True)))

        with pytest.raises(ValueError, match="cycle_length"):
            trainer.train_step(batch)
        with pytest.raises(ValueError, match="cycle_length"):
            trainer.checkpoint_state()

    def test_an_explicit_horizon_is_used_as_given(self, tmp_path: Path) -> None:
        trainer = _trainer(tmp_path, config=_config(scheduler=_cosine(cycle_length=50)))

        assert trainer.schedule is not None
        assert trainer.schedule_horizon == 50
        assert float(trainer.schedule(50)) == pytest.approx(_LEARNING_RATE * _MIN_LR_RATIO)
        trainer.train(_data(), num_epochs=1, batch_size=_BATCH)
        assert trainer.schedule_horizon == 50

    def test_a_schedule_without_a_horizon_is_built_at_construction(self, tmp_path: Path) -> None:
        constant = SchedulerConfig(name="constant", scheduler_type="constant")
        trainer = _trainer(tmp_path, config=_config(scheduler=constant))

        assert trainer.optimizer is not None
        assert trainer.schedule_horizon is None
        batch = next(iter(create_data_pipeline(_source(), batch_size=_BATCH, drop_last=True)))
        assert "loss" in trainer.train_step(batch)


class TestEvaluationScoresTheRealRowsOnly:
    def test_the_average_is_over_the_records_not_the_padding(self, tmp_path: Path) -> None:
        trainer = _trainer(tmp_path, loss_fn=_id_mean_loss_fn)

        metrics = trainer.evaluate(_data(), batch_size=_BATCH)

        # The mean id over the 70 records; with the six padded rows it would be higher.
        assert metrics["loss"] == pytest.approx(float(np.mean(np.arange(_N))))

    def test_a_divisible_split_is_unchanged(self, tmp_path: Path) -> None:
        trainer = _trainer(tmp_path, loss_fn=_id_mean_loss_fn)
        data = {key: value[: 4 * _BATCH] for key, value in _data().items()}

        metrics = trainer.evaluate(data, batch_size=_BATCH)

        assert metrics["loss"] == pytest.approx(float(np.mean(np.arange(4 * _BATCH))))


class TestTrainEpochWithALoader:
    @staticmethod
    def _loader(batch_size: int) -> Iterator[dict[str, jax.Array]]:
        data = _data()
        return iter(
            {key: value[i * batch_size : (i + 1) * batch_size] for key, value in data.items()}
            for i in range(_FULL_BATCHES)
        )

    def test_an_epoch_runs_until_the_loader_is_exhausted(self, tmp_path: Path) -> None:
        trainer = _trainer(tmp_path, train_data_loader=self._loader)

        metrics = trainer.train_epoch()

        assert trainer.step == _FULL_BATCHES
        assert "loss" in metrics

    def test_an_explicit_step_count_stops_the_epoch_early(self, tmp_path: Path) -> None:
        trainer = _trainer(tmp_path, train_data_loader=self._loader)

        trainer.train_epoch(steps=2)

        assert trainer.step == 2


def test_data_config_carries_no_drop_remainder() -> None:
    """The final-batch policy is the pipeline's ``drop_last``; the field nothing read is gone."""
    assert "drop_remainder" not in DataConfig.__dataclass_fields__
